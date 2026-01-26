import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed

def load_scgpt_model(model_dir):
    set_seed(42)
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    
    model_config_file = model_dir / "args.json"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    
    model = TransformerModel(
        len(vocab),
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        vocab=vocab,
        pad_value=-2,
        n_input_bins=51,
    )
    
    model_file = model_dir / "best_model.pt"
    try:
        model.load_state_dict(torch.load(model_file))
        print(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model.to(device)
    return model, vocab, device

def create_and_save_laplacian(model_dir, scgpt_gene_info_path):
    print("Loading scGPT model...")
    model, vocab, device = load_scgpt_model(Path(model_dir))
    gene2idx = vocab.get_stoi()
    
    special_tokens = {"<pad>", "<cls>", "<eoc>"}
    filtered_gene2idx = {k: v for k, v in gene2idx.items() if k not in special_tokens}
    
    print(f"Getting embeddings for {len(filtered_gene2idx)} gene symbols...")
    gene_ids = torch.tensor(list(filtered_gene2idx.values()), dtype=torch.long).to(device)
    
    with torch.no_grad():
        gene_embeddings = model.encoder(gene_ids)
    gene_embeddings = gene_embeddings.detach().cpu().numpy()
    
    print("Computing CTS matrix...")
    cts_matrix = np.corrcoef(gene_embeddings)
    
    S_E = np.maximum(0, cts_matrix)
    np.fill_diagonal(S_E, 0)
    
    print("Saving matrices...")
        # Translate gene symbols to Ensembl IDs in the EXACT row/column order of S_E
    print("Translating gene symbols to Ensembl IDs and saving in-order list...")

    gene_info_df = pd.read_csv(scgpt_gene_info_path)
    symbol_to_ensg = pd.Series(gene_info_df.feature_id.values,
                            index=gene_info_df.feature_name).to_dict()

    # Reconstruct symbol order used to build embeddings/S_E
    idx_list = list(filtered_gene2idx.values())
    inv_idx_to_symbol = {v: k for k, v in filtered_gene2idx.items()}
    symbols_in_order = [inv_idx_to_symbol[i] for i in idx_list]

    # Map to Ensembl; drop any symbols we cannot map
    ensg_in_order = [symbol_to_ensg.get(sym) for sym in symbols_in_order]
    keep_idx = [i for i, e in enumerate(ensg_in_order) if e is not None]

    S_E_kept = S_E[np.ix_(keep_idx, keep_idx)]
    ensg_kept = [ensg_in_order[i] for i in keep_idx]

    # Now save in-order outputs (filtered S_E, ordered Ensembl list)
    script_dir = Path(__file__).parent
    
    np.save(script_dir / "S_E_relu.npy", S_E_kept)
    with open(script_dir / "genes_order.json", "w") as f:
        json.dump(ensg_kept, f)


if __name__ == '__main__':
    model_dir = "/hpc/group/singhlab/user/agk21/projects/deltaNMF/data/scGPT/scGPT_human"
    scgpt_gene_info_path = "/hpc/group/singhlab/user/agk21/projects/deltaNMF/data/scGPT/scGPT_gene_info.csv"
    
    create_and_save_laplacian(
        model_dir=model_dir,
        scgpt_gene_info_path=scgpt_gene_info_path
    )