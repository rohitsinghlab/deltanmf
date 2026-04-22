from pathlib import Path
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deltanmf.api import run_twostage_deltanmf
from deltanmf.io import h5ad_to_npy


def main():
    repo_root = Path(__file__).resolve().parents[1]

    h5ad_path = repo_root / "resources" / "test_data.h5ad"
    out = repo_root / "resources" / "out"
    out.mkdir(parents=True, exist_ok=True)

    X_ntc, X_spec, gene_names, ntc_barcodes, spec_barcodes = h5ad_to_npy(
        h5ad_path,
        ntc_key="NTC",
        condition_key="condition",
        case_key=None,
        layer=None,
    )

    resources = repo_root / "resources"
    S_E_PATH = resources / "scgpt" / "S_E_relu.npy" # can replace scgpt with transcriptformer
    S_E_GENES_PATH = resources / "scgpt" / "genes_order.json" # can replace scgpt with transcriptformer
    USE_FM = True

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing: {h5ad_path}")
    if USE_FM:
        if not S_E_PATH.exists():
            raise FileNotFoundError(f"Missing: {S_E_PATH}")
        if not S_E_GENES_PATH.exists():
            raise FileNotFoundError(f"Missing: {S_E_GENES_PATH}")
    else:
        S_E_PATH = None
        S_E_GENES_PATH = None

    res = run_twostage_deltanmf(
        X_ntc, X_spec, gene_names, S_E_PATH, S_E_GENES_PATH,
        K_stage1=30, K_stage2=60,
        MIN_CELLS=10,
        use_fm=USE_FM,
        stage1_rel_alpha=(0.05 if USE_FM else 0.0),
        stage2_rel_alpha=(0.05 if USE_FM else 0.0),
        stage2_rel_gamma=0.0,
        stage1_max_iter=10000,
        stage2_max_iter=10000,
        FM_NONNEG="softplus",
        FM_SOFTPLUS_BETA=5.0,
        lr=0.01,
        ntc_barcodes=ntc_barcodes,
        specific_barcodes=spec_barcodes,
    )

    np.save(out / "W_stage1.npy", res["W_stage1"])
    np.save(out / "H_stage1.npy", res["H_stage1"])
    np.save(out / "W_stage2.npy", res["W_stage2"])
    np.save(out / "H_stage2.npy", res["H_stage2"])
    np.save(out / "gene_names_aligned.npy", res["gene_names_aligned"])
    np.savetxt(out / "gene_names_aligned.tsv", res["gene_names_aligned"], fmt="%s")
    np.save(out / "ntc_cell_ids.npy", res["ntc_cell_ids"])
    np.savetxt(out / "ntc_cell_ids.tsv", res["ntc_cell_ids"], fmt="%s")
    np.save(out / "specific_cell_ids.npy", res["specific_cell_ids"])
    np.savetxt(out / "specific_cell_ids.tsv", res["specific_cell_ids"], fmt="%s")

    gene_names_aligned = res["gene_names_aligned"]
    K1 = res["W_stage1"].shape[1]
    K2 = res["W_stage2"].shape[1]
    baseline_names = [f"baseline_{i}" for i in range(K1)]
    case_names = [f"case_{i}" for i in range(K2)]
    program_names = baseline_names + case_names
    W_combined = np.hstack([res["W_stage1"], res["W_stage2"]])

    # H matrices: cells (rows) x programs (columns)
    pd.DataFrame(res["H_stage1"].T, index=res["ntc_cell_ids"], columns=baseline_names).to_csv(out / "H_stage1.csv")
    pd.DataFrame(res["H_stage2"].T, index=res["specific_cell_ids"], columns=program_names).to_csv(out / "H_stage2.csv")

    # W matrices: genes (rows) x programs (columns)
    pd.DataFrame(res["W_stage1"], index=gene_names_aligned, columns=baseline_names).to_csv(out / "W_stage1.csv")
    pd.DataFrame(W_combined, index=gene_names_aligned, columns=program_names).to_csv(out / "W_combined.csv")


if __name__ == "__main__":
    main()
