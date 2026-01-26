import json
from pathlib import Path
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

H5_PATH = Path("/hpc/group/singhlab/rawdata/transcriptformer/homo_sapiens_gene.h5")

def _to_py_str_array(ds):
    arr = np.array(ds)
    if arr.dtype.kind in ("S", "O"):
        arr = np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])
    return arr.astype(str).tolist()

def main():
    with h5py.File(H5_PATH, "r") as h5:
        arrays = h5["arrays"]
        keys   = _to_py_str_array(h5["keys"])

        N = len(keys)
        D = arrays[keys[0]].shape[0]
        print(f"[info] genes={N}, dim={D}, dtype={arrays[keys[0]].dtype}")

        # Build E: shape (N, D) where rows are genes, columns are embedding dims
        E = np.empty((N, D), dtype=np.float32)
        for i, ensg in enumerate(keys):
            E[i, :] = arrays[ensg][:]
            if (i + 1) % 10000 == 0:
                print(f"  loaded {i+1}/{N}")

    # Standardize each embedding dimension across genes (zero mean, unit variance)
    # After this, column j has mean ~0 and std ~1 over all genes
    print("Standardizing embedding dimensions with StandardScaler ...")
    scaler = StandardScaler(with_mean=True, with_std=True)
    E_std = scaler.fit_transform(E).astype(np.float32)

    # Pearson correlation across embedding dimensions (rows are variables = genes)
    print("Computing CTS = corrcoef(E_std) ...")
    CTS = np.corrcoef(E_std, rowvar=True).astype(np.float32)

    print("Building S_E with ReLU and zero diagonal ...")
    S_E = np.maximum(0.0, CTS)
    np.fill_diagonal(S_E, 0.0)
    S_E = S_E.astype(np.float32)

    # Save outputs
    script_dir = Path(__file__).parent
    np.save(script_dir / "S_E_relu.npy", S_E)


    gene_to_index = {ensg: i for i, ensg in enumerate(keys)}
    with open(script_dir / "genes_order.json", "w") as f:
        json.dump(keys, f)


if __name__ == "__main__":
    main()
