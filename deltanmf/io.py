import numpy as np
import anndata as ad


def h5ad_to_npy(
    h5ad_path,
    ntc_key="NTC",
    condition_key="condition",
    case_key=None,
    layer=None,
):
    adata = ad.read_h5ad(h5ad_path)

    if condition_key not in adata.obs:
        raise ValueError(f"adata.obs must contain '{condition_key}'")

    cond = adata.obs[condition_key].astype(str).to_numpy()
    mask_ntc = cond == str(ntc_key)
    if mask_ntc.sum() == 0:
        raise ValueError(f"No cells matched NTC: obs[{condition_key}] == {ntc_key}")

    if case_key is None:
        mask_spec = ~mask_ntc
    else:
        mask_spec = cond == str(case_key)
        if mask_spec.sum() == 0:
            raise ValueError(f"No cells matched case: obs[{condition_key}] == {case_key}")

    X = adata.layers[layer] if layer is not None else adata.X
    X_ntc = X[mask_ntc, :]
    X_spec = X[mask_spec, :] if mask_spec is not None else None

    if hasattr(X_ntc, "toarray"):
        X_ntc = X_ntc.toarray()
    if X_spec is not None and hasattr(X_spec, "toarray"):
        X_spec = X_spec.toarray()

    gene_names = np.asarray(adata.var_names).astype(str)

    X_ntc = X_ntc.T.astype(np.float32, copy=False)  # genes x cells
    if X_spec is not None:
        X_spec = X_spec.T.astype(np.float32, copy=False)

    return X_ntc, X_spec, gene_names
