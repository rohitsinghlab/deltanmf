"""
GPU-Accelerated Consensus NMF

Implements consensus NMF following Kotliar et al. (2019) with GPU acceleration.
Used to provide robust initialization for DeltaNMF
"""

import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from tqdm.auto import trange
import warnings

# Utility Functions
def _torch_device(use_gpu=True):
    """Selects the appropriate torch device."""
    if use_gpu and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        return torch.device("cuda")
    return torch.device("cpu")

def _to_dense_batch(X, col_idx):
    """Extracts a dense column-batch from a potentially sparse matrix."""
    if sp.issparse(X):
        if sp.isspmatrix_csr(X):
            X = X.tocsc(copy=False)
        return X[:, col_idx].toarray()
    return np.asarray(X[:, col_idx], dtype=np.float32)

def _normalize_W_L1(W, eps=1e-12):
    """L1 normalizes columns of W in-place and returns the column sums."""
    s = W.sum(dim=0).clamp_min(eps)
    W /= s.unsqueeze(0)
    return s

def _check_nonneg(X):
    """Checks if the input matrix is non-negative."""
    min_val = X.data.min() if sp.issparse(X) else X.min()
    if min_val < 0:
        raise ValueError("Input X contains negative values.")

# Core NMF Solvers
@torch.no_grad()
def _solve_H_batch_mu(W, Xb, Hb=None, inner_iters=15, eps=1e-9, gen=None):
    """Solves for H for a single batch using Multiplicative Update."""
    k, b = W.shape[1], Xb.shape[1]
    
    if Hb is None or Hb.shape[1] != b:
        Hb = torch.rand(k, b, device=W.device, dtype=W.dtype, generator=gen) if gen is not None \
             else torch.rand(k, b, device=W.device, dtype=W.dtype)

    Wt = W.transpose(0, 1)
    WtX = Wt @ Xb
    WtW = Wt @ W
    for _ in range(inner_iters):
        denom = (WtW @ Hb).clamp_min_(eps)
        Hb.mul_(WtX / denom)
    return Hb.clamp_min_(0)

def nmf_gpu_minibatch(
    X, k, epochs=50, batch_size=4096, h_inner_iters=15, seed=1337,
    use_gpu=True, dtype=torch.float32, return_H=True,
    tol=1e-4, verbose=False
):
    """Performs one full run of mini-batch NMF."""
    _check_nonneg(X)
    rng = np.random.default_rng(seed)
    device = _torch_device(use_gpu)

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    n_genes, n_cells = X.shape
    W = torch.rand(n_genes, k, device=device, dtype=dtype, generator=g)
    _normalize_W_L1(W)

    H_host = np.zeros((k, n_cells), dtype=np.float32) if return_H else None

    last_obj = np.inf
    history = {"epoch": [], "proxy_objective": []}
    column_indices = np.arange(n_cells, dtype=np.int64)

    iterator = trange(epochs, desc="NMF Replicate", leave=False) if verbose else range(epochs)

    for epoch in iterator:
        rng.shuffle(column_indices)
        
        XHt = torch.zeros(n_genes, k, device=device, dtype=dtype)
        HHt = torch.zeros(k, k, device=device, dtype=dtype)
        
        Hb_cache = None

        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            idx = column_indices[start:end]
            Xb = torch.from_numpy(_to_dense_batch(X, idx)).to(device=device, dtype=dtype)
            
            Hb_cache = _solve_H_batch_mu(W, Xb, Hb_cache, inner_iters=h_inner_iters, gen=g)

            if return_H:
                H_host[:, idx] = Hb_cache.detach().cpu().numpy()

            XHt.add_(Xb @ Hb_cache.transpose(0, 1))
            HHt.add_(Hb_cache @ Hb_cache.transpose(0, 1))
            del Xb

        denom = (W @ HHt).clamp_min_(1e-9)
        W.mul_(XHt / denom).clamp_min_(0)

        scale_factors = _normalize_W_L1(W)
        if return_H:
            H_host *= scale_factors.detach().cpu().numpy().reshape(k, 1)

        tr_WH_XHt = float((W * XHt).sum().detach().cpu().item())
        proxy_obj = float((W @ HHt @ W.transpose(0, 1)).trace().detach().cpu().item()) - 2.0 * tr_WH_XHt
        
        history["epoch"].append(epoch)
        history["proxy_objective"].append(proxy_obj)

        rel_impr = (last_obj - proxy_obj) / (abs(last_obj) + 1e-12)
        if verbose:
            iterator.set_postfix(proxy_obj=f"{proxy_obj:.4e}", rel_impr=f"{rel_impr:.3e}")
            
        last_obj = proxy_obj

    W_out = W.detach().cpu().numpy().astype(np.float32)
    return W_out, H_host, {"history": history, "final_epoch": epoch}

def refit_H_given_W(
    X, W, batch_size=8192, h_inner_iters=20, use_gpu=True, dtype=torch.float32, seed=1337
):
    """Refits H for a fixed W. Expects X to be (genes, cells)."""
    _check_nonneg(X)
    device = _torch_device(use_gpu)
    n_genes, n_cells = X.shape
    k = W.shape[1]
    H_host = np.zeros((k, n_cells), dtype=np.float32)
    W_t = torch.from_numpy(W).to(device=device, dtype=dtype)
    Hb = None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        Xb = torch.from_numpy(_to_dense_batch(X, np.arange(start, end))).to(device=device, dtype=dtype)
        Hb = _solve_H_batch_mu(W_t, Xb, Hb, inner_iters=h_inner_iters, gen=g)
        H_host[:, start:end] = Hb.detach().cpu().numpy()
        del Xb
    return H_host

def consensus_nmf_gpu(
    X, k, n_runs=20, epochs=40, batch_size=4096,
    use_gpu=True, seed=1337, verbose=False,
    gene_names=None, cell_names=None,
    consensus_mode='median',
    density_threshold_quantile=0.95,
    local_neighborhood_size=0.3
):
    """
    Performs consensus NMF with robust, quantile-based density filtering.
    """
    n_genes, n_cells = X.shape
    rng = np.random.default_rng(seed)
    all_W_columns = []

    iterator = trange(n_runs, desc="Consensus Runs") if verbose else range(n_runs)
    for r in iterator:
        run_seed = int(rng.integers(0, 2**31 - 1))
        W, _, _ = nmf_gpu_minibatch(
            X, k=k, epochs=epochs, batch_size=batch_size,
            seed=run_seed, use_gpu=use_gpu, return_H=False, verbose=False
        )
        all_W_columns.extend([W[:, j] for j in range(k)])

    # Quantile-based Density Filtering
    M_unfiltered = np.stack(all_W_columns, axis=0)
    M_l2_unfiltered = M_unfiltered / (np.linalg.norm(M_unfiltered, axis=1, keepdims=True) + 1e-12)

    n_neighbors = int(local_neighborhood_size * n_runs)
    if n_neighbors < 1: n_neighbors = 1
    
    dist_matrix = euclidean_distances(M_l2_unfiltered)
    dist_matrix.sort(axis=1)
    local_densities = dist_matrix[:, 1:n_neighbors+1].mean(axis=1)
    
    # Calculate the adaptive threshold based on the specified quantile
    threshold_value = np.quantile(local_densities, density_threshold_quantile)
    density_filter_mask = local_densities < threshold_value
    
    M_filtered = M_l2_unfiltered[density_filter_mask]
    
    num_filtered = M_unfiltered.shape[0] - M_filtered.shape[0]

    if M_filtered.shape[0] < k:
        raise RuntimeError(f"Fewer programs ({M_filtered.shape[0]}) remain after filtering than k ({k}). Try increasing n_runs or density_threshold_quantile.")

    # Clustering and Consensus on Filtered Data
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=seed)
    labels_filtered = kmeans.fit_predict(M_filtered)
    
    full_labels = np.full(M_unfiltered.shape[0], -1, dtype=int)
    full_labels[density_filter_mask] = labels_filtered
    

    # Build Consensus W matrix (medians computed in the same L2 space used for clustering)
    W_consensus = np.zeros((n_genes, k), dtype=np.float32)
    for cluster_idx in range(k):
        mask = full_labels == cluster_idx
        if np.any(mask):
            # Use the L2-normalized rows for aggregation (same space used for KMeans)
            cluster_programs = M_l2_unfiltered[mask]
            if consensus_mode == 'median':
                w = np.median(cluster_programs, axis=0)
            else:  # 'mean'
                w = cluster_programs.mean(axis=0)
        else:
            warnings.warn(f"Cluster {cluster_idx} was empty. Using nearest point to centroid as fallback.")
            centroid = kmeans.cluster_centers_[cluster_idx]    
            nearest_idx = np.argmin(np.linalg.norm(M_filtered - centroid, axis=1))
            w = M_filtered[nearest_idx]
        W_consensus[:, cluster_idx] = w

    W_consensus /= (np.sum(W_consensus, axis=0, keepdims=True) + 1e-12)

    # Refit H and Finalize
    H_consensus = refit_H_given_W(X, W_consensus, batch_size=max(batch_size, 8192), use_gpu=use_gpu)

    error = 0.0
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        X_batch = _to_dense_batch(X, np.arange(start, end))
        recon_batch = W_consensus @ H_consensus[:, start:end]
        error += np.sum((X_batch - recon_batch)**2)
    
    info = {"reconstruction_error": error, "k": k, "num_filtered_programs": num_filtered}
    
    if gene_names is not None and cell_names is not None:
        gep_labels = [f"GEP_{i+1}" for i in range(k)]
        W_out = pd.DataFrame(W_consensus, index=gene_names, columns=gep_labels)
        H_out = pd.DataFrame(H_consensus, index=gep_labels, columns=cell_names)
        return W_out, H_out, info
    else:
        return W_consensus, H_consensus, info
