"""
Data Preprocessing for DeltaNMF

Functions for loading and normalizing single-cell data from various formats
"""

import numpy as np
import pandas as pd
import json
from scipy.io import mmread
from pathlib import Path
import anndata as ad
from scipy.sparse import csr_matrix

def load_from_h5ad(h5ad_path, s_e_path, s_e_genes_path, control_key='target_gene', control_values=['negative-control', 'safe-targeting'],verbose=False):
    """
    Loads data from a single h5ad file and an external S_E matrix.
    Returns the data in the format expected by the parameter sweep script.
    """
    h5ad_path = Path(h5ad_path) 
    adata = ad.read_h5ad(h5ad_path)
    
    # Split the matrix back into NTC and Specific based on the target_gene metadata
    ntc_mask = adata.obs[control_key].isin(control_values)
    X_ntc = adata[ntc_mask, :].X.T.toarray() if isinstance(adata.X, csr_matrix) else adata[ntc_mask, :].X.T
    X_specific = adata[~ntc_mask, :].X.T.toarray() if isinstance(adata.X, csr_matrix) else adata[~ntc_mask, :].X.T
     
    gene_names = adata.var_names.tolist()

    # Load and align S_E matrix based on the h5ad gene names
    # Find common genes and align all matrices
    S_E_full = np.load(s_e_path)
    with open(s_e_genes_path, 'r') as f:
        s_e_full_genes_list = list(json.load(f).keys())
    s_e_full_gene_to_idx = {gene: i for i, gene in enumerate(s_e_full_genes_list)}
    common_genes = [gene for gene in gene_names if gene in s_e_full_gene_to_idx]
    
    x_indices = [gene_names.index(gene) for gene in common_genes]
    s_e_indices = [s_e_full_gene_to_idx[gene] for gene in common_genes]
    
    X_ntc_aligned = X_ntc[x_indices, :]
    X_specific_aligned = X_specific[x_indices, :]
    S_E_aligned = S_E_full[np.ix_(s_e_indices, s_e_indices)]
    
    return X_ntc_aligned, X_specific_aligned, S_E_aligned, common_genes


def load_data_onestage(X, gene_names, s_e_path, s_e_genes_path, min_cells=215, additional_genes_to_remove=None, verbose=False):
    hvg_gene_names_full = gene_names

    if min_cells > 0:
        gene_cell_counts = (X > 0).sum(axis=1)
        mask = gene_cell_counts >= min_cells
    else:
        mask = np.ones(len(hvg_gene_names_full), dtype=bool)

    if additional_genes_to_remove:
        remove = np.isin(hvg_gene_names_full, additional_genes_to_remove)
        mask &= ~remove
    
    if mask.dtype != bool or mask.shape[0] != hvg_gene_names_full.shape[0]:
        raise ValueError(f"Mask shape/type mismatch: mask {mask.shape} bool? {mask.dtype==bool}, "
                         f"genes {hvg_gene_names_full.shape}")

    # apply mask once to keep everything aligned
    X_aligned = X[mask, :]
    hvg_gene_names = hvg_gene_names_full[mask].tolist()
    print(f"DEBUG: Genes after min_cells filter: {len(hvg_gene_names)}")
    
    S_E_full = np.load(s_e_path, mmap_mode="r")
    with open(s_e_genes_path, 'r') as f:
        s_e_full_genes_list = json.load(f)
    
    print(f"DEBUG: Genes loaded from S_E_GENES_PATH: {len(s_e_full_genes_list)}")
   
    # choose desired gene order
    x_index = pd.Index(hvg_gene_names)
    se_index = pd.Index(s_e_full_genes_list)
    keep = x_index.intersection(se_index, sort=False)

    final_genes = keep.tolist()

    if len(final_genes) == 0:
        raise ValueError("No genes remain after intersection with S_E; check gene naming / filters.")

    x_gene_to_idx  = {g: i for i, g in enumerate(hvg_gene_names)}
    se_gene_to_idx = {g: i for i, g in enumerate(s_e_full_genes_list)}
    x_indices  = np.fromiter((x_gene_to_idx[g]  for g in final_genes), dtype=np.int64)
    se_indices = np.fromiter((se_gene_to_idx[g] for g in final_genes), dtype=np.int64)

    X_aligned = X[x_indices, :]
    S_E_aligned = S_E_full[np.ix_(se_indices, se_indices)]
    
    return X_aligned, S_E_aligned
    
def load_data_twostage(X_control, X_case, gene_names, s_e_path, s_e_genes_path, min_cells=215, additional_genes_to_remove=None, verbose=False):
    X_ntc_full = X_control
    X_specific_full = X_case
    hvg_gene_names_full = gene_names

    if min_cells > 0:
        gene_cell_counts = (X_ntc_full > 0).sum(axis=1) + (X_specific_full > 0).sum(axis=1)
        mask = gene_cell_counts >= min_cells
    else:
        mask = np.ones(len(hvg_gene_names_full), dtype=bool)

    if additional_genes_to_remove:
        remove = np.isin(hvg_gene_names_full, additional_genes_to_remove)
        mask &= ~remove
    
    if mask.dtype != bool or mask.shape[0] != hvg_gene_names_full.shape[0]:
        raise ValueError(f"Mask shape/type mismatch: mask {mask.shape} bool? {mask.dtype==bool}, "
                         f"genes {hvg_gene_names_full.shape}")

    # apply mask once to keep everything aligned
    X_ntc = X_ntc_full[mask, :]
    X_specific = X_specific_full[mask, :]
    del X_ntc_full, X_specific_full
    hvg_gene_names = hvg_gene_names_full[mask].tolist()
    print(f"DEBUG: Genes after min_cells filter: {len(hvg_gene_names)}")
    
    S_E_full = np.load(s_e_path, mmap_mode="r")
    with open(s_e_genes_path, 'r') as f:
        s_e_full_genes_list = json.load(f)
    
    print(f"DEBUG: Genes loaded from S_E_GENES_PATH: {len(s_e_full_genes_list)}")
   
    # choose desired gene order
    x_index = pd.Index(hvg_gene_names)
    se_index = pd.Index(s_e_full_genes_list)
    keep = x_index.intersection(se_index, sort=False)

    final_genes = keep.tolist()

    if len(final_genes) == 0:
        raise ValueError("No genes remain after intersection with S_E; check gene naming / filters.")

    x_gene_to_idx  = {g: i for i, g in enumerate(hvg_gene_names)}
    se_gene_to_idx = {g: i for i, g in enumerate(s_e_full_genes_list)}
    x_indices  = np.fromiter((x_gene_to_idx[g]  for g in final_genes), dtype=np.int64)
    se_indices = np.fromiter((se_gene_to_idx[g] for g in final_genes), dtype=np.int64)

    X_ntc_aligned = X_ntc[x_indices, :]
    X_specific_aligned = X_specific[x_indices, :]
    S_E_aligned = S_E_full[np.ix_(se_indices, se_indices)]
    
    return X_ntc_aligned, X_specific_aligned, S_E_aligned

def normalize_cells_to_median(matrix):
    # normalizes each cell (column) to have the same total count as the median cell
    cell_sums = matrix.sum(axis=0)
    median_sum = np.median(cell_sums)
    scaling_factors = median_sum / cell_sums
    return matrix * scaling_factors

def scale_genes_to_unit_variance(matrix):
    # scales each gene (row) to have unit variance
    std_devs = matrix.std(axis=1, ddof=1, keepdims=True)
    std_devs[std_devs == 0] = 1.0
    scaled_matrix = matrix / std_devs
    np.nan_to_num(scaled_matrix, copy=False)
    return scaled_matrix

def cnmf_tpm_like_normalization(X, target_sum=1e6, eps=1e-12):
    """Per-cell normalize columns to sum to 1e6 (genes x cells)."""
    col_sums = X.sum(axis=0, keepdims=True)
    scale = target_sum / np.maximum(col_sums, eps)
    return X * scale
