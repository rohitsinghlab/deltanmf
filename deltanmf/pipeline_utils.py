import numpy as np
from . import preprocessing
from sklearn.decomposition import non_negative_factorization


def solve_H_pi_then_cd(X, W, max_iter_cd=50, tol=1e-8, random_state=0):
    # Pseudoinverse 
    #    H0 = clip( pinv(W) @ X, 0 )
    W32 = np.maximum(0.0, np.asarray(W, dtype=np.float32))
    X32 = np.asarray(X, dtype=np.float32)
    W_pinv = np.linalg.pinv(W32)
    H0 = W_pinv @ X32
    H0 = np.maximum(0.0, H0).astype(np.float32)

    Z0, Hfix, Xt = H0.T, W32.T, X32.T
    Z, _, _ = non_negative_factorization(Xt, W=Z0, H=Hfix, n_components=W32.shape[1],
        init='custom', update_H=False,
        solver='cd', beta_loss='frobenius',
        alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,
        max_iter=max_iter_cd, tol=tol, random_state=random_state)
    return Z.T.astype(np.float32)

def solve_W_pi_then_cd(X, H, max_iter_cd=50, tol=1e-8, random_state=0):
    # Pseudoinverse 
    #    W0 = clip( X @ pinv(H), 0 )
    H32 = np.maximum(0.0, np.asarray(H, dtype=np.float32))
    X32 = np.asarray(X, dtype=np.float32)
    H_pinv = np.linalg.pinv(H32)
    W0 = X32 @ H_pinv
    W0 = np.maximum(0.0, W0).astype(np.float32)

    W, _, _ = non_negative_factorization(
        X32, W=W0, H=H32, n_components=H32.shape[0],
        init='custom', update_H=False,
        solver='cd', beta_loss='frobenius',
        alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,
        max_iter=max_iter_cd, tol=tol, random_state=random_state
    )
    return W.astype(np.float32)

def apply_normalization(X, use_tpm, use_median, use_unitvar, tpm_target):
    if use_tpm and use_median:
        raise ValueError("choose either TPM-like or median normalization")
    Xp = X.astype(np.float32, copy=True)
    if use_tpm:
        Xp = preprocessing.cnmf_tpm_like_normalization(Xp, target_sum=tpm_target)
    elif use_median:
        Xp = preprocessing.normalize_cells_to_median(Xp)
    if use_unitvar:
        Xp = preprocessing.scale_genes_to_unit_variance(Xp)
    return Xp


def build_init_from_H(X, H_in, *, row_normalize_usages=True,
                      cd_iters=50, tol=1e-8, random_state=None):
    H = H_in
    if row_normalize_usages:
        H = H / (H.sum(axis=0, keepdims=True) + 1e-12)
    W_init = solve_W_pi_then_cd(X, H, max_iter_cd=cd_iters, tol=tol, random_state=random_state)
    H_init = solve_H_pi_then_cd(X, W_init, max_iter_cd=cd_iters, tol=tol, random_state=random_state)
    
    return W_init.astype(np.float32), H_init.astype(np.float32)
