"""
High-level API for DeltaNMF
"""
import numpy as np
from . import preprocessing
from . import nmf_torch
from . import models
from . import pipeline_utils


def calculate_adaptive_hyperparams(
    X, S_E, k, hyperparam_set, init_W, init_H,
    max_scale=10.0, eps=1e-12):
    W_use = np.clip(init_W, 0, None)
    H_use = np.clip(init_H, 0, None)

    m = W_use.shape[0]
    recon_loss = np.mean((X - W_use @ H_use)**2)

    D_E = np.diag(np.sum(S_E, axis=1))
    L_S = D_E - S_E
    unscaled_fm = np.trace(W_use.T @ L_S @ W_use) / (m * k)

    out = {}
    if "relative_strength_alpha" in hyperparam_set:
        rs = float(hyperparam_set["relative_strength_alpha"])
        out["alpha_ntc"] = float(np.clip(rs * recon_loss / (unscaled_fm + eps), 0.0, max_scale)) if unscaled_fm > 0 else 0.0
    else:
        out["alpha_ntc"] = float(hyperparam_set.get("alpha_ntc", 0.0))

    return out

def calculate_adaptive_hyperparams_combined(
    X_combined, S_E, k_ntc, k_spec, hyperparam_set,
    W_init_combined, H_init_combined, eps=1e-12):
    W0 = np.clip(W_init_combined, 0, None)
    H0 = np.clip(H_init_combined, 0, None)

    m = W0.shape[0]
    recon_loss = np.mean((X_combined - W0 @ H0)**2)

    W_ntc = W0[:, :k_ntc]
    W_spec = W0[:, k_ntc:]

    D_E = np.diag(np.sum(S_E, axis=1))
    L_S = D_E - S_E
    unscaled_fm = np.trace(W_spec.T @ L_S @ W_spec) / (m * max(1, k_spec))

    Wn_ntc = W_ntc / (np.linalg.norm(W_ntc, axis=0) + eps)
    Wn_spec = W_spec / (np.linalg.norm(W_spec, axis=0) + eps)
    unscaled_ortho = np.mean((Wn_ntc.T @ Wn_spec)**2)

    out = {}
    if "relative_strength_alpha" in hyperparam_set:
        rs = float(hyperparam_set["relative_strength_alpha"])
        out["alpha_fm"] = float(rs * recon_loss / (unscaled_fm + eps)) if unscaled_fm > 0 else 0.0
    else:
        out["alpha_fm"] = float(hyperparam_set.get("alpha_fm", 0.0))

    if "relative_strength_gamma" in hyperparam_set:
        rs = float(hyperparam_set["relative_strength_gamma"])
        out["gamma"] = float(rs * recon_loss / (unscaled_ortho)) if unscaled_ortho > 0 else 0.0
    else:
        out["gamma"] = float(hyperparam_set.get("gamma", 0.0))

    return out


def run_onestage_deltanmf(
    X_control, gene_names, S_E_PATH, S_E_GENES_PATH, K,
    MIN_CELLS, REMOVE_GENES = [],
    USE_TPM = False, USE_MEDIAN = False, USE_UNITVAR = True, TPM_TARGET = 1e6,
    rel_alpha = 0.0, 
    max_iter = 10000, 
    batchsize = 5506, 
    FM_LAST_ITERS = 200,
    FM_NONNEG = "softplus", FM_SOFTPLUS_BETA = 5.0, lr = 0.01,
    BASE_SEED = 1337):
    X_ntc, S_E_aligned = preprocessing.load_data_onestage(
        X_control, gene_names, S_E_PATH, S_E_GENES_PATH,
        min_cells=MIN_CELLS,
        additional_genes_to_remove=(REMOVE_GENES if len(REMOVE_GENES) > 0 else None),
        verbose=False
    )

    X_ntc = pipeline_utils.apply_normalization(X_ntc, USE_TPM, USE_MEDIAN, USE_UNITVAR, TPM_TARGET).astype(np.float32, copy=False)

    W_cons, H_cons, info = nmf_torch.consensus_nmf_gpu(
        X_ntc, k=K, n_runs=20, epochs=100, batch_size=batchsize, use_gpu=True,
        seed=BASE_SEED, verbose=True, gene_names=None, cell_names=None,
        consensus_mode="median", density_threshold_quantile=0.95, local_neighborhood_size=0.30
    )

    W_init, H_init = pipeline_utils.build_init_from_H(X_ntc, H_cons.copy())

    alpha_params = calculate_adaptive_hyperparams(
        X_ntc, S_E_aligned, K,
        {"relative_strength_alpha": rel_alpha},
        init_W=W_init, init_H=H_init
    )
    alpha_val = alpha_params["alpha_ntc"]

    W_ntc, H_ntc, loss_fm = models.solve_ntc_regularized(
        X_ntc, k=K, S_E=S_E_aligned,
        alpha_ntc=alpha_val,
        init_W=W_init, init_H=H_init,
        max_iter=max_iter, tol=0,
        nonneg=FM_NONNEG, softplus_beta=FM_SOFTPLUS_BETA,
        normalize_W=False, init_fix_scale=False,
        seed=BASE_SEED,
        lr_start=lr, fm_target_ratio=rel_alpha,
        fm_apply_late=True, fm_last_iters=FM_LAST_ITERS
    )

    return {
        "W": W_ntc,
        "H": H_ntc,
    }


def run_twostage_deltanmf(
    X_control, X_case, gene_names, S_E_PATH, S_E_GENES_PATH, K_stage1, K_stage2,
    MIN_CELLS, REMOVE_GENES = [],
    USE_TPM = False, USE_MEDIAN = False, USE_UNITVAR = True, TPM_TARGET = 1e6,
    stage1_rel_alpha = 0.0, stage2_rel_alpha = 0.0, stage2_rel_gamma = 0.0,
    stage1_max_iter = 10000, stage2_max_iter = 10000,
    stage1_batchsize = 5506, stage2_batchsize = 40960, 
    FM_LAST_ITERS = 200,
    FM_NONNEG = "softplus", FM_SOFTPLUS_BETA = 5.0, lr = 0.01,
    BASE_SEED = 1337):
    X_ntc, X_spec, S_E_aligned = preprocessing.load_data_twostage(
        X_control, X_case, gene_names, S_E_PATH, S_E_GENES_PATH,
        min_cells=MIN_CELLS,
        additional_genes_to_remove=(REMOVE_GENES if len(REMOVE_GENES) > 0 else None),
        verbose=False
    )

    X_ntc = pipeline_utils.apply_normalization(X_ntc, USE_TPM, USE_MEDIAN, USE_UNITVAR, TPM_TARGET).astype(np.float32, copy=False)
    X_spec = pipeline_utils.apply_normalization(X_spec, USE_TPM, USE_MEDIAN, USE_UNITVAR, TPM_TARGET).astype(np.float32, copy=False)


    W_cons, H_cons, info = nmf_torch.consensus_nmf_gpu(
        X_ntc, k=K_stage1, n_runs=20, epochs=100, batch_size=stage1_batchsize, use_gpu=True,
        seed=BASE_SEED, verbose=True, gene_names=None, cell_names=None,
        consensus_mode="median", density_threshold_quantile=0.95, local_neighborhood_size=0.30
    )

    W_init, H_init = pipeline_utils.build_init_from_H(X_ntc, H_cons.copy())

    alpha_params = calculate_adaptive_hyperparams(
        X_ntc, S_E_aligned, K_stage1,
        {"relative_strength_alpha": stage1_rel_alpha},
        init_W=W_init, init_H=H_init
    )
    alpha_val = alpha_params["alpha_ntc"]

    W_ntc, H_ntc, loss_fm = models.solve_ntc_regularized(
        X_ntc, k=K_stage1, S_E=S_E_aligned,
        alpha_ntc=alpha_val,
        init_W=W_init, init_H=H_init,
        max_iter=stage1_max_iter, tol=0,
        nonneg=FM_NONNEG, softplus_beta=FM_SOFTPLUS_BETA,
        normalize_W=False, init_fix_scale=False,
        seed=BASE_SEED,
        lr_start=lr, fm_target_ratio=stage1_rel_alpha,
        fm_apply_late=True, fm_last_iters=FM_LAST_ITERS
    )

    H_n_spec = pipeline_utils.solve_H_pi_then_cd(
        X_spec, W_ntc, max_iter_cd=50, tol=1e-8, random_state=BASE_SEED
    )
    R = np.maximum(0.0, X_spec - (W_ntc @ H_n_spec))

    W_tc_init, H_tc_init, info_tc = nmf_torch.consensus_nmf_gpu(
        R, k=K_stage2, n_runs=20, epochs=100, batch_size=stage1_batchsize,
        use_gpu=True, seed=BASE_SEED + 17, verbose=True, gene_names=None, cell_names=None,
        consensus_mode="median", density_threshold_quantile=0.95, local_neighborhood_size=0.30
    )
    H_init_spec = np.vstack([H_n_spec, H_tc_init])  

    # L2 column sums
    s2_ntc = np.linalg.norm(W_ntc, axis=0)
    s2_spec = np.linalg.norm(W_tc_init, axis=0)

    t_ntc = np.median(s2_ntc[s2_ntc > 0])
    t_spec = np.median(s2_spec[s2_spec > 0])

    scale = t_ntc / (t_spec)
    W_ntc_bal  = W_ntc
    W_spec_bal = W_tc_init * scale

    W_combined_init = np.hstack([W_ntc_bal, W_spec_bal])
    H_combined_init = pipeline_utils.solve_H_pi_then_cd(
        X_spec, W_combined_init, max_iter_cd=50, tol=1e-8, random_state=BASE_SEED
    )

    hyper_rel = {
        "relative_strength_alpha": stage2_rel_alpha,
        "relative_strength_gamma": stage2_rel_gamma
    }
    final_hypers = calculate_adaptive_hyperparams_combined(
        X_spec, S_E_aligned, K_stage1, K_stage2, hyper_rel,
        W_combined_init, H_combined_init
    )

    # fit W_specific on full X_spec with W_ntc fixed
    W_spec_final, H_spec_final, loss_df = models.solve_specific_with_fixed_ntc(
        X_specific=X_spec,
        W_ntc=W_ntc_bal,
        k_specific=K_stage2,
        S_E=S_E_aligned,
        hyperparameters=final_hypers,
        guide_aggregation_map=None,
        epochs=stage2_max_iter,
        lr_W_specific=lr,
        tol=0,
        batch_size=stage2_batchsize,
        init_W_specific=W_spec_bal,
        H_init=H_combined_init,
        nonneg=FM_NONNEG,
        softplus_beta=FM_SOFTPLUS_BETA
    )

    return {
        "W_stage1": W_ntc,
        "H_stage1": H_ntc,
        "W_stage2": W_spec_final,
        "H_stage2": H_spec_final,
    }