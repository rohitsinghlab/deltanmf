import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from tqdm.auto import trange

class NTCOptimizer(nn.Module):
    def __init__(self, m, n, k, init_W=None, init_H=None):
        super().__init__()
        if init_W is not None and init_H is not None:
            self.W = nn.Parameter(torch.tensor(init_W, dtype=torch.float32))
            self.H = nn.Parameter(torch.tensor(init_H, dtype=torch.float32))
        else:
            self.W = nn.Parameter(torch.rand(m, k, dtype=torch.float32))
            self.H = nn.Parameter(torch.rand(k, n, dtype=torch.float32))

    def forward(self):
        return self.W, self.H

def solve_ntc_regularized(
    X, k, S_E, 
    alpha_ntc=0.0,
    init_W=None, init_H=None,
    max_iter=1000, tol=1e-8,
    nonneg="softplus", softplus_beta=10.0,
    normalize_W=False,
    init_fix_scale=False,
    seed=None,
    lr_start=0.01,
    fm_target_ratio=None,
    fm_apply_late=False,
    fm_last_iters=None
):
    """
    Fit NTC programs by minimizing reconstruction loss plus optional regularizers.
    X is expected to be (genes x cells). Returns (W, H, loss_history_df).
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    m, n = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    D_E = np.diag(np.sum(S_E, axis=1)) if S_E is not None else None

    # init scaling fix: L1-normalize W cols once, rescale H rows to keep WH
    if init_W is not None and init_H is not None and init_fix_scale:
        W0 = torch.as_tensor(init_W, dtype=torch.float32)
        H0 = torch.as_tensor(init_H, dtype=torch.float32)
        s0 = W0.sum(dim=0, keepdim=True).clamp_min(1e-12)
        W0 = W0 / s0
        H0 = H0 * s0.squeeze(0).unsqueeze(1)
        init_W = W0.cpu().numpy()
        init_H = H0.cpu().numpy()

    model = NTCOptimizer(m, n, k, init_W=init_W, init_H=init_H).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_start)

    S_E_tensor = L_S_tensor = None
    if (S_E is not None) and (D_E is not None):
        L_S_tensor = torch.tensor(D_E - S_E, dtype=torch.float32).to(device)

    # choose nonnegativity activation
    if nonneg.lower() == "relu":
        def _act(t): return F.relu(t)
    else:
        def _act(t): return F.softplus(t, beta=softplus_beta)

    loss_history = []
    prev_loss = np.inf
    pbar = trange(max_iter, desc="Regularized NTC Discovery", leave=False)
    eps = torch.as_tensor(1e-12, dtype=torch.float32, device=device)

    total_iters = int(max_iter)
    _last = int(fm_last_iters) if (fm_last_iters is not None) else 0
    fm_start_iter = (total_iters - _last) if (fm_apply_late and _last > 0) else 0
    if fm_start_iter < 0:
        fm_start_iter = 0

    for i in pbar:
        optimizer.zero_grad()

        W_raw, H_raw = model()
        W_eff = _act(W_raw)
        H_eff = _act(H_raw)

        if normalize_W:
            s = W_eff.sum(dim=0, keepdim=True).clamp_min(eps)
            W_eff = W_eff / s
            H_eff = H_eff * s.squeeze(0).unsqueeze(1)

        recon = W_eff @ H_eff
        recon_loss = torch.mean((X_tensor - recon) ** 2)

        alpha_t = torch.as_tensor(alpha_ntc, dtype=X_tensor.dtype, device=X_tensor.device)

        alpha_updated = False

        if fm_apply_late and (i == fm_start_iter) and (fm_target_ratio is not None) and (L_S_tensor is not None):
            fm_unscaled = torch.trace(W_eff.T @ L_S_tensor @ W_eff) / (m * k)
            ru = float(recon_loss.item())
            fu = float(fm_unscaled.item())
            if fu > 0.0:
                alpha_ntc = float((fm_target_ratio * ru) / (fu + 1e-12))
                alpha_t = torch.as_tensor(alpha_ntc, dtype=X_tensor.dtype, device=X_tensor.device)
                alpha_updated = True

        fm_loss = torch.zeros([], dtype=X_tensor.dtype, device=X_tensor.device)

        if (alpha_ntc > 0) and (L_S_tensor is not None) and (i >= fm_start_iter):
            fm_loss = alpha_t * torch.trace(W_eff.T @ L_S_tensor @ W_eff) / (m * k)

        total_loss = recon_loss + fm_loss
        total_loss.backward()
        optimizer.step()
                    
        cur = float(total_loss.item())
        cur_lr = optimizer.param_groups[0]["lr"]
        
        loss_history.append({
            "iteration": i,
            "lr": float(cur_lr),
            "total_loss": cur,
            "recon_loss": float(recon_loss.item()),
            "fm_loss": float(fm_loss.item()),
            "alpha_ntc": float(alpha_ntc),
            "alpha_updated": bool(alpha_updated),
        })
        pbar.set_postfix(loss=cur, lr=f"{cur_lr:.5f}")

        if np.abs(prev_loss - cur) < tol * prev_loss:
            break
        prev_loss = cur

    with torch.no_grad():
        W_raw, H_raw = model()
        W_eff = _act(W_raw); H_eff = _act(H_raw)
        if normalize_W:
            s = W_eff.sum(dim=0, keepdim=True).clamp_min(eps)
            W_eff = W_eff / s
            H_eff = H_eff * s.squeeze(0).unsqueeze(1)

    return W_eff.detach().cpu().numpy(), H_eff.detach().cpu().numpy(), pd.DataFrame(loss_history)

def solve_ntc_regularized_minibatch(
    X, k, S_E,
    alpha_ntc=0.0,
    init_W=None, init_H=None,
    max_iter=1000, tol=1e-8,
    nonneg="softplus", softplus_beta=10.0,
    normalize_W=False,
    init_fix_scale=False,
    seed=None,
    lr_start=0.01,
    fm_target_ratio=None,
    fm_apply_late=False,
    fm_last_iters=None,
    batch_size=40960
):
    """
    Minibatched variant of solve_ntc_regularized.
    X is expected to be (genes x cells). Returns (W, H, loss_history_df)
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    m, n = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hybrid memory strategy:
    # - Try full X on device for fast on-device slicing.
    # - Fall back to CPU-resident X with per-batch transfers on OOM.
    X_tensor_cpu = torch.tensor(X, dtype=torch.float32)
    X_tensor = None
    x_on_device = False
    if device.type == "cuda":
        try:
            X_tensor = X_tensor_cpu.to(device)
            x_on_device = True
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                torch.cuda.empty_cache()
                x_on_device = False
            else:
                raise
    else:
        X_tensor = X_tensor_cpu
        x_on_device = True

    D_E = np.diag(np.sum(S_E, axis=1)) if S_E is not None else None

    # init scaling fix: L1-normalize W cols once, rescale H rows to keep WH
    if init_W is not None and init_H is not None and init_fix_scale:
        W0 = torch.as_tensor(init_W, dtype=torch.float32)
        H0 = torch.as_tensor(init_H, dtype=torch.float32)
        s0 = W0.sum(dim=0, keepdim=True).clamp_min(1e-12)
        W0 = W0 / s0
        H0 = H0 * s0.squeeze(0).unsqueeze(1)
        init_W = W0.cpu().numpy()
        init_H = H0.cpu().numpy()

    model = NTCOptimizer(m, n, k, init_W=init_W, init_H=init_H).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_start)

    L_S_tensor = None
    if (S_E is not None) and (D_E is not None):
        L_S_tensor = torch.tensor(D_E - S_E, dtype=torch.float32).to(device)

    # choose nonnegativity activation
    if nonneg.lower() == "relu":
        def _act(t): return F.relu(t)
    else:
        def _act(t): return F.softplus(t, beta=softplus_beta)

    loss_history = []
    prev_loss = np.inf
    pbar = trange(max_iter, desc="Regularized NTC Discovery (minibatch)", leave=False)
    eps = torch.as_tensor(1e-12, dtype=torch.float32, device=device)

    total_iters = int(max_iter)
    _last = int(fm_last_iters) if (fm_last_iters is not None) else 0
    fm_start_iter = (total_iters - _last) if (fm_apply_late and _last > 0) else 0
    if fm_start_iter < 0:
        fm_start_iter = 0

    for epoch in pbar:
        epoch_losses = {key: 0.0 for key in ["recon_loss", "fm_loss", "total_loss"]}
        num_batches = 0
        alpha_updated = False

        permutation = torch.randperm(n, device=device) if x_on_device else torch.randperm(n)
        for start in range(0, n, batch_size):
            optimizer.zero_grad()

            idx = permutation[start:start + batch_size]
            if x_on_device:
                X_b = X_tensor[:, idx]
                H_idx = idx
                batch_n = idx.numel()
            else:
                X_b = X_tensor_cpu[:, idx].to(device, non_blocking=True)
                H_idx = idx.to(device, non_blocking=True)
                batch_n = idx.numel()

            W_eff = _act(model.W)
            H_b = _act(model.H[:, H_idx])

            if normalize_W:
                s = W_eff.sum(dim=0, keepdim=True).clamp_min(eps)
                W_eff = W_eff / s
                H_b = H_b * s.squeeze(0).unsqueeze(1)
            
            recon_loss = torch.mean((X_b - W_eff @ H_b) ** 2)

            alpha_t = torch.as_tensor(alpha_ntc, dtype=X_b.dtype, device=device)

            if fm_apply_late and (epoch == fm_start_iter) and (not alpha_updated) and (fm_target_ratio is not None) and (L_S_tensor is not None):
                fm_unscaled = torch.trace(W_eff.T @ L_S_tensor @ W_eff) / (m * k)
                ru = float(recon_loss.item())
                fu = float(fm_unscaled.item())
                if fu > 0.0:
                    alpha_ntc = float((fm_target_ratio * ru) / (fu + 1e-12))
                    alpha_t = torch.as_tensor(alpha_ntc, dtype=X_b.dtype, device=device)
                    alpha_updated = True

            fm_loss = torch.zeros([], dtype=X_b.dtype, device=device)
            if (alpha_ntc > 0) and (L_S_tensor is not None) and (epoch >= fm_start_iter):
                # Scale by the batch fraction so FM weight matches full-batch objective.
                batch_frac = float(batch_n) / float(n)
                fm_loss = alpha_t * torch.trace(W_eff.T @ L_S_tensor @ W_eff) / (m * k)
                fm_loss = fm_loss * batch_frac

            total_loss = recon_loss + fm_loss
            total_loss.backward()
            optimizer.step()

            epoch_losses["recon_loss"] += float(recon_loss.item())
            epoch_losses["fm_loss"] += float(fm_loss.item())
            epoch_losses["total_loss"] += float(total_loss.item())
            num_batches += 1

        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        cur_lr = optimizer.param_groups[0]["lr"]
        cur = avg_losses["total_loss"]

        loss_history.append({
            "iteration": int(epoch),
            "lr": float(cur_lr),
            "total_loss": float(cur),
            "recon_loss": float(avg_losses["recon_loss"]),
            "fm_loss": float(avg_losses["fm_loss"]),
            "alpha_ntc": float(alpha_ntc),
            "alpha_updated": bool(alpha_updated),
        })
        pbar.set_postfix(loss=cur, lr=f"{cur_lr:.5f}")

        if np.abs(prev_loss - cur) < tol * prev_loss:
            break
        prev_loss = cur

    with torch.no_grad():
        W_raw, H_raw = model()
        W_eff = _act(W_raw)
        H_eff = _act(H_raw)
        if normalize_W:
            s = W_eff.sum(dim=0, keepdim=True).clamp_min(eps)
            W_eff = W_eff / s
            H_eff = H_eff * s.squeeze(0).unsqueeze(1)

    return W_eff.detach().cpu().numpy(), H_eff.detach().cpu().numpy(), pd.DataFrame(loss_history)
        
def solve_specific_with_fixed_ntc(
    X_specific, W_ntc, k_specific, S_E, hyperparameters,
    guide_aggregation_map=None, epochs=200, lr_W_specific=0.01,
    tol=1e-8, batch_size=40960,
    init_W_specific=None, H_init=None,
    nonneg="softplus", softplus_beta=5.0
):
    """
    Fit specific programs on full X_specific with W_ntc fixed.
    Expects H_init to have shape (k_ntc + k_specific, n_spec), rows [0:k_ntc] for NTC usage in spec cells.
    """
    m, n_spec = X_specific.shape
    k_ntc = W_ntc.shape[1]

    D_E = np.diag(np.sum(S_E, axis=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if nonneg.lower() == "relu":
        def _act(t): return F.relu(t)
    else:
        def _act(t): return F.softplus(t, beta=softplus_beta)

    W_ntc_used = W_ntc.copy()

    if init_W_specific is None or H_init is None:
        raise ValueError("init_W_specific and H_init must be provided for solve_specific_with_fixed_ntc")

    X_tensor = torch.tensor(X_specific, dtype=torch.float32).to(device)
    W_ntc_const = torch.tensor(W_ntc_used, dtype=torch.float32, device=device)

    L_S_tensor = None
    if hyperparameters.get("alpha_fm", 0.0) > 0 and S_E is not None and D_E is not None:
        L_S_tensor = torch.tensor(D_E - S_E, dtype=torch.float32).to(device)

    guide_agg_map_tensor = None
    if guide_aggregation_map is not None:
        M = guide_aggregation_map.tocoo()
        guide_agg_map_tensor = torch.sparse_coo_tensor(
            indices=torch.from_numpy(np.vstack((M.row, M.col))),
            values=torch.from_numpy(M.data),
            size=M.shape
        ).to(device, dtype=torch.float32)

    class FixedNTCOptimizer(nn.Module):
        def __init__(self, W_ntc_fixed, W_spec_init, H_init_full):
            super().__init__()
            self.W_ntc_fixed = W_ntc_fixed  # not a Parameter
            self.W_spec = nn.Parameter(torch.tensor(W_spec_init, dtype=torch.float32))
            self.H = nn.Parameter(torch.tensor(H_init_full, dtype=torch.float32))
        def forward(self):
            W_ntc_nonneg = _act(self.W_ntc_fixed)
            W_spec_nonneg = _act(self.W_spec)
            H_nonneg = _act(self.H)
            W_combined_nonneg = torch.cat([W_ntc_nonneg, W_spec_nonneg], dim=1)
            return W_combined_nonneg, H_nonneg

    model = FixedNTCOptimizer(W_ntc_const, init_W_specific, H_init).to(device)

    optimizer = optim.Adam([
        {'params': [model.W_spec], 'lr': lr_W_specific},
        {'params': [model.H], 'lr': lr_W_specific}
    ])

    alpha_fm = float(hyperparameters.get("alpha_fm", 0.0))
    gamma    = float(hyperparameters.get("gamma", 0.0))
    eps = 1e-12

    loss_history = []
    prev_loss = np.inf
    pbar = trange(epochs, desc="Specific-with-Fixed-NTC", leave=False)

    for epoch in pbar:
        epoch_losses = {k: 0.0 for k in ['recon_loss', 'fm_loss', 'ortho_loss', 'total_loss']}
        num_batches = 0

        permutation = torch.randperm(n_spec, device=device)
        for i in range(0, n_spec, batch_size):
            optimizer.zero_grad()

            W_combined = torch.cat([_act(model.W_ntc_fixed), _act(model.W_spec)], dim=1)
            idx = permutation[i:i+batch_size]
            X_b = X_tensor[:, idx]
            H_b = _act(model.H[:, idx])

            recon_loss = torch.mean((X_b - W_combined @ H_b)**2)

            fm_loss = torch.zeros((), dtype=torch.float32, device=device)
            ortho_loss = torch.zeros((), dtype=torch.float32, device=device)

            if alpha_fm > 0 and L_S_tensor is not None:
                W_spec_part = W_combined[:, k_ntc:]
                fm_loss = alpha_fm * torch.trace(W_spec_part.T @ L_S_tensor @ W_spec_part) / (m * k_specific)

            if gamma > 0:
                W_ntc_part = W_combined[:, :k_ntc]
                W_spec_part = W_combined[:, k_ntc:]
                W_ntc_norm = F.normalize(W_ntc_part, p=2, dim=0)
                W_spec_norm = F.normalize(W_spec_part, p=2, dim=0)
                ortho_loss = gamma * torch.mean((W_ntc_norm.T @ W_spec_norm)**2)

        
            total_loss = recon_loss + fm_loss + ortho_loss
            total_loss.backward()
            optimizer.step()

            epoch_losses['recon_loss'] += float(recon_loss.item())
            epoch_losses['fm_loss'] += float(fm_loss.item())
            epoch_losses['ortho_loss'] += float(ortho_loss.item())
            epoch_losses['total_loss'] += float(total_loss.item())
            num_batches += 1

        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        avg_losses['epoch'] = int(epoch)
        loss_history.append(avg_losses)
        pbar.set_postfix(loss=avg_losses['total_loss'])

        cur = avg_losses['total_loss']
        if np.abs(prev_loss - cur) < tol * max(prev_loss, 1.0):
            break
        prev_loss = cur

    W_final_spec = _act(model.W_spec).detach().cpu().numpy()
    H_final = _act(model.H).detach().cpu().numpy()
    return W_final_spec, H_final, pd.DataFrame(loss_history)

def solve_specific_with_fixed_ntc_hybrid(
    X_specific, W_ntc, k_specific, S_E, hyperparameters,
    guide_aggregation_map=None, epochs=200, lr_W_specific=0.01,
    tol=1e-8, batch_size=40960,
    init_W_specific=None, H_init=None,
    nonneg="softplus", softplus_beta=5.0
):
    """
    Hybrid-memory variant of solve_specific_with_fixed_ntc.
    Tries full X_specific on device first, falls back to CPU batch transfers on OOM
    """
    m, n_spec = X_specific.shape
    k_ntc = W_ntc.shape[1]

    D_E = np.diag(np.sum(S_E, axis=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if nonneg.lower() == "relu":
        def _act(t): return F.relu(t)
    else:
        def _act(t): return F.softplus(t, beta=softplus_beta)

    W_ntc_used = W_ntc.copy()

    if init_W_specific is None or H_init is None:
        raise ValueError("init_W_specific and H_init must be provided for solve_specific_with_fixed_ntc_hybrid")

    # Memory strategy:
    # - Keep full X_specific on CPU.
    # - On CUDA, transfer per-batch slices to device to preserve VRAM for backward.
    X_tensor_cpu = torch.tensor(X_specific, dtype=torch.float32)
    X_tensor = None
    x_on_device = False
    if device.type == "cuda":
        x_on_device = False
    else:
        X_tensor = X_tensor_cpu
        x_on_device = True

    W_ntc_const = torch.tensor(W_ntc_used, dtype=torch.float32, device=device)

    L_S_tensor = None
    if hyperparameters.get("alpha_fm", 0.0) > 0 and S_E is not None and D_E is not None:
        L_S_tensor = torch.tensor(D_E - S_E, dtype=torch.float32).to(device)

    guide_agg_map_tensor = None
    if guide_aggregation_map is not None:
        M = guide_aggregation_map.tocoo()
        guide_agg_map_tensor = torch.sparse_coo_tensor(
            indices=torch.from_numpy(np.vstack((M.row, M.col))),
            values=torch.from_numpy(M.data),
            size=M.shape
        ).to(device, dtype=torch.float32)

    class FixedNTCOptimizer(nn.Module):
        def __init__(self, W_ntc_fixed, W_spec_init, H_init_full):
            super().__init__()
            self.W_ntc_fixed = W_ntc_fixed  # not a Parameter
            self.W_spec = nn.Parameter(torch.tensor(W_spec_init, dtype=torch.float32))
            self.H = nn.Parameter(torch.tensor(H_init_full, dtype=torch.float32))
        def forward(self):
            W_ntc_nonneg = _act(self.W_ntc_fixed)
            W_spec_nonneg = _act(self.W_spec)
            H_nonneg = _act(self.H)
            W_combined_nonneg = torch.cat([W_ntc_nonneg, W_spec_nonneg], dim=1)
            return W_combined_nonneg, H_nonneg

    model = FixedNTCOptimizer(W_ntc_const, init_W_specific, H_init).to(device)

    optimizer = optim.Adam([
        {'params': [model.W_spec], 'lr': lr_W_specific},
        {'params': [model.H], 'lr': lr_W_specific}
    ])

    alpha_fm = float(hyperparameters.get("alpha_fm", 0.0))
    gamma    = float(hyperparameters.get("gamma", 0.0))
    eps = 1e-12

    loss_history = []
    prev_loss = np.inf
    pbar = trange(epochs, desc="Specific-with-Fixed-NTC (hybrid)", leave=False)

    for epoch in pbar:
        epoch_losses = {k: 0.0 for k in ['recon_loss', 'fm_loss', 'ortho_loss', 'total_loss']}
        num_batches = 0

        permutation = torch.randperm(n_spec, device=device) if x_on_device else torch.randperm(n_spec)
        for i in range(0, n_spec, batch_size):
            optimizer.zero_grad()

            W_combined = torch.cat([_act(model.W_ntc_fixed), _act(model.W_spec)], dim=1)
            idx = permutation[i:i+batch_size]
            if x_on_device:
                X_b = X_tensor[:, idx]
                H_idx = idx
            else:
                X_b = X_tensor_cpu[:, idx].to(device, non_blocking=True)
                H_idx = idx.to(device, non_blocking=True)
            H_b = _act(model.H[:, H_idx])

            recon_loss = torch.mean((X_b - W_combined @ H_b)**2)

            fm_loss = torch.zeros((), dtype=torch.float32, device=device)
            ortho_loss = torch.zeros((), dtype=torch.float32, device=device)

            if alpha_fm > 0 and L_S_tensor is not None:
                W_spec_part = W_combined[:, k_ntc:]
                fm_loss = alpha_fm * torch.trace(W_spec_part.T @ L_S_tensor @ W_spec_part) / (m * k_specific)

            if gamma > 0:
                W_ntc_part = W_combined[:, :k_ntc]
                W_spec_part = W_combined[:, k_ntc:]
                W_ntc_norm = F.normalize(W_ntc_part, p=2, dim=0)
                W_spec_norm = F.normalize(W_spec_part, p=2, dim=0)
                ortho_loss = gamma * torch.mean((W_ntc_norm.T @ W_spec_norm)**2)

            total_loss = recon_loss + fm_loss + ortho_loss
            total_loss.backward()
            optimizer.step()

            epoch_losses['recon_loss'] += float(recon_loss.item())
            epoch_losses['fm_loss'] += float(fm_loss.item())
            epoch_losses['ortho_loss'] += float(ortho_loss.item())
            epoch_losses['total_loss'] += float(total_loss.item())
            num_batches += 1

        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        avg_losses['epoch'] = int(epoch)
        loss_history.append(avg_losses)
        pbar.set_postfix(loss=avg_losses['total_loss'])

        cur = avg_losses['total_loss']
        if np.abs(prev_loss - cur) < tol * max(prev_loss, 1.0):
            break
        prev_loss = cur

    W_final_spec = _act(model.W_spec).detach().cpu().numpy()
    H_final = _act(model.H).detach().cpu().numpy()
    return W_final_spec, H_final, pd.DataFrame(loss_history)

def solve_specific_with_fixed_ntc_hybrid_fast(
    X_specific, W_ntc, k_specific, S_E, hyperparameters,
    guide_aggregation_map=None, epochs=200, lr_W_specific=0.01,
    tol=1e-8, batch_size=40960,
    init_W_specific=None, H_init=None,
    nonneg="softplus", softplus_beta=5.0
):
    """
    Hybrid-memory variant of solve_specific_with_fixed_ntc.
    Tries full X on GPU first; falls back to per-batch CPU->GPU transfer on OOM.
    """
    m, n_spec = X_specific.shape
    k_ntc = W_ntc.shape[1]

    D_E = np.diag(np.sum(S_E, axis=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if nonneg.lower() == "relu":
        def _act(t): return F.relu(t)
    else:
        def _act(t): return F.softplus(t, beta=softplus_beta)

    W_ntc_used = W_ntc.copy()

    if init_W_specific is None or H_init is None:
        raise ValueError("init_W_specific and H_init must be provided for solve_specific_with_fixed_ntc_hybrid")

    X_np = np.ascontiguousarray(X_specific, dtype=np.float32)
    X_gpu = None
    X_T = None
    X_pin = None
    x_on_device = False

    _x_bytes = m * n_spec * 4
    print(f"[hybrid_fast] X shape=({m}, {n_spec}), X_bytes={_x_bytes / 1e9:.2f} GB, "
          f"requested batch_size={batch_size}, device={device}", flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        try:
            X_gpu = torch.tensor(X_np, dtype=torch.float32, device=device)
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_ratio = free_mem / total_mem
            _k_total = W_ntc.shape[1] + k_specific
            _model_bytes = 4 * (m * k_specific + _k_total * n_spec) * 4
            _safety = 1024 * 1024 * 1024
            _per_col_peak = 8 * m * 4
            _remaining = free_mem - _model_bytes - _safety
            print(f"[hybrid_fast] X on GPU. free={free_mem/1e9:.2f} GB ({free_ratio:.1%}), "
                  f"total={total_mem/1e9:.2f} GB, model_est={_model_bytes/1e9:.2f} GB, "
                  f"remaining={_remaining/1e9:.2f} GB", flush=True)
            if free_ratio >= 0.40 and _remaining >= _per_col_peak * 128:
                max_bs = int(_remaining // _per_col_peak)
                if max_bs < batch_size:
                    print(f"[hybrid_fast] Tier-1: reducing batch_size {batch_size} -> {max_bs}", flush=True)
                    batch_size = max_bs
                else:
                    print(f"[hybrid_fast] Tier-1: batch_size {batch_size} fits (max_bs={max_bs})", flush=True)
                x_on_device = True
            else:
                print(f"[hybrid_fast] Free ratio {free_ratio:.1%} < 40%, falling back to Tier-2", flush=True)
                del X_gpu
                torch.cuda.empty_cache()
                X_gpu = None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[hybrid_fast] X_gpu OOM, falling back to Tier-2", flush=True)
                torch.cuda.empty_cache()
            else:
                raise
    if not x_on_device and device.type == "cuda":
        print(f"[hybrid_fast] Tier-2: CPU batches with pinned memory", flush=True)
        X_T = np.ascontiguousarray(X_np.T)
        _actual_bs = min(batch_size, n_spec)
        X_pin = torch.empty(m, _actual_bs, dtype=torch.float32, pin_memory=True)
    elif not x_on_device:
        X_gpu = torch.from_numpy(X_np)
        x_on_device = True

    W_ntc_const = torch.tensor(W_ntc_used, dtype=torch.float32, device=device)

    L_S_tensor = None
    if hyperparameters.get("alpha_fm", 0.0) > 0 and S_E is not None and D_E is not None:
        L_S_tensor = torch.tensor(D_E - S_E, dtype=torch.float32).to(device)

    guide_agg_map_tensor = None
    if guide_aggregation_map is not None:
        M = guide_aggregation_map.tocoo()
        guide_agg_map_tensor = torch.sparse_coo_tensor(
            indices=torch.from_numpy(np.vstack((M.row, M.col))),
            values=torch.from_numpy(M.data),
            size=M.shape
        ).to(device, dtype=torch.float32)

    class FixedNTCOptimizer(nn.Module):
        def __init__(self, W_ntc_fixed, W_spec_init, H_init_full):
            super().__init__()
            self.W_ntc_fixed = W_ntc_fixed
            self.W_spec = nn.Parameter(torch.tensor(W_spec_init, dtype=torch.float32))
            self.H = nn.Parameter(torch.tensor(H_init_full, dtype=torch.float32))

    model = FixedNTCOptimizer(W_ntc_const, init_W_specific, H_init).to(device)

    optimizer = optim.Adam([
        {'params': [model.W_spec], 'lr': lr_W_specific},
        {'params': [model.H], 'lr': lr_W_specific}
    ])

    alpha_fm = float(hyperparameters.get("alpha_fm", 0.0))
    gamma    = float(hyperparameters.get("gamma", 0.0))
    eps = 1e-12

    rng = np.random.default_rng(42)

    loss_history = []
    prev_loss = np.inf
    pbar = trange(epochs, desc="Specific-with-Fixed-NTC (hybrid)", leave=False)

    for epoch in pbar:
        epoch_losses = {k: 0.0 for k in ['recon_loss', 'fm_loss', 'ortho_loss', 'total_loss']}
        num_batches = 0

        perm_np = rng.permutation(n_spec)
        if x_on_device:
            perm_dev = torch.tensor(perm_np, dtype=torch.long, device=device)

        for i in range(0, n_spec, batch_size):
            optimizer.zero_grad()

            W_combined = torch.cat([_act(model.W_ntc_fixed), _act(model.W_spec)], dim=1)

            if x_on_device:
                idx = perm_dev[i:i+batch_size]
                X_b = X_gpu[:, idx]
            else:
                batch_idx = perm_np[i:i+batch_size]
                actual_bs = len(batch_idx)
                rows = torch.from_numpy(X_T[batch_idx])
                X_pin[:, :actual_bs].copy_(rows.T)
                X_b = X_pin[:, :actual_bs].to(device, non_blocking=True)
                idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
            H_b = _act(model.H[:, idx])

            diff = X_b - W_combined @ H_b
            del X_b
            recon_loss = diff.pow(2).mean()
            del diff

            fm_loss = torch.zeros((), dtype=torch.float32, device=device)
            ortho_loss = torch.zeros((), dtype=torch.float32, device=device)

            if alpha_fm > 0 and L_S_tensor is not None:
                W_spec_part = W_combined[:, k_ntc:]
                fm_loss = alpha_fm * torch.trace(W_spec_part.T @ L_S_tensor @ W_spec_part) / (m * k_specific)

            if gamma > 0:
                W_ntc_part = W_combined[:, :k_ntc]
                W_spec_part = W_combined[:, k_ntc:]
                W_ntc_norm = F.normalize(W_ntc_part, p=2, dim=0)
                W_spec_norm = F.normalize(W_spec_part, p=2, dim=0)
                ortho_loss = gamma * torch.mean((W_ntc_norm.T @ W_spec_norm)**2)

            total_loss = recon_loss + fm_loss + ortho_loss
            if x_on_device and device.type == "cuda":
                torch.cuda.empty_cache()
            total_loss.backward()
            optimizer.step()

            epoch_losses['recon_loss'] += float(recon_loss.item())
            epoch_losses['fm_loss'] += float(fm_loss.item())
            epoch_losses['ortho_loss'] += float(ortho_loss.item())
            epoch_losses['total_loss'] += float(total_loss.item())
            num_batches += 1

        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        avg_losses['epoch'] = int(epoch)
        loss_history.append(avg_losses)
        pbar.set_postfix(loss=avg_losses['total_loss'])

        cur = avg_losses['total_loss']
        if np.abs(prev_loss - cur) < tol * max(prev_loss, 1.0):
            break
        prev_loss = cur

    with torch.no_grad():
        W_final_spec = _act(model.W_spec).detach().cpu().numpy()
        H_final = _act(model.H).detach().cpu().numpy()
    return W_final_spec, H_final, pd.DataFrame(loss_history)