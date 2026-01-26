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

    D_E = np.diag(np.sum(S_E, axis=1))

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

            W_combined, H = model()
            idx = permutation[i:i+batch_size]
            X_b = X_tensor[:, idx]
            H_b = H[:, idx]

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