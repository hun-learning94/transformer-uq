"""
walker_comparison.py

Side-by-side comparison of TransformerUQ vs Empirical Bayes GP on Walker Lake.
Layout: 2x3 Grid
  Row 1: Transformer (Mean, Lower 5%, Upper 95%)
  Row 2: GP Baseline (Mean, Lower 5%, Upper 95%)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import model
import data

# -----------------------------------------------------------------------------
# 1. Helpers: Normalization & Plotting
# -----------------------------------------------------------------------------
def normalize_marginals_to_I_over_2(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    std = Xc.std(axis=0, ddof=1, keepdims=True) + 1e-12
    Xn = Xc / std / np.sqrt(2.0) 
    params = {"mu": mu, "std": std}
    return Xn, params

def apply_normalize_marginals(X: np.ndarray, params):
    return ((X - params["mu"]) / params["std"]) / np.sqrt(2.0)

def standardize_y(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    mu = np.nanmean(y)
    sd = np.nanstd(y, ddof=1) + 1e-12
    return (y - mu) / sd, {"mu_y": mu, "sd_y": sd}

def to_grid(x, y, values):
    x, y, values = np.asarray(x), np.asarray(y), np.asarray(values)
    ux = np.unique(x)
    uy = np.unique(y)

    if len(ux) * len(uy) != len(values):
        x = np.round(x, 6)
        y = np.round(y, 6)
        ux = np.unique(x)
        uy = np.unique(y)

    if len(ux) * len(uy) != len(values):
        return None, ux, uy 

    ix = np.searchsorted(ux, x)
    iy = np.searchsorted(uy, y)

    grid = np.full((len(uy), len(ux)), np.nan, dtype=float)
    grid[iy, ix] = values
    return grid, ux, uy

def plot_grid_or_tri(ax, x, y, values, title, cmap="terrain", vmin=None, vmax=None):
    """Plot values on (x,y) using grid imshow or fallback to tricontourf."""
    grid, ux, uy = to_grid(x, y, values)

    if grid is not None:
        m = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[ux.min(), ux.max(), uy.min(), uy.max()],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        m = ax.tricontourf(x, y, values, levels=40, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    # ax.set_xticks([])
    # ax.set_yticks([])
    return m

def quantiles_from_binned_probs(probs: torch.Tensor, edges: torch.Tensor, qs: list) -> torch.Tensor:
    """Compute quantiles from binned probabilities via linear interpolation."""
    if probs.ndim == 2: probs = probs.unsqueeze(1)
    B, _, C = probs.shape
    device = probs.device
    
    edges_left = edges[:-1].view(1, 1, C).expand(B, 1, C)
    edges_right = edges[1:].view(1, 1, C).expand(B, 1, C)

    cdf = probs.cumsum(dim=-1)
    cdf = torch.clamp(cdf, 0.0, 1.0)
    cdf[..., -1] = 1.0

    out = []
    for q in qs:
        q_t = torch.tensor(q, device=device, dtype=probs.dtype).view(1, 1, 1)
        ge = (cdf >= q_t).to(torch.int64)
        idx = torch.argmax(ge, dim=-1, keepdim=True)
        
        left = torch.gather(edges_left, -1, idx).squeeze(-1)
        right = torch.gather(edges_right, -1, idx).squeeze(-1)
        p_bin = torch.gather(probs, -1, idx).squeeze(-1)

        prev_idx = torch.clamp(idx - 1, min=0)
        prev_cdf = torch.gather(cdf, -1, prev_idx).squeeze(-1)
        prev_cdf = torch.where((idx.squeeze(-1) > 0), prev_cdf, torch.zeros_like(prev_cdf))

        frac = (q_t.squeeze(-1) - prev_cdf) / torch.clamp(p_bin, min=1e-12)
        qval = left + torch.clamp(frac, 0.0, 1.0) * (right - left)
        out.append(qval)

    return torch.cat(out, dim=-1)

# -----------------------------------------------------------------------------
# 2. Main Comparison Suite
# -----------------------------------------------------------------------------
def run_comparison_suite(
    checkpoint_path,
    walker_df,
    walker_exh_df,
    seeds=[0, 1, 2, 3, 4],
    n_train=256,
    y_col="V",
    device="cuda",
    cmap="plasma",
    save_dir="figs_comparison"
):
    os.makedirs(save_dir, exist_ok=True)
    
    # --- A. Load Transformer (ONCE) ---
    print(f"Loading Transformer checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    if "feature_scale_init_vec" not in cfg_dict: cfg_dict["feature_scale_init_vec"] = None
    
    cfg = model.ModelConfig(**cfg_dict)
    args_dict = ckpt["args"]
    edges = data.make_bin_edges(args_dict["a"], args_dict["b"], args_dict["C"], device=device)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    
    net = model.TransformerUQ(cfg=cfg, edges=edges).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    
    # --- B. Prepare Data Arrays ---
    X_all = walker_df[["x", "y"]].to_numpy(dtype=float)
    y_all = walker_df[y_col].to_numpy(dtype=float)
    X_exh_raw = walker_exh_df[["x", "y"]].to_numpy(dtype=float)
    y_exh_raw = walker_exh_df[y_col].to_numpy(dtype=float)
    
    # --- C. Loop Over Seeds ---
    for seed in seeds:
        print(f"\n--- Running Comparison Seed {seed} ---")
        rng = np.random.default_rng(seed)
        
        # 1. Sample Context
        n_ctx = min(n_train, len(X_all))
        idx = rng.choice(len(X_all), size=n_ctx, replace=False)
        X_train_raw = X_all[idx]
        y_train_raw = y_all[idx]
        
        # 2. Normalize (Data-Dependent per seed)
        X_train_n, xparams = normalize_marginals_to_I_over_2(X_train_raw)
        X_exh_n = apply_normalize_marginals(X_exh_raw, xparams)
        y_train, yparams = standardize_y(y_train_raw)
        
        # ==========================================
        # Part 1: Transformer Inference
        # ==========================================
        X_ctx = torch.tensor(X_train_n, dtype=torch.float32, device=device).unsqueeze(0)
        Y_ctx = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(0)
        X_q_all = torch.tensor(X_exh_n, dtype=torch.float32, device=device)
        
        probs_list = []
        batch_size = 4096 
        with torch.no_grad():
            for i in range(0, len(X_exh_n), batch_size):
                chunk_xq = X_q_all[i : i+batch_size]
                B_curr = chunk_xq.shape[0]
                logits, _ = net(X_ctx.expand(B_curr, -1, -1), Y_ctx.expand(B_curr, -1), chunk_xq)
                probs_list.append(torch.softmax(logits, dim=-1))
            
        probs_all = torch.cat(probs_list, dim=0)
        mu_tf = (probs_all * bin_centers).sum(dim=-1).cpu().numpy()
        qs_tf = quantiles_from_binned_probs(probs_all, edges, [0.05, 0.95])
        q05_tf = qs_tf[:, 0].cpu().numpy()
        q95_tf = qs_tf[:, 1].cpu().numpy()

        # ==========================================
        # Part 2: GP Baseline Inference
        # ==========================================
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=0.0, normalize_y=False, n_restarts_optimizer=5, random_state=seed
        )
        gp.fit(X_train_n, y_train)
        mu_gp, std_gp = gp.predict(X_exh_n, return_std=True)
        q05_gp = mu_gp - 1.645 * std_gp
        q95_gp = mu_gp + 1.645 * std_gp

        # --- NEW: Un-standardize all predictions back to original scale ---
        # Formula: y_orig = y_std * std + mean
        mu_y, sd_y = yparams["mu_y"], yparams["sd_y"]

        # Transformer
        mu_tf = mu_tf * sd_y + mu_y
        q05_tf = q05_tf * sd_y + mu_y
        q95_tf = q95_tf * sd_y + mu_y

        # GP
        mu_gp = mu_gp * sd_y + mu_y
        q05_gp = q05_gp * sd_y + mu_y
        q95_gp = q95_gp * sd_y + mu_y

        # Note: y_exh is currently standardized. Revert it to original y_exh_raw for consistency if plotted.
        # (Though y_exh isn't strictly plotted in the 2x3 layout, we use it for vmin/vmax logic)
        
        # --- Update Plot Limits and Labels ---
        # Update colorbar label in the plotting section below from "Standardized V" to "Variable V (ppm)"

        # ==========================================
        # Part 3: Plotting (2x3)
        # ==========================================
        # Determine shared vmin/vmax from Means to keep color scale consistent
        pooled = np.concatenate([mu_tf, mu_gp])
        vmin, vmax = np.percentile(pooled, [2, 98])
        
        fig, axes = plt.subplots(2, 3, figsize=(9, 5), constrained_layout=True)
        
        def overlay_obs(ax):
            ax.scatter(X_train_n[:, 0], X_train_n[:, 1], s=15, c="#00FFFF", marker="x", alpha=0.75, label="Context")

        # Row 1: Transformer
        m0 = plot_grid_or_tri(axes[0, 0], X_exh_n[:,0], X_exh_n[:,1], mu_tf, "PFN PPD Mean", cmap, vmin, vmax)
        overlay_obs(axes[0, 0])
        plot_grid_or_tri(axes[0, 1], X_exh_n[:,0], X_exh_n[:,1], q05_tf, "PFN PPD Lower 5%", cmap, vmin, vmax)
        plot_grid_or_tri(axes[0, 2], X_exh_n[:,0], X_exh_n[:,1], q95_tf, "PFN PPD Upper 95%", cmap, vmin, vmax)

        # Row 2: GP
        plot_grid_or_tri(axes[1, 0], X_exh_n[:,0], X_exh_n[:,1], mu_gp, "GP PPD Mean", cmap, vmin, vmax)
        overlay_obs(axes[1, 0])
        plot_grid_or_tri(axes[1, 1], X_exh_n[:,0], X_exh_n[:,1], q05_gp, "GP PPD Lower 5%", cmap, vmin, vmax)
        plot_grid_or_tri(axes[1, 2], X_exh_n[:,0], X_exh_n[:,1], q95_gp, "GP PPD Upper 95%", cmap, vmin, vmax)

        # Add single colorbar
        cbar = fig.colorbar(m0, ax=axes, location="right", shrink=0.6, pad=0.02)
        cbar.set_label("Mineral Grade (ppm)")

        save_path = os.path.join(save_dir, f"walker_compare_seed_{seed}.pdf")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    WALKER_CSV = "data/walker.csv"
    WALKER_EXH_CSV = "data/walker_exh.csv"
    # Update to your best checkpoint
    CKPT_PATH = "runs/walker_lake_eb/best.pt"

    if os.path.exists(CKPT_PATH) and os.path.exists(WALKER_CSV):
        walker_df = pd.read_csv(WALKER_CSV)
        walker_exh_df = pd.read_csv(WALKER_EXH_CSV)
        
        run_comparison_suite(
            checkpoint_path=CKPT_PATH,
            walker_df=walker_df,
            walker_exh_df=walker_exh_df,
            seeds=[3], 
            n_train=200,
            save_dir="walker_plots"
        )
    else:
        print(f"File not found.\nCheck: {CKPT_PATH}")