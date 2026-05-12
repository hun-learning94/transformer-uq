import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import model
import data

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# -----------------------------------------------------------------------------
# 1. Helpers: Normalization, grid construction, and plotting
# -----------------------------------------------------------------------------
def normalize_marginals_with_constant(X: np.ndarray, x_constant: float = 0.3):
    """Center/scale each coordinate, then divide by x_constant.

    Matches the Sacramento EB fitting convention you settled on:
      X_proc = (X - mean) / (std * x_constant)
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    std = Xc.std(axis=0, ddof=1, keepdims=True) + 1e-12
    Xn = Xc / (std * x_constant)
    params = {"mu": mu, "std": std, "x_constant": float(x_constant)}
    return Xn, params


def apply_normalize_marginals(X: np.ndarray, params):
    return (X - params["mu"]) / (params["std"] * params["x_constant"])


def standardize_y(y: np.ndarray, y_constant: float = 1.0):
    y = np.asarray(y, dtype=float)
    mu = np.nanmean(y)
    sd = np.nanstd(y, ddof=1) + 1e-12
    return y_constant * (y - mu) / sd, {"mu_y": mu, "sd_y": sd, "y_constant": float(y_constant)}


def invert_standardize_y(y_std: np.ndarray, params):
    return y_std * (params["sd_y"] / params["y_constant"]) + params["mu_y"]


def make_regular_2d_grid(
    X_raw: np.ndarray,
    grid_size_x: int = 100,
    grid_size_y: int = 100,
    pad_frac: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
):
    """
    Construct a regular 2D grid over user-specified coordinate bounds.

    If any bound is None, it defaults to the observed min/max of that coordinate.
    This only changes the grid extent, not the context points used for fitting.
    """
    x = np.asarray(X_raw[:, 0], dtype=float)
    y = np.asarray(X_raw[:, 1], dtype=float)

    xmin = float(np.min(x)) if x_min is None else float(x_min)
    xmax = float(np.max(x)) if x_max is None else float(x_max)
    ymin = float(np.min(y)) if y_min is None else float(y_min)
    ymax = float(np.max(y)) if y_max is None else float(y_max)

    if xmax < xmin:
        raise ValueError(f"x_max must be >= x_min, got {xmax} < {xmin}")
    if ymax < ymin:
        raise ValueError(f"y_max must be >= y_min, got {ymax} < {ymin}")

    xr = xmax - xmin
    yr = ymax - ymin
    xpad = pad_frac * xr
    ypad = pad_frac * yr

    if xr == 0.0:
        xmin -= 0.5
        xmax += 0.5
    else:
        xmin -= xpad
        xmax += xpad

    if yr == 0.0:
        ymin -= 0.5
        ymax += 0.5
    else:
        ymin -= ypad
        ymax += ypad

    xs = np.linspace(xmin, xmax, grid_size_x)
    ys = np.linspace(ymin, ymax, grid_size_y)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return xx.reshape(-1), yy.reshape(-1)

def truncate_points_by_bounds(
    X_raw: np.ndarray,
    y_raw: np.ndarray | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
):
    """
    Keep only rows whose spatial coordinates lie within the given bounds.
    If y_raw is provided, it is truncated with the same mask.
    """
    x = np.asarray(X_raw[:, 0], dtype=float)
    y = np.asarray(X_raw[:, 1], dtype=float)

    keep = np.ones(len(X_raw), dtype=bool)
    if x_min is not None:
        keep &= (x >= x_min)
    if x_max is not None:
        keep &= (x <= x_max)
    if y_min is not None:
        keep &= (y >= y_min)
    if y_max is not None:
        keep &= (y <= y_max)

    X_out = X_raw[keep]
    if y_raw is None:
        return X_out, keep
    return X_out, np.asarray(y_raw)[keep], keep

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


def plot_grid_or_tri(ax, x, y, values, title, cmap="plasma", vmin=None, vmax=None):
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
    return m


def quantiles_from_binned_probs(probs: torch.Tensor, edges: torch.Tensor, qs: list[float]) -> torch.Tensor:
    if probs.ndim == 2:
        probs = probs.unsqueeze(1)
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
# 2. Main comparison
# -----------------------------------------------------------------------------
def run_comparison(
    checkpoint_path,
    sacra_df,
    y_col="V",
    x_constant=0.3,
    y_constant=1.0,
    grid_size_x=100,
    grid_size_y=100,
    pad_frac=0.0,
    grid_x_min=None,
    grid_x_max=None,
    grid_y_min=None,
    grid_y_max=None,
    device="cuda",
    cmap="plasma",
    save_dir="sacra_plots",
):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading Transformer checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    if "feature_scale_init_vec" not in cfg_dict:
        cfg_dict["feature_scale_init_vec"] = None

    cfg = model.ModelConfig(**cfg_dict)
    args_dict = ckpt["args"]
    edges = data.make_bin_edges(args_dict["a"], args_dict["b"], args_dict["C"], device=device)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    net = model.TransformerUQ(cfg=cfg, edges=edges).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    # Full dataset context
    X_all_raw = sacra_df[["x", "y"]].to_numpy(dtype=float)
    y_all_raw = sacra_df[y_col].to_numpy(dtype=float)

    X_plot_raw, y_plot_raw, plot_keep = truncate_points_by_bounds(
        X_all_raw,
        y_all_raw,
        x_min=grid_x_min,
        x_max=grid_x_max,
        y_min=grid_y_min,
        y_max=grid_y_max,
    )

    # Build 100x100 grid over observed spatial extent
    gx, gy = make_regular_2d_grid(
        X_all_raw,
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        pad_frac=pad_frac,
        x_min=grid_x_min,
        x_max=grid_x_max,
        y_min=grid_y_min,
        y_max=grid_y_max,
    )
    X_grid_raw = np.column_stack([gx, gy])

    # Normalize using full-data context
    X_ctx_n, xparams = normalize_marginals_with_constant(X_all_raw, x_constant=x_constant)
    X_grid_n = apply_normalize_marginals(X_grid_raw, xparams)
    y_ctx, yparams = standardize_y(y_all_raw, y_constant=y_constant)

    # ==========================================
    # Part 1: Transformer inference on full grid
    # ==========================================
    X_ctx = torch.tensor(X_ctx_n, dtype=torch.float32, device=device).unsqueeze(0)
    Y_ctx = torch.tensor(y_ctx, dtype=torch.float32, device=device).unsqueeze(0)
    X_q_all = torch.tensor(X_grid_n, dtype=torch.float32, device=device)

    probs_list = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_grid_n), batch_size):
            chunk_xq = X_q_all[i : i + batch_size]
            B_curr = chunk_xq.shape[0]
            logits, _ = net(X_ctx.expand(B_curr, -1, -1), Y_ctx.expand(B_curr, -1), chunk_xq)
            probs_list.append(torch.softmax(logits, dim=-1))

    probs_all = torch.cat(probs_list, dim=0)
    mu_tf = (probs_all * bin_centers).sum(dim=-1).cpu().numpy()
    qs_tf = quantiles_from_binned_probs(probs_all, edges, [0.05, 0.95])
    q05_tf = qs_tf[:, 0].cpu().numpy()
    q95_tf = qs_tf[:, 1].cpu().numpy()

    # ==========================================
    # Part 2: GP baseline on full grid
    # ==========================================
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=0.0, normalize_y=False, n_restarts_optimizer=5, random_state=0
    )
    gp.fit(X_ctx_n, y_ctx)
    mu_gp, std_gp = gp.predict(X_grid_n, return_std=True)
    q05_gp = mu_gp - 1.645 * std_gp
    q95_gp = mu_gp + 1.645 * std_gp

    # Unstandardize back to original scale
    mu_tf = invert_standardize_y(mu_tf, yparams)
    q05_tf = invert_standardize_y(q05_tf, yparams)
    q95_tf = invert_standardize_y(q95_tf, yparams)
    mu_gp = invert_standardize_y(mu_gp, yparams)
    q05_gp = invert_standardize_y(q05_gp, yparams)
    q95_gp = invert_standardize_y(q95_gp, yparams)

    pooled = np.concatenate([mu_tf, mu_gp])
    vmin, vmax = np.percentile(pooled, [2, 98])

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

    def overlay_obs(ax):
        ax.scatter(X_plot_raw[:, 0], X_plot_raw[:, 1], s=10, c="#00FFFF", marker="x", alpha=0.65)

    m0 = plot_grid_or_tri(axes[0, 0], gx, gy, mu_tf, "PFN PPD Mean", cmap, vmin, vmax)
    overlay_obs(axes[0, 0])
    plot_grid_or_tri(axes[0, 1], gx, gy, q05_tf, "PFN PPD Lower 5%", cmap, vmin, vmax)
    plot_grid_or_tri(axes[0, 2], gx, gy, q95_tf, "PFN PPD Upper 95%", cmap, vmin, vmax)

    plot_grid_or_tri(axes[1, 0], gx, gy, mu_gp, "GP PPD Mean", cmap, vmin, vmax)
    overlay_obs(axes[1, 0])
    plot_grid_or_tri(axes[1, 1], gx, gy, q05_gp, "GP PPD Lower 5%", cmap, vmin, vmax)
    plot_grid_or_tri(axes[1, 2], gx, gy, q95_gp, "GP PPD Upper 95%", cmap, vmin, vmax)

    cbar = fig.colorbar(m0, ax=axes, location="right", shrink=0.6, pad=0.02)
    cbar.set_label(y_col)

    save_path = os.path.join(save_dir, "sacra_compare_full.pdf")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

    out_df = pd.DataFrame({
        "x": gx,
        "y": gy,
        "tf_mean": mu_tf,
        "tf_q05": q05_tf,
        "tf_q95": q95_tf,
        "gp_mean": mu_gp,
        "gp_q05": q05_gp,
        "gp_q95": q95_gp,
    })
    out_df.to_csv(os.path.join(save_dir, "sacra_compare_full_grid.csv"), index=False)
    print(f"Saved: {os.path.join(save_dir, 'sacra_compare_full_grid.csv')}")


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SACRA_CSV = "data/Sacramento.csv"
    CKPT_PATH = "runs/sacramento_eb/best.pt"

    if os.path.exists(CKPT_PATH) and os.path.exists(SACRA_CSV):
        sacra_df = pd.read_csv(SACRA_CSV)
        run_comparison(
            checkpoint_path=CKPT_PATH,
            sacra_df=sacra_df,
            y_col="V",
            x_constant=0.3,
            y_constant=1.0,
            grid_size_x=100,
            grid_size_y=100,
            grid_x_min = -121.53,
            grid_x_max = -121.33,
            grid_y_min = 38.38,
            grid_y_max = 38.69,
            save_dir="sacra_plots",
        )
    else:
        print(f"File not found.\nCheck: {CKPT_PATH} and {SACRA_CSV}")
