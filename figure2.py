"""figure2.py

Generalization plots across training context caps (n_max) and depth (L),
evaluated at various sample sizes n. Supports evaluating under different x_dist
(e.g. normal vs uniform) and caches results with a cache key that includes
evaluation distribution + hypers to avoid collisions.

This version changes/extends metrics:
  - Keeps: TV distance (discretized), MSE(y), MSE(mu)
  - Replaces MSE(tau) with MSE(mu2) where mu2 is the second moment E[Y^2]
  - Adds: 90% interval coverage, computed analytically under the true Normal.

Plotting:
  - Produces TWO figures: 
      1. Main: MSE(y), Coverage, Width
      2. Supp: TV, MSE(mu), MSE(mu2)
  - Visuals: 
      - Oracle lines are RED.
      - Down arrows added to loss labels.
      - Pretraining range (n_min, n_max) is shaded in gray.
      - Oracle legend appears only on relevant metric plots.
"""

import os
import glob
import json
import hashlib
from typing import List, Tuple, Optional, Dict, Callable

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import data
import model
import ppd_viewer as pv


# ---------------------------
# Utilities
# ---------------------------

def _seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _banner(msg: str) -> str:
    return f"\n{'='*90}\n{msg}\n{'='*90}"


def _fmt_path(p: str, root: str = "runs") -> str:
    try:
        if p.startswith(root + os.sep):
            return p[len(root) + 1 :]
    except Exception:
        pass
    return p


def resolve_checkpoint(root_dir: str, folder_pat: str) -> Dict[str, Optional[str]]:
    """Resolve checkpoint path from a run folder pattern."""
    base_dir = os.path.join(root_dir, folder_pat)

    # (1) finetune dirs
    ft_dirs = sorted(glob.glob(base_dir + "_finetune*"))
    best_ft_ckpt, best_ft_mtime, best_ft_dir = None, -1.0, None
    for d in ft_dirs:
        p = os.path.join(d, "best.pt")
        if os.path.isfile(p):
            mt = os.path.getmtime(p)
            if mt > best_ft_mtime:
                best_ft_mtime = mt
                best_ft_ckpt = p
                best_ft_dir = d

    if best_ft_ckpt is not None:
        note = f"FINETUNE ✓ ({os.path.basename(best_ft_dir)}/best.pt)"
        return {"ckpt_path": best_ft_ckpt, "note": note, "base_dir": base_dir}

    # (2) base best/final
    base_best = os.path.join(base_dir, "best.pt")
    if os.path.isfile(base_best):
        return {"ckpt_path": base_best, "note": "BASE best.pt", "base_dir": base_dir}

    base_final = os.path.join(base_dir, "final.pt")
    if os.path.isfile(base_final):
        return {"ckpt_path": base_final, "note": "BASE final.pt", "base_dir": base_dir}

    return {"ckpt_path": None, "note": None, "base_dir": base_dir}


def quantiles_from_binned_probs(
    probs: torch.Tensor,
    edges: torch.Tensor,
    qs: List[float],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute quantiles from binned probabilities with linear interpolation."""
    if probs.ndim == 2:
        probs = probs.unsqueeze(1)
    B, _, C = probs.shape
    edges_left = edges[:-1].view(1, 1, C).to(device=probs.device, dtype=probs.dtype).expand(B, 1, C)
    edges_right = edges[1:].view(1, 1, C).to(device=probs.device, dtype=probs.dtype).expand(B, 1, C)

    cdf = probs.cumsum(dim=-1)
    cdf = torch.clamp(cdf, 0.0, 1.0)
    cdf[..., -1] = 1.0

    out = []
    for q in qs:
        q_t = torch.tensor(q, device=probs.device, dtype=probs.dtype).view(1, 1, 1)
        ge = (cdf >= q_t).to(torch.int64)
        idx = torch.argmax(ge, dim=-1, keepdim=True)
        left = torch.gather(edges_left, -1, idx).squeeze(-1)
        right = torch.gather(edges_right, -1, idx).squeeze(-1)
        p_bin = torch.gather(probs, -1, idx).squeeze(-1)
        prev_idx = torch.clamp(idx - 1, min=0)
        prev_cdf = torch.gather(cdf, -1, prev_idx).squeeze(-1)
        prev_cdf = torch.where((idx.squeeze(-1) > 0), prev_cdf, torch.zeros_like(prev_cdf))
        frac = (q_t.squeeze(-1) - prev_cdf) / torch.clamp(p_bin, min=eps)
        frac = torch.clamp(frac, 0.0, 1.0)
        qval = left + frac * (right - left)
        out.append(qval)

    return torch.stack(out, dim=-1)


# ---------------------------
# Robust Evaluation
# ---------------------------

@torch.inference_mode()
def evaluate_robust(
    net: torch.nn.Module,
    eval_loader: list,
    device,
    dtype,
    edges: torch.Tensor,
    *,
    interval_level: float = 0.90,
) -> Dict[str, float]:
    net.eval()
    tvs, mse_mus, mse_mu2s, mse_ys = [], [], [], []
    covs_emp, width_models, width_trues = [], [], []

    edges = edges.to(device=device, dtype=torch.float64)
    a = float(edges[0].item())
    b = float(edges[-1].item())

    bin_centers = ((edges[:-1] + edges[1:]) / 2.0).view(1, -1)
    bin_centers2 = bin_centers ** 2

    for batch_tuple in eval_loader:
        X_pad, Y_pad, xq, yq, c_true, q_true, mu_true, tau_true = batch_tuple

        X_in = X_pad.to(device=device, dtype=dtype)
        Y_in = Y_pad.to(device=device, dtype=dtype)
        xq_in = xq.to(device=device, dtype=dtype)

        logits, _aux = net(X_in, Y_in, xq_in, padding_mask=None)
        probs = torch.softmax(logits, dim=-1).to(dtype=torch.float64)

        mu_numeric = (probs * bin_centers).sum(dim=-1)
        mu2_numeric = (probs * bin_centers2).sum(dim=-1)

        yq_in = yq.to(device=device, dtype=torch.float64).reshape(-1)
        mu_true = mu_true.to(device=device, dtype=torch.float64).reshape(-1)
        tau_true = tau_true.to(device=device, dtype=torch.float64).reshape(-1)
        mu2_true = mu_true ** 2 + tau_true

        mse_ys.append(F.mse_loss(mu_numeric, yq_in).item())
        mse_mus.append(F.mse_loss(mu_numeric, mu_true).item())
        mse_mu2s.append(F.mse_loss(mu2_numeric, mu2_true).item())

        if q_true is not None:
            q_true_dev = q_true.to(device=device, dtype=torch.float64)
            if q_true_dev.ndim == 3:
                q_true_dev = q_true_dev.squeeze(1)
            tv = 0.5 * torch.abs(probs - q_true_dev).sum(dim=-1).mean()
            tvs.append(tv.item())
        else:
            tvs.append(0.0)

        out_m = pv.interval_coverage_from_bins(
            probs=probs, edges=edges, y=yq_in, level=float(interval_level),
        )
        covs_emp.append(float(out_m["coverage"]))
        width_models.append(float(out_m["width_mean"]))

        out_t = pv.interval_coverage_trunc_normal(
            y=yq_in, mu=mu_true, tau=tau_true, a=a, b=b, level=float(interval_level),
        )
        width_trues.append(float(out_t["width_mean"]))

    return {
        "eval_tv": float(np.mean(tvs)),
        "eval_mse_y": float(np.mean(mse_ys)),
        "eval_mse_mu": float(np.mean(mse_mus)),
        "eval_mse_mu2": float(np.mean(mse_mu2s)),
        "eval_cov_emp": float(np.mean(covs_emp)),
        "eval_width_model": float(np.mean(width_models)),
        "eval_width_true": float(np.mean(width_trues)),
    }


# ---------------------------
# GPU Eval Loader
# ---------------------------

def get_eval_loader(
    task: str, B: int, n: int, d: int, device, dtype, seed: int, edges: torch.Tensor,
    sigma: float, num_batches: int = 64, x_dist: str = "normal",
    rbf_alpha: float = 1.0, rbf_ell: float = 0.8, rbf_jitter: float = 1e-6,
    blr_diag_var=None,
) -> List[Tuple]:
    _seed_all(seed)
    loader = []

    if task == "blr":
        hypers = data.BLR_hyprms(sigma=sigma, diag_var=blr_diag_var)
    elif task == "rbf":
        hypers = data.RBF_hyprms(sigma=sigma, alpha=rbf_alpha, ell=rbf_ell, jitter=rbf_jitter)
    elif task == "softmax":
        hypers = data.Softmax_hyprms(sigma=sigma, alpha=rbf_alpha, jitter=rbf_jitter)
    else:
        raise ValueError(f"Unknown task {task}")

    for _ in range(num_batches):
        if task == "blr":
            batch_dict = data.draw_BLR_batch(
                B, n, d, hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges
            )
        elif task == "rbf":
            batch_dict = data.draw_RBF_batch(
                B, n, d, hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges
            )
        elif task == "softmax":
            batch_dict = data.draw_softmax_batch(
                B, n, d, hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges
            )
        else:
            raise ValueError(f"Unknown task {task}")

        X = batch_dict["X"]
        Y = batch_dict["Y"]
        xq = batch_dict["xq"]
        if Y.ndim == 3: Y = Y.squeeze(-1)
        if xq.ndim == 3: xq = xq.squeeze(1)

        loader.append((
            X, Y, xq,
            batch_dict["yq"], batch_dict["c_true"], batch_dict["q_true"],
            batch_dict["mu"], batch_dict["tau"],
        ))

    return loader


# ---------------------------
# Metrics driver
# ---------------------------

def choose_eval_params(n: int, target_total: int = 5000):
    if n <= 200: B = 256
    elif n <= 600: B = 128
    else: B = 64
    num_batches = max(1, int(np.ceil(target_total / B)))
    return B, num_batches


def compute_metrics(
    net, task: str, sample_sizes, d: int, device, dtype, edges: torch.Tensor,
    *, x_dist: str, sigma: float, rbf_alpha: float = 1.0, rbf_ell: float = 0.8,
    rbf_jitter: float = 1e-6, seed: int = 0, interval_level: float = 0.90,
):
    results = {
        "n": [], "tv": [], "mse_y": [], "mse_mu": [], "mse_mu2": [],
        "cov90": [], "width90": [], "width90_true": [],
    }

    print("    ↳ eval n:", end=" ")
    for n_eval in sample_sizes:
        n_eval = int(n_eval)
        eval_B, num_batches = choose_eval_params(n_eval)
        print(f"{n_eval}(B={eval_B},nb={num_batches})", end="  ", flush=True)

        dl = get_eval_loader(
            task=task, B=eval_B, n=n_eval, d=d, device=device, dtype=dtype, seed=seed + n_eval,
            edges=edges, sigma=sigma, num_batches=num_batches, x_dist=x_dist,
            rbf_alpha=rbf_alpha, rbf_ell=rbf_ell, rbf_jitter=rbf_jitter,
        )

        metrics = evaluate_robust(
            net, eval_loader=dl, device=device, dtype=dtype, edges=edges, interval_level=interval_level,
        )

        results["n"].append(n_eval)
        results["tv"].append(metrics["eval_tv"])
        results["mse_y"].append(metrics["eval_mse_y"])
        results["mse_mu"].append(metrics["eval_mse_mu"])
        results["mse_mu2"].append(metrics["eval_mse_mu2"])
        results["cov90"].append(metrics["eval_cov_emp"])
        results["width90"].append(metrics["eval_width_model"])
        results["width90_true"].append(metrics["eval_width_true"])

    print("")
    return results


# ---------------------------
# Caching
# ---------------------------

def get_cache_filename(
    ckpt_path: str, n_eval_list, *, x_dist: str, sigma: float, task: str, d_eval: int,
    C: int, rbf_alpha: float = 1.0, rbf_ell: float = 0.8, rbf_jitter: float = 1e-6,
    cache_version: str = "v4", interval_level: float = 0.90,
):
    payload = {
        "v": cache_version, "task": task, "d": int(d_eval), "C": int(C),
        "x_dist": str(x_dist), "sigma": float(sigma), "n_eval": list(map(int, list(n_eval_list))),
        "rbf_alpha": float(rbf_alpha), "rbf_ell": float(rbf_ell),
        "rbf_jitter": float(rbf_jitter), "interval_level": float(interval_level),
    }
    s = json.dumps(payload, sort_keys=True)
    h = hashlib.md5(s.encode()).hexdigest()[:10]
    base = os.path.basename(os.path.dirname(ckpt_path))
    return os.path.join(os.path.dirname(ckpt_path), f"metrics_cache_{base}_{h}.json")


# ---------------------------
# Plotting helpers
# ---------------------------

def _apply_pub_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.linestyle": "--",
        "grid.alpha": 0.25,
    })


def _force_shared_ylims_per_row(axes_2d: np.ndarray):
    rows = axes_2d.shape[0]
    cols = axes_2d.shape[1]
    for r in range(rows):
        ymins, ymaxs = [], []
        for c in range(cols):
            ymin, ymax = axes_2d[r, c].get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
        row_ymin, row_ymax = min(ymins), max(ymaxs)
        for c in range(cols):
            axes_2d[r, c].set_ylim(row_ymin, row_ymax)


# ---------------------------
# Main routine
# ---------------------------

def validate_config(task: str, kernel: str, normalize: int, norm_decay: int):
    if task == "blr" and kernel != "linear":
        raise ValueError("BLR requires linear kernel")
    if task == "softmax" and kernel != "softmax":
        raise ValueError("Softmax task should use softmax kernel")
    print(_banner(f"Figure 2 config | task={task} kernel={kernel} norm={normalize} decay={norm_decay}"))


def figure2_generalization(
    task: str, kernel: str, mode: str, normalize: int, norm_decay: int,
    C: int, list_of_Ls: List[int], list_of_nmax: List[int], sample_sizes,
    *, root_dir: str = "runs", device: str = "cuda", d: int = 2,
    n_min: int = 64, x_dist: str = "normal", cache: bool = True,
    out_dir: str = "figs", interval_level: float = 0.90,
    seed: int = 0,
):
    validate_config(task, kernel, normalize, norm_decay)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    t_device = torch.device(device)
    dtype = torch.float64

    _apply_pub_style()

    def _log(x: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(x, 1e-12))

    # Updated metric definitions with arrows
    metric_defs: Dict[str, Dict] = {
        "tv": {"label": r"Log TV $\downarrow$", "transform": _log, "hline": None},
        "mse_y": {"label": r"Log MSE $y$ $\downarrow$", "transform": _log, "hline": None},
        "mse_mu": {"label": r"Log MSE $\mu$ $\downarrow$", "transform": _log, "hline": None},
        "mse_mu2": {"label": r"Log MSE $\mathbb{E}[Y^2]$ $\downarrow$", "transform": _log, "hline": None},
        "cov90": {"label": f"{int(interval_level*100)}% Coverage", "transform": lambda x: x, "hline": interval_level},
        "width90": {"label": f"{int(interval_level*100)}% Width", "transform": lambda x: x, "hline": None},
    }

    plot_groups = [
        ("main", ["mse_y", "cov90", "width90"]),
        ("supp", ["tv", "mse_mu", "mse_mu2"]),
    ]
    
    nd_tag = "_nd" if norm_decay else ""
    norm_tag = "_norm" if normalize else ""
    k_tag = "_softmax" if kernel == "softmax" else ""

    colors = sns.color_palette("viridis", len(list_of_Ls))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    ncols = len(list_of_nmax)
    used_ckpts: List[Tuple[int, int, str]] = []

    metrics_bank: Dict[Tuple[int, int], Dict[str, List[float]]] = {}

    # --- 1. Compute/Load Metrics ---
    for col_idx, n_max in enumerate(list_of_nmax):
        print(f"\n🧩 Column {col_idx+1}/{ncols} | Train context cap N_max={n_max}")
        for l_idx, L in enumerate(list_of_Ls):
            folder_pat = (
                f"{task}_d{d}_C{C}_L{L}_mixture_n{n_min}-{n_max}_{mode}"
                f"{norm_tag}{nd_tag}{k_tag}_seed{seed}"
            )
            ckpt_info = resolve_checkpoint(root_dir, folder_pat)
            ckpt_path = ckpt_info["ckpt_path"]
            if ckpt_path is None:
                continue

            try:
                state = torch.load(ckpt_path, map_location="cpu")
                ckpt_args = state["args"]
                d_ckpt = int(ckpt_args.get("d", d))
                C_eval = int(ckpt_args.get("C", C))
                
                sigma_eval = float(ckpt_args.get("sigma", 0.2))
                rbf_alpha = float(ckpt_args.get("rbf_alpha", 1.0))
                rbf_ell = float(ckpt_args.get("rbf_ell", 0.8))
                rbf_jitter = float(ckpt_args.get("rbf_jitter", 1e-6))

                cache_file = get_cache_filename(
                    ckpt_path, sample_sizes, x_dist=x_dist, sigma=sigma_eval, task=task,
                    d_eval=d_ckpt, C=C_eval, rbf_alpha=rbf_alpha, rbf_ell=rbf_ell,
                    rbf_jitter=rbf_jitter, interval_level=interval_level,
                )
            except Exception:
                continue

            key = (n_max, L)

            if cache and os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    metrics_bank[key] = json.load(f)
            else:
                try:
                    edges = data.make_bin_edges(
                        ckpt_args["a"], ckpt_args["b"], C_eval, device=t_device, dtype=dtype
                    )
                    cfg_dict = state["cfg"]
                    if "gradient_checkpointing" not in cfg_dict:
                        cfg_dict["gradient_checkpointing"] = False
                    model_cfg = model.ModelConfig(**cfg_dict)

                    net = model.TransformerUQ(model_cfg, edges).to(t_device, dtype=dtype)
                    net.load_state_dict(state["model_state"], strict=False)

                    metrics_bank[key] = compute_metrics(
                        net, task, sample_sizes, d=d_ckpt, device=t_device, dtype=dtype,
                        edges=edges, x_dist=x_dist, sigma=sigma_eval, rbf_alpha=rbf_alpha,
                        rbf_ell=rbf_ell, rbf_jitter=rbf_jitter, interval_level=interval_level,
                    )

                    with open(cache_file, "w") as f:
                        json.dump(metrics_bank[key], f)
                except Exception:
                    continue

            pretty_ckpt = _fmt_path(ckpt_path, root=root_dir)
            used_ckpts.append((n_max, L, pretty_ckpt))

    if len(metrics_bank) == 0:
        raise SystemExit("No metrics computed.")

    # --- 2. Generate Plots ---
    os.makedirs(out_dir, exist_ok=True)
    n_s, n_e = int(sample_sizes[0]), int(sample_sizes[-1])

    for group_name, group_rows in plot_groups:
        nrows = len(group_rows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols * 1.1, 2.2 * nrows * 1.1), sharex=True)
        if ncols == 1: axes = np.array(axes).reshape(nrows, 1)
        if nrows == 1: axes = np.array(axes).reshape(1, ncols)

        for col_idx, n_max in enumerate(list_of_nmax):
            for l_idx, L in enumerate(list_of_Ls):
                key = (n_max, L)
                if key not in metrics_bank: continue
                data_metrics = metrics_bank[key]
                x_vals = data_metrics["n"]

                for row_idx, metric_key in enumerate(group_rows):
                    ax = axes[row_idx, col_idx]
                    y_raw = np.array(data_metrics[metric_key], dtype=float)
                    y_vals = metric_defs[metric_key]["transform"](y_raw)

                    ax.plot(
                        x_vals, y_vals, color=colors[l_idx], marker=markers[l_idx % len(markers)],
                        markersize=4, linewidth=1.3, alpha=0.9,
                        label=f"L={L}" if (row_idx == 0 and col_idx == 0) else None,
                    )

                    # Oracle Width Line
                    if metric_key == "width90":
                        baseline_series, x_vals0 = None, None
                        # Grab baseline from any valid L
                        for L_cand in list_of_Ls:
                            key_cand = (n_max, L_cand)
                            if key_cand in metrics_bank:
                                baseline_series = metrics_bank[key_cand].get("width90_true", None)
                                x_vals0 = metrics_bank[key_cand].get("n", None)
                                if baseline_series is not None: break
                        
                        if baseline_series is not None:
                            # Color RED, no label in plot (handled by legend manually)
                            ax.plot(x_vals0, baseline_series, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.9)

        # Formatting & Oracle Legends
        oracle_line = mlines.Line2D([], [], color='#d62728', linestyle='--', linewidth=1.5, label='True PPD')

        for col_idx, n_max in enumerate(list_of_nmax):
            for row_idx, metric_key in enumerate(group_rows):
                ax = axes[row_idx, col_idx]
                ax.grid(True, which="both", linestyle="--", alpha=0.25)
                
                if row_idx == 0:
                    ax.set_title(f"$n_{{max}}={n_max}$", fontsize=12)
                if col_idx == 0:
                    ax.set_ylabel(metric_defs[metric_key]["label"], fontsize=11)
                if row_idx == nrows - 1:
                    ax.set_xlabel("Eval Sample Size ($n$)", fontsize=11)
                
                # --- NEW: Shaded Training Range ---
                ax.axvspan(n_min, n_max, color="gray", alpha=0.1, lw=0)
                ax.axvline(n_min, color="gray", linestyle=":", alpha=0.5)
                ax.axvline(n_max, color="gray", linestyle=":", alpha=0.5)
                
                # Oracle Coverage Line
                hline = metric_defs[metric_key].get("hline", None)
                if hline is not None and metric_key == "cov90":
                    ax.axhline(hline, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.9)

                # Add Oracle Legend ONLY to relevant metrics
                if metric_key in ["cov90", "width90"] and col_idx == 0:
                    # If it's the very first plot (top-left) where Depth legend lives
                    if row_idx == 0:
                        # Re-add Depth legend to preserve it
                        leg_depth = ax.get_legend()
                        ax.legend(handles=[oracle_line], loc="best", frameon=False, fontsize=9)
                        if leg_depth:
                            ax.add_artist(leg_depth)
                    else:
                        # Otherwise just add it normally
                        ax.legend(handles=[oracle_line], loc="best", frameon=False, fontsize=9)

        _force_shared_ylims_per_row(axes)
        
        # 1. Main Legend (Depth) - Top Left
        leg1 = axes[0, 0].legend(loc="best", frameon=False)
        
        filename = (
            f"figure2_{group_name}_{task}_{mode}_{x_dist}_{kernel}_C{C}{norm_tag}{nd_tag}"
            f"_d{d}_nmin{n_min}_n{n_s}-{n_e}.pdf"
        )
        save_path = os.path.join(out_dir, filename)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(_banner(f"✅ Saved figure ({group_name}) → {os.path.abspath(save_path)}"))
        plt.close(fig)

    print(_banner("Checkpoints used:"))
    for (n_max, L, p) in sorted(set(used_ckpts), key=lambda x: (x[0], x[1])):
        print(f"  n_max={n_max:>4} | L={L:>3} | {p}")


if __name__ == "__main__":
    task = "rbf"
    C = 256
    LIST_OF_Ls = [4, 8, 16, 32]
    n_min = 64
    LIST_OF_NMAX = [128, 256, 512]
    SAMPLE_SIZES = [64 * i for i in range(1, 16)]
    cache = False

    for d in [16, 8, 4]:
        for mode in ["learnable"]:
            for ker in ["softmax"]:
                for x_dist in ["normal"]:
                    figure2_generalization(
                        task=task,
                        kernel=ker,
                        mode=mode,
                        normalize=1,
                        norm_decay=1,
                        C=C,
                        list_of_Ls=LIST_OF_Ls,
                        list_of_nmax=LIST_OF_NMAX,
                        sample_sizes=SAMPLE_SIZES,
                        d=d,
                        n_min=n_min,
                        x_dist=x_dist,
                        cache=cache,
                        interval_level=0.90,
                        seed=0,
                    )