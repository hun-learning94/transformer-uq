#!/usr/bin/env python3
"""figure3.py
Generalization Failure of Unnormalized Attention.
Layout: 1x2 Plots 
Metric: Log TV Distance (discretized)
"""

import os
import glob
import json
import hashlib
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

import data
import model

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

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
    base_dir = os.path.join(root_dir, folder_pat)
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

    base_best = os.path.join(base_dir, "best.pt")
    if os.path.isfile(base_best):
        return {"ckpt_path": base_best, "note": "BASE best.pt", "base_dir": base_dir}

    base_final = os.path.join(base_dir, "final.pt")
    if os.path.isfile(base_final):
        return {"ckpt_path": base_final, "note": "BASE final.pt", "base_dir": base_dir}

    return {"ckpt_path": None, "note": None, "base_dir": base_dir}

# -----------------------------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_tv(
    net: torch.nn.Module,
    eval_loader: list,
    device,
    dtype,
    edges: torch.Tensor,
) -> float:
    """Compute mean TV distance between model bins and oracle bins."""
    net.eval()
    tvs = []
    
    for batch_tuple in eval_loader:
        # Tuple unpacking matches collate_pad in train.py
        # X, Y, xq, yq, c_true, q_true, mu, tau, mask
        X_pad, Y_pad, xq, _, _, q_true, _, _, mask = batch_tuple
        
        X_in = X_pad.to(device=device, dtype=dtype)
        Y_in = Y_pad.to(device=device, dtype=dtype)
        xq_in = xq.to(device=device, dtype=dtype)
        mask_in = mask.to(device=device)

        logits, _ = net(X_in, Y_in, xq_in, padding_mask=mask_in)
        probs = torch.softmax(logits, dim=-1).to(dtype=torch.float64)

        # q_true is the oracle binned probability vector (from data.draw_... functions)
        if q_true is not None:
            q_true_dev = q_true.to(device=device, dtype=torch.float64)
            if q_true_dev.ndim == 3:
                q_true_dev = q_true_dev.squeeze(1)
            
            # TV = 0.5 * sum |p - q|
            tv = 0.5 * torch.abs(probs - q_true_dev).sum(dim=-1).mean()
            tvs.append(tv.item())
        else:
            tvs.append(0.0)

    return float(np.mean(tvs))

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
    else:
        raise ValueError(f"Unknown task {task}")

    for _ in range(num_batches):
        if task == "blr":
            batch = data.draw_BLR_batch(B, n, d, hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges)
        else:
            batch = data.draw_RBF_batch(B, n, d, hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges)
        
        # Manually create the tuple structure expected by evaluate_tv
        # We simulate the padding mask as all-ones since we generate fixed size n here
        X = batch["X"]
        Y = batch["Y"]
        xq = batch["xq"]
        B_curr = X.shape[0]
        n_curr = X.shape[1]
        
        # Fix dimensions
        if Y.ndim == 3: Y = Y.squeeze(-1)
        if xq.ndim == 3: xq = xq.squeeze(1)
        
        # Mask: (B, n+1, 1)
        mask = torch.ones(B_curr, n_curr + 1, 1, device=device, dtype=torch.bool)

        loader.append((
            X, Y, xq, 
            batch["yq"], batch["c_true"], batch["q_true"], batch["mu"], batch["tau"], mask
        ))
    return loader

def choose_eval_params(n: int, target_total: int = 4096):
    if n <= 200: B = 256
    elif n <= 600: B = 128
    else: B = 64
    num_batches = max(1, int(np.ceil(target_total / B)))
    return B, num_batches

def compute_metrics(
    net, task: str, sample_sizes, d: int, device, dtype, edges: torch.Tensor,
    *, x_dist: str, sigma: float,
    rbf_alpha: float = 1.0, rbf_ell: float = 0.8, rbf_jitter: float = 1e-6,
    seed: int = 0
):
    results = {"n": [], "tv": []}
    
    print("    ↳ eval n:", end=" ")
    for n_eval in sample_sizes:
        n_eval = int(n_eval)
        B, num_batches = choose_eval_params(n_eval)
        print(f"{n_eval}", end=" ", flush=True)
        
        dl = get_eval_loader(
            task=task, B=B, n=n_eval, d=d, device=device, dtype=dtype, 
            seed=seed + n_eval, edges=edges, sigma=sigma, num_batches=num_batches,
            x_dist=x_dist, rbf_alpha=rbf_alpha, rbf_ell=rbf_ell, rbf_jitter=rbf_jitter
        )
        
        tv = evaluate_tv(net, dl, device, dtype, edges)
        results["n"].append(n_eval)
        results["tv"].append(tv)
        
    print("")
    return results

def get_cache_filename(ckpt_path, n_eval_list, **kwargs):
    # Create a unique cache filename based on config
    payload = {k: str(v) for k, v in kwargs.items()}
    payload["n_eval"] = list(map(int, list(n_eval_list)))
    s = json.dumps(payload, sort_keys=True)
    h = hashlib.md5(s.encode()).hexdigest()[:10]
    base = os.path.basename(os.path.dirname(ckpt_path))
    return os.path.join(os.path.dirname(ckpt_path), f"metrics_cache_fig3_{base}_{h}.json")

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _apply_pub_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.25,
    })

def figure3_plot(
    configs: List[Dict[str, Any]], 
    out_dir: str = "figs",
    root_dir: str = "runs",
    cache: bool = True,
    device: str = "cuda"
):
    _apply_pub_style()
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    t_device = torch.device(device)
    dtype = torch.float64
    
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3), constrained_layout=True, sharey=True)
    
    # Visual Styling
    colors = {"Unnormalized": "black", "Normalized": "black"} 
    linestyles = {"Unnormalized": "-", "Normalized": "-"}

    for i, cfg in enumerate(configs):
        ax = axes[i]
        task = cfg["task"]
        d = cfg["d"]
        C = cfg["C"]
        L = cfg["L"]
        n_min = cfg["n_min"]
        n_max = cfg["n_max"]
        sigma = cfg["sigma"]
        x_dist = cfg["x_dist"]
        eval_n = cfg["eval_n"]
        variants = cfg["variants"]
        
        print(_banner(f"Plotting Panel {i+1}: {task.upper()} (d={d}, L={L}, n={n_min}-{n_max})"))
        
        for var in variants:
            label = var["label"]
            normalize = var["normalize"]
            norm_decay = var["norm_decay"]
            
            # Auto-kernel logic matches training script conventions
            kernel_arg = var.get("kernel", "auto")
            if kernel_arg == "auto":
                kernel = "linear" if task == "blr" else "rbf"
            else:
                kernel = kernel_arg
            
            # Construct folder pattern to find checkpoint
            norm_tag = "_norm" if normalize else ""
            nd_tag = "_nd" if norm_decay else ""
            k_tag = "_softmax" if kernel == "softmax" else ""
            
            folder_pat = (
                f"{task}_d{d}_C{C}_L{L}_mixture_n{n_min}-{n_max}_learnable"
                f"{norm_tag}{nd_tag}{k_tag}_seed0"
            )
            
            ckpt_info = resolve_checkpoint(root_dir, folder_pat)
            ckpt_path = ckpt_info["ckpt_path"]
            
            if ckpt_path is None:
                print(f"  ❌ Missing Checkpoint: {folder_pat}")
                # Plot dummy line to show missing data if needed, or skip
                continue
                
            print(f"  🔍 Found: {ckpt_info['note']} | {label}")
            
            # --- Caching / Computation ---
            # Load checkpoint args to ensure we use correct sigma/hypers for eval
            state = torch.load(ckpt_path, map_location="cpu")
            ckpt_args = state["args"]
            sigma_eval = float(ckpt_args.get("sigma", 0.2))
            rbf_alpha = float(ckpt_args.get("rbf_alpha", 1.0))
            rbf_ell = float(ckpt_args.get("rbf_ell", 0.8))
            rbf_jitter = float(ckpt_args.get("rbf_jitter", 1e-6))
            
            cache_file = get_cache_filename(
                ckpt_path, eval_n, task=task, d=d, C=C, x_dist=x_dist, 
                sigma=sigma_eval, normalize=normalize, norm_decay=norm_decay
            )
            
            if cache and os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    metrics = json.load(f)
                print(f"    ↳ Cache Hit ✅")
            else:
                # Need to compute
                edges = data.make_bin_edges(ckpt_args["a"], ckpt_args["b"], C, device=t_device, dtype=dtype)
                
                # Reconstruct Model
                cfg_dict = state["cfg"]
                if "gradient_checkpointing" not in cfg_dict: cfg_dict["gradient_checkpointing"] = False
                model_cfg = model.ModelConfig(**cfg_dict)
                
                net = model.TransformerUQ(model_cfg, edges).to(t_device, dtype=dtype)
                net.load_state_dict(state["model_state"], strict=False)
                
                metrics = compute_metrics(
                    net, task, eval_n, d, t_device, dtype, edges,
                    x_dist=x_dist, sigma=sigma_eval,
                    rbf_alpha=rbf_alpha, rbf_ell=rbf_ell, rbf_jitter=rbf_jitter
                )
                
                with open(cache_file, "w") as f:
                    json.dump(metrics, f)
            
            # --- Plotting Line ---
            x_vals = metrics["n"]
            # Log TV
            y_vals = np.log(np.maximum(metrics["tv"], 1e-12))
            
            ax.plot(x_vals, y_vals, label=label, 
                    color=colors.get(label, "black"), 
                    linestyle=linestyles.get(label, "-"),
                    marker="o", markersize=4, alpha=0.8, linewidth=1.5)

        # --- Formatting ---
        var_label = variants[0]["label"]
        ax.set_title(f"{task.upper()} {var_label}")
        ax.set_xlabel("Eval Sample Size ($n$)")
        if i == 0:
            ax.set_ylabel(r"Log TV $\downarrow$")
        
        # Mark Training Range
        ax.axvspan(n_min, n_max, color="gray", alpha=0.1, label="Train Range" if i == 0 else None)
        ax.axvline(n_min, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(n_max, color="gray", linestyle=":", alpha=0.5)
        
        # # Legend (only if labels exist)
        # handles, labels = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend()

    save_path = os.path.join(out_dir, f"figure3_d{d}.pdf")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(_banner(f"✅ Saved figure → {os.path.abspath(save_path)}"))


if __name__ == "__main__":
    # -----------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------
    
    # 1. Evaluation Sample Sizes
    # Dense grid for BLR around 128-512
    eval_n_blr = [64 * i for i in range(1, 16)]
    
    # Dense grid for RBF around 64-128
    eval_n_rbf = np.arange(1, 18) * 10 + 30
    d = 16
    # 2. Panel Definitions
    panels = [
        # LEFT: RBF (Unnormalized)
        {
            "task": "rbf",
            "d": d, "C": 256, "L": 32,
            "n_min": 64, "n_max": 128,
            "sigma": 0.2, "x_dist": "normal",
            "eval_n": eval_n_rbf,
            "variants": [
                {"label": "Unnormalized", "normalize": 0, "norm_decay": 0, "kernel": "rbf"},
            ]
        },
        # RIGHT: RBF (Normallized)
        {
            "task": "rbf",
            "d": d, "C": 256, "L": 32,
            "n_min": 64, "n_max": 128,
            "sigma": 0.2, "x_dist": "normal",
            "eval_n": eval_n_rbf,
            "variants": [
                {"label": "Normalized", "normalize": 1, "norm_decay": 1, "kernel": "rbf"},
            ]
        }
    ]

    figure3_plot(panels, cache=False)