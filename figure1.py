"""
figure1.py

GPU-only evaluation for a grid of checkpoints, using mixture of sample sizes within pretraining range:
  n ~ Uniform{n_min,...,n_max} per eval batch.

- Loads checkpoint (prefers finetune best.pt if present)
- Rebuilds model from ckpt cfg (backward-compatible)
- Generates eval data on GPU (float64 for rbf, float32 for blr)
- Computes:
    eval_nll, eval_kl (continuous), eval_tv (continuous), eval_mse_mu, eval_mse_tau
- Caches per-run metrics into: evaluation_metrics.json

Usage (example):
    python figure1.py --task blr --d 5 --n_min 128 --n_max 512 --metric eval_tv --normalize 0 --norm_decay 0
    python figure1.py --task rbf --d 2 --n_min 64 --n_max 128 --metric eval_tv --normalize 0 --norm_decay 0
"""

import os
import glob
import json
import argparse
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

import data
import model
from train import continuous_kl_tv_between_oracle_and_tf


# --------------------------------------------------------------------------------------
# checkpoint selection
# --------------------------------------------------------------------------------------

def resolve_best_checkpoint(base_dir: str) -> Tuple[Optional[str], str]:
    """
    Priority:
      1) latest-modified base_dir_finetune*/best.pt
      2) base_dir/best.pt
      3) base_dir/final.pt
    """
    ft_dirs = sorted(glob.glob(base_dir + "_finetune*"))
    best_ft, best_mtime, best_ft_dir = None, -1.0, None
    for d in ft_dirs:
        p = os.path.join(d, "best.pt")
        if os.path.isfile(p):
            mt = os.path.getmtime(p)
            if mt > best_mtime:
                best_ft, best_mtime, best_ft_dir = p, mt, d
    if best_ft is not None:
        return best_ft, f"FINETUNE best.pt ({os.path.basename(best_ft_dir)})"

    p = os.path.join(base_dir, "best.pt")
    if os.path.isfile(p):
        return p, "BASE best.pt"

    p = os.path.join(base_dir, "final.pt")
    if os.path.isfile(p):
        return p, "BASE final.pt"

    return None, "MISSING"


# --------------------------------------------------------------------------------------
# GPU eval
# --------------------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_mixture_on(
    net: torch.nn.Module,
    *,
    task: str,
    d: int,
    edges: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    eval_batches: int,
    batch_size: int,
    n_mode: str,
    n: int,
    n_min: int,
    n_max: int,
    sigma: float,
    x_dist: str,
    blr_diag_var: Optional[torch.Tensor],
    rbf_alpha: float,
    rbf_ell: float,
    rbf_jitter: float,
    num_grid: int = 512,
) -> Dict[str, float]:
    """
    Generate eval batches on GPU and compute continuous KL/TV vs oracle.
    Mixture: n ~ Uniform[n_min, n_max] per batch if n_mode != "single".
    """
    net.eval()

    nlls, kls, tvs, mse_mus, mse_taus = [], [], [], [], []

    # ensure edges on device (train's continuous_kl_tv... will cast to float64 internally)
    edges_dev = edges.to(device=device)

    # hypers
    if task == "blr":
        hypers = data.BLR_hyprms(sigma=sigma, diag_var=blr_diag_var)
    elif task == "rbf":
        hypers = data.RBF_hyprms(sigma=sigma, alpha=rbf_alpha, ell=rbf_ell, jitter=rbf_jitter)
    else:
        raise ValueError(task)

    for _ in range(eval_batches):
        if n_mode == "single":
            nn = int(n)
        else:
            nn = int(torch.randint(low=n_min, high=n_max + 1, size=(1,), device=device).item())

        if task == "blr":
            batch = data.draw_BLR_batch(
                B=batch_size, n=nn, d=d, hypers=hypers,
                device=device, dtype=dtype, x_dist=x_dist, edges=edges_dev
            )
        else:
            batch = data.draw_RBF_batch(
                B=batch_size, n=nn, d=d, hypers=hypers,
                device=device, dtype=dtype, x_dist=x_dist, edges=edges_dev
            )

        X = batch["X"]          # (B, nn, d)
        Y = batch["Y"]          # (B, nn) or (B, nn, 1)
        xq = batch["xq"]        # (B, d) or (B, 1, d)
        c_true = batch["c_true"]
        mu_true = batch["mu"]
        tau_true = batch["tau"]

        if Y.ndim == 3:
            Y = Y.squeeze(-1)
        if xq.ndim == 3:
            xq = xq.squeeze(1)

        logits, aux = net(X, Y, xq, padding_mask=None)

        # NLL (discrete)
        nll = torch.nn.functional.cross_entropy(logits, c_true)
        nlls.append(float(nll.item()))

        # probs in float64 for accurate continuous metrics
        probs = torch.softmax(logits, dim=-1).to(dtype=torch.float64)

        # parameter MSEs: use aux (model internal belief)
        aux_mu = aux["mu"].to(dtype=torch.float64)
        aux_tau = aux["tau"].to(dtype=torch.float64)
        mse_mus.append(float(torch.nn.functional.mse_loss(aux_mu, mu_true.to(aux_mu)).item()))
        mse_taus.append(float(torch.nn.functional.mse_loss(aux_tau, tau_true.to(aux_tau)).item()))

        # continuous KL/TV
        cont = continuous_kl_tv_between_oracle_and_tf(
            probs=probs, mu=mu_true, tau=tau_true, edges=edges_dev, num_grid=num_grid
        )
        kls.append(float(cont["kl_cont"].item()))
        tvs.append(float(cont["tv_cont"].item()))

    print(f"    done: nll={np.mean(nlls):.4g}, tv={np.mean(tvs):.4g}, kl={np.mean(kls):.4g}")
    return {
        "eval_nll": float(np.mean(nlls)),
        "eval_kl": float(np.mean(kls)),
        "eval_tv": float(np.mean(tvs)),
        "eval_mse_mu": float(np.mean(mse_mus)),
        "eval_mse_tau": float(np.mean(mse_taus)),
        # helpful extras
        "eval_batches": int(eval_batches),
        "batch_size": int(batch_size),
        "n_mode": str(n_mode),
        "n_min": int(n_min),
        "n_max": int(n_max),
    }


def run_evaluation_on_ckpt(
    ckpt_path: str,
    *,
    device: str = "cuda",
    eval_batches: int = 64,
    batch_size: int = 128,
    seed: int = 0,
    use_cache: bool = True,
    cache_name: str = "evaluation_metrics.json",
) -> Optional[Dict[str, float]]:
    if not os.path.exists(ckpt_path):
        return None

    run_dir = os.path.dirname(ckpt_path)
    cache_file = os.path.join(run_dir, cache_name)

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Corrupt cache file at {cache_file}. Recomputing...")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    ckpt_cfg = ckpt.get("cfg", {})

    task = ckpt_args.get("task", "rbf")

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    # dtype policy: rbf float64, blr float32 (you can force float64 if you want)
    dt = torch.float64 if task == "rbf" else torch.float32

    # bins
    a = ckpt_args["a"]
    b = ckpt_args["b"]
    C = ckpt_args["C"]
    edges = data.make_bin_edges(a, b, C, device=dev, dtype=dt)

    # cfg filtering for backward compat
    valid_keys = {f.name for f in fields(model.ModelConfig)}
    filtered_cfg = {k: v for k, v in ckpt_cfg.items() if k in valid_keys}

    if "mode" not in filtered_cfg:
        filtered_cfg["mode"] = ckpt_args.get("mode", "learnable")
    if "normalize" not in filtered_cfg:
        filtered_cfg["normalize"] = bool(ckpt_args.get("normalize", 0))
    if "norm_decay" not in filtered_cfg:
        filtered_cfg["norm_decay"] = bool(ckpt_args.get("normdecay", 0))

    # IMPORTANT: do NOT default kernel to "auto" (invalid). Choose by task.
    if "kernel" not in filtered_cfg:
        filtered_cfg["kernel"] = "linear" if task == "blr" else "rbf"

    # If checkpoint cfg predates this field
    if "gradient_checkpointing" not in filtered_cfg:
        filtered_cfg["gradient_checkpointing"] = False

    cfg = model.ModelConfig(**filtered_cfg)

    net = model.TransformerUQ(cfg=cfg, edges=edges).to(dev, dtype=dt)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt.get("state_dict", {})
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError:
        net.load_state_dict(state, strict=False)
    net.eval()

    # evaluation settings from ckpt
    d = int(ckpt_cfg.get("d", ckpt_args.get("d", 2)))
    sigma_true = float(ckpt_args.get("sigma", 0.2))
    x_dist = ckpt_args.get("x_dist", "normal")

    n_mode = ckpt_args.get("n_mode", "mixture")
    n_min = int(ckpt_args.get("n_min", 64))
    n_max = int(ckpt_args.get("n_max", 128))
    n_single = int(ckpt_args.get("n", 64))

    if task == "blr":
        blr_diag_var = data._default_diag_var(d=d, device=dev, dtype=dt)
        rbf_alpha, rbf_ell, rbf_jitter = 1.0, 0.8, 1e-6
    else:
        blr_diag_var = None
        rbf_alpha = float(ckpt_args.get("rbf_alpha", 1.0))
        rbf_ell = float(ckpt_args.get("rbf_ell", 0.8))
        rbf_jitter = float(ckpt_args.get("rbf_jitter", 1e-6))

    metrics = evaluate_mixture_on(
        net,
        task=task,
        d=d,
        edges=edges,
        device=dev,
        dtype=dt,
        eval_batches=eval_batches,
        batch_size=batch_size,
        n_mode=n_mode,
        n=n_single,
        n_min=n_min,
        n_max=n_max,
        sigma=sigma_true,
        x_dist=x_dist,
        blr_diag_var=blr_diag_var,
        rbf_alpha=rbf_alpha,
        rbf_ell=rbf_ell,
        rbf_jitter=rbf_jitter,
        num_grid=512,
    )

    # cache
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    try:
        with open(cache_file, "w") as f:
            json.dump(metrics, f, default=convert, indent=2)
    except Exception as e:
        print(f"⚠️  Could not write cache file {cache_file}: {e}")

    return metrics


# --------------------------------------------------------------------------------------
# grid analysis + plotting (keeps your original style)
# --------------------------------------------------------------------------------------

def analyze_grid_results(
    *,
    task: str = "rbf",
    d: int = 2,
    n_mode: str = "mixture",
    n_min: int = 64,
    n_max: int = 128,
    metric: str = "eval_tv",
    root: str = "runs",
    normalize: int = 0,
    norm_decay: int = 0,
    save_plot: bool = True,
    use_cache: bool = True,
    figsize=(16, 4),
    eval_batches: int = 64,
    batch_size: int = 128,
    device: str = "cuda",
):
    seed = 0

    if task == "blr" and normalize != 0:
        print("⚠️  BLR task detected: forcing normalize=0 and norm_decay=0.")
        normalize = 0
        norm_decay = 0

    Ls = [2, 4, 8, 16, 32]
    Cs = [16, 32, 64, 128, 256]

    is_kernel_comparison = (normalize == 1)

    if is_kernel_comparison:
        scan_kernels = ["rbf", "softmax"]   # folder tags use _softmax for softmax
        scan_modes = ["learnable"]
    else:
        scan_kernels = ["rbf"]              # baseline folder naming in your runs uses "auto" before; but cfg is rbf
        scan_modes = ["theory", "learnable"]

    norm_tag = "_norm" if normalize == 1 else ""
    nd_tag = "_nd" if norm_decay == 1 else ""
    kernel_tag = lambda k: "_softmax" if k == "softmax" else ""

    metric_key = metric.lower()
    display_label = f"log({metric_key.replace('eval_', '')})"

    print(f"\nRe-evaluating (GPU) checkpoints under '{root}' for metric={metric_key} | "
          f"normalize={normalize}, norm_decay={norm_decay}, device={device}")
    print(f"Comparison: {'Kernel (RBF vs Softmax)' if is_kernel_comparison else 'Mode (Theory vs Learnable)'}")

    results = []

    for k in scan_kernels:
        for mode in scan_modes:
            if k == "softmax" and mode == "theory":
                continue
            if mode == "theory" and normalize == 1:
                continue

            for C in Cs:
                for L in Ls:
                    folder_name = (
                        f"{task}_d{d}_C{C}_L{L}_{n_mode}_n{n_min}-{n_max}_{mode}"
                        f"{norm_tag}{nd_tag}{kernel_tag(k)}_seed{seed}"
                    )
                    base_dir = os.path.join(root, folder_name)

                    ckpt_path, note = resolve_best_checkpoint(base_dir)
                    val = np.nan
                    if ckpt_path is not None:
                        metrics_dict = run_evaluation_on_ckpt(
                            ckpt_path,
                            device=device,
                            eval_batches=eval_batches,
                            batch_size=batch_size,
                            seed=seed,
                            use_cache=use_cache,
                        )
                        if metrics_dict and metric_key in metrics_dict:
                            val = float(metrics_dict[metric_key])

                    if is_kernel_comparison:
                        group_label = "RBF" if k == "rbf" else "Softmax"
                    else:
                        group_label = mode.capitalize()

                    results.append({"C": C, "L": L, "Group": group_label, "Value": val})

    df = pd.DataFrame(results)
    if df.empty or df["Value"].isna().all():
        print(f"❌ No data found/computed for metric='{metric_key}'")
        return

    df["LogValue"] = np.log(np.maximum(df["Value"].to_numpy(), 1e-300))

    target_L = max(Ls)
    target_C = max(Cs)

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    cmap = "rocket"

    title_norm = "Normalized Attention" if normalize == 1 else "Standard Attention"
    if norm_decay:
        title_norm += " + Decay (Eq 10)"

    plot_title = f"Task: {task.upper()} | {title_norm} | Metric: {display_label}"
    plot_title += " | " + ("RBF vs Softmax" if is_kernel_comparison else "Theory vs Learnable")
    # fig.suptitle(plot_title, fontsize=14, y=0.95)

    if is_kernel_comparison:
        left_label, right_label = "RBF", "Softmax"
        palette = {"RBF": "tab:blue", "Softmax": "tab:red"}
    else:
        left_label, right_label = "Theory", "Learnable"
        palette = {"Theory": "tab:orange", "Learnable": "tab:blue"}

    left_df = df[df["Group"] == left_label]
    if not left_df.empty:
        pivot_left = left_df.pivot(index="C", columns="L", values="LogValue")
        sns.heatmap(pivot_left, ax=axes[0], annot=True, fmt=".2f", cmap=cmap,
                    cbar_kws={"label": display_label})
        # axes[0].set_title(left_label)
        axes[0].set_title(f"{task.upper()} {left_label}")
        axes[0].invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, f"No Data: {left_label}", ha="center")

    right_df = df[df["Group"] == right_label]
    if not right_df.empty:
        pivot_right = right_df.pivot(index="C", columns="L", values="LogValue")
        sns.heatmap(pivot_right, ax=axes[1], annot=True, fmt=".2f", cmap=cmap,
                    cbar_kws={"label": display_label})
        # axes[1].set_title(right_label)
        axes[1].set_title(f"{task.upper()} {right_label}")
        axes[1].invert_yaxis()
    else:
        axes[1].text(0.5, 0.5, f"No Data: {right_label}", ha="center")

    data_fixed_L = df[df["L"] == target_L]
    if not data_fixed_L.empty:
        sns.lineplot(
            data=data_fixed_L, x="C", y="LogValue", hue="Group", style="Group",
            markers=True, dashes=False, ax=axes[2], palette=palette
        )
        axes[2].set_title(f"Fixed L = {target_L}")
        axes[2].set_xlabel("Bins (C)")
        axes[2].set_ylabel(display_label)
        axes[2].set_xscale("log", base=2)
        axes[2].grid(True, which="both", ls="--", alpha=0.3)
        axes[2].legend()

    data_fixed_C = df[df["C"] == target_C]
    if not data_fixed_C.empty:
        sns.lineplot(
            data=data_fixed_C, x="L", y="LogValue", hue="Group", style="Group",
            markers=True, dashes=False, ax=axes[3], palette=palette
        )
        axes[3].set_title(f"Fixed C = {target_C}")
        axes[3].set_xlabel("Layers (L)")
        axes[3].set_ylabel(display_label)
        axes[3].set_xscale("log", base=2)
        axes[3].grid(True, which="both", ls="--", alpha=0.3)
        axes[3].legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    if save_plot:
        os.makedirs("figs", exist_ok=True)
        norm_str = "norm" if normalize == 1 else "no_norm"
        nd_str = "_nd" if norm_decay == 1 else ""
        comparison_tag = "kernel_comp" if is_kernel_comparison else "mode_comp"
        out_name = f"figs/figure1_{task}_d{d}_n{n_min}-{n_max}_{metric_key}_{norm_str}{nd_str}_{comparison_tag}.pdf"
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved to: {os.path.abspath(out_name)}")

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs")
    ap.add_argument("--task", type=str, default="rbf") # rbf, blr
    ap.add_argument("--d", type=int, default=2) # 2, 5
    ap.add_argument("--n_mode", type=str, default="mixture")
    ap.add_argument("--n_min", type=int, default=64) # 64, 128
    ap.add_argument("--n_max", type=int, default=128) # 128, 512
    ap.add_argument("--metric", type=str, default="eval_tv")
    ap.add_argument("--normalize", type=int, default=0)
    ap.add_argument("--norm_decay", type=int, default=0)
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval_batches", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--figsize_w", type=float, default=16.5)
    ap.add_argument("--figsize_h", type=float, default=3.5)
    args = ap.parse_args()

    analyze_grid_results(
        task=args.task,
        d=args.d,
        n_mode=args.n_mode,
        n_min=args.n_min,
        n_max=args.n_max,
        metric=args.metric,
        root=args.root,
        normalize=args.normalize,
        norm_decay=args.norm_decay,
        save_plot=(not args.no_save),
        use_cache=args.use_cache,
        figsize=(args.figsize_w, args.figsize_h),
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
