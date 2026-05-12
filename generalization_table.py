import os
import glob
import json
import math
import hashlib
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

import data
import model
import ppd_viewer as pv


# ============================================================
# Utilities
# ============================================================
def fmt3(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.3g}"

def _seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt_path(p: str, root: str = "runs") -> str:
    try:
        if p.startswith(root + os.sep):
            return p[len(root) + 1:]
    except Exception:
        pass
    return p


def resolve_checkpoint(root_dir: str, folder_pat: str) -> Dict[str, Optional[str]]:
    """
    Resolve checkpoint path from a run folder pattern.

    Priority:
      (1) newest finetune best.pt
      (2) base best.pt
      (3) base final.pt
    """
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


def choose_eval_params(n: int, target_total: int = 5000):
    if n <= 200:
        B = 256
    elif n <= 600:
        B = 128
    else:
        B = 64
    num_batches = max(1, int(np.ceil(target_total / B)))
    return B, num_batches


def get_cache_filename(
    ckpt_path: str,
    n_eval_list,
    *,
    x_dist: str,
    sigma: float,
    task: str,
    d_eval: int,
    C: int,
    rbf_alpha: float = 1.0,
    rbf_ell: float = 0.8,
    rbf_jitter: float = 1e-6,
    rbfmix_ell_grid: Optional[List[float]] = None,
    rbfmix_sigma_grid: Optional[List[float]] = None,
    cache_version: str = "rebuttal_v3_rbfmix",
):
    payload = {
        "v": cache_version,
        "task": task,
        "d": int(d_eval),
        "C": int(C),
        "x_dist": str(x_dist),
        "sigma": float(sigma),
        "n_eval": list(map(int, list(n_eval_list))),
        "rbf_alpha": float(rbf_alpha),
        "rbf_ell": float(rbf_ell),
        "rbf_jitter": float(rbf_jitter),
        "rbfmix_ell_grid": None if rbfmix_ell_grid is None else list(map(float, rbfmix_ell_grid)),
        "rbfmix_sigma_grid": None if rbfmix_sigma_grid is None else list(map(float, rbfmix_sigma_grid)),
    }
    s = json.dumps(payload, sort_keys=True)
    h = hashlib.md5(s.encode()).hexdigest()[:10]
    base = os.path.basename(os.path.dirname(ckpt_path))
    return os.path.join(os.path.dirname(ckpt_path), f"rebuttal_metrics_cache_{base}_{h}.json")


def candidate_folder_patterns(
    *,
    task: str,
    d: int,
    C: int,
    L: int,
    n_min: int,
    n_max: int,
    mode: str,
    normalize: int,
    norm_decay: int,
    kernel: str,
    seed: int,
    ell_filter: Optional[float],
    sigma_filter: Optional[float],
    exact_task_tag: Optional[str] = None,
) -> List[str]:
    """
    Build candidate folder patterns.

    Compatibility:
      - baseline ell=0.8 may have NO ell tag
      - baseline sigma=0.2 may have NO sigma tag
      - kernel='rbf' may have no explicit suffix
      - for rbfmix, prefer exact_task_tag when supplied
    """
    nd_tag = "_nd" if norm_decay else ""
    norm_tag = "_norm" if normalize else ""
    k_tag = "_softmax" if kernel == "softmax" else ""

    if exact_task_tag is not None:
        return [
            f"{task}_d{d}{exact_task_tag}_C{C}_L{L}_mixture_n{n_min}-{n_max}_{mode}"
            f"{norm_tag}{nd_tag}{k_tag}_seed{seed}"
        ]

    ell_tags = [""]
    if ell_filter is not None and abs(float(ell_filter) - 0.8) > 1e-12:
        ell_tags = [f"_ell{float(ell_filter):g}"]

    sigma_tags = [""]
    if sigma_filter is not None and abs(float(sigma_filter) - 0.2) > 1e-12:
        sigma_tags = [f"_sig{float(sigma_filter):g}"]
    elif sigma_filter is not None and abs(float(sigma_filter) - 0.2) < 1e-12:
        sigma_tags = ["", f"_sig{float(sigma_filter):g}"]

    cands = []
    for ell_tag in ell_tags:
        for sig_tag in sigma_tags:
            cands.append(
                f"{task}_d{d}{ell_tag}{sig_tag}_C{C}_L{L}_mixture_n{n_min}-{n_max}_{mode}"
                f"{norm_tag}{nd_tag}{k_tag}_seed{seed}"
            )

    out = []
    seen = set()
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ============================================================
# Data loader
# ============================================================

def get_eval_loader(
    task: str,
    B: int,
    n: int,
    d: int,
    device,
    dtype,
    seed: int,
    edges: torch.Tensor,
    sigma: float,
    num_batches: int = 64,
    x_dist: str = "normal",
    rbf_alpha: float = 1.0,
    rbf_ell: float = 0.8,
    rbf_jitter: float = 1e-6,
    blr_diag_var=None,
    rbfmix_ell_grid: Optional[List[float]] = None,
    rbfmix_sigma_grid: Optional[List[float]] = None,
) -> List[Tuple]:
    _seed_all(seed)
    loader = []

    if task == "blr":
        hypers = data.BLR_hyprms(sigma=sigma, diag_var=blr_diag_var)
    elif task == "rbf":
        hypers = data.RBF_hyprms(sigma=sigma, alpha=rbf_alpha, ell=rbf_ell, jitter=rbf_jitter)
    elif task == "rbfmix":
        if rbfmix_ell_grid is None or rbfmix_sigma_grid is None:
            raise ValueError("rbfmix evaluation requires rbfmix_ell_grid and rbfmix_sigma_grid.")
        hypers = data.RBF_Mixture_hyprms(
            sigma_grid=tuple(float(x) for x in rbfmix_sigma_grid),
            ell_grid=tuple(float(x) for x in rbfmix_ell_grid),
            alpha=rbf_alpha,
            jitter=rbf_jitter,
        )
    else:
        raise ValueError(f"Unsupported task {task}.")

    for _ in range(num_batches):
        if task == "blr":
            batch_dict = data.draw_BLR_batch(
                B, n, d, hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges
            )
        elif task == "rbf":
            batch_dict = data.draw_RBF_batch(
                B, n, d, hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges
            )
        elif task == "rbfmix":
            batch_dict = data.draw_RBF_mixture_batch(
                B, n, d, hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges
            )
        else:
            raise ValueError(f"Unsupported task {task}")

        X = batch_dict["X"]
        Y = batch_dict["Y"]
        xq = batch_dict["xq"]
        if Y.ndim == 3:
            Y = Y.squeeze(-1)
        if xq.ndim == 3:
            xq = xq.squeeze(1)

        loader.append((
            X,
            Y,
            xq,
            batch_dict["yq"],
            batch_dict["c_true"],
            batch_dict["q_true"],
            batch_dict["mu"],
            batch_dict["tau"],
        ))

    return loader


# ============================================================
# Metrics
# ============================================================

def crps_from_binned_probs(probs: torch.Tensor, edges: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    device = probs.device
    dtype = probs.dtype

    edges = edges.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype).reshape(-1)

    left = edges[:-1].view(1, -1)
    right = edges[1:].view(1, -1)
    widths = right - left

    cdf_left = torch.cumsum(probs, dim=-1) - probs
    yv = y.view(-1, 1)

    fully_right = (yv <= left).to(dtype)
    crossing = ((yv > left) & (yv < right))
    fully_left_or_touching = (~crossing & ~(yv <= left)).to(dtype)

    crps = (
        widths * (
            fully_right * (cdf_left - 1.0) ** 2
            + fully_left_or_touching * (cdf_left ** 2)
        )
    ).sum(dim=-1)

    if crossing.any():
        idx = torch.argmax(crossing.to(torch.int64), dim=-1)
        l_star = edges[:-1][idx]
        r_star = edges[1:][idx]
        F_star = cdf_left.gather(1, idx.view(-1, 1)).squeeze(1)

        left_part = torch.clamp(y - l_star, min=0.0)
        right_part = torch.clamp(r_star - y, min=0.0)

        full_width = r_star - l_star
        crps = crps - full_width * (F_star ** 2)
        crps = crps + left_part * (F_star ** 2) + right_part * ((F_star - 1.0) ** 2)

    return crps


def coverage_from_bins_ppdviewer(
    probs: torch.Tensor,
    edges: torch.Tensor,
    y: torch.Tensor,
    level: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = pv.interval_coverage_from_bins(
        probs=probs,
        edges=edges,
        y=y,
        level=float(level),
    )
    coverage = torch.tensor(float(out["coverage"]), device=probs.device, dtype=torch.float64)
    width_mean = torch.tensor(float(out["width_mean"]), device=probs.device, dtype=torch.float64)
    return coverage, width_mean


@torch.inference_mode()
def evaluate_table_metrics(
    net: torch.nn.Module,
    eval_loader: list,
    device,
    dtype,
    edges: torch.Tensor,
) -> Dict[str, float]:
    net.eval()

    log_mse_y_vals = []
    log_tv_vals = []
    crps_vals = []
    cov50_vals = []
    cov90_vals = []
    cov95_vals = []

    edges = edges.to(device=device, dtype=torch.float64)
    bin_centers = ((edges[:-1] + edges[1:]) / 2.0).view(1, -1)

    for batch_tuple in eval_loader:
        X_pad, Y_pad, xq, yq, c_true, q_true, mu_true, tau_true = batch_tuple

        X_in = X_pad.to(device=device, dtype=dtype)
        Y_in = Y_pad.to(device=device, dtype=dtype)
        xq_in = xq.to(device=device, dtype=dtype)

        logits, _aux = net(X_in, Y_in, xq_in, padding_mask=None)
        probs = torch.softmax(logits, dim=-1).to(torch.float64)

        yq_in = yq.to(device=device, dtype=torch.float64).reshape(-1)
        pred_mean = (probs * bin_centers).sum(dim=-1).reshape(-1)

        mse_y = F.mse_loss(pred_mean, yq_in).item()
        log_mse_y_vals.append(float(np.log(max(mse_y, 1e-12))))

        if q_true is not None:
            q_true_dev = q_true.to(device=device, dtype=torch.float64)
            if q_true_dev.ndim == 3:
                q_true_dev = q_true_dev.squeeze(1)
            tv = 0.5 * torch.abs(probs - q_true_dev).sum(dim=-1).mean().item()
            log_tv_vals.append(float(np.log(max(tv, 1e-12))))
        else:
            log_tv_vals.append(float("nan"))

        crps_batch = crps_from_binned_probs(probs, edges, yq_in).mean().item()
        crps_vals.append(float(crps_batch))

        cov50, _ = coverage_from_bins_ppdviewer(probs, edges, yq_in, level=0.50)
        cov90, _ = coverage_from_bins_ppdviewer(probs, edges, yq_in, level=0.90)
        cov95, _ = coverage_from_bins_ppdviewer(probs, edges, yq_in, level=0.95)

        cov50_vals.append(float(cov50.item()))
        cov90_vals.append(float(cov90.item()))
        cov95_vals.append(float(cov95.item()))

    return {
        # "log_mse_y": float(np.mean(log_mse_y_vals)),
        # "log_tv": float(np.mean(log_tv_vals)),
        "crps": float(np.mean(crps_vals)),
        "cov50": float(np.mean(cov50_vals)),
        "cov90": float(np.mean(cov90_vals)),
        "cov95": float(np.mean(cov95_vals)),
    }


def compute_table_metrics(
    net,
    task: str,
    sample_sizes,
    d: int,
    device,
    dtype,
    edges: torch.Tensor,
    *,
    x_dist: str,
    sigma: float,
    rbf_alpha: float = 1.0,
    rbf_ell: float = 0.8,
    rbf_jitter: float = 1e-6,
    rbfmix_ell_grid: Optional[List[float]] = None,
    rbfmix_sigma_grid: Optional[List[float]] = None,
    seed: int = 0,
):
    results = {
        "n": [],
        "log_mse_y": [],
        "log_tv": [],
        "crps": [],
        "cov50": [],
        "cov90": [],
        "cov95": [],
    }

    print("    ↳ eval n:", end=" ")
    for n_eval in sample_sizes:
        n_eval = int(n_eval)
        eval_B, num_batches = choose_eval_params(n_eval)
        print(f"{n_eval}(B={eval_B},nb={num_batches})", end="  ", flush=True)

        dl = get_eval_loader(
            task=task,
            B=eval_B,
            n=n_eval,
            d=d,
            device=device,
            dtype=dtype,
            seed=seed + n_eval,
            edges=edges,
            sigma=sigma,
            num_batches=num_batches,
            x_dist=x_dist,
            rbf_alpha=rbf_alpha,
            rbf_ell=rbf_ell,
            rbf_jitter=rbf_jitter,
            rbfmix_ell_grid=rbfmix_ell_grid,
            rbfmix_sigma_grid=rbfmix_sigma_grid,
        )

        metrics = evaluate_table_metrics(
            net=net,
            eval_loader=dl,
            device=device,
            dtype=dtype,
            edges=edges,
        )

        results["n"].append(n_eval)
        # results["log_mse_y"].append(metrics["log_mse_y"])
        # results["log_tv"].append(metrics["log_tv"])
        results["crps"].append(metrics["crps"])
        results["cov50"].append(metrics["cov50"])
        results["cov90"].append(metrics["cov90"])
        results["cov95"].append(metrics["cov95"])

    print("")
    return results


# ============================================================
# Markdown writing
# ============================================================

def markdown_table_grouped_by_L(
    title: str,
    results_by_L: Dict[int, Dict[str, List[float]]],
) -> str:
    Ls_sorted = sorted(results_by_L.keys())
    n_list = results_by_L[Ls_sorted[0]]["n"]

    def cell(metric: str, i: int) -> str:
        vals = [fmt3(results_by_L[L][metric][i]) for L in Ls_sorted]
        return "" + ", ".join(vals) + ""

    lines = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"Entries are ordered by L = {Ls_sorted}.")
    lines.append("")
    # lines.append("| n | log MSE y | CRPS | 50% cov | 90% cov | 95% cov |")
    # lines.append("|---:|---|---|---|---|---|")
    lines.append("| n | CRPS | 50% cov | 90% cov | 95% cov |")
    lines.append("|---:|---|---|---|---|")

    for i, n in enumerate(n_list):
        lines.append(
            f"| {int(n)} | "
            # f"{cell('log_mse_y', i)} | "
            # f"{cell('log_tv', i)} | "
            f"{cell('crps', i)} | "
            f"{cell('cov50', i)} | "
            f"{cell('cov90', i)} | "
            f"{cell('cov95', i)} |"
        )
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main driver
# ============================================================

def rebuttal_tables(
    task: str,
    kernel: str,
    mode: str,
    normalize: int,
    norm_decay: int,
    C: int,
    list_of_Ls: List[int],
    list_of_nmax: List[int],
    sample_sizes,
    *,
    root_dir: str = "runs",
    out_dir: str = "tables",
    device: str = "cuda",
    d: int = 16,
    n_min: int = 64,
    x_dist: str = "normal",
    cache: bool = True,
    seed: int = 0,
    ell_filter: Optional[float] = None,
    sigma_filter: Optional[float] = None,
    exact_task_tag: Optional[str] = None,
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    t_device = torch.device(device)
    dtype = torch.float64

    nd_tag = "_nd" if norm_decay else ""
    norm_tag = "_norm" if normalize else ""

    os.makedirs(out_dir, exist_ok=True)

    all_sections = []
    used_ckpts = []
    grouped_results = {}   # grouped_results[n_max][L] = results

    for n_max in list_of_nmax:
        print(f"\n=== Train context cap n_max={n_max} ===")
        for L in list_of_Ls:
            folder_candidates = candidate_folder_patterns(
                task=task,
                d=d,
                C=C,
                L=L,
                n_min=n_min,
                n_max=n_max,
                mode=mode,
                normalize=normalize,
                norm_decay=norm_decay,
                kernel=kernel,
                seed=seed,
                ell_filter=ell_filter,
                sigma_filter=sigma_filter,
                exact_task_tag=exact_task_tag,
            )

            ckpt_path = None
            for folder_pat in folder_candidates:
                ckpt_info = resolve_checkpoint(root_dir, folder_pat)
                if ckpt_info["ckpt_path"] is not None:
                    ckpt_path = ckpt_info["ckpt_path"]
                    break

            if ckpt_path is None:
                print(f"  [missing] candidates={folder_candidates}")
                continue

            pretty_ckpt = _fmt_path(ckpt_path, root=root_dir)
            print(f"  [load] n_max={n_max}, L={L}: {pretty_ckpt}")

            state = torch.load(ckpt_path, map_location="cpu")
            ckpt_args = state["args"]
            d_ckpt = int(ckpt_args.get("d", d))
            C_eval = int(ckpt_args.get("C", C))

            sigma_eval = float(ckpt_args.get("sigma", 0.2))
            rbf_alpha = float(ckpt_args.get("rbf_alpha", 1.0))
            rbf_ell = float(ckpt_args.get("rbf_ell", 0.8))
            rbf_jitter = float(ckpt_args.get("rbf_jitter", 1e-6))
            rbfmix_ell_grid = ckpt_args.get("rbfmix_ell_grid", None)
            rbfmix_sigma_grid = ckpt_args.get("rbfmix_sigma_grid", None)

            cache_file = get_cache_filename(
                ckpt_path,
                sample_sizes,
                x_dist=x_dist,
                sigma=sigma_eval,
                task=task,
                d_eval=d_ckpt,
                C=C_eval,
                rbf_alpha=rbf_alpha,
                rbf_ell=rbf_ell,
                rbf_jitter=rbf_jitter,
                rbfmix_ell_grid=rbfmix_ell_grid,
                rbfmix_sigma_grid=rbfmix_sigma_grid,
            )

            if cache and os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    results = json.load(f)
            else:
                edges = data.make_bin_edges(
                    ckpt_args["a"], ckpt_args["b"], C_eval, device=t_device, dtype=dtype
                )

                cfg_dict = state["cfg"]
                if "gradient_checkpointing" not in cfg_dict:
                    cfg_dict["gradient_checkpointing"] = False

                model_cfg = model.ModelConfig(**cfg_dict)
                net = model.TransformerUQ(model_cfg, edges).to(t_device, dtype=dtype)
                net.load_state_dict(state["model_state"], strict=False)

                results = compute_table_metrics(
                    net=net,
                    task=task,
                    sample_sizes=sample_sizes,
                    d=d_ckpt,
                    device=t_device,
                    dtype=dtype,
                    edges=edges,
                    x_dist=x_dist,
                    sigma=sigma_eval,
                    rbf_alpha=rbf_alpha,
                    rbf_ell=rbf_ell,
                    rbf_jitter=rbf_jitter,
                    rbfmix_ell_grid=rbfmix_ell_grid,
                    rbfmix_sigma_grid=rbfmix_sigma_grid,
                    seed=seed,
                )

                with open(cache_file, "w") as f:
                    json.dump(results, f)

            if n_max not in grouped_results:
                grouped_results[n_max] = {}
            grouped_results[n_max][L] = results

            used_ckpts.append((n_max, L, pretty_ckpt))

    if len(grouped_results) == 0:
        raise RuntimeError("No checkpoints found / evaluated.")

    for n_max in sorted(grouped_results.keys()):
        title = f"n_max={n_max}"
        if ell_filter is not None:
            title += f", ell={ell_filter:g}"
        if sigma_filter is not None:
            title += f", sigma={sigma_filter:g}"
        if task == "rbfmix":
            # just use the last-loaded grid description if you want it in the title
            title += f", mixture hyperprior"

        all_sections.append(
            markdown_table_grouped_by_L(title, grouped_results[n_max])
        )

    n_s, n_e = int(sample_sizes[0]), int(sample_sizes[-1])
    ell_name = f"_ell{ell_filter:g}" if ell_filter is not None else ""
    sig_name = f"_sig{sigma_filter:g}" if sigma_filter is not None else ""
    exact_name = exact_task_tag.replace("/", "_") if exact_task_tag is not None else ""

    filename = (
        f"rebuttal_tables_{task}_{mode}_{x_dist}_{kernel}_C{C}{norm_tag}{nd_tag}"
        f"_d{d}{ell_name}{sig_name}{exact_name}_nmin{n_min}_n{n_s}-{n_e}.md"
    )
    save_path = os.path.join(out_dir, filename)

    header = []
    header.append("# Rebuttal evaluation tables")
    header.append("")
    header.append(f"- task: `{task}`")
    header.append(f"- kernel: `{kernel}`")
    header.append(f"- mode: `{mode}`")
    header.append(f"- normalize: `{normalize}`")
    header.append(f"- norm_decay: `{norm_decay}`")
    header.append(f"- x_dist: `{x_dist}`")
    header.append(f"- d: `{d}`")
    header.append(f"- C: `{C}`")
    header.append(f"- sample sizes: `{list(sample_sizes)}`")
    if ell_filter is not None:
        header.append(f"- ell filter in folder name: `{ell_filter}`")
    if sigma_filter is not None:
        header.append(f"- sigma filter in folder name: `{sigma_filter}`")
    if exact_task_tag is not None:
        header.append(f"- exact task tag: `{exact_task_tag}`")
    header.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(header))
        f.write("\n\n")
        f.write("\n\n".join(all_sections))
        f.write("\n")

    print(f"\nSaved markdown tables to: {os.path.abspath(save_path)}")
    print("\nCheckpoints used:")
    for n_max, L, p in used_ckpts:
        print(f"  n_max={n_max:>4} | L={L:>3} | {p}")


if __name__ == "__main__":
    task = "rbfmix"
    kernel = "rbf"
    mode = "learnable"
    normalize = 1
    norm_decay = 1
    C = 256

    LIST_OF_LS = [8, 16, 32]
    LIST_OF_NMAX = [512]
    SAMPLE_SIZES = [128 * i for i in range(1, 10)]

    ell_filter = None
    sigma_filter = None
    exact_task_tag = "_ellmix0.4-0.8-1.2_sigmix0.1-0.2-0.3"  # e.g. "_ellmix0.4-0.8-1.2_sigmix0.1-0.2-0.3" for rbfmix
    # exact_task_tag = None  # e.g. "_ellmix0.4-0.8-1.2_sigmix0.1-0.2-0.3" for rbfmix

    rebuttal_tables(
        task=task,
        kernel=kernel,
        mode=mode,
        normalize=normalize,
        norm_decay=norm_decay,
        C=C,
        list_of_Ls=LIST_OF_LS,
        list_of_nmax=LIST_OF_NMAX,
        sample_sizes=SAMPLE_SIZES,
        d=16,
        n_min=64,
        x_dist="normal",
        cache=True,
        seed=0,
        ell_filter=ell_filter,
        sigma_filter=sigma_filter,
        exact_task_tag=exact_task_tag,
    )