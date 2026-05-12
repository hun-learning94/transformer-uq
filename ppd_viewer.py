import math
import os
from typing import Optional, Literal

import torch
import matplotlib.pyplot as plt

import data
import model


def gaussian_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def truncated_normal_logpdf_grid(
    x: torch.Tensor, mu: torch.Tensor, tau: torch.Tensor,
    a: float, b: float, eps: float = 1e-12
):
    sd = torch.sqrt(torch.clamp(tau, min=eps))
    z = (x - mu[:, None]) / sd[:, None]
    log_base = -0.5 * z**2 - torch.log(sd[:, None]) - 0.5 * math.log(2.0 * math.pi)
    alpha = (a - mu) / sd
    beta  = (b - mu) / sd
    Z = torch.clamp(gaussian_cdf(beta) - gaussian_cdf(alpha), min=eps)
    return log_base - torch.log(Z)[:, None]


def piecewise_density_on_grid(probs: torch.Tensor, edges: torch.Tensor, x_mid: torch.Tensor, eps: float = 1e-12):
    widths = (edges[1:] - edges[:-1]).clamp_min(eps)
    dens_bins = probs / widths[None, :]
    idx = torch.bucketize(x_mid, edges) - 1
    idx = idx.clamp(0, probs.shape[1]-1)
    return dens_bins[:, idx]


def binned_ppf(probs: torch.Tensor, edges: torch.Tensor, q_levels: torch.Tensor, eps: float = 1e-12):
    probs = probs.clamp_min(0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    cdf = torch.cumsum(probs, dim=-1)
    B, C = probs.shape
    qs = []
    for q in q_levels:
        ge = (cdf >= q).to(torch.int64)
        idx = torch.argmax(ge, dim=-1)
        prev_idx = torch.clamp(idx - 1, min=0)
        prev_cdf = cdf[torch.arange(B), prev_idx]
        prev_cdf = torch.where(idx > 0, prev_cdf, torch.zeros_like(prev_cdf))
        p_bin = probs[torch.arange(B), idx].clamp_min(eps)
        left = edges[:-1][idx]
        right = edges[1:][idx]
        frac = ((q - prev_cdf) / p_bin).clamp(0.0, 1.0)
        xq = left + frac * (right - left)
        qs.append(xq)
    return torch.stack(qs, dim=-1)


def interval_coverage_from_bins(
    probs: torch.Tensor,
    edges: torch.Tensor,
    y: torch.Tensor,
    level: float = 0.90,
    eps: float = 1e-12,
):
    alpha = 1.0 - level
    q_levels = torch.tensor([alpha/2.0, 1.0 - alpha/2.0], device=probs.device, dtype=probs.dtype)
    ints = binned_ppf(probs, edges, q_levels, eps=eps)
    lo, hi = ints[:, 0], ints[:, 1]
    yy = y.reshape(-1).to(device=probs.device, dtype=probs.dtype)
    covered = ((yy >= lo) & (yy <= hi)).to(torch.float32)
    return {
        "coverage": covered.mean().item(),
        "width_mean": (hi - lo).mean().item(),
        "lo": lo,
        "hi": hi,
    }


def make_hypers_from_ckpt_args(args: dict):
    task = args["task"]
    if task == "blr":
        return data.BLR_hyprms(
            sigma=float(args["sigma"]),
            diag_var=torch.tensor(args["blr_diag_var"], dtype=torch.float64) if args.get("blr_diag_var", None) is not None else None,
        )
    if task == "rbf":
        return data.RBF_hyprms(
            sigma=float(args["sigma"]),
            alpha=float(args.get("rbf_alpha", 1.0)),
            ell=float(args.get("rbf_ell", 0.8)),
            jitter=float(args.get("rbf_jitter", 1e-6)),
        )
    if task == "rbfmix":
        return data.RBF_Mixture_hyprms(
            sigma_grid=tuple(float(x) for x in args["rbfmix_sigma_grid"]),
            ell_grid=tuple(float(x) for x in args["rbfmix_ell_grid"]),
            alpha=float(args.get("rbf_alpha", 1.0)),
            jitter=float(args.get("rbf_jitter", 1e-6)),
        )
    raise ValueError(f"Unsupported task={task}")


def sample_eval_batch(
    task: str,
    B: int,
    n: int,
    d: int,
    hypers,
    *,
    device,
    dtype,
    x_dist: str,
    edges: Optional[torch.Tensor],
):
    if task == "blr":
        return data.draw_BLR_batch(B=B, n=n, d=d, hypers=hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges)
    if task == "rbf":
        return data.draw_RBF_batch(B=B, n=n, d=d, hypers=hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges)
    if task == "rbfmix":
        return data.draw_RBF_mixture_batch(B=B, n=n, d=d, hypers=hypers, device=device, dtype=dtype, x_dist=x_dist, edges=edges)
    raise ValueError(f"Unsupported task={task}")


@torch.no_grad()
def plot_ppd(
    batch: dict,
    probs_model: torch.Tensor,
    edges: torch.Tensor,
    idx: int = 0,
    title: Optional[str] = None,
):
    probs_model = probs_model.detach().cpu().to(torch.float64)
    edges = edges.detach().cpu().to(torch.float64)

    x_mid = torch.linspace(float(edges[0]), float(edges[-1]), 800, dtype=torch.float64)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    q_model_piece = piecewise_density_on_grid(
        probs_model[idx:idx+1], edges, x_mid
    )[0].cpu()
    (line_model,) = ax.plot(x_mid, q_model_piece, linewidth=2.0, label="model density")
    color_model = line_model.get_color()

    probs_true = batch.get("q_true", None)
    mu_true = batch.get("mu", None)
    tau_true = batch.get("tau", None)
    yq = batch.get("yq", None)

    color_true = None

    if probs_true is not None:
        probs_true = probs_true.detach().cpu().to(torch.float64)
        q_true_piece = piecewise_density_on_grid(
            probs_true[idx:idx+1], edges, x_mid
        )[0].cpu()
        (line_true,) = ax.plot(
            x_mid, q_true_piece, linewidth=1.8, label="true density (oracle bins)", alpha=0.9
        )
        color_true = line_true.get_color()
    elif (mu_true is not None) and (tau_true is not None):
        mu_true = mu_true.detach().cpu().to(torch.float64).reshape(-1)
        tau_true = tau_true.detach().cpu().to(torch.float64).reshape(-1)
        logp = truncated_normal_logpdf_grid(
            x_mid[None, :], mu_true[idx:idx+1], tau_true[idx:idx+1],
            a=float(edges[0]), b=float(edges[-1])
        )
        p = torch.exp(logp[0]).cpu()
        (line_true,) = ax.plot(
            x_mid, p, linewidth=1.8, label="true density (trunc normal)", alpha=0.9
        )
        color_true = line_true.get_color()

    q_levels = torch.tensor([0.05, 0.95], dtype=torch.float64)

    model_int = binned_ppf(probs_model[idx:idx+1], edges, q_levels)[0]
    m_lo, m_hi = float(model_int[0]), float(model_int[1])
    ax.axvline(m_lo, linestyle="--", linewidth=1.4, color=color_model, label="model 90% interval")
    ax.axvline(m_hi, linestyle="--", linewidth=1.4, color=color_model)

    if probs_true is not None:
        true_int = binned_ppf(probs_true[idx:idx+1], edges, q_levels)[0]
        t_lo, t_hi = float(true_int[0]), float(true_int[1])
        ax.axvline(t_lo, linestyle=":", linewidth=1.4, color=color_true, label="true 90% interval")
        ax.axvline(t_hi, linestyle=":", linewidth=1.4, color=color_true)

    if yq is not None:
        yy = float(yq[idx].detach().cpu().reshape(-1)[0])
        ax.axvline(yy, linestyle="-", linewidth=1.2, alpha=0.75, label="realized y")

    ax.set_xlabel("y")
    ax.set_ylabel("density")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


@torch.no_grad()
def ppd_demo(
    run_dir: str,
    n_eval: int,
    *,
    B: int = 16,
    idx: int = 0,
    device: str = "cuda",
):
    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(run_dir, "final.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No best.pt/final.pt found under {run_dir}")

    state = torch.load(ckpt_path, map_location="cpu")
    args = state["args"]
    cfg = state["cfg"]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    t_device = torch.device(device)
    dtype = torch.float64

    edges = data.make_bin_edges(args["a"], args["b"], args["C"], device=t_device, dtype=dtype)

    if "gradient_checkpointing" not in cfg:
        cfg["gradient_checkpointing"] = False
    net = model.TransformerUQ(model.ModelConfig(**cfg), edges).to(t_device, dtype=dtype)
    net.load_state_dict(state["model_state"], strict=False)
    net.eval()

    hypers = make_hypers_from_ckpt_args(args)
    batch = sample_eval_batch(
        task=args["task"],
        B=B,
        n=n_eval,
        d=args["d"],
        hypers=hypers,
        device=t_device,
        dtype=dtype,
        x_dist=args.get("x_dist", "normal"),
        edges=edges,
    )

    X = batch["X"].to(t_device, dtype=dtype)
    Y = batch["Y"].to(t_device, dtype=dtype)
    xq = batch["xq"].to(t_device, dtype=dtype)
    if Y.ndim == 3:
        Y = Y.squeeze(-1)
    if xq.ndim == 3:
        xq = xq.squeeze(1)

    logits, aux = net(X, Y, xq, padding_mask=None)
    probs_model = torch.softmax(logits, dim=-1)

    fig, ax = plot_ppd(
        batch=batch,
        probs_model=probs_model,
        edges=edges,
        idx=idx,
        title=f"{args['task']} | n={n_eval} | idx={idx}",
    )

    yq = batch["yq"].to(t_device, dtype=torch.float64).reshape(-1)
    cov_model = interval_coverage_from_bins(
        probs=probs_model.to(torch.float64),
        edges=edges.to(torch.float64),
        y=yq,
        level=0.90,
    )
    print(f"model 90% coverage (batch empirical): {cov_model['coverage']:.4f}")

    if batch.get("q_true", None) is not None:
        cov_true = interval_coverage_from_bins(
            probs=batch["q_true"].to(torch.float64),
            edges=edges.to(torch.float64),
            y=yq,
            level=0.90,
        )
        print(f"oracle-bin 90% coverage (batch empirical): {cov_true['coverage']:.4f}")

    return fig, ax, batch, probs_model, edges


if __name__ == "__main__":
    task = "rbf"
    mode = "learnable"
    d = 16
    L = 16
    nmin = 64
    nmax = 256

    # Example fixed-rbf run:
    run = f"runs/{task}_d{d}_sig0.2_C256_L{L}_mixture_n{nmin}-{nmax}_{mode}_norm_nd_seed0"

    # Example rbfmix run:
    # run = f"runs/rbfmix_d16_ellmix0.4-0.8-1.6_sigmix0.1-0.2-0.4_C256_L{L}_mixture_n{nmin}-{nmax}_{mode}_norm_nd_seed0"

    fig, ax, batch, probs_model, edges = ppd_demo(run, n_eval=256, B=32, idx=0, device="cuda")
    plt.show()