"""
NLL training for Transformer-UQ
Supports --mode theory (fixed structure/constraints) and --mode learnable (fixed structure, free parameters).

Fixes:
- Eval now pads to n_max to prevent torch.compile recompilation.
- Only moves necessary tensors to GPU during training.
- Saves/Resumes Scheduler state correctly.
- Asserts optimizer exists before training.
"""
from __future__ import annotations
import os
# Force single-threaded workers to avoid CPU oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import csv
import argparse
import math
import time
from dataclasses import asdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import autocast

import data
import model

SQRT_2PI = math.sqrt(2.0 * math.pi)
# Enable TF32 for speed on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def truncated_normal_logpdf_grid(
    x: torch.Tensor,          # (G,)
    mu: torch.Tensor,         # (B,)
    tau: torch.Tensor,        # (B,)
    a: float,
    b: float,
    eps: float = 1e-16,
) -> torch.Tensor:
    """
    log p(x | a<=X<b) for a Normal(mu, tau) truncated to [a,b).
    Returns logp: (B,G)
    """
    dtype = x.dtype
    device = x.device
    
    sd = torch.sqrt(torch.clamp(tau, min=eps))  # (B,)
    z = (x[None, :] - mu[:, None]) / sd[:, None]  # (B,G)

    # log φ(z) - log sd
    log_phi = -0.5 * z**2 - 0.5 * math.log(2.0 * math.pi)
    log_unnorm = log_phi - torch.log(sd)[:, None]  # (B,G)

    # truncation constant Z = Φ((b-mu)/sd) - Φ((a-mu)/sd)
    za = (torch.tensor(a, device=device, dtype=dtype) - mu) / sd
    zb = (torch.tensor(b, device=device, dtype=dtype) - mu) / sd
    Z = torch.clamp(0.5 * (1.0 + torch.erf(zb / math.sqrt(2.0))) -
                    0.5 * (1.0 + torch.erf(za / math.sqrt(2.0))),
                    min=eps)
    logZ = torch.log(Z)  # (B,)

    return log_unnorm - logZ[:, None]  # (B,G)


@torch.no_grad()
def piecewise_constant_q_on_grid(
    probs: torch.Tensor,   # (B,C) bin masses from TF (sum=1)
    edges: torch.Tensor,   # (C+1,)
    x: torch.Tensor,       # (G,) points in [a,b)
    eps: float = 1e-16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q(x) = probs[c]/width[c] for x in [edges[c], edges[c+1]).
    Returns:
      q: (B,G) density values
      logq: (B,G)
    """
    device = probs.device
    dtype = probs.dtype
    edges = edges.to(device=device, dtype=dtype)

    widths = edges[1:] - edges[:-1]  # (C,)
    widths = torch.clamp(widths, min=eps)

    # bucketize into bins 0..C-1 using interior edges
    idx = torch.bucketize(x, edges[1:-1])  # (G,), in {0..C-1}

    q = probs[:, idx] / widths[idx][None, :]  # (B,G)
    q = torch.clamp(q, min=eps)
    logq = torch.log(q)
    return q, logq


@torch.no_grad()
def continuous_kl_tv_between_oracle_and_tf(
    *,
    probs: torch.Tensor,        # (B,C) TF bin masses
    mu: torch.Tensor,           # (B,) oracle mu
    tau: torch.Tensor,          # (B,) oracle tau
    edges: torch.Tensor,        # (C+1,)
    num_grid: int = 512,
    eps: float = 1e-15,
) -> dict[str, torch.Tensor]:
    """
    Compute via midpoint-rule integration over [a,b).
    Force float64 for precision.
    """
    device = probs.device
    
    # FORCE float64
    probs = probs.to(device=device, dtype=torch.float64)
    mu    = mu.to(device=device, dtype=torch.float64)
    tau   = tau.to(device=device, dtype=torch.float64)
    edges = edges.to(device=device, dtype=torch.float64)
    dtype = torch.float64

    a = edges[0].item()
    b = edges[-1].item()

    # uniform integration grid on [a,b)
    xs = torch.linspace(a, b, steps=num_grid + 1, device=device, dtype=dtype)
    x_mid = 0.5 * (xs[:-1] + xs[1:])                 # (G,)
    dx = (xs[1] - xs[0])                             # tensor scalar, float64

    logp = truncated_normal_logpdf_grid(x_mid, mu, tau, a=a, b=b, eps=eps)  # (B,G)
    p = torch.exp(logp)  # (B,G)

    q, logq = piecewise_constant_q_on_grid(probs, edges, x_mid, eps=eps)    # (B,G)

    kl = (p * (logp - logq)).sum(dim=-1) * dx            # (B,)
    l1 = (p - q).abs().sum(dim=-1) * dx                  # (B,)
    tv = 0.5 * l1                                        # (B,)

    return {
        "kl_cont": kl.mean(),
        "tv_cont": tv.mean(),
    }


def set_seed(seed:int)->None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def estimate_ab_from_pretrain(
    *,
    task: str,
    d: int,
    n_mode: str,
    n: int,
    n_min: int,
    n_max: int,
    B: int,
    num_samples: int,
    q_low: float,
    q_high: float,
    hypers,
    device,
    dtype,
    x_dist: str,
    unit_norm_x: bool = False,
) -> tuple[float, float]:
    """Estimate bin range [a,b] by sampling y_q from the *untruncated* pretraining distribution."""
    ys = []
    remaining = num_samples
    while remaining > 0:
        Bb = min(B, remaining)
        if n_mode == "single":
            nn = n
        else:
            nn = int(torch.randint(low=n_min, high=n_max + 1, size=(1,)).item())

        if task == "blr":
            batch = data.draw_BLR_batch(
                B=Bb, n=nn, d=d, hypers=hypers,
                device=device, dtype=dtype, x_dist=x_dist, unit_norm_x=unit_norm_x, edges=None
            )
        elif task == "rbf":
            batch = data.draw_RBF_batch(
                B=Bb, n=nn, d=d, hypers=hypers,
                device=device, dtype=dtype, x_dist=x_dist, unit_norm_x=unit_norm_x, edges=None
            )
        elif task == "rbfmix":
            batch = data.draw_RBF_mixture_batch(
                B=Bb, n=nn, d=d, hypers=hypers,
                device=device, dtype=dtype, x_dist=x_dist, unit_norm_x=unit_norm_x, edges=None
            )
        else:
            raise ValueError(task)

        y = batch.get("yq_untrunc", batch["yq"])
        ys.append(y.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
        remaining -= Bb

    y_all = torch.cat(ys, dim=0)  # (num_samples,)
    a = float(torch.quantile(y_all, q_low).item())
    b = float(torch.quantile(y_all, q_high).item())

    if not (a < b):
        m = float(y_all.mean().item())
        s = float(y_all.std(unbiased=False).item())
        a, b = m - 4.0 * s, m + 4.0 * s
    return a, b

def freeze_all_params(m: torch.nn.Module)-> None:
    for p in m.parameters():
        p.requires_grad_(False)

def unfreeze_by_name(m: torch.nn.Module, name_substrings: list[str])->None:
    for name, p in m.named_parameters():
        if any(s in name for s in name_substrings):
            p.requires_grad_(True)

def make_optimizer(m: torch.nn.Module, lr: float, weight_decay: float=0.0)->torch.optim.Optimizer:
    params = [p for p in m.parameters() if p.requires_grad]
    if len(params) == 0:
        return None
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

def sample_batch(
    task: str,
    B: int, n: int, d: int,
    device, dtype,
    edges: torch.Tensor,
    sigma: float,
    blr_diag_var: torch.Tensor | None,
    rbf_alpha: float,
    rbf_ell: float,
    rbf_jitter: float,
    x_dist: str,
    unit_norm_x: bool = False,
    rbfmix_ell_grid=None,
    rbfmix_sigma_grid=None,
) -> dict[str, torch.Tensor]:
    if task == "blr":
        hypers = data.BLR_hyprms(sigma=sigma, diag_var=blr_diag_var)
        batch = data.draw_BLR_batch(
            B=B, n=n, d=d, hypers=hypers,
            device=device, dtype=dtype, x_dist=x_dist, unit_norm_x=unit_norm_x, edges=edges
        )
    elif task == "rbf":
        hypers = data.RBF_hyprms(sigma=sigma, alpha=rbf_alpha, ell=rbf_ell, jitter=rbf_jitter)
        batch = data.draw_RBF_batch(
            B=B, n=n, d=d, hypers=hypers,
            device=device, dtype=dtype, x_dist=x_dist, unit_norm_x=unit_norm_x, edges=edges
        )
    elif task == "rbfmix":
        hypers = data.RBF_Mixture_hyprms(
            sigma_grid=tuple(float(x) for x in rbfmix_sigma_grid),
            ell_grid=tuple(float(x) for x in rbfmix_ell_grid),
            alpha=rbf_alpha,
            jitter=rbf_jitter,
        )
        batch = data.draw_RBF_mixture_batch(
            B=B, n=n, d=d, hypers=hypers,
            device=device, dtype=dtype, x_dist=x_dist, unit_norm_x=unit_norm_x, edges=edges
        )
    else:
        raise ValueError(f"Unknown task={task}")
    return batch

# --- Collate Function to Pad Batches ---
def collate_pad(batch_list, n_max):
    # Dataset yields full batches, so batch_list has 1 item (the dict)
    data_dict = batch_list[0] if isinstance(batch_list, list) else batch_list

    X = data_dict["X"]  # (B, n, d)
    B, n, d = X.shape
    dev = X.device

    Y = data_dict["Y"]
    xq = data_dict["xq"]
    c_true = data_dict["c_true"]
    mu = data_dict["mu"]
    tau = data_dict["tau"]

    # If n < n_max, we pad
    if n < n_max:
        pad_len = n_max - n
        X_pad = F.pad(X, (0, 0, 0, pad_len), value=0.0)
        Y_pad = F.pad(Y, (0, pad_len), value=0.0)

        # Create Padding Mask (B, n_max + 1, 1)
        # 1 for valid, 0 for pad. Last token (query) is always valid.
        mask = torch.zeros(B, n_max + 1, 1, dtype=torch.bool, device=dev)
        mask[:, :n, :] = True
        mask[:, n_max, :] = True

        return X_pad, Y_pad, xq, data_dict["yq"], c_true, data_dict["q_true"], mu, tau, mask

    else:
        # No padding needed, return full True mask
        mask = torch.ones(B, n_max + 1, 1, dtype=torch.bool, device=dev)
        return X, Y, xq, data_dict["yq"], c_true, data_dict["q_true"], mu, tau, mask

def use_gpu_batchgen(args, device: torch.device) -> bool:
    return (args.task == "rbfmix") and (device.type == "cuda")

def make_fixed_eval_loader(args, device, dtype, edges, blr_diag_var):
    print("Generating fixed evaluation set (this takes a moment)...")
    eval_data = []

    # For rbfmix on CUDA: generate on GPU (so Cholesky is on GPU),
    # then immediately move stored eval batches back to CPU to save VRAM.
    eval_gen_on_gpu = (args.task == "rbfmix") and (torch.device(device).type == "cuda")
    eval_gen_device = device if eval_gen_on_gpu else "cpu"

    edges_gen = edges.to(device=eval_gen_device, dtype=torch.float64)
    blr_diag_var_gen = (
        blr_diag_var.to(device=eval_gen_device, dtype=torch.float64)
        if blr_diag_var is not None else None
    )

    for _ in range(args.eval_batches):
        if args.n_mode == "single":
            n = args.n
        else:
            n = int(torch.randint(
                low=args.n_min,
                high=args.n_max + 1,
                size=(1,),
                device=device if torch.device(device).type == "cuda" else "cpu"
            ).item())

        # Generate batch in float64 on the chosen generation device
        batch = sample_batch(
            task=args.task,
            B=args.batch,
            n=n,
            d=args.d,
            device=eval_gen_device,
            dtype=torch.float64,
            edges=edges_gen,
            sigma=args.sigma,
            blr_diag_var=blr_diag_var_gen,
            rbf_alpha=args.rbf_alpha,
            rbf_ell=args.rbf_ell,
            rbf_jitter=args.rbf_jitter,
            x_dist=args.x_dist,
            unit_norm_x=bool(args.unit_norm_x),
            rbfmix_ell_grid=args.rbfmix_ell_grid,
            rbfmix_sigma_grid=args.rbfmix_sigma_grid,
        )

        # Cast model inputs to eval dtype before padding
        batch["X"] = batch["X"].to(dtype=dtype)
        batch["Y"] = batch["Y"].to(dtype=dtype)
        batch["xq"] = batch["xq"].to(dtype=dtype)

        # Pad on generation device
        batch_tuple = collate_pad(batch, args.n_max)

        # Store on CPU so the fixed eval set does not occupy VRAM
        batch_tuple_cpu = tuple(
            t.to("cpu") if isinstance(t, torch.Tensor) else t
            for t in batch_tuple
        )
        eval_data.append(batch_tuple_cpu)

    print(f"Generated {len(eval_data)} fixed eval batches.")
    return eval_data

@torch.no_grad()
def evaluate(
    net: torch.nn.Module,
    eval_loader: list, 
    device,
    dtype,
    edges: torch.Tensor,
) -> dict[str, float]:
    net.eval()
    nlls, kls, tvs, mse_mus, mse_taus= [], [], [], [], []

    # edges: (C+1,) -> centers: (C,) -> (1, 1, C) for broadcasting
    # Ensure edges are on the computation device
    edges = edges.to(device)

    # Iterate over pre-generated data
    for batch_tuple in eval_loader:
        # Unpack tuple (already padded)
        # (X_pad, Y_pad, xq, yq, c_true, q_true, mu, tau, mask)
        X_pad, Y_pad, xq, yq, c_true, q_true, mu_true, tau_true, mask = batch_tuple

        # Move inputs to GPU/float32
        X_in = X_pad.to(device=device, dtype=dtype)
        Y_in = Y_pad.to(device=device, dtype=dtype)
        xq_in = xq.to(device=device, dtype=dtype)
        mask_in = mask.to(device=device)

        # Forward
        logits, aux = net(X_in, Y_in, xq_in, padding_mask=mask_in)
        
        # Metrics
        nll = F.cross_entropy(logits, c_true.to(device))
        
        # Convert logits to probabilities (High Precision for Math)
        probs = torch.softmax(logits, dim=-1).to(dtype=torch.float64) 

        # 2. Parameter Errors (MSE_mu, MSE_tau)
        # We stick to aux["mu"] (Analytic/Learned Mean) for parameter comparison 
        # because it represents the model's internal "belief" state.
        aux_mu = aux["mu"].to(dtype=torch.float64)
        aux_tau = aux["tau"].to(dtype=torch.float64)
        
        mu_true = mu_true.to(device)
        tau_true = tau_true.to(device)
        
        mse_mus.append(F.mse_loss(aux_mu, mu_true).item())
        mse_taus.append(F.mse_loss(aux_tau, tau_true).item())

        # 3. Distributional Metrics (KL, TV)
        nlls.append(nll.item())

        if q_true is not None:
            q_true_dev = q_true.to(device=device, dtype=torch.float64)
            if q_true_dev.ndim == 3:
                q_true_dev = q_true_dev.squeeze(1)

            eps = 1e-12
            q_safe = torch.clamp(q_true_dev, min=eps)
            p_safe = torch.clamp(probs, min=eps)

            kl_disc = (q_safe * (torch.log(q_safe) - torch.log(p_safe))).sum(dim=-1).mean().item()
            tv_disc = 0.5 * torch.abs(q_true_dev - probs).sum(dim=-1).mean().item()

            kls.append(kl_disc)
            tvs.append(tv_disc)
        else:
            kls.append(float("nan"))
            tvs.append(float("nan"))

    return {
        "eval_nll": float(sum(nlls) / len(nlls)),
        "eval_kl": float(sum(kls) / len(kls)),
        "eval_tv": float(sum(tvs) / len(tvs)),
        "eval_mse_mu": float(sum(mse_mus) / len(mse_mus)),
        "eval_mse_tau": float(sum(mse_taus) / len(mse_taus))
    }

def save_ckpt(path: str, net: torch.nn.Module, opt, scheduler, step: int, cfg: model.ModelConfig, args, best_nll: float) -> None:
    payload = {
        "step": step,
        "cfg": asdict(cfg),
        "args": vars(args),
        "model_state": net.state_dict(),
        "opt_state": opt.state_dict() if opt is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None, # [FIX] Save scheduler
        "best_eval_nll": best_nll,
    }
    torch.save(payload, path)

# --- New Dataset Class for Stable Training ---
class UQIterableDataset(IterableDataset):
    def __init__(self, args, edges, blr_diag_var):
        self.args = args
        # [FIX] Move tensors to CPU immediately. 
        # Workers cannot access GPU tensors, so we must store CPU copies here.
        self.edges = edges.cpu() 
        self.blr_diag_var = blr_diag_var.cpu() if blr_diag_var is not None else None
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.args.seed
        if worker_info is not None:
            seed += worker_info.id
        set_seed(seed)
        
        while True:
            # Select n
            if self.args.n_mode == "single":
                n = int(self.args.n)
            else:
                n = int(torch.randint(low=self.args.n_min, high=self.args.n_max + 1, size=(1,)).item())
            
            # Generate in float64 (CPU)
            # [FIX] self.edges and self.blr_diag_var are already on CPU now.
            batch = sample_batch(
                task=self.args.task, B=self.args.batch, n=n, d=self.args.d,
                device="cpu", dtype=torch.float64,
                edges=self.edges,
                sigma=self.args.sigma,
                blr_diag_var=self.blr_diag_var,
                rbf_alpha=self.args.rbf_alpha,
                rbf_ell=self.args.rbf_ell,
                rbf_jitter=self.args.rbf_jitter,
                x_dist=self.args.x_dist,
                unit_norm_x=bool(self.args.unit_norm_x),
                rbfmix_ell_grid=self.args.rbfmix_ell_grid,
                rbfmix_sigma_grid=self.args.rbfmix_sigma_grid,
            )
            
            # Cast inputs to float32 before yielding
            batch["X"] = batch["X"].to(dtype=torch.float32)
            batch["Y"] = batch["Y"].to(dtype=torch.float32)
            batch["xq"] = batch["xq"].to(dtype=torch.float32)
            
            yield batch

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        # 1. Warmup Phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 2. Cosine Decay Phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["blr", "rbf", "rbfmix"], default="blr")

    # model/training sizes
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)

    # n selection
    ap.add_argument("--n_mode", type=str, choices=["single", "mixture"], default="mixture")
    ap.add_argument("--n", type=int, default=64, help="Used when --n_mode single")
    ap.add_argument("--n_min", type=int, default=64, help="Used when --n_mode mixture")
    ap.add_argument("--n_max", type=int, default=128, help="Used when --n_mode mixture")

    # bins
    ap.add_argument("--a", type=float, default=-4.0)
    ap.add_argument("--b", type=float, default=4.0)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--auto_ab", action="store_true", default=True)
    ap.add_argument("--no_auto_ab", dest="auto_ab", action="store_false")
    ap.add_argument("--ab_samples", type=int, default=2000)
    ap.add_argument("--ab_q_low", type=float, default=0.001)
    ap.add_argument("--ab_q_high", type=float, default=0.999)

    # pretrain/test data params
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--rbf_alpha", type=float, default=1.0)
    ap.add_argument("--rbf_ell", type=float, default=0.8)
    ap.add_argument("--rbf_jitter", type=float, default=1e-6)
    ap.add_argument("--rbfmix_ell_grid", type=float, nargs="+", default=[0.4, 0.8, 1.2])
    ap.add_argument("--rbfmix_sigma_grid", type=float, nargs="+", default=[0.1, 0.2, 0.3])

    # Model Mode
    ap.add_argument("--mode", type=str, choices=["learnable", "theory"], default="learnable")
    ap.add_argument("--learn_convert_scales", type=int, default=1)
    ap.add_argument("--eta_shared", type=int, default=0)
    
    # Normalization Option
    ap.add_argument("--normalize", type=int, default=0)
    ap.add_argument("--normdecay", type=int, default=0)
    ap.add_argument("--kernel", type=str, default="auto", choices=["auto", "linear", "rbf", "softmax"])

    # init
    ap.add_argument("--feature_scale_init", type=float, default=0.83)
    ap.add_argument("--eta_init", type=float, default=0.01)

    # optimization
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=32)

    # checkpointing
    ap.add_argument("--save_dir", type=str, default="", help="If empty, auto-generate from args.")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--save_best", type=int, default=1)
    ap.add_argument("--reset_scheduler", action="store_true", help="Reset scheduler when resuming.")

    # numeric
    # ap.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--x_dist", type=str, choices=["normal", "uniform"], default="normal")
    ap.add_argument("--unit_norm_x", type=int, default=0,
                    help="If 1, normalize each covariate/query vector to unit norm after sampling.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    ap.add_argument('--finetune_from', type=str, default=None, help='Path to checkpoint to load weights from (resets optimizer/scheduler)')

    args = ap.parse_args()
    
    if args.kernel == "auto":
        if args.task == "blr":
            args.kernel = "linear"
        elif args.task in {"rbf", "rbfmix"}:
            args.kernel = "rbf"
        else:
            raise ValueError(f"Unknown task={args.task}")

    """
    logic
    if A is true => must be this
    task blr => kernel linear & normalize 0 & normdecay 0
    task rbf => kernel [rbf, softmax]
    normdecay 1 => normalize 1
    kernel softmax => normalize 1
    """
    if args.task == "blr":
        args.kernel = "linear"
        args.normalize = 0
        args.normdecay = 0
        print(f"[warning] blr, forced to --kernel linear --normalize 0 --normdecay 0")
    if args.normdecay == 1:
        args.normalize = 1
        print(f"[warning] forced to --normalize 1 since --normdecay 1")
    if args.kernel == "softmax":
        args.normalize = 1
        print(f"[warning] forced --normalize 1 since --kernel softmax")
    if args.task in {"rbf", "rbfmix"} and args.kernel == "softmax":
        args.unit_norm_x = 1
        print(f"[warning] forced --unit_norm_x 1 since --task rbf with --kernel softmax")

    do_normalize = bool(args.normalize)
    do_normdecay = bool(args.normdecay)

    # make saving directory
    if args.save_dir == "":
        if args.n_mode == "single":
            n_tag = f"n{args.n}"
        else:
            n_tag = f"n{args.n_min}-{args.n_max}"
        
        norm_tag = "_norm" if do_normalize else ""
        kernel_tag = f"_{args.kernel}" if args.kernel == "softmax" else ""
        nd_tag = "_nd" if do_normdecay else ""
        xnorm_tag = "_xunit" if bool(args.unit_norm_x) else ""

        if args.task == "rbf":
            sigma_tag = f"_sig{args.sigma:g}" if abs(args.sigma - 0.2) > 1e-12 else ""
            ell_tag = f"_ell{args.rbf_ell:g}" if abs(args.rbf_ell - 0.8) > 1e-12 else ""
            task_tag = f"{ell_tag}{sigma_tag}"
        elif args.task == "rbfmix":
            ell_grid_tag = "-".join(f"{x:g}" for x in args.rbfmix_ell_grid)
            sigma_grid_tag = "-".join(f"{x:g}" for x in args.rbfmix_sigma_grid)
            task_tag = f"_ellmix{ell_grid_tag}_sigmix{sigma_grid_tag}"
        else:
            task_tag = ""

        args.save_dir = os.path.join(
            "runs",
            f"{args.task}_d{args.d}{task_tag}_C{args.C}_L{args.L}_{args.n_mode}_{n_tag}_{args.mode}{norm_tag}{nd_tag}{kernel_tag}_seed{args.seed}"
        )
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[ckpt] save_dir={args.save_dir}")

    set_seed(args.seed)
    device = torch.device(args.device)
    # dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dtype = torch.float32
    gpu_batchgen = use_gpu_batchgen(args, device)
    if gpu_batchgen:
        print("[batchgen] using direct GPU batch generation for rbfmix")

    if args.task == "blr":
        blr_diag_var = data._default_diag_var(d = args.d, device=device, dtype=dtype)
    else:
        blr_diag_var = None

    if args.auto_ab:
        if args.task == "blr":
            hypers_for_ab = data.BLR_hyprms(
                sigma=args.sigma, diag_var=blr_diag_var
            )
        elif args.task == "rbf":
            hypers_for_ab = data.RBF_hyprms(
                sigma=args.sigma, alpha=args.rbf_alpha,
                ell=args.rbf_ell, jitter=args.rbf_jitter
            )
        elif args.task == "rbfmix":
            hypers_for_ab = data.RBF_Mixture_hyprms(
                sigma_grid=tuple(float(x) for x in args.rbfmix_sigma_grid),
                ell_grid=tuple(float(x) for x in args.rbfmix_ell_grid),
                alpha=args.rbf_alpha,
                jitter=args.rbf_jitter,
            )
        else:
            raise ValueError(f"Unknown task={args.task}")
        a_hat, b_hat = estimate_ab_from_pretrain(
            task=args.task, d=args.d,
            n_mode=args.n_mode, n=args.n, n_min=args.n_min, n_max=args.n_max,
            B=min(args.batch, 256),
            num_samples=args.ab_samples, q_low=args.ab_q_low, q_high=args.ab_q_high,
            hypers=hypers_for_ab, device=device, dtype=torch.float64, x_dist=args.x_dist,
            unit_norm_x=bool(args.unit_norm_x),
        )
        args.a, args.b = a_hat, b_hat
        print(f"[auto_ab] estimated a={args.a:.4f}, b={args.b:.4f}")

    edges = data.make_bin_edges(args.a, args.b, args.C, device=device, dtype=dtype)

    cfg = model.ModelConfig(
        d=args.d,
        L=args.L,
        mode=args.mode,             
        normalize=do_normalize,     
        norm_decay=do_normdecay,
        kernel=args.kernel,         
        sigma_init=args.sigma,
        feature_scale_init=args.feature_scale_init,
        feature_scale_init_vec=None,
        eta_shared=bool(args.eta_shared),
        eta_init=args.eta_init,
        # SPEED: Disable checkpointing to trade memory for speed (remove if OOM)
        gradient_checkpointing=True, 
    )

    # [FIX] Keep a reference to the raw module for saving and custom methods
    net_raw = model.TransformerUQ(cfg=cfg, edges=edges).to(device=device, dtype=dtype)
    freeze_all_params(net_raw)
    to_unfreeze = []
    if args.learn_convert_scales:
        to_unfreeze += ["convert.s1", "convert.s2"]
    if args.mode == "theory":
        to_unfreeze.append("raw_eta_k")
        to_unfreeze.append("raw_eta_r")
        to_unfreeze.append("log_feature_scale")
        to_unfreeze.append("log_sigma")
    elif args.mode == "learnable":
        to_unfreeze.append("v0_kx_label")
        to_unfreeze.append("layers")
        to_unfreeze.append("w_readout")
        to_unfreeze.append("log_feature_scales")
    unfreeze_by_name(net_raw, to_unfreeze)
    # for name, p in net_raw.named_parameters():
    #     if p.requires_grad:
    #         print(f"  [trainable] {name:40s} {tuple(p.shape)}  ({p.numel():,})")
    net = net_raw

    opt = make_optimizer(net_raw, lr=args.lr, weight_decay=args.weight_decay)
    
    # Safety check to ensure parameters were actually unfrozen
    assert opt is not None, "Optimizer is None! No parameters matched 'unfreeze_by_name' patterns. Check substrings vs model.named_parameters()."

    # --- Scheduler Setup ---
    warmup_steps = int(args.steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=args.steps, min_lr_ratio=0.1
    )

    # Resume Logic (Moved AFTER scheduler creation to allow scheduler load)
    start_step = 1
    best_eval_nll = float("inf")

    if args.finetune_from:
        print(f"✨ Loading weights for fine-tuning from: {args.finetune_from}")
        checkpoint = torch.load(args.finetune_from, map_location=device)
        
        # Load model weights only
        # Use strict=False in case you want to slightly change architecture later, but strict=True is safer for now
        net.load_state_dict(checkpoint['model_state'], strict=True)
        
        # Explicitly do NOT load optimizer/scheduler/step
        # This allows the new LR and scheduler settings to take effect

    elif args.resume:
        if os.path.isfile(args.resume):
            print(f"==> Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            # Load state into net_raw
            state = ckpt["model_state"]

            # --- Backward compat: old checkpoints had raw_eta, new model has raw_eta_k/raw_eta_r ---
            if "raw_eta" in state and ("raw_eta_k" not in state and "raw_eta_r" not in state):
                print("[ckpt] migrating raw_eta -> raw_eta_k/raw_eta_r")
                state["raw_eta_k"] = state["raw_eta"].clone()
                state["raw_eta_r"] = state["raw_eta"].clone()
                del state["raw_eta"]

            net_raw.load_state_dict(state, strict=True)
            
            # Resume scheduler properly
            if not args.reset_scheduler:
                if ckpt.get("opt_state") is not None:
                    opt.load_state_dict(ckpt["opt_state"])
                if ckpt.get("scheduler_state") is not None:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                    print("==> Scheduler state restored.")
                else:
                    print("!! Checkpoint missing scheduler state. Scheduler starting fresh.")
            else:
                print("==> Resetting optimizer/scheduler to fresh start.")

            start_step = ckpt["step"] + 1
            if "best_eval_nll" in ckpt:
                best_eval_nll = ckpt["best_eval_nll"]
        else:
            print(f"!! Checkpoint {args.resume} not found. Starting from scratch.")

    # --- Setup Data ---
    train_iter = None
    if not gpu_batchgen:
        train_ds = UQIterableDataset(args, edges, blr_diag_var)
        train_dl = DataLoader(
            train_ds,
            batch_size=None,  # Dataset yields full batches
            num_workers=1,
            pin_memory=True,
            collate_fn=lambda x: collate_pad(x, args.n_max)
        )
        train_iter = iter(train_dl)
    
    # --- Setup CSV Logging ---
    log_path = os.path.join(args.save_dir, "log.csv")
    
    # Only write headers if starting from scratch (step 1)
    if start_step == 1:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "lr", "train_loss", 
                "eval_nll", "eval_kl", "eval_tv", 
                "eval_mse_mu", "eval_mse_tau", "time"
            ])

    # Generate Fixed Eval Set ONCE
    fixed_eval_data = make_fixed_eval_loader(args, device, dtype, edges, blr_diag_var)
    if gpu_batchgen:
        torch.cuda.empty_cache()
    t0 = time.time()
    # TRAINING LOOP
    for step in range(start_step, args.steps + 1):
        net.train()

        if gpu_batchgen:
            # Sample directly on GPU so rbfmix kernel build + Cholesky happen on GPU
            if args.n_mode == "single":
                n = int(args.n)
            else:
                n = int(torch.randint(
                    low=args.n_min,
                    high=args.n_max + 1,
                    size=(1,),
                    device=device
                ).item())

            batch = sample_batch(
                task=args.task,
                B=args.batch,
                n=n,
                d=args.d,
                device=device,
                dtype=torch.float64,   # keep GP / Cholesky numerics in float64
                edges=edges.to(device=device, dtype=torch.float64),
                sigma=args.sigma,
                blr_diag_var=blr_diag_var.to(device=device, dtype=torch.float64) if blr_diag_var is not None else None,
                rbf_alpha=args.rbf_alpha,
                rbf_ell=args.rbf_ell,
                rbf_jitter=args.rbf_jitter,
                x_dist=args.x_dist,
                unit_norm_x=bool(args.unit_norm_x),
                rbfmix_ell_grid=args.rbfmix_ell_grid,
                rbfmix_sigma_grid=args.rbfmix_sigma_grid,
            )

            # Cast model inputs down to train dtype after batch generation
            batch["X"] = batch["X"].to(dtype=dtype)
            batch["Y"] = batch["Y"].to(dtype=dtype)
            batch["xq"] = batch["xq"].to(dtype=dtype)

            batch_data = collate_pad(batch, args.n_max)

            X, Y, xq, _, c_true, _, _, _, pad_mask = batch_data
        else:
            try:
                batch_data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch_data = next(train_iter)

            # tuple: (X_pad, Y_pad, xq, yq, c_true, q_true, mu, tau, mask)
            X_pad, Y_pad, xq, _, c_true, _, _, _, mask = batch_data

            X = X_pad.to(device, non_blocking=True)
            Y = Y_pad.to(device, non_blocking=True)
            xq = xq.to(device, non_blocking=True)
            c_true = c_true.to(device, non_blocking=True)
            pad_mask = mask.to(device, non_blocking=True)

        # --- [CHANGE START] Mixed Precision Wrapper ---
        # We wrap ONLY the Forward Pass and Loss Calculation.
        # This runs the heavy math in bfloat16 (fast) but keeps the graph 
        # compatible with the float32 optimizer.
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=autocast_device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            logits, aux = net(X, Y, xq, padding_mask=pad_mask)
            loss = F.cross_entropy(logits, c_true)
        # --- [CHANGE END] ---

        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in net_raw.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            scheduler.step()
            if hasattr(net_raw, "project_parameters_"):
                net_raw.project_parameters_()

        if step % args.eval_every == 0 or step == 1:
            metrics = evaluate(
                net, 
                eval_loader=fixed_eval_data, # <--- Pass the list
                device=device, 
                dtype=dtype,
                edges=edges
            )
            elapsed = time.time() - t0
            # Get current Learning Rate
            current_lr = scheduler.get_last_lr()[0]

            # 1. Print to Terminal
            print(f"[step {step:6d}] LR={current_lr:.2e} NLL={loss.item():.3f} "
                  f"eval_NLL={metrics['eval_nll']:.3f} eval_TV={metrics['eval_tv']:.3e} ({elapsed:.1f}s)")

            # 2. Save to CSV
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, 
                    current_lr, 
                    loss.item(), 
                    metrics['eval_nll'], 
                    metrics['eval_kl'], 
                    metrics['eval_tv'], 
                    metrics['eval_mse_mu'], 
                    metrics['eval_mse_tau'], 
                    elapsed
                ])

            if args.save_best and metrics["eval_nll"] < best_eval_nll:
                best_eval_nll = metrics["eval_nll"]
                save_ckpt(os.path.join(args.save_dir, "best.pt"), net_raw, opt, scheduler, step, cfg, args, best_eval_nll)

            if args.save_every and (step % args.save_every == 0):
                save_ckpt(os.path.join(args.save_dir, f"ckpt_step_{step}.pt"), net_raw, opt, scheduler, step, cfg, args, best_eval_nll)

    save_ckpt(os.path.join(args.save_dir, "final.pt"), net_raw, opt, scheduler, args.steps, cfg, args, best_eval_nll)
    print(f"Done. Saved to {args.save_dir}")

if __name__ == "__main__":
    main()