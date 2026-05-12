"""
train_walker.py

Streamlined training script for Empirical Bayes (EB) demonstration on Walker Lake.
Includes Checkpoint/Resume AND "Best Checkpoint" tracking.
"""

from __future__ import annotations
import os
# Force single-threaded workers
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

# -----------------------------------------------------------------------------
# Walker Lake Empirical Bayes Constants
# -----------------------------------------------------------------------------
WALKER_LAKE_PARAMS = {
    "sigma": 0.611,          # sqrt(0.373)
    "alpha": 0.782,          # Amplitude
    "ell": [0.146, 0.252],   # Length scales (d=2)
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def collate_pad(batch_data, n_max):
    # 1. Handle dict (batch_size=None) vs list (batch_size=1)
    data_dict = batch_data[0] if isinstance(batch_data, list) else batch_data

    X, Y = data_dict["X"], data_dict["Y"]
    B, n, d = X.shape
    
    # 2. Extract all 9 expected variables
    xq = data_dict["xq"]
    yq = data_dict.get("yq")
    c_true = data_dict.get("c_true")
    q_true = data_dict.get("q_true")  
    mu, tau = data_dict["mu"], data_dict["tau"]

    # 3. Create Padding
    if n < n_max:
        pad_len = n_max - n
        X_pad = F.pad(X, (0, 0, 0, pad_len), value=0.0)
        Y_pad = F.pad(Y, (0, pad_len), value=0.0)
        
        # Mask: 1 for valid, 0 for pad. Last token (query) is always valid.
        mask = torch.zeros(B, n_max + 1, 1, dtype=torch.bool)
        mask[:, :n, :] = True
        mask[:, n_max, :] = True
        
        return X_pad, Y_pad, xq, yq, c_true, q_true, mu, tau, mask
    else:
        mask = torch.ones(B, n_max + 1, 1, dtype=torch.bool)
        return X, Y, xq, yq, c_true, q_true, mu, tau, mask

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class WalkerLakeDataset(IterableDataset):
    def __init__(self, args, edges):
        self.args = args
        self.edges = edges.cpu()
        self.base_ell = torch.tensor(WALKER_LAKE_PARAMS["ell"], dtype=torch.float64)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.args.seed + (worker_info.id if worker_info else 0)
        set_seed(seed)
        
        while True:
            # 1. Sample N (Context Size)
            n = int(torch.randint(low=self.args.n_min, high=self.args.n_max + 1, size=(1,)).item())
            
            # 2. Robust Empirical Bayes: Jitter length scales
            ell_jitter = self.base_ell * (0.8 + 0.4 * torch.rand(2, dtype=torch.float64))
            
            hypers = data.ARD_RBF_hyprms(
                sigma=WALKER_LAKE_PARAMS["sigma"],
                alpha=WALKER_LAKE_PARAMS["alpha"],
                ell=ell_jitter
            )
            
            # 3. Generate Data
            batch = data.draw_ARD_RBF_batch(
                B=self.args.batch, n=n, d=self.args.d,
                hypers=hypers,
                device="cpu", dtype=torch.float64,
                x_dist=self.args.x_dist,
                edges=self.edges
            )
            
            batch["X"] = batch["X"].to(dtype=torch.float32)
            batch["Y"] = batch["Y"].to(dtype=torch.float32)
            batch["xq"] = batch["xq"].to(dtype=torch.float32)
            
            yield batch

# -----------------------------------------------------------------------------
# Checkpointing Helpers
# -----------------------------------------------------------------------------
def save_checkpoint(path, net, opt, scheduler, step, cfg, args, best_nll):
    torch.save({
        "step": step,
        "model_state": net.state_dict(),
        "opt_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "cfg": asdict(cfg),
        "args": vars(args),
        "best_nll": best_nll  # Save the metric to persist across resumes
    }, path)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    
    # Data Params
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--n_min", type=int, default=128)
    parser.add_argument("--n_max", type=int, default=384)
    
    # Binning
    parser.add_argument("--C", type=int, default=256)
    parser.add_argument("--auto_ab", action="store_true", default=True)
    parser.add_argument("--a", type=float, default=-4.0)
    parser.add_argument("--b", type=float, default=4.0)

    # Model Params
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--eta_init", type=float, default=0.05)
    
    # Training
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="runs/walker_lake_eb")
    parser.add_argument("--x_dist", type=str, default="normal")
    
    # Resume
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt file to resume")

    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"--- Walker Lake Empirical Bayes Training ---")
    print(f"Target Hypers: {WALKER_LAKE_PARAMS}")

    # 1. Estimate [a, b] or load from args
    if args.auto_ab:
        print("Estimating output range [a,b]...")
        hypers_base = data.ARD_RBF_hyprms(
            sigma=WALKER_LAKE_PARAMS["sigma"],
            alpha=WALKER_LAKE_PARAMS["alpha"],
            ell=torch.tensor(WALKER_LAKE_PARAMS["ell"])
        )
        ys = []
        for _ in range(10):
            b_tmp = data.draw_ARD_RBF_batch(B=256, n=args.n_max, d=args.d, hypers=hypers_base, device="cpu")
            ys.append(b_tmp["yq_untrunc"].view(-1))
        y_all = torch.cat(ys)
        args.a = float(torch.quantile(y_all, 0.001).item())
        args.b = float(torch.quantile(y_all, 0.999).item())
        print(f"Auto-range: [{args.a:.3f}, {args.b:.3f}]")

    edges = data.make_bin_edges(args.a, args.b, args.C, device=device)

    # 2. Configure Model
    fitted_ell = torch.tensor(WALKER_LAKE_PARAMS["ell"])
    init_scales = (1.0 / fitted_ell).tolist()

    cfg = model.ModelConfig(
        d=args.d,
        L=args.L,
        mode="learnable",
        normalize=True,
        norm_decay=False, 
        kernel="rbf",
        sigma_init=WALKER_LAKE_PARAMS["sigma"],
        feature_scale_init_vec=init_scales,
        eta_init=args.eta_init,
        gradient_checkpointing=True
    )

    net = model.TransformerUQ(cfg=cfg, edges=edges).to(device)
    
    # 3. Setup Optimizer
    for p in net.parameters(): p.requires_grad_(False)
    
    trainable_names = ["layers", "w_readout", "log_feature_scales", "convert", "v0_kx_label"]
    trainable_params = []
    for name, p in net.named_parameters():
        if any(t in name for t in trainable_names):
            p.requires_grad_(True)
            trainable_params.append(p)
    
    opt = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(opt, int(args.steps*0.1), args.steps, 0.1)

    # 4. Resume Logic
    start_step = 1
    best_nll = float('inf')  # Initialize best metric
    
    if args.resume and os.path.isfile(args.resume):
        print(f"==> Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        
        net.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = ckpt["step"] + 1
        best_nll = ckpt.get("best_nll", float('inf'))  # Recover best_nll if available
        print(f"==> Resumed at step {start_step} (Best NLL so far: {best_nll:.4f})")
    elif args.resume:
        print(f"!! Checkpoint {args.resume} not found. Starting from scratch.")

    # 5. Data Loader
    ds = WalkerLakeDataset(args, edges)
    dl = DataLoader(ds, batch_size=None, num_workers=4, pin_memory=True, 
                    collate_fn=lambda x: collate_pad(x, args.n_max))
    train_iter = iter(dl)

    # 6. Training Loop
    print(f"Starting training loop from step {start_step} to {args.steps}...")
    t0 = time.time()
    
    mode = "a" if start_step > 1 else "w"
    log_f = open(os.path.join(args.save_dir, "log.csv"), mode, newline="")
    writer = csv.writer(log_f)
    if start_step == 1:
        writer.writerow(["step", "loss", "eval_nll", "eval_mse_mu"])

    for step in range(start_step, args.steps + 1):
        net.train()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dl)
            batch = next(train_iter)

        X_pad, Y_pad, xq, _, c_true, q_true, mu_true, _, mask = batch
        
        X = X_pad.to(device, non_blocking=True)
        Y = Y_pad.to(device, non_blocking=True)
        xq = xq.to(device, non_blocking=True)
        c_true = c_true.to(device, non_blocking=True)
        q_true = q_true.to(device, non_blocking=True) # Move Ground Truth Probs to GPU
        pad_mask = mask.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, aux = net(X, Y, xq, padding_mask=pad_mask)
            loss = F.cross_entropy(logits, c_true)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        scheduler.step()
        net.project_parameters_()

        # Logging & Best Checkpoint
        if step % 200 == 0 or step == 1:
            current_loss = loss.item()
            with torch.no_grad():
                # 1. MSE
                mse_mu = F.mse_loss(aux["mu"].float().cpu(), mu_true.float())
                # 2. TV Distance (0.5 * sum|p - q|)
                probs = torch.softmax(logits, dim=-1)
                # Ensure float32 for stable summation
                tv_dist = 0.5 * (probs.float() - q_true.float()).abs().sum(dim=-1).mean()
            
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            # Print with TV
            print(f"[step {step}] Loss: {current_loss:.4f} | TV: {tv_dist:.4f} | MSE_mu: {mse_mu:.4f} | LR: {lr:.2e}")
            writer.writerow([step, current_loss, tv_dist.item(), mse_mu.item()])
            log_f.flush()

            # --- Save Best Logic ---
            if current_loss < best_nll:
                print(f"    New best loss: {current_loss:.4f} (was {best_nll:.4f}). Saving best.pt...")
                best_nll = current_loss
                save_checkpoint(os.path.join(args.save_dir, "best.pt"), 
                              net, opt, scheduler, step, cfg, args, best_nll)

        # Periodic Checkpoint
        if step % 500 == 0 or step == args.steps:
            save_path = os.path.join(args.save_dir, f"ckpt_step_{step}.pt")
            save_checkpoint(save_path, net, opt, scheduler, step, cfg, args, best_nll)
            save_checkpoint(os.path.join(args.save_dir, "latest.pt"), 
                          net, opt, scheduler, step, cfg, args, best_nll)

    print(f"Training complete. Saved to {args.save_dir}")
    log_f.close()

if __name__ == "__main__":
    main()