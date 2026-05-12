from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import csv
import argparse
import math
import time
import re
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.amp import autocast

import data
import model

# -----------------------------------------------------------------------------
# Hard-coded Sacramento EB hyperparameters (fit on full dataset after rescaling)
# -----------------------------------------------------------------------------
SACRAMENTO_PARAMS = {
    'sigma': 0.8356424725,
    'alpha': 0.6657966795,
    'ell': [1.2260828077648718, 0.7696499474208177],
    'x_constant': 0.3,
    'y_constant': 1.0,
}

# Enable TF32 for speed on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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


def freeze_all_params(m: torch.nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def collate_pad(batch_list, n_max: int):
    data_dict = batch_list[0] if isinstance(batch_list, list) else batch_list

    X = data_dict['X']
    B, n, d = X.shape
    dev = X.device

    Y = data_dict['Y']
    xq = data_dict['xq']
    c_true = data_dict['c_true']
    mu = data_dict['mu']
    tau = data_dict['tau']

    if n < n_max:
        pad_len = n_max - n
        X_pad = F.pad(X, (0, 0, 0, pad_len), value=0.0)
        Y_pad = F.pad(Y, (0, pad_len), value=0.0)
        mask = torch.zeros(B, n_max + 1, 1, dtype=torch.bool, device=dev)
        mask[:, :n, :] = True
        mask[:, n_max, :] = True
        return X_pad, Y_pad, xq, data_dict['yq'], c_true, data_dict['q_true'], mu, tau, mask
    else:
        mask = torch.ones(B, n_max + 1, 1, dtype=torch.bool, device=dev)
        return X, Y, xq, data_dict['yq'], c_true, data_dict['q_true'], mu, tau, mask




def sample_ard_batch(args, n: int, device, dtype, edges: torch.Tensor | None):
    ell = torch.tensor(SACRAMENTO_PARAMS['ell'], device=device, dtype=torch.float64)
    hypers = data.ARD_RBF_hyprms(
        sigma=float(SACRAMENTO_PARAMS['sigma']),
        alpha=float(SACRAMENTO_PARAMS['alpha']),
        ell=ell,
        jitter=args.rbf_jitter,
    )
    batch = data.draw_ARD_RBF_batch(
        B=args.batch,
        n=n,
        d=args.d,
        hypers=hypers,
        device=device,
        dtype=torch.float64,
        x_dist=args.x_dist,
        edges=edges,
    )
    batch['X'] = batch['X'].to(dtype=dtype)
    batch['Y'] = batch['Y'].to(dtype=dtype)
    batch['xq'] = batch['xq'].to(dtype=dtype)
    return batch
@torch.no_grad()
def estimate_ab_from_pretrain(args, device) -> tuple[float, float]:
    hypers = data.ARD_RBF_hyprms(
        sigma=float(SACRAMENTO_PARAMS['sigma']),
        alpha=float(SACRAMENTO_PARAMS['alpha']),
        ell=torch.tensor(SACRAMENTO_PARAMS['ell'], device=device, dtype=torch.float64),
        jitter=args.rbf_jitter,
    )

    ys = []
    remaining = args.ab_samples
    while remaining > 0:
        Bb = min(min(args.batch, 256), remaining)
        if args.n_mode == 'single':
            nn = args.n
        else:
            nn = int(torch.randint(low=args.n_min, high=args.n_max + 1, size=(1,)).item())

        batch = data.draw_ARD_RBF_batch(
            B=Bb,
            n=nn,
            d=args.d,
            hypers=hypers,
            device=device,
            dtype=torch.float64,
            x_dist=args.x_dist,
            edges=None,
        )
        y = batch.get('yq_untrunc', batch['yq'])
        ys.append(y.detach().reshape(-1).to(device='cpu', dtype=torch.float64))
        remaining -= Bb

    y_all = torch.cat(ys, dim=0)
    a = float(torch.quantile(y_all, args.ab_q_low).item())
    b = float(torch.quantile(y_all, args.ab_q_high).item())
    if not (a < b):
        m = float(y_all.mean().item())
        s = float(y_all.std(unbiased=False).item())
        a, b = m - 4.0 * s, m + 4.0 * s
    return a, b


def make_fixed_eval_loader(args, device, dtype, edges):
    print('Generating fixed evaluation set on GPU...')
    eval_data = []
    edges_gen = edges.to(device=device, dtype=torch.float64)

    for _ in range(args.eval_batches):
        if args.n_mode == 'single':
            n = args.n
        else:
            n = int(torch.randint(low=args.n_min, high=args.n_max + 1, size=(1,), device=device).item())

        batch = sample_ard_batch(args, n=n, device=device, dtype=dtype, edges=edges_gen)
        batch_tuple = collate_pad(batch, args.n_max)
        eval_data.append(batch_tuple)

    print(f'Generated {len(eval_data)} fixed eval batches on GPU.')
    return eval_data


@torch.no_grad()
def evaluate(net: torch.nn.Module, eval_loader: list, device, dtype, edges: torch.Tensor) -> dict[str, float]:
    net.eval()
    nlls, kls, tvs, mse_mus, mse_taus = [], [], [], [], []
    edges = edges.to(device)

    for batch_tuple in eval_loader:
        X_pad, Y_pad, xq, yq, c_true, q_true, mu_true, tau_true, mask = batch_tuple
        X_in = X_pad.to(device=device, dtype=dtype)
        Y_in = Y_pad.to(device=device, dtype=dtype)
        xq_in = xq.to(device=device, dtype=dtype)
        mask_in = mask.to(device=device)

        logits, aux = net(X_in, Y_in, xq_in, padding_mask=mask_in)
        nll = F.cross_entropy(logits, c_true.to(device))
        probs = torch.softmax(logits, dim=-1).to(dtype=torch.float64)

        aux_mu = aux['mu'].to(dtype=torch.float64)
        aux_tau = aux['tau'].to(dtype=torch.float64)
        mu_true = mu_true.to(device)
        tau_true = tau_true.to(device)
        mse_mus.append(F.mse_loss(aux_mu, mu_true).item())
        mse_taus.append(F.mse_loss(aux_tau, tau_true).item())
        nlls.append(nll.item())

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

    return {
        'eval_nll': float(sum(nlls) / len(nlls)),
        'eval_kl': float(sum(kls) / len(kls)),
        'eval_tv': float(sum(tvs) / len(tvs)),
        'eval_mse_mu': float(sum(mse_mus) / len(mse_mus)),
        'eval_mse_tau': float(sum(mse_taus) / len(mse_taus)),
    }


def save_ckpt(path: str, net: torch.nn.Module, opt, scheduler, step: int, cfg: model.ModelConfig, args, best_nll: float) -> None:
    payload = {
        'step': step,
        'cfg': asdict(cfg),
        'args': vars(args),
        'model_state': net.state_dict(),
        'opt_state': opt.state_dict() if opt is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'best_eval_nll': best_nll,
        'sacramento_params': SACRAMENTO_PARAMS,
    }
    torch.save(payload, path)


def find_latest_step_checkpoint(save_dir: str) -> str:
    if not os.path.isdir(save_dir):
        return ''
    candidates = []
    pat = re.compile(r'^ckpt_step_(\d+)\.pt$')
    for name in os.listdir(save_dir):
        m = pat.match(name)
        if m:
            candidates.append((int(m.group(1)), os.path.join(save_dir, name)))
    if not candidates:
        return ''
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def parse_args():
    ap = argparse.ArgumentParser(description='Dedicated Sacramento ARD-RBF PFN training script.')
    ap.add_argument('--d', type=int, default=2)
    ap.add_argument('--L', type=int, default=32)
    ap.add_argument('--batch', type=int, default=128)

    ap.add_argument('--n_mode', type=str, choices=['single', 'mixture'], default='mixture')
    ap.add_argument('--n', type=int, default=552)
    ap.add_argument('--n_min', type=int, default=128)
    ap.add_argument('--n_max', type=int, default=552)

    ap.add_argument('--a', type=float, default=-4.0)
    ap.add_argument('--b', type=float, default=4.0)
    ap.add_argument('--C', type=int, default=256)
    ap.add_argument('--auto_ab', action='store_true', default=True)
    ap.add_argument('--no_auto_ab', dest='auto_ab', action='store_false')
    ap.add_argument('--ab_samples', type=int, default=2000)
    ap.add_argument('--ab_q_low', type=float, default=0.001)
    ap.add_argument('--ab_q_high', type=float, default=0.999)
    ap.add_argument('--rbf_jitter', type=float, default=1e-6)

    ap.add_argument('--eta_init', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--eval_batches', type=int, default=32)

    ap.add_argument('--save_dir', type=str, default='runs/sacramento_eb')
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--save_best', type=int, default=1)
    ap.add_argument('--reset_scheduler', action='store_true')

    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--x_dist', type=str, choices=['normal', 'uniform'], default='normal')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--auto_resume', type=int, default=1, help='If 1 and --resume is empty, resume from the latest ckpt_step_*.pt in save_dir if available.')
    ap.add_argument('--finetune_from', type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    assert args.d == 2, 'Sacramento script currently assumes d=2 spatial coordinates.'

    os.makedirs(args.save_dir, exist_ok=True)
    print(f'[ckpt] save_dir={args.save_dir}')
    print(f'[sacramento eb] {SACRAMENTO_PARAMS}')

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32

    if args.auto_ab:
        a_hat, b_hat = estimate_ab_from_pretrain(args, device)
        args.a, args.b = a_hat, b_hat
        print(f'[auto_ab] estimated a={args.a:.4f}, b={args.b:.4f}')

    edges = data.make_bin_edges(args.a, args.b, args.C, device=device, dtype=dtype)
    feature_scale_init_vec = (1.0 / torch.tensor(SACRAMENTO_PARAMS['ell'], dtype=torch.float32)).tolist()

    cfg = model.ModelConfig(
        d=args.d,
        L=args.L,
        mode='learnable',
        normalize=True,
        norm_decay=True,
        kernel='rbf',
        sigma_init=float(SACRAMENTO_PARAMS['sigma']),
        feature_scale_init=1.0,
        feature_scale_init_vec=feature_scale_init_vec,
        eta_shared=False,
        eta_init=args.eta_init,
        gradient_checkpointing=True,
    )

    net_raw = model.TransformerUQ(cfg=cfg, edges=edges).to(device=device, dtype=dtype)
    freeze_all_params(net_raw)
    for name, p in net_raw.named_parameters():
        if any(s in name for s in ['convert.s1', 'convert.s2', 'v0_kx_label', 'layers', 'w_readout', 'log_feature_scales']):
            p.requires_grad_(True)
    net = net_raw

    params = [p for p in net_raw.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = int(args.steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=args.steps, min_lr_ratio=0.1)

    start_step = 1
    best_eval_nll = float('inf')

    resume_path = args.resume.strip()
    if not resume_path and bool(args.auto_resume):
        resume_path = find_latest_step_checkpoint(args.save_dir)
        if resume_path:
            print(f'==> Auto-resume found latest checkpoint: {resume_path}')

    if args.finetune_from:
        print(f'Loading weights for fine-tuning from: {args.finetune_from}')
        checkpoint = torch.load(args.finetune_from, map_location=device)
        net.load_state_dict(checkpoint['model_state'], strict=True)
    elif resume_path:
        if os.path.isfile(resume_path):
            print(f'==> Resuming from checkpoint: {resume_path}')
            ckpt = torch.load(resume_path, map_location=device)
            net_raw.load_state_dict(ckpt['model_state'], strict=True)
            if not args.reset_scheduler:
                if ckpt.get('opt_state') is not None:
                    opt.load_state_dict(ckpt['opt_state'])
                if ckpt.get('scheduler_state') is not None:
                    scheduler.load_state_dict(ckpt['scheduler_state'])
                    print('==> Scheduler state restored.')
            else:
                print('==> Resetting optimizer/scheduler to fresh start.')
            start_step = ckpt['step'] + 1
            if 'best_eval_nll' in ckpt:
                best_eval_nll = ckpt['best_eval_nll']
        else:
            print(f'!! Checkpoint {resume_path} not found. Starting from scratch.')

    log_path = os.path.join(args.save_dir, 'log.csv')
    if start_step == 1:
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'lr', 'train_loss',
                'eval_nll', 'eval_kl', 'eval_tv',
                'eval_mse_mu', 'eval_mse_tau', 'time'
            ])

    fixed_eval_data = make_fixed_eval_loader(args, device, dtype, edges)
    t0 = time.time()

    for step in range(start_step, args.steps + 1):
        net.train()
        if args.n_mode == 'single':
            n = int(args.n)
        else:
            n = int(torch.randint(low=args.n_min, high=args.n_max + 1, size=(1,), device=device).item())

        batch = sample_ard_batch(args, n=n, device=device, dtype=dtype, edges=edges.to(device=device, dtype=torch.float64))
        batch_data = collate_pad(batch, args.n_max)

        X, Y, xq, _, c_true, _, _, _, pad_mask = batch_data

        autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=autocast_device, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
            logits, aux = net(X, Y, xq, padding_mask=pad_mask)
            loss = F.cross_entropy(logits, c_true)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in net_raw.parameters() if p.requires_grad], args.grad_clip)
        opt.step()
        scheduler.step()
        if hasattr(net_raw, 'project_parameters_'):
            net_raw.project_parameters_()

        if step % args.eval_every == 0 or step == 1:
            metrics = evaluate(net, eval_loader=fixed_eval_data, device=device, dtype=dtype, edges=edges)
            elapsed = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]
            print(f"[step {step:6d}] LR={current_lr:.2e} NLL={loss.item():.3f} "
                  f"eval_NLL={metrics['eval_nll']:.3f} eval_TV={metrics['eval_tv']:.3e} ({elapsed:.1f}s)")
            with open(log_path, 'a', newline='') as f:
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
                    elapsed,
                ])

            if args.save_best and metrics['eval_nll'] < best_eval_nll:
                best_eval_nll = metrics['eval_nll']
                save_ckpt(os.path.join(args.save_dir, 'best.pt'), net_raw, opt, scheduler, step, cfg, args, best_eval_nll)

            if args.save_every and (step % args.save_every == 0):
                save_ckpt(os.path.join(args.save_dir, f'ckpt_step_{step}.pt'), net_raw, opt, scheduler, step, cfg, args, best_eval_nll)

    save_ckpt(os.path.join(args.save_dir, 'final.pt'), net_raw, opt, scheduler, args.steps, cfg, args, best_eval_nll)
    print(f'Done. Saved to {args.save_dir}')


if __name__ == '__main__':
    main()
