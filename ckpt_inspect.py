import argparse
import os
import torch
import numpy as np

"""
# Inspect finetune best checkpoint
python ckpt_inspect.py --ckpt_path runs/<your_run>_finetune.../best.pt

# Inspect base run best checkpoint
python ckpt_inspect.py --save_dir runs/<your_run> --which best
python ckpt_inspect.py --save_dir runs/rbf_d2_C256_L32_mixture_n64-512_learnable_norm_softmax_seed0 --which best

# If you want to see all parameter keys
python ckpt_inspect.py --ckpt_path ... --show_keys
"""

def _maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _tinfo(t: torch.Tensor):
    return f"shape={tuple(t.shape)}, dtype={t.dtype}, min={t.min().item():.4g}, max={t.max().item():.4g}"

def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, default="", help="Path to checkpoint .pt (overrides --save_dir/--which).")
    ap.add_argument("--save_dir", type=str, default="", help="Directory containing best.pt/final.pt.")
    ap.add_argument("--which", type=str, choices=["best", "final"], default="final")
    ap.add_argument("--show_keys", action="store_true", help="Print all state_dict keys.")
    args = ap.parse_args()

    # Resolve checkpoint path
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        if not args.save_dir:
            raise SystemExit("Provide --ckpt_path or --save_dir.")
        ckpt_path = os.path.join(args.save_dir, f"{args.which}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    _print_header(f"Checkpoint inspection: {ckpt_path}")
    print(f"Step: {ckpt.get('step', 'Unknown')}")
    print(f"Best eval NLL (stored): {ckpt.get('best_eval_nll', 'Unknown')}")

    state = ckpt.get("model_state", ckpt.get("state_dict", {}))
    cfg = ckpt.get("cfg", {}) or {}
    ckpt_args = ckpt.get("args", {}) or {}

    # Prefer cfg/args for mode detection
    mode = cfg.get("mode", ckpt_args.get("mode", "unknown"))
    kernel = cfg.get("kernel", ckpt_args.get("kernel", "unknown"))
    normalize = cfg.get("normalize", ckpt_args.get("normalize", None))
    norm_decay = cfg.get("norm_decay", ckpt_args.get("normdecay", None))
    L = cfg.get("L", ckpt_args.get("L", None))
    d = cfg.get("d", ckpt_args.get("d", None))
    C = ckpt_args.get("C", None)

    print("\n--- Config (from ckpt metadata) ---")
    print(f"mode      : {mode}")
    print(f"kernel    : {kernel}")
    print(f"normalize : {normalize}")
    print(f"norm_decay: {norm_decay}")
    print(f"d         : {d}")
    print(f"L         : {L}")
    print(f"C (bins)  : {C}")

    # learn_convert_scales is an ARG, not in cfg
    learn_convert_scales = ckpt_args.get("learn_convert_scales", None)
    if learn_convert_scales is not None:
        print(f"learn_convert_scales (trainable?) : {int(learn_convert_scales)}")

    if args.show_keys:
        _print_header("State dict keys")
        for k in sorted(state.keys()):
            print(k)

    # -------------------------------------------------------------------------
    # Shared / always-present learned parameters in your current model.py
    # -------------------------------------------------------------------------
    _print_header("Shared / core learned parameters")

    # Analytic conversion scales
    if "convert.s1" in state and "convert.s2" in state:
        s1 = state["convert.s1"].item()
        s2 = state["convert.s2"].item()
        tag = "TRAINABLE" if learn_convert_scales else "FROZEN/NOT-TRAINED"
        if learn_convert_scales is None:
            tag = "unknown-trainability"
        print(f"Analytic conversion scales: s1={s1:.6f}, s2={s2:.6f}   [{tag}]")
    else:
        print("Analytic conversion scales: NOT FOUND (unexpected for current model.py)")

    # Sigma
    if "log_sigma" in state:
        sigma = state["log_sigma"].exp().item()
        print(f"sigma = exp(log_sigma) = {sigma:.8f} (meaningless in learnable)")
    else:
        print("log_sigma: not found")

    # Eta(s)
    if "raw_eta" in state:
        raw_eta = state["raw_eta"]
        etas = torch.sigmoid(raw_eta)
        if etas.numel() == 1:
            print(f"eta = sigmoid(raw_eta) = {etas.item():.8f}")
        else:
            print(f"eta vector: {_tinfo(etas)}")
    else:
        print("raw_eta: not found")

    # Feature scale(s)
    if "log_feature_scale" in state:
        fs = state["log_feature_scale"].exp().detach().cpu()
        print(f"feature_scale (shared): {fs.numpy()}")
    if "log_feature_scales" in state:
        fs = state["log_feature_scales"].exp().detach().cpu()  # (L, d)
        print(f"feature_scales (per-layer): {tuple(fs.shape)} (showing first 10 rows)")
        # print(fs[:10].numpy())
        print(fs.numpy())

    # -------------------------------------------------------------------------
    # THEORY MODE
    # -------------------------------------------------------------------------
    if mode == "theory":
        _print_header("THEORY-mode extras (should be minimal)")
        # Nothing else beyond the shared items in current implementation
        # (theory still has raw_eta/log_sigma/log_feature_scale)
        print("No per-layer V/S parameters in theory mode (expected).")

    # -------------------------------------------------------------------------
    # LEARNABLE MODE
    # -------------------------------------------------------------------------
    elif mode == "learnable":
        _print_header("LEARNABLE-mode parameters")

        # v0 init
        if "v0_kx_label" in state:
            print(f"v0_kx_label = {state['v0_kx_label'].item():.8f}")

        # readout weights
        if "w_readout.w_mu" in state:
            w_mu = state.get("w_readout.w_mu", torch.tensor(float("nan"))).item()
            w_sigma2 = state.get("w_readout.w_sigma2", torch.tensor(float("nan"))).item()
            w_k = state.get("w_readout.w_k", torch.tensor(float("nan"))).item()
            w_tau = state.get("w_readout.w_tau", torch.tensor(float("nan"))).item()
            print("Readout weights:")
            print(f"  w_mu     = {w_mu:.8f}")
            print(f"  w_sigma2 = {w_sigma2:.8f}")
            print(f"  w_k      = {w_k:.8f}")
            print(f"  w_tau    = {w_tau:.8f}")
        else:
            print("Readout weights not found.")

        # Layer recurrence parameters
        layer_rows = []
        max_layer = -1
        for k in state.keys():
            if k.startswith("layers."):
                parts = k.split(".")
                if len(parts) >= 3 and parts[1].isdigit():
                    max_layer = max(max_layer, int(parts[1]))

        if max_layer >= 0:
            params = ["v_mu_label", "v_tau_kx", "v_mu_mu", "v_tau_tau", "s_mu_mu", "s_tau_tau"]
            for l in range(max_layer + 1):
                row = {"layer": l}
                for p in params:
                    key = f"layers.{l}.{p}"
                    row[p] = state[key].item() if key in state else float("nan")
                layer_rows.append(row)

            # Pretty print: use pandas if available, else manual
            try:
                import pandas as pd
                df = pd.DataFrame(layer_rows)
                print("\nLayer recurrence parameters:")
                print(df.head(10).to_string(index=False, float_format=lambda x: f"{x: .6f}"))
                if len(df) > 10:
                    print(f"... ({len(df)} total layers)")
            except Exception:
                print("\nLayer recurrence parameters (manual, first 10 layers):")
                hdr = ["layer"] + params
                print(" ".join([f"{h:>12}" for h in hdr]))
                for row in layer_rows[:10]:
                    line = f"{row['layer']:12d}" + "".join([f"{row[p]:12.6f}" for p in params])
                    print(line)
                if len(layer_rows) > 10:
                    print(f"... ({len(layer_rows)} total layers)")
        else:
            print("No layer parameters found (unexpected for learnable mode).")

    else:
        _print_header("WARNING: unknown mode")
        print("Could not determine mode from ckpt['cfg']/ckpt['args'].")
        print("Consider passing a checkpoint from a recent run, or inspect keys with --show_keys.")

if __name__ == "__main__":
    main()
