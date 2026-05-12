"""
python sacra_ebfit.py \
  --csv data/Sacramento.csv \
  --feature_cols x y \
  --y_col V \
  --n_fit 512 \
  --x_constant 0.3 \
  --y_constant 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


def normalize_X_with_constant(X: np.ndarray, constant: float = 1.0):
    """Center and standardize each column, then divide by a user constant.

    X_proc = (X - mean) / (std * constant)
    """
    if constant <= 0:
        raise ValueError("constant must be positive")
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    std = Xc.std(axis=0, ddof=1, keepdims=True) + 1e-12
    Xn = Xc / std / constant
    return Xn, {"mu": mu, "std": std, "constant": float(constant)}


def standardize_y_with_constant(y: np.ndarray, constant: float = 1.0):
    """Center and standardize y, then multiply by a user constant.

    y_proc = constant * (y - mean) / std

    With constant > 1, the standardized response has larger scale.
    With constant < 1, it has smaller scale.
    """
    if constant <= 0:
        raise ValueError("constant must be positive")
    y = np.asarray(y, dtype=float)
    mu = np.nanmean(y)
    sd = np.nanstd(y, ddof=1) + 1e-12
    yn = constant * (y - mu) / sd
    return yn, {"mu_y": float(mu), "sd_y": float(sd), "constant": float(constant)}


def fit_eb_hypers(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    y_col: str,
    x_constant: float,
    y_constant: float,
    n_fit: int,
    seed: int,
    n_restarts_optimizer: int,
):
    X_all = df[list(feature_cols)].to_numpy(dtype=float)
    y_all = df[y_col].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n_fit = min(int(n_fit), len(df))
    idx = rng.choice(len(df), size=n_fit, replace=False)

    X_fit_raw = X_all[idx]
    y_fit_raw = y_all[idx]

    X_fit_proc, xparams = normalize_X_with_constant(X_fit_raw, constant=x_constant)
    y_fit_proc, yparams = standardize_y_with_constant(y_fit_raw, constant=y_constant)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(len(feature_cols)), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=seed,
    )
    gp.fit(X_fit_proc, y_fit_proc)

    k = gp.kernel_
    result = {
        "sigma": float(np.sqrt(k.k2.noise_level)),
        "alpha": float(np.sqrt(k.k1.k1.constant_value)),
        "ell": np.asarray(k.k1.k2.length_scale, dtype=float).tolist(),
        "kernel": str(k),
        "feature_cols": list(feature_cols),
        "y_col": y_col,
        "n_fit": int(n_fit),
        "seed": int(seed),
        "x_constant": float(x_constant),
        "y_constant": float(y_constant),
        "fit_idx": idx.tolist(),
        "x_mean": xparams["mu"].reshape(-1).tolist(),
        "x_std": xparams["std"].reshape(-1).tolist(),
        "y_mean": yparams["mu_y"],
        "y_std": yparams["sd_y"],
    }
    return result


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Load a real dataset, standardize spatial coordinates and y with user-provided constants, "
            "fit an empirical-Bayes ARD RBF GP, and print the fitted hyperparameters."
        )
    )
    ap.add_argument("--csv", type=str, required=True, help="Input CSV file.")
    ap.add_argument("--feature_cols", type=str, nargs="+", required=True, help="Spatial/input columns.")
    ap.add_argument("--y_col", type=str, required=True, help="Response column.")
    ap.add_argument("--n_fit", type=int, default=200, help="Subset size used for EB fitting.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for subset selection and optimizer restart randomness.")
    ap.add_argument("--gp_restarts", type=int, default=5, help="Number of optimizer restarts in sklearn GP.")
    ap.add_argument(
        "--x_constant",
        type=float,
        default=1.0,
        help="After coordinatewise standardization of X, divide by this constant. Smaller values make fitted ell larger.",
    )
    ap.add_argument(
        "--y_constant",
        type=float,
        default=1.0,
        help="After standardizing y, multiply by this constant. This rescales fitted alpha and sigma.",
    )
    ap.add_argument("--save_json", type=str, default="", help="Optional path to save the fitted EB results as JSON.")
    return ap.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    result = fit_eb_hypers(
        df=df,
        feature_cols=args.feature_cols,
        y_col=args.y_col,
        x_constant=args.x_constant,
        y_constant=args.y_constant,
        n_fit=args.n_fit,
        seed=args.seed,
        n_restarts_optimizer=args.gp_restarts,
    )

    print("Fitted EB hyperparameters")
    print(f"  sigma : {result['sigma']:.10f}")
    print(f"  alpha : {result['alpha']:.10f}")
    print(f"  ell   : {result['ell']}")
    print(f"  kernel: {result['kernel']}")
    print(f"  x_constant: {result['x_constant']}")
    print(f"  y_constant: {result['y_constant']}")

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON to {save_path}")


if __name__ == "__main__":
    main()
