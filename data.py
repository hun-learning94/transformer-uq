"""
Codes for Simulation Data Generation for Transformer-UQ

This file contains:
    1) Task samplers:
        - Bayesian Linear Regression (BLR) with non-isotropic (e.g. diagonal) prior, i.e., k(x,x') = x^T Σ x'
        - Radial Basis Function (RBF-GP) regression, fixed kernel length-scale/variance
        We generate *prompts* Z^{(0)} = (X,Y,x_q) and an oracle PPD
            y_q | X,Y,x_q  ~  Normal(mu, tau)
        Then (when bin edges are provided) we define the *truncated* conditional distribution on [a,b):
            π_{[a,b)}(dy | Z^{(0)})  ∝  1_{a <= y < b} Normal(mu,tau)(dy)
        Training/evaluation label is produced by sampling y_q ~ π_{[a,b)}(·|Z^{(0)}) and binning it.
        This makes cross-entropy over bins a correct MC estimator of the truncated population NLL:
            E_{Z^{(0)}} E_{Y~π_{[a,b)}}[-log q(Y|Z^{(0)})]
        where q is the model's categorical distribution over bins.
    2) Oracle PPD mean (mu), variance (tau) given by Gaussian conditionals (y_{n+1} | X, Y, x_{n+1})
    3) Oracle PPD bin probabilities q*(c|Z) (discretized true PPD)
        Not needed for training, but to evaluate "did it approximate the true PPD?"

Tasks implemented:
  - BLR: Bayesian linear regression with diagonal prior variance (anisotropic linear kernel)
  - RBF-GP: GP regression with fixed RBF kernel (alpha, ell) and observation noise sigma

Shapes (batch-first):
    X:   (B, n, d)
    y:   (B, n)
    xq:  (B, d)  query input x_{n+1}
    yq:  (B, )   query label y_{n+1}
    mu:  (B, )   oracle PPD mean
    tau: (B, )  oracle PPD var
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import torch

# -------------------------------------------------------------------------------------------------
# Binning + Normal CDF/PPF + truncated sampling
# -------------------------------------------------------------------------------------------------
def make_bin_edges(a: float, b: float, num_bins: int, device=None, dtype=None) -> torch.Tensor:
    """Uniform bin edges [a=u0 < ... < u_C=b] defining C bins."""
    return torch.linspace(a, b, num_bins + 1, device=device, dtype=dtype)

def gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard Normal CDF Φ(x), element-wise."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gaussian_ppf(u: torch.Tensor) -> torch.Tensor:
    """Standard Normal inverse CDF Φ^{-1}(u), element-wise."""
    # Φ^{-1}(u) = sqrt(2) * erfinv(2u-1)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

def truncated_normal_sample(mu: torch.Tensor, tau: torch.Tensor, a: float, b: float, eps: float = 1e-12) -> torch.Tensor:
    """Sample Y ~ Normal(mu, tau) conditioned on Y in [a,b).
    We use inverse-CDF sampling:
      u ~ Uniform(Φ((a-mu)/sd), Φ((b-mu)/sd)), y = mu + sd * Φ^{-1}(u)
    """
    sd = torch.sqrt(torch.clamp(tau, min=eps))
    Fa = gaussian_cdf((a - mu) / sd)
    Fb = gaussian_cdf((b - mu) / sd)
    width = torch.clamp(Fb - Fa, min=eps)
    u = Fa + width * torch.rand_like(mu)
    u = torch.clamp(u, min=eps, max=1.0 - eps)
    return mu + sd * gaussian_ppf(u)

def gaussian_bin_probs(mu: torch.Tensor, tau: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Bin probs under the *truncated* normal on [a,b), where a=edges[0], b=edges[-1].
    Returns probs[b,c] = P(u_c <= Y < u_{c+1} | a <= Y < b), sum_c probs[b,c] = 1.
    """
    eps = torch.finfo(mu.dtype).tiny
    edges = edges.to(device=mu.device, dtype=mu.dtype)
    sd = torch.sqrt(torch.clamp(tau, min=eps))
    z = (edges[None, :] - mu[:, None]) / sd[:, None]  # (B, C+1)
    Phi = gaussian_cdf(z)
    masses = torch.clamp(Phi[:, 1:] - Phi[:, :-1], min=0.0)     # (B,C)
    denom = torch.clamp(Phi[:, -1] - Phi[:, 0], min=eps)        # (B,)
    probs = masses / denom[:, None]
    probs = torch.clamp(probs, min=eps)                         # optional, after truncation
    probs = probs / probs.sum(dim=-1, keepdim=True)             # optional final renorm
    return probs

def bin_index(y: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Map y to bin index c in {0,...,C-1}, assuming y in [edges[0], edges[-1])."""
    i = torch.bucketize(y, edges, right=False)  # edges[i-1] <= y < edges[i]
    c = (i - 1).clamp(min=0, max=edges.numel() - 2)
    return c.to(torch.long)


# -------------------------------------------------------------------------------------------------
# Kernels and oracle posterior predictive moments
# -------------------------------------------------------------------------------------------------
def linear_kernel(X: torch.Tensor, X2: torch.Tensor, diag_var: torch.Tensor) -> torch.Tensor:
    """Linear (anisotropic) kernel: k(x,x') = x^T diag(diag_var) x'."""
    return (X * diag_var[None, None, :]) @ X2.transpose(-1, -2)

def RBF_kernel(X: torch.Tensor, X2: torch.Tensor, alpha: float, ell: float) -> torch.Tensor:
    """RBF kernel: k(x,x') = alpha^2 * exp(-||x-x'||^2 / (2 ell^2))."""
    x_norm2 = (X ** 2).sum(dim=-1, keepdim=True)   # (B,n,1)
    x2_norm2 = (X2 ** 2).sum(dim=-1, keepdim=True) # (B,m,1)
    cross = X @ X2.transpose(-1, -2)               # (B,n,m)
    dist2 = x_norm2 + x2_norm2.transpose(-1, -2) - 2.0 * cross
    dist2 = torch.clamp(dist2, min=0.0)
    return (alpha ** 2) * torch.exp(-0.5 * dist2 / (ell ** 2))

def gp_posterior_moments(
    K: torch.Tensor,     # (B,n,n)
    kx: torch.Tensor,    # (B,n,1)
    kxx: torch.Tensor,   # (B,1,1)
    Y: torch.Tensor,     # (B,n)
    sigma: float,
    jitter: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle posterior predictive mean/variance for y_q | X,Y,x_q under GP prior with obs noise sigma.
    mu = kx^T (K + sigma^2 I)^{-1} Y
    tau = (kxx - kx^T (K + sigma^2 I)^{-1} kx) + sigma^2
    """
    # print("K device:", K.device)
    B, n, _ = K.shape
    device, dtype = K.device, K.dtype
    eye = torch.eye(n, device=device, dtype=dtype)[None, :, :]

    A = K + (sigma ** 2 + jitter) * eye  # (B,n,n)
    L = torch.linalg.cholesky(A)         # (B,n,n)

    # Solve A^{-1} Y and A^{-1} kx via cholesky_solve
    Y_col = Y[:, :, None]  # (B,n,1)
    AinvY = torch.cholesky_solve(Y_col, L)  # (B,n,1)
    Ainvkx = torch.cholesky_solve(kx, L)    # (B,n,1)

    mu = (kx.transpose(-1, -2) @ AinvY).squeeze(-1).squeeze(-1)  # (B,)
    var_f = (kxx - (kx.transpose(-1, -2) @ Ainvkx)).squeeze(-1).squeeze(-1)  # (B,)
    tau = torch.clamp(var_f + (sigma ** 2), min=1e-12)
    return mu, tau

def gp_log_marginal_likelihood(
    K: torch.Tensor,     # (B,n,n)
    Y: torch.Tensor,     # (B,n)
    sigma: float,
    jitter: float = 0.0,
) -> torch.Tensor:
    """
    log p(Y | X, h) under GP prior with kernel matrix K and observation noise sigma.
    Returns shape (B,).
    """
    B, n, _ = K.shape
    device, dtype = K.device, K.dtype
    eye = torch.eye(n, device=device, dtype=dtype)[None, :, :]

    A = K + (sigma ** 2 + jitter) * eye
    L = torch.linalg.cholesky(A)

    Y_col = Y[:, :, None]  # (B,n,1)
    AinvY = torch.cholesky_solve(Y_col, L).squeeze(-1)  # (B,n)

    quad = (Y * AinvY).sum(dim=-1)  # (B,)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (B,)

    return -0.5 * (
        n * math.log(2.0 * math.pi) + logdet + quad
    )


def gaussian_bin_probs_untruncated(mu: torch.Tensor, tau: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Untruncated Gaussian bin probabilities:
      probs[b,c] = P(u_c <= Y < u_{c+1}) under N(mu, tau)
    Returns (B, C), rows sum to <= 1 if tails lie outside [a,b].
    """
    edges = edges.to(device=mu.device, dtype=mu.dtype)
    sd = torch.sqrt(torch.clamp(tau, min=eps))
    z = (edges[None, :] - mu[:, None]) / sd[:, None]  # (B, C+1)
    Phi = gaussian_cdf(z)
    masses = torch.clamp(Phi[:, 1:] - Phi[:, :-1], min=0.0)
    return masses


def truncated_mixture_sample_from_bin_probs(
    probs: torch.Tensor,   # (B, C), already normalized over bins
    edges: torch.Tensor,   # (C+1,)
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from the piecewise-constant categorical-over-bins target:
      1) sample bin c ~ probs
      2) sample uniformly within that bin
    Returns:
      y: sampled value in [a,b)
      c: sampled bin index
    """
    probs = torch.clamp(probs, min=eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    B, C = probs.shape
    device, dtype = probs.device, probs.dtype
    edges = edges.to(device=device, dtype=dtype)

    c = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
    left = edges[:-1][c]
    right = edges[1:][c]
    u = torch.rand(B, device=device, dtype=dtype)
    y = left + (right - left) * u
    return y, c

# -------------------------------------------------------------------------------------------------
# Feature samplers
# -------------------------------------------------------------------------------------------------
def normalize_rows(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize the last dimension of X to unit norm."""
    norms = torch.linalg.norm(X, dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=eps)
    return X / norms

def draw_X(
    B: int,
    n: int,
    d: int,
    device=None,
    dtype=None,
    dist: str = "normal",
    unit_norm: bool = False,
) -> torch.Tensor:
    """Draw covariates; optionally normalize each token to unit norm."""
    if dist == "normal":
        X = torch.randn(B, n, d, device=device, dtype=dtype) / math.sqrt(d)
    elif dist == "uniform":
        X = (2.0 * torch.rand(B, n, d, device=device, dtype=dtype) - 1.0) / math.sqrt(d)
    else:
        raise ValueError(f"Unknown dist='{dist}'")

    if unit_norm:
        X = normalize_rows(X)

    return X

def draw_xnew(
    B: int,
    d: int,
    device=None,
    dtype=None,
    dist: str = "normal",
    unit_norm: bool = False,
) -> torch.Tensor:
    if dist == "normal":
        xq = torch.randn(B, d, device=device, dtype=dtype) / math.sqrt(d)
    elif dist == "uniform":
        xq = (2.0 * torch.rand(B, d, device=device, dtype=dtype) - 1.0) / math.sqrt(d)
    else:
        raise ValueError(f"Unknown dist='{dist}'")

    if unit_norm:
        xq = normalize_rows(xq)

    return xq

# -------------------------------------------------------------------------------------------------
# Global hyperparameters (fixed across pretrain/test)
# -------------------------------------------------------------------------------------------------
def _default_diag_var(d: int, device=None, dtype=None) -> torch.Tensor:
    diag = torch.ones(d, device=device, dtype=dtype)
    a = d // 3
    b = (2 * d) // 3
    diag[:a] = 2.0
    diag[a:b] = 1.0
    diag[b:] = 0.4
    return diag

@dataclass
class BLR_hyprms:
    sigma: float = 0.2
    diag_var: Optional[torch.Tensor] = None  # (d,)

@dataclass
class RBF_hyprms:
    sigma: float = 0.2
    alpha: float = 1.0
    ell: float = 0.8
    jitter: float = 1e-6

@dataclass
class RBF_Mixture_hyprms:
    sigma_grid: Tuple[float, ...] = (0.1, 0.2, 0.4)
    ell_grid: Tuple[float, ...] = (0.4, 0.8, 1.6)
    alpha: float = 1.0
    jitter: float = 1e-6

# -------------------------------------------------------------------------------------------------
# Task samplers
# -------------------------------------------------------------------------------------------------
def draw_BLR_batch(
    B: int, n: int, d: int,
    hypers: BLR_hyprms,
    *,
    device=None, dtype=torch.float64,
    x_dist: str = "normal",
    unit_norm_x: bool = False,
    edges: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Draw BLR prompt and oracle posterior predictive.
    Generative:
      beta ~ N(0, diag_var)
      y_i = x_i^T beta + eps_i
      test label y_q is sampled from the *posterior predictive* N(mu,tau) given (X,Y,xq)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    diag_var = hypers.diag_var
    if diag_var is None:
        diag_var = _default_diag_var(d, device=device, dtype=dtype)
    else:
        diag_var = diag_var.to(device=device, dtype=dtype)
        assert diag_var.shape == (d,)
    sigma = float(hypers.sigma)

    X = draw_X(B, n, d, device=device, dtype=dtype, dist=x_dist, unit_norm=unit_norm_x)
    xq = draw_xnew(B, d, device=device, dtype=dtype, dist=x_dist, unit_norm=unit_norm_x)

    beta = torch.randn(B, d, device=device, dtype=dtype) * torch.sqrt(diag_var[None, :])
    Y = (X * beta[:, None, :]).sum(dim=-1) + sigma * torch.randn(B, n, device=device, dtype=dtype)

    # Oracle posterior predictive moments
    K = linear_kernel(X, X, diag_var)
    kx = linear_kernel(X, xq[:, None, :], diag_var)
    kxx = linear_kernel(xq[:, None, :], xq[:, None, :], diag_var)
    mu, tau = gp_posterior_moments(K=K, kx=kx, kxx=kxx, Y=Y, sigma=sigma, jitter=1e-10)

    # Sample test label from (possibly truncated) PPD
    yq_untrunc = mu + torch.sqrt(tau) * torch.randn(B, device=device, dtype=dtype)
    if edges is not None:
        edges = edges.to(device=device, dtype=dtype)
        a = float(edges[0].item())
        b = float(edges[-1].item())
        yq = truncated_normal_sample(mu, tau, a=a, b=b)
        q_true = gaussian_bin_probs(mu, tau, edges)
        c_true = bin_index(yq, edges)
    else:
        yq = yq_untrunc
        q_true = None
        c_true = None

    out: Dict[str, torch.Tensor] = {
        "X": X, "Y": Y, "xq": xq,
        "yq": yq,
        "yq_untrunc": yq_untrunc,
        "mu": mu, "tau": tau,
        "diag_var": diag_var,
    }
    if edges is not None:
        out["edges"] = edges
        out["q_true"] = q_true
        out["c_true"] = c_true
    return out


def draw_RBF_batch(
    B: int, n: int, d: int,
    hypers: RBF_hyprms,
    *,
    device=None, dtype=torch.float64,
    x_dist: str = "normal",
    unit_norm_x: bool = False,
    edges: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Draw RBF-GP prompt and oracle posterior predictive."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    alpha = float(hypers.alpha)
    ell = float(hypers.ell)
    jitter = float(hypers.jitter)
    sigma = float(hypers.sigma)

    X = draw_X(B, n, d, device=device, dtype=dtype, dist=x_dist, unit_norm=unit_norm_x)
    xq = draw_xnew(B, d, device=device, dtype=dtype, dist=x_dist, unit_norm=unit_norm_x)

    # GP prior sample f ~ N(0, K)
    K = RBF_kernel(X, X, alpha=alpha, ell=ell)
    K = 0.5 * (K + K.transpose(-1, -2))
    diag = K.diagonal(dim1=-2, dim2=-1)
    min_diag = diag.min().item()
    max_diag = diag.max().item()

    eye = torch.eye(n, device=device, dtype=dtype)[None, :, :]
    if torch.isnan(K).any():
        raise RuntimeError("NaNs in K before Cholesky")
    
    base_jitter = float(hypers.jitter)
    jitter_eff = base_jitter
    if min_diag <= 0:
        jitter_eff = base_jitter + (-min_diag + 1e-6)

    L, info = torch.linalg.cholesky_ex(K + jitter_eff * eye)
    if (info > 0).any():
        # You can escalate jitter or throw a clean error
        raise RuntimeError(f"Cholesky failed, info={info}")

    z = torch.randn(B, n, device=device, dtype=dtype)
    f = (L @ z[:, :, None]).squeeze(-1)  # (B,n)

    Y = f + sigma * torch.randn(B, n, device=device, dtype=dtype)

    # Oracle posterior predictive
    kx = RBF_kernel(X, xq[:, None, :], alpha=alpha, ell=ell)
    kxx = RBF_kernel(xq[:, None, :], xq[:, None, :], alpha=alpha, ell=ell)
    mu, tau = gp_posterior_moments(K=K, kx=kx, kxx=kxx, Y=Y, sigma=sigma, jitter=jitter)

    yq_untrunc = mu + torch.sqrt(tau) * torch.randn(B, device=device, dtype=dtype)
    if edges is not None:
        edges = edges.to(device=device, dtype=dtype)
        a = float(edges[0].item())
        b = float(edges[-1].item())
        yq = truncated_normal_sample(mu, tau, a=a, b=b)
        q_true = gaussian_bin_probs(mu, tau, edges)
        c_true = bin_index(yq, edges)
    else:
        yq = yq_untrunc
        q_true = None
        c_true = None

    out: Dict[str, torch.Tensor] = {
        "X": X, "Y": Y, "xq": xq,
        "yq": yq,
        "yq_untrunc": yq_untrunc,
        "mu": mu, "tau": tau,
        "alpha": torch.tensor(alpha, device=device, dtype=dtype),
        "ell": torch.tensor(ell, device=device, dtype=dtype),
    }
    if edges is not None:
        out["edges"] = edges
        out["q_true"] = q_true
        out["c_true"] = c_true
    return out

def draw_RBF_mixture_batch(
    B: int, n: int, d: int,
    hypers: RBF_Mixture_hyprms,
    *,
    device=None, dtype=torch.float64,
    x_dist: str = "normal",
    unit_norm_x: bool = False,
    edges: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Draw prompt from a hierarchical RBF-GP prior with discrete latent hyperparameters:
      ell   ~ Uniform(hypers.ell_grid)
      sigma ~ Uniform(hypers.sigma_grid)

    Oracle PPD is the exact finite mixture over all (ell, sigma) pairs:
      p(y_q | X,Y,x_q) = sum_h p(h | X,Y) N(mu_h, tau_h)

    If edges are provided, q_true is the discretized truncated oracle PPD on [a,b).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    alpha = float(hypers.alpha)
    jitter = float(hypers.jitter)
    ell_grid = tuple(float(x) for x in hypers.ell_grid)
    sigma_grid = tuple(float(x) for x in hypers.sigma_grid)

    ell_vals = torch.tensor(ell_grid, device=device, dtype=dtype)        # (E,)
    sigma_vals = torch.tensor(sigma_grid, device=device, dtype=dtype)    # (S,)

    ell_mesh, sigma_mesh = torch.meshgrid(ell_vals, sigma_vals, indexing="ij")
    ell_all = ell_mesh.reshape(-1)       # (H,)
    sigma_all = sigma_mesh.reshape(-1)   # (H,)
    H_size = ell_all.numel()

    X = draw_X(B, n, d, device=device, dtype=dtype, dist=x_dist, unit_norm=unit_norm_x)
    xq = draw_xnew(B, d, device=device, dtype=dtype, dist=x_dist, unit_norm=unit_norm_x)

    # ------------------------------------------------------------------
    # Sample latent hyperparameter pair uniformly for data generation
    # ------------------------------------------------------------------
    ell_vals = torch.tensor(ell_grid, device=device, dtype=dtype)        # (E,)
    sigma_vals = torch.tensor(sigma_grid, device=device, dtype=dtype)    # (S,)

    num_ell = len(ell_grid)
    num_sigma = len(sigma_grid)

    h_idx = torch.randint(low=0, high=H_size, size=(B,), device=device)

    ell_idx = h_idx // num_sigma
    sigma_idx = h_idx % num_sigma

    ell_true = ell_vals[ell_idx]         # (B,)
    sigma_true = sigma_vals[sigma_idx]   # (B,)

    # Build kernel K_true batchwise using per-example ell_true
    x_norm2 = (X ** 2).sum(dim=-1, keepdim=True)                 # (B,n,1)
    cross = X @ X.transpose(-1, -2)                              # (B,n,n)
    dist2 = x_norm2 + x_norm2.transpose(-1, -2) - 2.0 * cross
    dist2 = torch.clamp(dist2, min=0.0)

    K_true = (alpha ** 2) * torch.exp(-0.5 * dist2 / (ell_true[:, None, None] ** 2))
    K_true = 0.5 * (K_true + K_true.transpose(-1, -2))

    eye = torch.eye(n, device=device, dtype=dtype)[None, :, :]
    A_true = K_true + (jitter + 1e-6) * eye
    L_true, info = torch.linalg.cholesky_ex(A_true)
    if (info > 0).any():
        raise RuntimeError(f"Cholesky failed in draw_RBF_mixture_batch, info={info}")

    z = torch.randn(B, n, device=device, dtype=dtype)
    f = (L_true @ z[:, :, None]).squeeze(-1)
    Y = f + sigma_true[:, None] * torch.randn(B, n, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Oracle mixture posterior predictive
    # ------------------------------------------------------------------
    logw_list = []
    mu_list = []
    tau_list = []
    q_comp_list = []

    for j in range(H_size):
        ell = float(ell_all[j].item())
        sigma = float(sigma_all[j].item())
        K = RBF_kernel(X, X, alpha=alpha, ell=ell)
        K = 0.5 * (K + K.transpose(-1, -2))

        kx = RBF_kernel(X, xq[:, None, :], alpha=alpha, ell=ell)
        kxx = RBF_kernel(xq[:, None, :], xq[:, None, :], alpha=alpha, ell=ell)

        mu_h, tau_h = gp_posterior_moments(K=K, kx=kx, kxx=kxx, Y=Y, sigma=sigma, jitter=jitter)
        loglik_h = gp_log_marginal_likelihood(K=K, Y=Y, sigma=sigma, jitter=jitter)

        # Uniform prior over H, so add constant log prior
        logprior_h = -math.log(H_size)
        logw_h = loglik_h + logprior_h  # (B,)

        logw_list.append(logw_h)
        mu_list.append(mu_h)
        tau_list.append(tau_h)

        if edges is not None:
            # untruncated Gaussian masses first; truncation is applied after mixture
            q_comp_h = gaussian_bin_probs_untruncated(mu_h, tau_h, edges)
            q_comp_list.append(q_comp_h)

    logw = torch.stack(logw_list, dim=-1)     # (B, H)
    w = torch.softmax(logw, dim=-1)           # posterior weights p(h | X,Y)

    mu_all = torch.stack(mu_list, dim=-1)     # (B, H)
    tau_all = torch.stack(tau_list, dim=-1)   # (B, H)

    # Mixture moments
    mu_mix = (w * mu_all).sum(dim=-1)  # (B,)
    second_moment = (w * (tau_all + mu_all ** 2)).sum(dim=-1)
    tau_mix = torch.clamp(second_moment - mu_mix ** 2, min=1e-12)

    # ------------------------------------------------------------------
    # Build oracle q_true and sampled yq
    # ------------------------------------------------------------------
    if edges is not None:
        edges = edges.to(device=device, dtype=dtype)

        q_comp = torch.stack(q_comp_list, dim=-1)      # (B, C, H)
        q_true_untrunc = (q_comp * w[:, None, :]).sum(dim=-1)   # (B, C)

        # truncate to [a,b) by renormalizing over bins
        eps = torch.finfo(dtype).tiny
        q_true = torch.clamp(q_true_untrunc, min=0.0)
        q_true = q_true / torch.clamp(q_true.sum(dim=-1, keepdim=True), min=eps)

        yq, c_true = truncated_mixture_sample_from_bin_probs(q_true, edges)
        yq_untrunc = yq.clone()   # here sampled within [a,b), consistent with training target
    else:
        # sample from Gaussian mixture exactly:
        comp_idx = torch.multinomial(w, num_samples=1).squeeze(-1)  # (B,)
        mu_sel = mu_all.gather(1, comp_idx[:, None]).squeeze(1)
        tau_sel = tau_all.gather(1, comp_idx[:, None]).squeeze(1)
        yq_untrunc = mu_sel + torch.sqrt(tau_sel) * torch.randn(B, device=device, dtype=dtype)
        yq = yq_untrunc
        q_true = None
        c_true = None

    out: Dict[str, torch.Tensor] = {
        "X": X,
        "Y": Y,
        "xq": xq,
        "yq": yq,
        "yq_untrunc": yq_untrunc,
        "mu": mu_mix,
        "tau": tau_mix,
        "q_true": q_true if edges is not None else None,
        "c_true": c_true if edges is not None else None,
        "edges": edges if edges is not None else None,
        "ell_true": ell_true,
        "sigma_true": sigma_true,
        "post_w": w,              # (B, H)
        "mu_components": mu_all,  # (B, H)
        "tau_components": tau_all # (B, H)
    }
    return out

# -------------------------------------------------------------------------------------------------
# New Kernel: Anisotropic RBF (ARD)
# -------------------------------------------------------------------------------------------------
def ARD_RBF_kernel(X: torch.Tensor, X2: torch.Tensor, alpha: float, ell: torch.Tensor) -> torch.Tensor:
    """
    ARD RBF kernel: k(x,x') = alpha^2 * exp(-0.5 * sum_k ((x_k - x'_k)/ell_k)^2)
    
    Args:
        X: (B, n, d)
        X2: (B, m, d)
        alpha: Amplitude scalar
        ell: Length-scale vector (d,)
    """
    # Scale inputs by length-scales: x_k / ell_k
    # We rely on broadcasting: (B, n, d) / (d,) -> (B, n, d)
    X_scaled = X / ell
    X2_scaled = X2 / ell
    
    # Compute squared Euclidean distance on the scaled features
    x_norm2 = (X_scaled ** 2).sum(dim=-1, keepdim=True)    # (B, n, 1)
    x2_norm2 = (X2_scaled ** 2).sum(dim=-1, keepdim=True)  # (B, m, 1)
    cross = X_scaled @ X2_scaled.transpose(-1, -2)         # (B, n, m)
    
    dist2 = x_norm2 + x2_norm2.transpose(-1, -2) - 2.0 * cross
    dist2 = torch.clamp(dist2, min=0.0)
    
    return (alpha ** 2) * torch.exp(-0.5 * dist2)

# -------------------------------------------------------------------------------------------------
# New Dataclass & Sampler
# -------------------------------------------------------------------------------------------------
@dataclass
class ARD_RBF_hyprms:
    sigma: float = 0.61            # Derived from sqrt(0.373)
    alpha: float = 0.78            # Derived from 0.782
    ell: Optional[torch.Tensor] = None # Expected shape (d,), e.g. [0.146, 0.252]
    jitter: float = 1e-6

def draw_ARD_RBF_batch(
    B: int, n: int, d: int,
    hypers: ARD_RBF_hyprms,
    *,
    device=None, dtype=torch.float64,
    x_dist: str = "normal",
    edges: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Draw Anisotropic RBF-GP prompt and oracle posterior predictive.
    Uses vector length-scales 'ell' for each dimension.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    alpha = float(hypers.alpha)
    sigma = float(hypers.sigma)
    jitter = float(hypers.jitter)
    
    # Validate and move ell to device
    if hypers.ell is None:
        # Default fallback if not provided: isotropic 1.0
        ell = torch.ones(d, device=device, dtype=dtype)
    else:
        ell = hypers.ell.to(device=device, dtype=dtype)
        if ell.shape != (d,):
            raise ValueError(f"ARD length-scale 'ell' must be shape ({d},), got {ell.shape}")

    X = draw_X(B, n, d, device=device, dtype=dtype, dist=x_dist)
    xq = draw_xnew(B, d, device=device, dtype=dtype, dist=x_dist)

    # 1. GP prior sample f ~ N(0, K_ARD)
    K = ARD_RBF_kernel(X, X, alpha=alpha, ell=ell)
    
    # Symmetrize and stabilize
    K = 0.5 * (K + K.transpose(-1, -2))
    eye = torch.eye(n, device=device, dtype=dtype)[None, :, :]
    
    # Robust Cholesky for sampling
    L, info = torch.linalg.cholesky_ex(K + (jitter + 1e-6) * eye)
    if (info > 0).any():
        raise RuntimeError(f"Cholesky failed in draw_ARD_RBF_batch, info={info}")

    z = torch.randn(B, n, device=device, dtype=dtype)
    f = (L @ z[:, :, None]).squeeze(-1)

    # 2. Add observation noise
    Y = f + sigma * torch.randn(B, n, device=device, dtype=dtype)

    # 3. Oracle posterior predictive moments
    kx = ARD_RBF_kernel(X, xq[:, None, :], alpha=alpha, ell=ell)
    kxx = ARD_RBF_kernel(xq[:, None, :], xq[:, None, :], alpha=alpha, ell=ell)
    
    # Note: gp_posterior_moments is generic, it just needs the kernel matrices
    mu, tau = gp_posterior_moments(K=K, kx=kx, kxx=kxx, Y=Y, sigma=sigma, jitter=jitter)

    # 4. Sample test label
    yq_untrunc = mu + torch.sqrt(tau) * torch.randn(B, device=device, dtype=dtype)
    
    if edges is not None:
        edges = edges.to(device=device, dtype=dtype)
        a = float(edges[0].item())
        b = float(edges[-1].item())
        yq = truncated_normal_sample(mu, tau, a=a, b=b)
        q_true = gaussian_bin_probs(mu, tau, edges)
        c_true = bin_index(yq, edges)
    else:
        yq = yq_untrunc
        q_true = None
        c_true = None

    out: Dict[str, torch.Tensor] = {
        "X": X, "Y": Y, "xq": xq,
        "yq": yq,
        "yq_untrunc": yq_untrunc,
        "mu": mu, "tau": tau,
        "alpha": torch.tensor(alpha, device=device, dtype=dtype),
        "ell": ell,  # Return the vector ell
    }
    if edges is not None:
        out["edges"] = edges
        out["q_true"] = q_true
        out["c_true"] = c_true
    return out









