"""
Transform-UQ architecture:
    Z0 -> Attention -> MLP -> Linear readout
    Returns: logit

Input format:
  Z0 has shape (B, D=d+4, T=n+1) where *columns are tokens*, *rows are channels*
  Z^{(0)} = [ X   xq   :d   feature rows, x_i (1<=i<=n+1)
              Y   1    d    label row, y_i (1<=i<=n) & 1 (for 1st attention)
              0   0    d+1  kernel row, k(xq, x_i) (1<=i<=n+1) (filled by 1st attention)
              0   0    d+2  mu scratch space
              0   0 ]  d+3  tau scratch space
  First attention populates k_x row using only the query token as the "value source".
  Subsequent layers implement the recursions for mean and variance using kernel-weighted sums.

- Attention is "kernel attention":
    A_{ij} = kappa(K z_i, Q z_j) * mask_{ij}
    Attn(Z) = (V Z) @ A
  where @ is matrix multiplication over token dimension.
  optionally, column normalize A (for rbf, softmax, not for linear)

Optional weight decay: 
    normdecay = 1, divide skip connection SZ by A.sum(dim=1, keepdim=True)
    to implement the ideal preconditioned iteration
    (only if rbf + learnable + softmax or rbf + normalize = 1)

Modes:
  - "theory": Learned eta/sigma, but matrices V and S are strictly constrained by the 
              theoretical recursion formulas (V ~ eta, S ~ -eta*sigma^2).
              Feature scales K/Q are shared across layers.
  - "learnable": V, S, W sparsity is fixed, but non-zero entries are independent learnable parameters per layer.
                 Feature scales K/Q are independent per layer (still diagonal).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Sequence
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def midpoints_from_edges(edges: torch.Tensor) -> torch.Tensor:
    return 0.5 * (edges[:-1] + edges[1:])

def fixed_U_from_edges(edges:torch.Tensor)->torch.Tensor:
    m = midpoints_from_edges(edges) 
    U = torch.stack([m, m**2], dim=-1) 
    return U

def linear_kernel(x: torch.Tensor, x2:torch.Tensor)->torch.Tensor:
    return torch.bmm(x, x2.transpose(1,2))

def rbf_kernel(x:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
    x_norm2 = (x ** 2).sum(dim=-1, keepdim=True)
    x2_norm2 = (x2 ** 2).sum(dim=-1, keepdim=True)
    cross = torch.bmm(x, x2.transpose(1, 2))
    dist2 = x_norm2 + x2_norm2.transpose(1, 2) - 2.0 * cross
    dist2 = torch.clamp(dist2, min=0.0)
    return torch.exp(-0.5 * dist2)

class KernelAttention(nn.Module):
    """
    Kernel Attention Layer
    """
    def __init__(
        self,
        d: int,
        kernel: str,
        normalize: bool = False,
        attn_clip: Optional[float] = None,
    ):
        super().__init__()
        assert kernel in ("linear", "rbf", "softmax")
        self.d = d
        self.kernel = kernel
        self.normalize = normalize
        self.attn_clip = attn_clip

    def forward(
        self, 
        Z: torch.Tensor, 
        V: torch.Tensor, 
        Kdiag: torch.Tensor, 
        Qdiag: torch.Tensor, 
        mask: torch.Tensor 
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        returns: (Output, Normalizer)
        Output: (B, D, T)
        Normalizer: (B, 1, T) or None (if not normalized)
        """
        B, D, T = Z.shape
        
        # Values
        VZ = torch.einsum("ij, bjt -> bit", V, Z) 
        
        # Keys/Queries
        Z_X = Z[:, :self.d, :].transpose(1,2) 
        KZ_X = Z_X * Kdiag[None, None, :]
        QZ_X = Z_X * Qdiag[None, None, :]
        
        # softmax
        if self.kernel == "softmax":
            # 1. Compute scores using dot product (linear kernel)
            scores = linear_kernel(KZ_X, QZ_X)
            # 2. Clip logits (optional)
            if self.attn_clip is not None:
                scores = torch.clamp(scores, -self.attn_clip, self.attn_clip)
            
            # 3. Masking before softmax (fill with -inf)
            scores = scores.masked_fill(~mask, -1e9)

            # 4. Apply Softmax over Source/Key dimension (dim=1)
            A = torch.softmax(scores, dim=1)

            # 5. Return weighted sum
            normalizer = None
            if self.normalize:
                # logZ = log sum_i exp(score_iq)
                logZ = torch.logsumexp(scores, dim=1, keepdim=True)  # (B, 1, T)
                # If a query has no valid keys (can happen with padding), avoid inf scaling
                has_any = mask.any(dim=1, keepdim=True)              # (B, 1, T)
                logZ = torch.where(has_any, logZ, torch.zeros_like(logZ))
                normalizer = logZ

            return torch.bmm(VZ, A), normalizer

        else:
            # Original RBF/Linear Logic
            if self.kernel == "linear":
                A = linear_kernel(KZ_X, QZ_X)
                if self.attn_clip is not None:
                    A = torch.clamp(A, -self.attn_clip, self.attn_clip)
            elif self.kernel == "rbf":
                A = rbf_kernel(KZ_X, QZ_X)
            
            # Apply mask (zeroing out)
            A = A * mask.to(dtype=A.dtype) 

            # --- MEMORY OPTIMIZATION: In-place Normalization ---
            normalizer = None
            if self.normalize:
                normalizer = A.sum(dim=1, keepdim=True) # Calculate sum (s_x), (B, 1, T)
                # Use in-place division (div_) to avoid allocating a new tensor for A
                A.div_(normalizer + 1e-6)

            return torch.bmm(VZ, A), normalizer

class AnalyticConversion(nn.Module):
    def __init__(self, learn_scales:bool = True):
        super().__init__()
        if learn_scales:
            self.s1 = nn.Parameter(torch.tensor(1.0))
            self.s2 = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("s1", torch.tensor(1.0))
            self.register_buffer("s2", torch.tensor(1.0))
    
    def forward(self, mu:torch.Tensor, tau:torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(tau, min=1e-8)
        theta1 = self.s1 * (mu/tau)
        theta2 = self.s2 * (-0.5/tau)
        return torch.stack([theta1, theta2], dim=-1)

# ---------------------------------------------------------------------------------------------------------------
# Layer Modules for "Learnable" Mode
# ---------------------------------------------------------------------------------------------------------------
class LearnableLayerParams(nn.Module):
    def __init__(self, eta_init: float, sigma_init: float):
        super().__init__()
        eta = eta_init
        sigma2 = sigma_init ** 2
        self.v_mu_label = nn.Parameter(torch.tensor(eta))
        self.v_tau_kx   = nn.Parameter(torch.tensor(eta))
        self.v_mu_mu    = nn.Parameter(torch.tensor(-eta))
        self.v_tau_tau  = nn.Parameter(torch.tensor(-eta))
        self.s_mu_mu    = nn.Parameter(torch.tensor(-eta * sigma2))
        self.s_tau_tau  = nn.Parameter(torch.tensor(-eta * sigma2))

class LearnableReadoutParams(nn.Module):
    def __init__(self, sigma_init: float):
        super().__init__()
        sigma2 = sigma_init ** 2
        self.w_mu = nn.Parameter(torch.tensor(1.0))
        self.w_sigma2 = nn.Parameter(torch.tensor(sigma2))
        self.w_k      = nn.Parameter(torch.tensor(1.0))
        self.w_tau    = nn.Parameter(torch.tensor(-1.0))

# ---------------------------------------------------------------------------------------------------------------
# Main Model: TransformerUQ
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    d: int
    L: int = 32
    mode: str = "learnable"
    normalize: bool = False
    norm_decay: bool = False
    kernel: str = "rbf"
    sigma_init: float = 0.2
    gradient_checkpointing: bool = True
    feature_scale_init: float = 1.0
    feature_scale_init_vec: Optional[Sequence[float]] = None
    eta_shared: bool = True
    eta_init: float = 0.1
    tau_floor: float = 1e-6
    state_clip: Optional[float] = 1e3
    logits_clip: Optional[float] = 1e3
    attn_clip: Optional[float] = 50.0
    log_feature_scale_min: float = -5.0
    log_feature_scale_max: float = 5.0
    log_sigma_min: float = -10.0
    log_sigma_max: float = 2.0


class TransformerUQ(nn.Module):
    def __init__(self, cfg: ModelConfig, edges: torch.Tensor):
        super().__init__()
        assert cfg.L >= 2
        
        self.cfg = cfg
        self.D = cfg.d + 4
        self.attn = KernelAttention(
            d=cfg.d,
            kernel=cfg.kernel,
            normalize=cfg.normalize,
            attn_clip=cfg.attn_clip,
        )

        U = fixed_U_from_edges(edges.detach().clone())
        self.register_buffer("U", U)
        self.convert = AnalyticConversion(learn_scales=True)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(cfg.sigma_init)))

        if cfg.feature_scale_init_vec is not None:
            init = torch.tensor(list(cfg.feature_scale_init_vec))
        else:
            init = torch.ones(cfg.d) * cfg.feature_scale_init

        if self.cfg.mode == "learnable":
            self.log_feature_scales = nn.Parameter(init.log().unsqueeze(0).repeat(cfg.L, 1))
        else:
            self.log_feature_scale = nn.Parameter(init.log())
        
        num_solver_layers = cfg.L - 1
        init_eta_raw = self._inv_sigmoid(cfg.eta_init)

        def _make_raw_eta():
            if cfg.eta_shared:
                return torch.tensor(init_eta_raw)
            else:
                return torch.ones(num_solver_layers) * init_eta_raw

        # Separate step sizes:
        # eta_k controls kernel residual term (V entries)
        # eta_r controls ridge term (S entries)
        self.raw_eta_k = nn.Parameter(_make_raw_eta(), requires_grad=True)
        self.raw_eta_r = nn.Parameter(_make_raw_eta(), requires_grad=True)


        if self.cfg.mode == "learnable":
            self.v0_kx_label = nn.Parameter(torch.tensor(1.0))
            self.layers = nn.ModuleList([
                LearnableLayerParams(eta_init=cfg.eta_init, sigma_init=cfg.sigma_init)
                for _ in range(num_solver_layers)
            ])
            self.w_readout = LearnableReadoutParams(sigma_init=cfg.sigma_init)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        if self.cfg.mode == "learnable":
            self.log_feature_scales.clamp_(self.cfg.log_feature_scale_min, self.cfg.log_feature_scale_max)
        else:
            self.log_feature_scale.clamp_(self.cfg.log_feature_scale_min, self.cfg.log_feature_scale_max)
        if isinstance(self.log_sigma, torch.Tensor):
            self.log_sigma.clamp_(self.cfg.log_sigma_min, self.cfg.log_sigma_max)

    @staticmethod
    def _inv_sigmoid(x:float) -> float: 
        x = min(max(x, 1e-6), 1-1e-6)
        return math.log(x / (1 - x))
    
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def make_mask_M0(self, T: int, device) -> torch.Tensor:
        mask = torch.zeros(1, T, T, device=device, dtype=torch.bool)
        mask[:, T-1, :] = True
        return mask

    def make_mask_M(self, T: int, device) -> torch.Tensor:
        mask = torch.zeros(1, T, T, device=device, dtype=torch.bool)
        mask[:, :T-1, :] = True
        return mask
    
    def get_V0(self, device, dtype) -> torch.Tensor:
        d = self.cfg.d
        D = d + 4
        V0 = torch.zeros(D, D, device=device, dtype=dtype)
        if self.cfg.mode == "learnable":
            V0[d + 1, d] = self.v0_kx_label
        else:
            V0[d + 1, d] = 1.0
        return V0

    def get_VS_layer(self, layer_idx: int, eta_k: torch.Tensor, eta_r: torch.Tensor, sigma2: torch.Tensor, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.cfg.d
        D = d + 4
        V = torch.zeros(D, D, device=device, dtype=dtype)
        S = torch.zeros(D, D, device=device, dtype=dtype)
        
        label = d; kx = d + 1; mu = d + 2; tau = d + 3

        if self.cfg.mode == "learnable":
            params = self.layers[layer_idx]
            V[mu, label] = params.v_mu_label
            V[tau, kx]   = params.v_tau_kx
            V[mu, mu]    = params.v_mu_mu
            V[tau, tau]  = params.v_tau_tau
            S[mu, mu]    = params.s_mu_mu
            S[tau, tau]  = params.s_tau_tau
        else:
            # kernel residual step size
            V[mu, label] = eta_k
            V[tau, kx]   = eta_k
            V[mu, mu]    = -eta_k
            V[tau, tau]  = -eta_k
            # ridge step size (separate)
            S[mu, mu]  = -eta_r * sigma2
            S[tau, tau]= -eta_r * sigma2
        return V, S

    def get_scales_for_layer(self, layer_idx: int) -> torch.Tensor:
        if self.cfg.mode == "learnable":
            return torch.exp(self.log_feature_scales[layer_idx])
        else:
            return torch.exp(self.log_feature_scale)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, xq: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            X: (B, N_max, d) - Padded input features
            Y: (B, N_max)    - Padded labels
            xq: (B, d)       - Query features
            padding_mask: (B, N_max+1, 1) - Optional mask. 1.0 for valid tokens, 0.0 for padding.
                                          Note: The +1 accounts for the query token appended at the end.
        """
        B, n, d = X.shape
        device = X.device
        dtype = X.dtype
        T = n + 1
        D = d + 4

        Z = torch.zeros(B, D, T, device=device, dtype=dtype)
        Z[:, :d, :n] = X.transpose(1, 2)
        Z[:, :d, n] = xq
        Z[:, d, :n] = Y
        Z[:, d, n] = 1.0

        scale0 = self.get_scales_for_layer(0)
        
        # Base masks from architecture
        mask0_algo = self.make_mask_M0(T, device=device) # (1, T, T)
        mask_algo = self.make_mask_M(T, device=device)   # (1, T, T)

        # Combine with padding mask if provided
        if padding_mask is not None:
            # padding_mask is (B, T, 1). 
            # We need to mask invalid Keys (columns) AND invalid Queries (rows).
            # valid_mask shape: (B, T, T)
            valid_mask = padding_mask & padding_mask.transpose(1, 2)
            mask0 = mask0_algo & valid_mask
            mask = mask_algo & valid_mask
        else:
            mask0 = mask0_algo
            mask = mask_algo

        V0 = self.get_V0(device=device, dtype=dtype)
        
        attn0, _ = self.attn(Z, V=V0, Kdiag=scale0, Qdiag=scale0, mask=mask0)
        Z = Z + attn0  

        sigma2 = (self.sigma() ** 2).to(dtype=dtype)
        
        if self.cfg.eta_shared:
            etas_k = torch.sigmoid(self.raw_eta_k).repeat(self.cfg.L - 1)
            etas_r = torch.sigmoid(self.raw_eta_r).repeat(self.cfg.L - 1)
        else:
            etas_k = torch.sigmoid(self.raw_eta_k)
            etas_r = torch.sigmoid(self.raw_eta_r)
        
        mu_row = d + 2
        tau_row = d + 3

        use_norm_decay = (
            self.cfg.norm_decay 
            and self.cfg.normalize 
            and self.cfg.kernel in {"rbf", "softmax"}
        )

        for t in range(self.cfg.L - 1):
            scale_t = self.get_scales_for_layer(t + 1)
            eta_k = etas_k[t]
            eta_r = etas_r[t]
            V, S = self.get_VS_layer(t, eta_k, eta_r, sigma2, device, dtype)
            
            if self.cfg.gradient_checkpointing and self.training:
                attn, normalizer = checkpoint(self.attn, Z, V, scale_t, scale_t, mask, use_reentrant=False)
            else:
                attn, normalizer = self.attn(Z, V=V, Kdiag=scale_t, Qdiag=scale_t, mask=mask)
            
            resid = torch.einsum("ij, bjt -> bit", S, Z)

            if use_norm_decay and normalizer is not None:
                # normalizer is logZ for softmax, or actual sum for rbf
                if self.cfg.kernel == "softmax":
                    resid = resid * torch.exp(-normalizer)     # == 1 / exp(logZ)
                else:
                    resid = resid * (1.0 / (normalizer + 1e-6))
            
            Z = Z + attn + resid
            
            if self.cfg.state_clip is not None:
                Z[:, mu_row, :].clamp_(-self.cfg.state_clip, self.cfg.state_clip)
                Z[:, tau_row, :].clamp_(-self.cfg.state_clip, self.cfg.state_clip)

        z_out = Z[:, :, n]

        if self.cfg.mode == "learnable":
            mu = self.w_readout.w_mu * z_out[:, mu_row]
            tau_val = (self.w_readout.w_sigma2 * z_out[:, d] + 
                       self.w_readout.w_k      * z_out[:, d+1] + 
                       self.w_readout.w_tau    * z_out[:, tau_row])
            tau = torch.clamp(tau_val, min=self.cfg.tau_floor)
        else:
            mu = z_out[:, mu_row]
            tau = torch.clamp(sigma2 + z_out[:, d + 1] - z_out[:, tau_row], min=self.cfg.tau_floor)

        theta = self.convert(mu, tau)
        logits = theta @ self.U.t()

        if self.cfg.logits_clip is not None:
            logits = torch.clamp(logits, -self.cfg.logits_clip, self.cfg.logits_clip)

        return logits, {"mu": mu, "tau": tau}