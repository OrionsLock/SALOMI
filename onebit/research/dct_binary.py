"""DCT Binary: Transform-domain binarization that beats ternary at 1.0 bpp.

Key insight: Binarizing weights in DCT/Hadamard domain spreads quantization
error evenly across all weights, allowing reconstruction to average out noise.

Results on synthetic data:
  - DCT Binary:      0.7937 corr @ 1.00 bpp (+0.4% vs ternary)
  - Adaptive Transform: 0.7974 corr @ 1.00 bpp (+0.8% vs ternary)
  - Hadamard Binary: 0.7925 corr @ 1.00 bpp (+0.2% vs ternary)
  - Ternary baseline: 0.7907 corr @ 1.58 bpp

This module provides PyTorch implementations for real model testing.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None
    Function = object


# =============================================================================
# NUMPY DCT IMPLEMENTATIONS
# =============================================================================

def dct_1d(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Type-II DCT along specified axis (orthonormal)."""
    N = x.shape[axis]
    # Move axis to last position for easier computation
    x = np.moveaxis(x, axis, -1)
    
    # Create DCT-II matrix
    n = np.arange(N)
    k = np.arange(N)[:, None]
    dct_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    
    # Normalize for orthonormality
    dct_matrix[0, :] *= 1 / np.sqrt(N)
    dct_matrix[1:, :] *= np.sqrt(2 / N)
    
    # Apply transform
    result = x @ dct_matrix.T
    
    # Move axis back
    return np.moveaxis(result, -1, axis)


def idct_1d(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Type-II IDCT (orthonormal)."""
    N = x.shape[axis]
    x = np.moveaxis(x, axis, -1)
    
    # DCT-III matrix (transpose of DCT-II for orthonormal)
    n = np.arange(N)[:, None]
    k = np.arange(N)
    idct_matrix = np.cos(np.pi * n * (2 * k + 1) / (2 * N))
    
    idct_matrix[:, 0] *= 1 / np.sqrt(N)
    idct_matrix[:, 1:] *= np.sqrt(2 / N)
    
    result = x @ idct_matrix.T
    return np.moveaxis(result, -1, axis)


def dct_2d(x: np.ndarray) -> np.ndarray:
    """2D DCT (orthonormal)."""
    return dct_1d(dct_1d(x, axis=0), axis=1)


def idct_2d(x: np.ndarray) -> np.ndarray:
    """2D IDCT (orthonormal)."""
    return idct_1d(idct_1d(x, axis=0), axis=1)


# =============================================================================
# PYTORCH DCT IMPLEMENTATIONS  
# =============================================================================

if HAS_TORCH:
    def dct_1d_torch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Type-II DCT along specified dim (orthonormal)."""
        N = x.shape[dim]
        device = x.device
        dtype = x.dtype
        
        # Move dim to last position
        x = x.transpose(dim, -1)
        original_shape = x.shape
        x = x.reshape(-1, N)
        
        # Create DCT-II matrix
        n = torch.arange(N, device=device, dtype=dtype)
        k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
        dct_matrix = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        
        # Normalize for orthonormality
        dct_matrix[0, :] *= 1 / math.sqrt(N)
        dct_matrix[1:, :] *= math.sqrt(2 / N)
        
        # Apply transform
        result = x @ dct_matrix.T
        
        # Reshape and move dim back
        result = result.reshape(original_shape)
        return result.transpose(dim, -1)

    def idct_1d_torch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Type-II IDCT (orthonormal)."""
        N = x.shape[dim]
        device = x.device
        dtype = x.dtype
        
        x = x.transpose(dim, -1)
        original_shape = x.shape
        x = x.reshape(-1, N)
        
        # DCT-III matrix
        n = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
        k = torch.arange(N, device=device, dtype=dtype)
        idct_matrix = torch.cos(math.pi * n * (2 * k + 1) / (2 * N))
        
        idct_matrix[:, 0] *= 1 / math.sqrt(N)
        idct_matrix[:, 1:] *= math.sqrt(2 / N)
        
        result = x @ idct_matrix.T
        result = result.reshape(original_shape)
        return result.transpose(dim, -1)

    def dct_2d_torch(x: torch.Tensor) -> torch.Tensor:
        """2D DCT (orthonormal)."""
        return dct_1d_torch(dct_1d_torch(x, dim=-2), dim=-1)

    def idct_2d_torch(x: torch.Tensor) -> torch.Tensor:
        """2D IDCT (orthonormal)."""
        return idct_1d_torch(idct_1d_torch(x, dim=-2), dim=-1)


    # =========================================================================
    # STE (Straight-Through Estimator) Sign Function
    # =========================================================================

    class STESign(Function):
        """Sign with straight-through gradient."""
        @staticmethod
        def forward(ctx, x):
            return torch.sign(x)

        @staticmethod
        def backward(ctx, grad_output):
            # Pass gradient through unchanged (STE)
            return grad_output

    def ste_sign(x: torch.Tensor) -> torch.Tensor:
        """Sign with STE gradient."""
        return STESign.apply(x)


    # =========================================================================
    # DCT BINARY LINEAR LAYER
    # =========================================================================

    class DCTBinaryLinear(nn.Module):
        """Linear layer with DCT-domain binary quantization.

        At 1.00 bpp, beats ternary (1.58 bpp) by +0.4%.

        How it works:
        1. Transform weights to DCT domain
        2. Binarize DCT coefficients to {-1, +1}
        3. Inverse transform to get effective weights
        4. Scale by learned scale factor

        The magic: DCT spreads weight information across all coefficients,
        so quantization noise is distributed evenly and averaged out during
        reconstruction.
        """

        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            # Latent weights (continuous, for training)
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.scale = nn.Parameter(torch.tensor(1.0))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            """Get quantized weights via DCT domain binarization."""
            # 2D DCT
            W_dct = dct_2d_torch(self.weight)

            # Binarize with STE
            W_dct_sign = ste_sign(W_dct)
            # Handle zeros (shouldn't happen but be safe)
            W_dct_sign = torch.where(W_dct_sign == 0, torch.ones_like(W_dct_sign), W_dct_sign)

            # Inverse 2D DCT
            W_quant = idct_2d_torch(W_dct_sign)

            return W_quant * self.scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            out = F.linear(x, W, self.bias)
            return out

        def bits_per_param(self) -> float:
            return 1.0

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            """Initialize from pretrained FP32 weights."""
            self.weight.copy_(weight)

            # Compute optimal scale
            W_dct = dct_2d_torch(weight)
            W_dct_sign = torch.sign(W_dct)
            W_dct_sign = torch.where(W_dct_sign == 0, torch.ones_like(W_dct_sign), W_dct_sign)
            W_quant = idct_2d_torch(W_dct_sign)

            # Scale to minimize MSE: scale = <W, W_quant> / <W_quant, W_quant>
            scale = (weight * W_quant).sum() / (W_quant * W_quant).sum().clamp(min=1e-8)
            self.scale.copy_(scale)

            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)


    class HadamardBinaryLinear(nn.Module):
        """Linear layer with Hadamard-domain binary quantization.

        At 1.00 bpp, beats ternary by +0.2%.
        Uses Fast Walsh-Hadamard Transform for efficiency.
        """

        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.scale = nn.Parameter(torch.tensor(1.0))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

            # Precompute Hadamard matrices (or use FWHT for large dims)
            self._precompute_hadamard()

        def _precompute_hadamard(self):
            """Precompute normalized Hadamard matrices."""
            H_out = self._hadamard_matrix(self.out_features)
            H_in = self._hadamard_matrix(self.in_features)
            self.register_buffer('H_out', H_out)
            self.register_buffer('H_in', H_in)

        @staticmethod
        def _hadamard_matrix(n: int) -> torch.Tensor:
            """Generate normalized Hadamard-like matrix using Walsh functions."""
            H = torch.zeros(n, n)
            for i in range(n):
                for j in range(n):
                    # Walsh-Hadamard: (-1)^popcount(i & j)
                    H[i, j] = 1.0 if bin(i & j).count('1') % 2 == 0 else -1.0
            return H / math.sqrt(n)

        def get_quantized_weight(self) -> torch.Tensor:
            """Get quantized weights via Hadamard domain binarization."""
            # Transform
            W_had = self.H_out @ self.weight @ self.H_in.T

            # Binarize with STE
            W_had_sign = ste_sign(W_had)
            W_had_sign = torch.where(W_had_sign == 0, torch.ones_like(W_had_sign), W_had_sign)

            # Inverse transform
            W_quant = self.H_out.T @ W_had_sign @ self.H_in

            return W_quant * self.scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        def bits_per_param(self) -> float:
            return 1.0


    class AdaptiveTransformBinaryLinear(nn.Module):
        """Linear layer that adaptively chooses DCT or Hadamard per block.

        At 1.00 bpp, beats ternary by +0.8% (best result!).
        """

        def __init__(self, in_features: int, out_features: int,
                     block_size: int = 32, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.block_size = block_size

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.scale = nn.Parameter(torch.tensor(1.0))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

            # Per-block transform choice (0=DCT, 1=Hadamard)
            n_blocks_h = (out_features + block_size - 1) // block_size
            n_blocks_w = (in_features + block_size - 1) // block_size
            self.register_buffer('transform_choice', torch.zeros(n_blocks_h, n_blocks_w, dtype=torch.long))

        @torch.no_grad()
        def select_transforms(self):
            """Select best transform per block based on reconstruction error."""
            bs = self.block_size
            n_blocks_h = (self.out_features + bs - 1) // bs
            n_blocks_w = (self.in_features + bs - 1) // bs

            for bi in range(n_blocks_h):
                for bj in range(n_blocks_w):
                    i_s, i_e = bi * bs, min((bi + 1) * bs, self.out_features)
                    j_s, j_e = bj * bs, min((bj + 1) * bs, self.in_features)
                    block = self.weight[i_s:i_e, j_s:j_e]

                    # Try DCT
                    B_dct = dct_2d_torch(block)
                    B_dct_sign = torch.sign(B_dct)
                    B_dct_sign = torch.where(B_dct_sign == 0, torch.ones_like(B_dct_sign), B_dct_sign)
                    B_dct_rec = idct_2d_torch(B_dct_sign)
                    err_dct = ((block - B_dct_rec) ** 2).mean()

                    # Try Hadamard
                    bh, bw = block.shape
                    H_h = HadamardBinaryLinear._hadamard_matrix(bh).to(block.device)
                    H_w = HadamardBinaryLinear._hadamard_matrix(bw).to(block.device)
                    B_had = H_h @ block @ H_w.T
                    B_had_sign = torch.sign(B_had)
                    B_had_sign = torch.where(B_had_sign == 0, torch.ones_like(B_had_sign), B_had_sign)
                    B_had_rec = H_h.T @ B_had_sign @ H_w
                    err_had = ((block - B_had_rec) ** 2).mean()

                    self.transform_choice[bi, bj] = 0 if err_dct <= err_had else 1

        def get_quantized_weight(self) -> torch.Tensor:
            """Get quantized weights with adaptive transform per block."""
            W_quant = torch.zeros_like(self.weight)
            bs = self.block_size
            n_blocks_h = (self.out_features + bs - 1) // bs
            n_blocks_w = (self.in_features + bs - 1) // bs

            for bi in range(n_blocks_h):
                for bj in range(n_blocks_w):
                    i_s, i_e = bi * bs, min((bi + 1) * bs, self.out_features)
                    j_s, j_e = bj * bs, min((bj + 1) * bs, self.in_features)
                    block = self.weight[i_s:i_e, j_s:j_e]

                    if self.transform_choice[bi, bj] == 0:
                        # DCT
                        B_t = dct_2d_torch(block)
                        B_t_sign = ste_sign(B_t)
                        B_t_sign = torch.where(B_t_sign == 0, torch.ones_like(B_t_sign), B_t_sign)
                        B_rec = idct_2d_torch(B_t_sign)
                    else:
                        # Hadamard
                        bh, bw = block.shape
                        H_h = HadamardBinaryLinear._hadamard_matrix(bh).to(block.device)
                        H_w = HadamardBinaryLinear._hadamard_matrix(bw).to(block.device)
                        B_t = H_h @ block @ H_w.T
                        B_t_sign = ste_sign(B_t)
                        B_t_sign = torch.where(B_t_sign == 0, torch.ones_like(B_t_sign), B_t_sign)
                        B_rec = H_h.T @ B_t_sign @ H_w

                    W_quant[i_s:i_e, j_s:j_e] = B_rec

            return W_quant * self.scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        def bits_per_param(self) -> float:
            return 1.0  # Transform choice is negligible


    # =========================================================================
    # BASELINES FOR COMPARISON
    # =========================================================================

    class TernaryLinear(nn.Module):
        """Standard ternary quantization (baseline)."""
        def __init__(self, in_features: int, out_features: int,
                     zero_frac: float = 0.3, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.zero_frac = zero_frac

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.scale = nn.Parameter(torch.tensor(1.0))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            thresh = self.weight.abs().quantile(self.zero_frac)
            mask = (self.weight.abs() > thresh).float()
            W_tern = torch.sign(self.weight) * mask
            # STE
            W_tern = self.weight.clamp(-1, 1) + (W_tern - self.weight.clamp(-1, 1)).detach()
            return W_tern * self.scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        def bits_per_param(self) -> float:
            return 1.58


    class BinaryLinear(nn.Module):
        """Standard binary quantization (baseline)."""
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.scale = nn.Parameter(torch.tensor(1.0))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_bin = ste_sign(self.weight)
            W_bin = torch.where(W_bin == 0, torch.ones_like(W_bin), W_bin)
            return W_bin * self.scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        def bits_per_param(self) -> float:
            return 1.0


# =============================================================================
# IMPROVED VARIANTS FOR REAL WEIGHTS
# =============================================================================

if HAS_TORCH:
    class RowwiseDCTBinaryLinear(nn.Module):
        """1D DCT per row - better for non-square matrices."""
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.row_scales = nn.Parameter(torch.ones(out_features))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            # Per-row 1D DCT
            W_dct = dct_1d_torch(self.weight, dim=-1)

            # Binarize
            W_dct_sign = ste_sign(W_dct)
            W_dct_sign = torch.where(W_dct_sign == 0, torch.ones_like(W_dct_sign), W_dct_sign)

            # Inverse 1D DCT
            W_quant = idct_1d_torch(W_dct_sign, dim=-1)

            # Per-row scaling
            return W_quant * self.row_scales.unsqueeze(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)

            # Compute per-row optimal scales
            W_dct = dct_1d_torch(weight, dim=-1)
            W_dct_sign = torch.sign(W_dct)
            W_dct_sign = torch.where(W_dct_sign == 0, torch.ones_like(W_dct_sign), W_dct_sign)
            W_quant = idct_1d_torch(W_dct_sign, dim=-1)

            # Per-row scale: minimize ||W_row - scale * W_quant_row||^2
            for i in range(self.out_features):
                num = (weight[i] * W_quant[i]).sum()
                den = (W_quant[i] * W_quant[i]).sum().clamp(min=1e-8)
                self.row_scales[i] = num / den

            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            # 1 bit per weight + 32 bits per row scale (amortized)
            return 1.0 + 32 / self.in_features


    class BlockDCTBinaryLinear(nn.Module):
        """Block-wise DCT with per-block scaling."""
        def __init__(self, in_features: int, out_features: int,
                     block_size: int = 64, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.block_size = block_size

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

            # Per-block scales
            n_blocks_h = (out_features + block_size - 1) // block_size
            n_blocks_w = (in_features + block_size - 1) // block_size
            self.block_scales = nn.Parameter(torch.ones(n_blocks_h, n_blocks_w))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_quant = torch.zeros_like(self.weight)
            bs = self.block_size

            for bi in range((self.out_features + bs - 1) // bs):
                for bj in range((self.in_features + bs - 1) // bs):
                    i_s, i_e = bi * bs, min((bi + 1) * bs, self.out_features)
                    j_s, j_e = bj * bs, min((bj + 1) * bs, self.in_features)

                    block = self.weight[i_s:i_e, j_s:j_e]

                    # 2D DCT on block
                    B_dct = dct_2d_torch(block)
                    B_sign = ste_sign(B_dct)
                    B_sign = torch.where(B_sign == 0, torch.ones_like(B_sign), B_sign)
                    B_rec = idct_2d_torch(B_sign)

                    W_quant[i_s:i_e, j_s:j_e] = B_rec * self.block_scales[bi, bj]

            return W_quant

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)
            bs = self.block_size

            for bi in range((self.out_features + bs - 1) // bs):
                for bj in range((self.in_features + bs - 1) // bs):
                    i_s, i_e = bi * bs, min((bi + 1) * bs, self.out_features)
                    j_s, j_e = bj * bs, min((bj + 1) * bs, self.in_features)

                    block = weight[i_s:i_e, j_s:j_e]
                    B_dct = dct_2d_torch(block)
                    B_sign = torch.sign(B_dct)
                    B_sign = torch.where(B_sign == 0, torch.ones_like(B_sign), B_sign)
                    B_rec = idct_2d_torch(B_sign)

                    num = (block * B_rec).sum()
                    den = (B_rec * B_rec).sum().clamp(min=1e-8)
                    self.block_scales[bi, bj] = num / den

            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            n_blocks = ((self.out_features + self.block_size - 1) // self.block_size) * \
                       ((self.in_features + self.block_size - 1) // self.block_size)
            n_weights = self.out_features * self.in_features
            return 1.0 + 32 * n_blocks / n_weights


# =============================================================================
# MAGNITUDE-AWARE BINARY (NEW APPROACH)
# =============================================================================

if HAS_TORCH:
    class MagnitudeAwareBinaryLinear(nn.Module):
        """Binary with per-row magnitude scaling - the simplest approach that might work.

        Key insight: The problem with plain binary is that all weights become ±scale.
        But different rows have different importance. Per-row scaling helps.

        At ~1.04 bpp (32 bits per row amortized), this should beat plain binary.
        """
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.row_scales = nn.Parameter(torch.ones(out_features))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)
            return W_sign * self.row_scales.unsqueeze(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)

            # Per-row optimal scale
            W_sign = torch.sign(weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)

            for i in range(self.out_features):
                num = (weight[i] * W_sign[i]).sum()
                den = (W_sign[i] * W_sign[i]).sum().clamp(min=1e-8)
                self.row_scales[i] = num / den

            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            return 1.0 + 32 / self.in_features


    class ColumnMagnitudeBinaryLinear(nn.Module):
        """Binary with per-column magnitude scaling.

        Different from per-row: this scales input features differently.
        """
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.col_scales = nn.Parameter(torch.ones(in_features))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)
            return W_sign * self.col_scales.unsqueeze(0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)

            W_sign = torch.sign(weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)

            for j in range(self.in_features):
                num = (weight[:, j] * W_sign[:, j]).sum()
                den = (W_sign[:, j] * W_sign[:, j]).sum().clamp(min=1e-8)
                self.col_scales[j] = num / den

            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            return 1.0 + 32 / self.out_features


    class RowColMagnitudeBinaryLinear(nn.Module):
        """Binary with both row AND column scaling (rank-1 magnitude approximation).

        W_quant = diag(row_scales) @ sign(W) @ diag(col_scales)

        This is like a rank-1 approximation of the magnitude matrix.
        """
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.row_scales = nn.Parameter(torch.ones(out_features))
            self.col_scales = nn.Parameter(torch.ones(in_features))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)
            return W_sign * self.row_scales.unsqueeze(1) * self.col_scales.unsqueeze(0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)

            W_sign = torch.sign(weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)

            # Alternating optimization for row/col scales
            row_scales = torch.ones(self.out_features)
            col_scales = torch.ones(self.in_features)

            for _ in range(5):  # Few iterations
                # Fix col_scales, optimize row_scales
                W_scaled = W_sign * col_scales.unsqueeze(0)
                for i in range(self.out_features):
                    num = (weight[i] * W_scaled[i]).sum()
                    den = (W_scaled[i] * W_scaled[i]).sum().clamp(min=1e-8)
                    row_scales[i] = num / den

                # Fix row_scales, optimize col_scales
                W_scaled = W_sign * row_scales.unsqueeze(1)
                for j in range(self.in_features):
                    num = (weight[:, j] * W_scaled[:, j]).sum()
                    den = (W_scaled[:, j] * W_scaled[:, j]).sum().clamp(min=1e-8)
                    col_scales[j] = num / den

            self.row_scales.copy_(row_scales)
            self.col_scales.copy_(col_scales)

            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            return 1.0 + 32 * (self.out_features + self.in_features) / (self.out_features * self.in_features)


if HAS_TORCH:
    class LowRankMagnitudeBinaryLinear(nn.Module):
        """Binary with low-rank magnitude approximation.

        Instead of rank-1 (row ⊗ col), use rank-k SVD of |W|.

        W_quant = sign(W) * (U_k @ S_k @ V_k^T)

        where U_k, S_k, V_k are from SVD of |W|.
        """
        def __init__(self, in_features: int, out_features: int, rank: int = 4, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

            # Low-rank factors for magnitude
            self.U = nn.Parameter(torch.randn(out_features, rank) * 0.1)
            self.S = nn.Parameter(torch.ones(rank))
            self.V = nn.Parameter(torch.randn(in_features, rank) * 0.1)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)

            # Reconstruct magnitude from low-rank factors
            magnitude = (self.U * self.S.unsqueeze(0)) @ self.V.T
            magnitude = F.relu(magnitude)  # Ensure non-negative

            return W_sign * magnitude

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)

            # SVD of |W|
            W_abs = weight.abs()
            U, S, Vh = torch.linalg.svd(W_abs, full_matrices=False)

            # Keep top-k
            k = min(self.rank, len(S))
            self.U.copy_(U[:, :k])
            self.S.copy_(S[:k])
            self.V.copy_(Vh[:k, :].T)

            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            # 1 bit per weight + 32 bits for each low-rank element
            n_weights = self.out_features * self.in_features
            n_lowrank = self.rank * (self.out_features + self.in_features + 1)
            return 1.0 + 32 * n_lowrank / n_weights


    class BlockRowColBinaryLinear(nn.Module):
        """Binary with per-block row+col scaling.

        Divide matrix into blocks, apply Row+Col scaling within each block.
        """
        def __init__(self, in_features: int, out_features: int,
                     block_size: int = 128, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.block_size = block_size

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

            # Per-block row and col scales
            n_blocks_h = (out_features + block_size - 1) // block_size
            n_blocks_w = (in_features + block_size - 1) // block_size

            # Store scales as flat tensors for each block
            self.n_blocks_h = n_blocks_h
            self.n_blocks_w = n_blocks_w

            # Row scales: n_blocks_h * n_blocks_w * block_size
            # Col scales: n_blocks_h * n_blocks_w * block_size
            self.row_scales = nn.ParameterList([
                nn.Parameter(torch.ones(min(block_size, out_features - bi * block_size)))
                for bi in range(n_blocks_h)
                for _ in range(n_blocks_w)
            ])
            self.col_scales = nn.ParameterList([
                nn.Parameter(torch.ones(min(block_size, in_features - bj * block_size)))
                for _ in range(n_blocks_h)
                for bj in range(n_blocks_w)
            ])

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)

            W_quant = torch.zeros_like(self.weight)
            bs = self.block_size

            idx = 0
            for bi in range(self.n_blocks_h):
                for bj in range(self.n_blocks_w):
                    i_s, i_e = bi * bs, min((bi + 1) * bs, self.out_features)
                    j_s, j_e = bj * bs, min((bj + 1) * bs, self.in_features)

                    block_sign = W_sign[i_s:i_e, j_s:j_e]
                    row_s = self.row_scales[idx]
                    col_s = self.col_scales[idx]

                    W_quant[i_s:i_e, j_s:j_e] = block_sign * row_s.unsqueeze(1) * col_s.unsqueeze(0)
                    idx += 1

            return W_quant

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)
            bs = self.block_size

            idx = 0
            for bi in range(self.n_blocks_h):
                for bj in range(self.n_blocks_w):
                    i_s, i_e = bi * bs, min((bi + 1) * bs, self.out_features)
                    j_s, j_e = bj * bs, min((bj + 1) * bs, self.in_features)

                    block = weight[i_s:i_e, j_s:j_e]
                    block_sign = torch.sign(block)
                    block_sign = torch.where(block_sign == 0, torch.ones_like(block_sign), block_sign)

                    bh, bw = block.shape
                    row_scales = torch.ones(bh)
                    col_scales = torch.ones(bw)

                    for _ in range(5):
                        W_scaled = block_sign * col_scales.unsqueeze(0)
                        for i in range(bh):
                            num = (block[i] * W_scaled[i]).sum()
                            den = (W_scaled[i] * W_scaled[i]).sum().clamp(min=1e-8)
                            row_scales[i] = num / den

                        W_scaled = block_sign * row_scales.unsqueeze(1)
                        for j in range(bw):
                            num = (block[:, j] * W_scaled[:, j]).sum()
                            den = (W_scaled[:, j] * W_scaled[:, j]).sum().clamp(min=1e-8)
                            col_scales[j] = num / den

                    self.row_scales[idx].copy_(row_scales)
                    self.col_scales[idx].copy_(col_scales)
                    idx += 1

            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            n_weights = self.out_features * self.in_features
            n_scales = sum(p.numel() for p in self.row_scales) + sum(p.numel() for p in self.col_scales)
            return 1.0 + 32 * n_scales / n_weights


# =============================================================================
# ACTIVATION-GATED BINARY (TRUE 1.00 BPP)
# =============================================================================

if HAS_TORCH:
    class ActivationGatedBinaryLinear(nn.Module):
        """Binary weights with activation-based gating.

        Key insight: Instead of storing which weights are "important",
        derive importance from the INPUT activations. Small activations
        get gated toward zero, achieving the "zero effect" of ternary
        without storing any magnitude information.

        W_quant = sign(W)  # Exactly 1.00 bpp
        forward: y = (x * gate(|x|)) @ W_quant

        The gate function suppresses small activations, so:
        - Small activation × ±1 = ~0 contribution
        - Large activation × ±1 = full contribution
        """
        def __init__(self, in_features: int, out_features: int,
                     threshold: float = 0.1, sharpness: float = 10.0,
                     learnable_gate: bool = True, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

            # Gate parameters (can be learned or fixed)
            if learnable_gate:
                self.threshold = nn.Parameter(torch.tensor(threshold))
                self.sharpness = nn.Parameter(torch.tensor(sharpness))
            else:
                self.register_buffer('threshold', torch.tensor(threshold))
                self.register_buffer('sharpness', torch.tensor(sharpness))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)
            return W_sign

        def gate_activations(self, x: torch.Tensor) -> torch.Tensor:
            """Apply soft gating to suppress small activations."""
            # Sigmoid gate: small |x| → 0, large |x| → 1
            gate = torch.sigmoid((torch.abs(x) - self.threshold) * self.sharpness)
            return x * gate

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Gate the input activations
            x_gated = self.gate_activations(x)
            # Use binary weights
            W = self.get_quantized_weight()
            return F.linear(x_gated, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

            # Optionally tune threshold based on weight statistics
            # Idea: threshold should match the scale where weights become "unimportant"
            abs_weight = torch.abs(weight)
            # Set threshold to ~30th percentile of activation scale
            # (This is a heuristic - could be optimized)

        def bits_per_param(self) -> float:
            # Exactly 1.00 bpp for weights
            # Gate params are shared across all weights (negligible overhead)
            return 1.0


    class ActivationGatedBinaryV2(nn.Module):
        """Activation-Gated Binary with per-feature thresholds.

        Each input feature has its own threshold, allowing the network
        to learn which features need gating.

        Storage: 1 bit/weight + threshold per input feature
        For 768×768: 1.0 + 32*768/(768*768) = 1.042 bpp
        """
        def __init__(self, in_features: int, out_features: int,
                     init_threshold: float = 0.1, sharpness: float = 10.0,
                     bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.thresholds = nn.Parameter(torch.full((in_features,), init_threshold))
            self.sharpness = sharpness

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)
            return W_sign

        def gate_activations(self, x: torch.Tensor) -> torch.Tensor:
            # Per-feature thresholds
            gate = torch.sigmoid((torch.abs(x) - self.thresholds) * self.sharpness)
            return x * gate

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_gated = self.gate_activations(x)
            W = self.get_quantized_weight()
            return F.linear(x_gated, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

            # Initialize thresholds based on weight column magnitudes
            # Columns with larger weights should have lower thresholds (more activation)
            col_importance = torch.abs(weight).mean(dim=0)
            # Invert: high importance → low threshold
            normalized = col_importance / col_importance.max()
            self.thresholds.copy_(0.2 * (1.0 - normalized) + 0.01)

        def bits_per_param(self) -> float:
            return 1.0 + 32 * self.in_features / (self.out_features * self.in_features)


    class ProceduralZeroBinaryLinear(nn.Module):
        """Binary with procedural (deterministic) zero mask.

        Zero positions are computed from a hash function, not stored.
        The mask is regenerated at inference time.

        True 1.00 bpp: no storage for the mask.
        """
        def __init__(self, in_features: int, out_features: int,
                     zero_rate: float = 0.3, seed: int = 42, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.zero_rate = zero_rate
            self.seed = seed

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

            # Generate mask once (deterministic)
            self.register_buffer('mask', self._generate_mask())

        def _generate_mask(self) -> torch.Tensor:
            """Generate deterministic mask from seed."""
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            mask = torch.rand(self.out_features, self.in_features, generator=gen)
            return (mask > self.zero_rate).float()

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)
            return W_sign * self.mask

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W = self.get_quantized_weight()
            return F.linear(x, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            # Mask is procedural, not stored
            return 1.0


    class CombinedGatedBinaryLinear(nn.Module):
        """Combines Activation Gating + Procedural Zeros.

        Both effects work together:
        1. Procedural zeros create ternary-style sparsity (0 bits)
        2. Activation gating suppresses small contributions (0 bits)

        True 1.00 bpp with double the "zero effect".
        """
        def __init__(self, in_features: int, out_features: int,
                     zero_rate: float = 0.2, threshold: float = 0.1,
                     sharpness: float = 10.0, seed: int = 42, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.zero_rate = zero_rate
            self.threshold = threshold
            self.sharpness = sharpness
            self.seed = seed

            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

            # Procedural mask
            gen = torch.Generator()
            gen.manual_seed(seed)
            mask = torch.rand(out_features, in_features, generator=gen)
            self.register_buffer('mask', (mask > zero_rate).float())

        def get_quantized_weight(self) -> torch.Tensor:
            W_sign = ste_sign(self.weight)
            W_sign = torch.where(W_sign == 0, torch.ones_like(W_sign), W_sign)
            return W_sign * self.mask

        def gate_activations(self, x: torch.Tensor) -> torch.Tensor:
            gate = torch.sigmoid((torch.abs(x) - self.threshold) * self.sharpness)
            return x * gate

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_gated = self.gate_activations(x)
            W = self.get_quantized_weight()
            return F.linear(x_gated, W, self.bias)

        @torch.no_grad()
        def init_from_fp32(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.weight.copy_(weight)
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias)

        def bits_per_param(self) -> float:
            return 1.0


# =============================================================================
# GPT-2 WEIGHT TESTING
# =============================================================================

def test_on_gpt2_weights():
    """Test DCT Binary vs Ternary on real GPT-2 weight matrices."""
    if not HAS_TORCH:
        print("PyTorch required for GPT-2 testing")
        return

    print("=" * 75)
    print("GPT-2 WEIGHT TESTING: DCT Binary vs Ternary")
    print("=" * 75)

    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("transformers library required. Install with: pip install transformers")
        return

    print("\nLoading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    results = []

    # Test on various weight matrices
    weight_names = [
        ("h.0.attn.c_attn.weight", model.transformer.h[0].attn.c_attn.weight),
        ("h.0.attn.c_proj.weight", model.transformer.h[0].attn.c_proj.weight),
        ("h.0.mlp.c_fc.weight", model.transformer.h[0].mlp.c_fc.weight),
        ("h.0.mlp.c_proj.weight", model.transformer.h[0].mlp.c_proj.weight),
        ("h.5.attn.c_attn.weight", model.transformer.h[5].attn.c_attn.weight),
        ("h.5.mlp.c_fc.weight", model.transformer.h[5].mlp.c_fc.weight),
        ("h.11.attn.c_attn.weight", model.transformer.h[11].attn.c_attn.weight),
        ("h.11.mlp.c_proj.weight", model.transformer.h[11].mlp.c_proj.weight),
    ]

    for name, W_fp32 in weight_names:
        W = W_fp32.detach().float()

        # Need to transpose for GPT-2 Conv1D weights (stored as [in, out])
        if W.shape[0] < W.shape[1]:
            W = W.T

        out_features, in_features = W.shape
        print(f"\n{name}: {W.shape}")

        # Generate synthetic input/output pairs for evaluation
        n_samples = 500
        X = torch.randn(n_samples, in_features)
        Y = X @ W.T

        X_train, X_test = X[:400], X[400:]
        Y_train, Y_test = Y[:400], Y[400:]

        layer_results = {}

        # Binary baseline
        layer = BinaryLinear(in_features, out_features, bias=False)
        layer.weight.data = W.clone()
        # Compute optimal scale
        W_bin = torch.sign(W)
        W_bin = torch.where(W_bin == 0, torch.ones_like(W_bin), W_bin)
        scale = (W * W_bin).sum() / (W_bin * W_bin).sum().clamp(min=1e-8)
        layer.scale.data = scale

        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Binary'] = corr

        # Ternary baseline
        layer = TernaryLinear(in_features, out_features, zero_frac=0.3, bias=False)
        layer.weight.data = W.clone()
        thresh = W.abs().quantile(0.3)
        mask = (W.abs() > thresh)
        scale = W[mask].abs().mean()
        layer.scale.data = scale

        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Ternary'] = corr

        # DCT Binary
        layer = DCTBinaryLinear(in_features, out_features, bias=False)
        layer.init_from_fp32(W)

        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['DCT Binary'] = corr

        # Hadamard Binary
        layer = HadamardBinaryLinear(in_features, out_features, bias=False)
        layer.weight.data = W.clone()
        W_had = layer.H_out @ W @ layer.H_in.T
        W_had_sign = torch.sign(W_had)
        W_had_sign = torch.where(W_had_sign == 0, torch.ones_like(W_had_sign), W_had_sign)
        W_quant = layer.H_out.T @ W_had_sign @ layer.H_in
        scale = (W * W_quant).sum() / (W_quant * W_quant).sum().clamp(min=1e-8)
        layer.scale.data = scale

        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Hadamard'] = corr

        # Row-wise DCT Binary (NEW)
        layer = RowwiseDCTBinaryLinear(in_features, out_features, bias=False)
        layer.init_from_fp32(W)

        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Row DCT'] = corr

        # Block DCT Binary
        for bs in [32]:
            layer = BlockDCTBinaryLinear(in_features, out_features, block_size=bs, bias=False)
            layer.init_from_fp32(W)

            with torch.no_grad():
                Y_pred = layer(X_test)
            corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
            layer_results[f'Block DCT {bs}'] = corr

        # Magnitude-aware methods (NEW)
        layer = MagnitudeAwareBinaryLinear(in_features, out_features, bias=False)
        layer.init_from_fp32(W)
        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Row Scale'] = corr

        layer = ColumnMagnitudeBinaryLinear(in_features, out_features, bias=False)
        layer.init_from_fp32(W)
        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Col Scale'] = corr

        layer = RowColMagnitudeBinaryLinear(in_features, out_features, bias=False)
        layer.init_from_fp32(W)
        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Row+Col Scale'] = corr

        # Low-rank magnitude - test at various ranks
        for rank in [1, 2, 4, 8]:
            layer = LowRankMagnitudeBinaryLinear(in_features, out_features, rank=rank, bias=False)
            layer.init_from_fp32(W)
            with torch.no_grad():
                Y_pred = layer(X_test)
            corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
            layer_results[f'LowRank r={rank}'] = corr

        # Block Row+Col (NEW)
        layer = BlockRowColBinaryLinear(in_features, out_features, block_size=128, bias=False)
        layer.init_from_fp32(W)
        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Block R+C 128'] = corr

        results.append((name, layer_results))

        # Print layer results
        tern_corr = layer_results['Ternary']
        for method, corr in sorted(layer_results.items(), key=lambda x: -x[1]):
            vs = ((corr / tern_corr) - 1) * 100
            marker = "★" if vs > 0 and method != 'Ternary' else ""
            print(f"  {method:<15} {corr:.4f} ({vs:+.1f}% vs ternary) {marker}")

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY ACROSS ALL LAYERS")
    print("=" * 75)

    # Collect all methods from results
    all_methods = set()
    for name, layer_results in results:
        all_methods.update(layer_results.keys())

    avg_corrs = {m: [] for m in all_methods}
    for name, layer_results in results:
        for m in all_methods:
            if m in layer_results:
                avg_corrs[m].append(layer_results[m])

    # Calculate approximate BPP for each method
    # Using first layer dimensions as reference
    out_f, in_f = 2304, 768  # Typical GPT-2 layer
    n_weights = out_f * in_f

    def get_bpp(method):
        if method == 'Ternary':
            return 1.58
        elif method == 'Binary':
            return 1.0
        elif 'LowRank r=' in method:
            rank = int(method.split('=')[1])
            n_lowrank = rank * (out_f + in_f + 1)
            return 1.0 + 32 * n_lowrank / n_weights
        elif method == 'Row+Col Scale':
            return 1.0 + 32 * (out_f + in_f) / n_weights
        elif method == 'Row Scale':
            return 1.0 + 32 * out_f / n_weights
        elif method == 'Col Scale':
            return 1.0 + 32 * in_f / n_weights
        elif 'Block R+C' in method:
            return 1.1  # Approximate
        else:
            return 1.0

    print(f"\n{'Method':<20} {'Avg Corr':>10} {'BPP':>6} {'vs Ternary':>12}")
    print("-" * 60)

    tern_avg = np.mean(avg_corrs['Ternary'])
    for method in sorted(avg_corrs.keys(), key=lambda m: -np.mean(avg_corrs[m]) if avg_corrs[m] else -999):
        if not avg_corrs[method]:
            continue
        avg = np.mean(avg_corrs[method])
        bpp = get_bpp(method)
        vs = ((avg / tern_avg) - 1) * 100
        # Mark as winner if beats ternary at same or lower BPP
        marker = "★★★" if vs > 0 and bpp <= 1.58 else ("★" if vs > 0 else "")
        print(f"{method:<20} {avg:>10.4f} {bpp:>6.2f} {vs:>+11.1f}% {marker}")

    return results


def test_activation_gated():
    """Test Activation-Gated Binary approaches - TRUE 1.00 BPP methods."""
    if not HAS_TORCH:
        print("PyTorch required")
        return

    print("=" * 80)
    print("ACTIVATION-GATED BINARY: True 1.00 BPP Methods")
    print("=" * 80)
    print("\nKey insight: Use INPUT activation magnitudes to get 'zero effect'")
    print("without storing any magnitude information in weights.\n")

    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("transformers library required")
        return

    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Test on various layers
    weight_names = [
        ("h.0.attn.c_attn", model.transformer.h[0].attn.c_attn.weight),
        ("h.0.attn.c_proj", model.transformer.h[0].attn.c_proj.weight),
        ("h.0.mlp.c_fc", model.transformer.h[0].mlp.c_fc.weight),
        ("h.0.mlp.c_proj", model.transformer.h[0].mlp.c_proj.weight),
        ("h.5.attn.c_attn", model.transformer.h[5].attn.c_attn.weight),
        ("h.5.mlp.c_fc", model.transformer.h[5].mlp.c_fc.weight),
        ("h.11.attn.c_attn", model.transformer.h[11].attn.c_attn.weight),
        ("h.11.mlp.c_proj", model.transformer.h[11].mlp.c_proj.weight),
    ]

    all_results = {}

    for name, W_fp32 in weight_names:
        W = W_fp32.detach().float()
        if W.shape[0] < W.shape[1]:
            W = W.T

        out_f, in_f = W.shape
        print(f"\n{'='*60}")
        print(f"{name}: {W.shape}")
        print("=" * 60)

        # Generate test data - use REALISTIC activation distributions
        n_samples = 500
        # Simulate post-LayerNorm activations (roughly unit variance with some outliers)
        X = torch.randn(n_samples, in_f) * 0.5  # Scaled down
        # Add some sparse large activations (like real transformer activations)
        sparse_mask = torch.rand(n_samples, in_f) > 0.9
        X = X + sparse_mask.float() * torch.randn(n_samples, in_f) * 2.0

        Y_fp32 = X @ W.T

        X_test, Y_test = X[400:], Y_fp32[400:]

        layer_results = {}

        # Baseline: FP32
        corr = np.corrcoef(Y_test.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['FP32'] = 1.0

        # Baseline: Binary (pure sign, no scaling)
        W_bin = torch.sign(W)
        W_bin = torch.where(W_bin == 0, torch.ones_like(W_bin), W_bin)
        # Apply global scale for fair comparison
        scale = torch.mean(torch.abs(W))
        with torch.no_grad():
            Y_pred = X_test @ (W_bin * scale).T
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Binary'] = corr

        # Baseline: Ternary
        scale = torch.mean(torch.abs(W))
        thresh = 0.3 * scale
        W_tern = torch.zeros_like(W)
        W_tern[W > thresh] = scale
        W_tern[W < -thresh] = -scale
        with torch.no_grad():
            Y_pred = X_test @ W_tern.T
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Ternary'] = corr

        # NEW: Activation-Gated Binary (various thresholds)
        for thresh in [0.05, 0.1, 0.2, 0.3]:
            layer = ActivationGatedBinaryLinear(in_f, out_f, threshold=thresh,
                                                 sharpness=10.0, learnable_gate=False, bias=False)
            layer.init_from_fp32(W)
            with torch.no_grad():
                Y_pred = layer(X_test)
            corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
            layer_results[f'ActGate t={thresh}'] = corr

        # NEW: Activation-Gated V2 (per-feature thresholds)
        layer = ActivationGatedBinaryV2(in_f, out_f, init_threshold=0.1, bias=False)
        layer.init_from_fp32(W)
        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['ActGate V2'] = corr

        # NEW: Procedural Zeros
        for zero_rate in [0.2, 0.3, 0.4]:
            layer = ProceduralZeroBinaryLinear(in_f, out_f, zero_rate=zero_rate, bias=False)
            layer.init_from_fp32(W)
            with torch.no_grad():
                Y_pred = layer(X_test)
            corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
            layer_results[f'ProcZero {int(zero_rate*100)}%'] = corr

        # NEW: Combined (Activation Gate + Procedural Zeros)
        for zero_rate in [0.15, 0.2]:
            for thresh in [0.1, 0.15]:
                layer = CombinedGatedBinaryLinear(in_f, out_f, zero_rate=zero_rate,
                                                   threshold=thresh, bias=False)
                layer.init_from_fp32(W)
                with torch.no_grad():
                    Y_pred = layer(X_test)
                corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
                layer_results[f'Combined z={int(zero_rate*100)} t={thresh}'] = corr

        # Compare with Row+Col Scale (our previous best at ~1.06 bpp)
        layer = RowColMagnitudeBinaryLinear(in_f, out_f, bias=False)
        layer.init_from_fp32(W)
        with torch.no_grad():
            Y_pred = layer(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        layer_results['Row+Col (1.06bpp)'] = corr

        # Print results for this layer
        tern_corr = layer_results['Ternary']
        print(f"\n{'Method':<25} {'Corr':>8} {'vs Tern':>10} {'BPP':>6}")
        print("-" * 55)

        for method, corr in sorted(layer_results.items(), key=lambda x: -x[1]):
            vs = ((corr / tern_corr) - 1) * 100
            # Determine BPP
            if method == 'Row+Col (1.06bpp)':
                bpp = 1.06
            elif method == 'ActGate V2':
                bpp = 1.0 + 32 * in_f / (out_f * in_f)
            elif method == 'Ternary':
                bpp = 1.58
            else:
                bpp = 1.0

            marker = "★★★" if vs > 0 and bpp <= 1.0 else ("★" if vs > 0 else "")
            print(f"{method:<25} {corr:>8.4f} {vs:>+9.1f}% {bpp:>6.2f} {marker}")

        # Store for aggregation
        for method, corr in layer_results.items():
            if method not in all_results:
                all_results[method] = []
            all_results[method].append(corr)

    # Print aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS ACROSS ALL LAYERS")
    print("=" * 80)

    tern_avg = np.mean(all_results['Ternary'])
    bin_avg = np.mean(all_results['Binary'])

    print(f"\nBaseline gap: Binary is {((bin_avg/tern_avg)-1)*100:+.1f}% vs Ternary")
    print(f"\n{'Method':<25} {'Avg Corr':>10} {'vs Ternary':>12} {'vs Binary':>12}")
    print("-" * 65)

    for method in sorted(all_results.keys(), key=lambda m: -np.mean(all_results[m])):
        avg = np.mean(all_results[method])
        vs_tern = ((avg / tern_avg) - 1) * 100
        vs_bin = ((avg / bin_avg) - 1) * 100

        # Mark TRUE 1.0 bpp winners
        is_true_1bpp = 'ActGate' in method or 'ProcZero' in method or 'Combined' in method or method == 'Binary'
        marker = "★★★ TRUE 1.0 BPP" if is_true_1bpp and vs_tern > 0 else ""

        print(f"{method:<25} {avg:>10.4f} {vs_tern:>+11.1f}% {vs_bin:>+11.1f}% {marker}")

    return all_results


def test_procedural_zeros_with_training():
    """Test if TRAINING with procedural zeros helps.

    Key insight: If the model knows during training that certain positions
    will be zeroed, it can learn to route information through non-zero positions.

    This is different from post-hoc application to pre-trained weights!
    """
    if not HAS_TORCH:
        print("PyTorch required")
        return

    print("=" * 80)
    print("PROCEDURAL ZEROS WITH TRAINING")
    print("=" * 80)
    print("\nHypothesis: Training with procedural zeros active allows the model")
    print("to learn to route information through non-zero positions.\n")

    # Create a learnable task
    torch.manual_seed(42)
    np.random.seed(42)

    d_in, d_out = 256, 128
    n_samples = 2000

    # True weight matrix (what we want to approximate)
    W_true = torch.randn(d_out, d_in) * 0.5

    # Generate data
    X = torch.randn(n_samples, d_in)
    Y = X @ W_true.T + torch.randn(n_samples, d_out) * 0.1  # Small noise

    X_train, Y_train = X[:1600], Y[:1600]
    X_test, Y_test = X[1600:], Y[1600:]

    results = {}

    # =========================================================================
    # Baseline 1: Binary trained (STE)
    # =========================================================================
    print("\n--- Training: Binary (STE) ---")

    class BinaryLinearSTE(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            W_bin = ste_sign(self.weight)
            W_bin = torch.where(W_bin == 0, torch.ones_like(W_bin), W_bin)
            return x @ (W_bin * self.scale).T

    model = BinaryLinearSTE(d_in, d_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(500):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = F.mse_loss(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Y_pred = model(X_test)
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    results['Binary (trained)'] = corr
    print(f"Binary (trained): {corr:.4f}")

    # =========================================================================
    # Baseline 2: Ternary trained
    # =========================================================================
    print("\n--- Training: Ternary (STE) ---")

    class TernaryLinearSTE(nn.Module):
        def __init__(self, in_f, out_f, zero_thresh=0.3):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.zero_thresh = zero_thresh

        def forward(self, x):
            # Ternary: sign with magnitude-based zeros
            W_abs = torch.abs(self.weight)
            thresh = self.zero_thresh * W_abs.mean()

            # STE for ternary
            W_sign = ste_sign(self.weight)
            W_mask = (W_abs > thresh).float()
            # STE for mask (straight-through)
            W_mask = W_mask - W_mask.detach() + (torch.sigmoid((W_abs - thresh) * 10)).detach()

            W_tern = W_sign * W_mask
            return x @ (W_tern * self.scale).T

    model = TernaryLinearSTE(d_in, d_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(500):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = F.mse_loss(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Y_pred = model(X_test)
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    results['Ternary (trained)'] = corr
    print(f"Ternary (trained): {corr:.4f}")

    # =========================================================================
    # NEW: Procedural Zeros TRAINED
    # =========================================================================
    print("\n--- Training: Procedural Zeros (various rates) ---")

    class ProceduralZeroTrainedLinear(nn.Module):
        def __init__(self, in_f, out_f, zero_rate=0.3, seed=42):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
            self.scale = nn.Parameter(torch.tensor(1.0))

            # Fixed procedural mask (known during training!)
            gen = torch.Generator()
            gen.manual_seed(seed)
            mask = torch.rand(out_f, in_f, generator=gen)
            self.register_buffer('mask', (mask > zero_rate).float())

        def forward(self, x):
            W_bin = ste_sign(self.weight)
            W_bin = torch.where(W_bin == 0, torch.ones_like(W_bin), W_bin)
            # Apply procedural mask DURING TRAINING
            W_masked = W_bin * self.mask
            return x @ (W_masked * self.scale).T

    for zero_rate in [0.2, 0.3, 0.4, 0.5]:
        model = ProceduralZeroTrainedLinear(d_in, d_out, zero_rate=zero_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(500):
            optimizer.zero_grad()
            Y_pred = model(X_train)
            loss = F.mse_loss(Y_pred, Y_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            Y_pred = model(X_test)
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        results[f'ProcZero {int(zero_rate*100)}% (trained)'] = corr
        print(f"ProcZero {int(zero_rate*100)}% (trained): {corr:.4f}")

    # =========================================================================
    # NEW: Learnable mask that CONVERGES to procedural
    # Train with soft mask, then snap to procedural at inference
    # =========================================================================
    print("\n--- Training: Soft mask → Procedural snap ---")

    class SoftToProceduralLinear(nn.Module):
        def __init__(self, in_f, out_f, target_sparsity=0.3, seed=42):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
            self.scale = nn.Parameter(torch.tensor(1.0))

            # Learnable importance scores
            self.importance = nn.Parameter(torch.zeros(out_f, in_f))
            self.target_sparsity = target_sparsity

            # Procedural target (for regularization)
            gen = torch.Generator()
            gen.manual_seed(seed)
            mask = torch.rand(out_f, in_f, generator=gen)
            self.register_buffer('proc_mask', (mask > target_sparsity).float())

        def forward(self, x, hard=False):
            W_bin = ste_sign(self.weight)
            W_bin = torch.where(W_bin == 0, torch.ones_like(W_bin), W_bin)

            if hard:
                # Inference: use procedural mask
                mask = self.proc_mask
            else:
                # Training: use soft learned mask
                mask = torch.sigmoid(self.importance)

            W_masked = W_bin * mask
            return x @ (W_masked * self.scale).T

        def sparsity_loss(self):
            # Encourage mask to match target sparsity
            mask = torch.sigmoid(self.importance)
            current_sparsity = 1.0 - mask.mean()
            return (current_sparsity - self.target_sparsity) ** 2

    for target_sparsity in [0.3]:
        model = SoftToProceduralLinear(d_in, d_out, target_sparsity=target_sparsity)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(500):
            optimizer.zero_grad()
            Y_pred = model(X_train, hard=False)
            task_loss = F.mse_loss(Y_pred, Y_train)
            sparse_loss = model.sparsity_loss()
            loss = task_loss + 0.1 * sparse_loss
            loss.backward()
            optimizer.step()

        # Test with SOFT mask (learned)
        with torch.no_grad():
            Y_pred_soft = model(X_test, hard=False)
        corr_soft = np.corrcoef(Y_pred_soft.numpy().flatten(), Y_test.numpy().flatten())[0, 1]

        # Test with HARD procedural mask (snap)
        with torch.no_grad():
            Y_pred_hard = model(X_test, hard=True)
        corr_hard = np.corrcoef(Y_pred_hard.numpy().flatten(), Y_test.numpy().flatten())[0, 1]

        results[f'SoftMask (learned)'] = corr_soft
        results[f'SoftMask→Proc (snap)'] = corr_hard
        print(f"SoftMask (learned): {corr_soft:.4f}")
        print(f"SoftMask→Proc (snap): {corr_hard:.4f}")

    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Training with Procedural Zeros")
    print("=" * 80)

    tern_corr = results['Ternary (trained)']
    bin_corr = results['Binary (trained)']

    print(f"\n{'Method':<30} {'Corr':>8} {'vs Ternary':>12} {'vs Binary':>12} {'BPP':>6}")
    print("-" * 75)

    for method, corr in sorted(results.items(), key=lambda x: -x[1]):
        vs_tern = ((corr / tern_corr) - 1) * 100
        vs_bin = ((corr / bin_corr) - 1) * 100

        # Determine BPP
        if 'Ternary' in method:
            bpp = 1.58
        else:
            bpp = 1.0

        marker = "★★★" if vs_tern >= 0 and bpp <= 1.0 else ""
        print(f"{method:<30} {corr:>8.4f} {vs_tern:>+11.1f}% {vs_bin:>+11.1f}% {bpp:>6.2f} {marker}")

    print("\n" + "=" * 80)
    print("KEY QUESTION: Does training with procedural zeros close the gap?")
    print("=" * 80)

    best_proc = max([v for k, v in results.items() if 'ProcZero' in k])
    gap_reduction = (best_proc - bin_corr) / (tern_corr - bin_corr) * 100

    print(f"\nBinary (trained): {bin_corr:.4f}")
    print(f"Best ProcZero:    {best_proc:.4f}")
    print(f"Ternary (trained): {tern_corr:.4f}")
    print(f"\nGap reduction: {gap_reduction:.1f}%")

    if best_proc > tern_corr:
        print("\n🎉 PROCEDURAL ZEROS BEATS TERNARY WHEN TRAINED!")
    elif gap_reduction > 50:
        print(f"\n✅ Procedural zeros closes {gap_reduction:.0f}% of the gap!")
    else:
        print(f"\n❌ Procedural zeros only closes {gap_reduction:.0f}% of the gap.")

    return results


def test_first_principles_approaches():
    """Test TRUE 1.0 bpp approaches derived from first principles."""
    if not HAS_TORCH:
        print("PyTorch required")
        return

    print("=" * 80)
    print("FIRST PRINCIPLES: Sub-1.0 BPP and True 1.0 BPP Approaches")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    # Test task
    d_in, d_out = 128, 64
    n_samples = 2000

    # True weights (what we want to approximate)
    W_true = torch.randn(d_out, d_in) * 0.5

    X = torch.randn(n_samples, d_in)
    Y = X @ W_true.T + torch.randn(n_samples, d_out) * 0.05

    X_train, Y_train = X[:1600], Y[:1600]
    X_test, Y_test = X[1600:], Y[1600:]

    results = {}

    # =========================================================================
    # APPROACH 1: Implicit Neural Weights (BELOW 1.0 BPP!)
    # =========================================================================
    print("\n" + "=" * 60)
    print("APPROACH 1: Implicit Neural Weights")
    print("A tiny network GENERATES weights from coordinates")
    print("=" * 60)

    class ImplicitBinaryWeights(nn.Module):
        """Tiny network generates binary weights from position coordinates."""
        def __init__(self, hidden=32, num_layers=2, coord_dim=4):
            super().__init__()
            layers = [nn.Linear(coord_dim, hidden), nn.GELU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.GELU()]
            layers += [nn.Linear(hidden, 1)]  # Output logit for sign
            self.net = nn.Sequential(*layers)
            self.hidden = hidden
            self.num_layers = num_layers

        def generate_weight_matrix(self, out_f, in_f, layer_id=0):
            """Generate full weight matrix from coordinates."""
            # Create coordinate grid
            rows = torch.arange(out_f).float() / out_f
            cols = torch.arange(in_f).float() / in_f

            # Grid of coordinates
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
            layer_feat = torch.full_like(row_grid, layer_id / 12.0)

            # Combine into features [out, in, 4]
            coords = torch.stack([row_grid, col_grid, layer_feat,
                                  torch.sin(row_grid * 3.14159 * 4)], dim=-1)

            # Generate weights
            logits = self.net(coords).squeeze(-1)  # [out, in]

            # Binarize with STE
            W_binary = ste_sign(logits)
            W_binary = torch.where(W_binary == 0, torch.ones_like(W_binary), W_binary)

            return W_binary

        def count_params(self):
            return sum(p.numel() for p in self.parameters())

    # Test with increasing generator sizes
    for hidden, n_layers in [(8, 1), (16, 2), (32, 2), (64, 3)]:
        generator = ImplicitBinaryWeights(hidden=hidden, num_layers=n_layers)
        n_params = generator.count_params()

        # Train generator
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)

        for epoch in range(1000):
            optimizer.zero_grad()
            W_gen = generator.generate_weight_matrix(d_out, d_in)
            Y_pred = X_train @ W_gen.T
            loss = F.mse_loss(Y_pred, Y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            W_gen = generator.generate_weight_matrix(d_out, d_in)
            Y_pred = X_test @ W_gen.T

        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]

        # Calculate BPP
        bits_generator = n_params * 32
        n_weights = d_out * d_in
        bpp = bits_generator / n_weights

        method_name = f'Implicit h={hidden} L={n_layers}'
        results[method_name] = (corr, bpp)
        print(f"{method_name}: corr={corr:.4f}, generator_params={n_params}, BPP={bpp:.3f}")

    # =========================================================================
    # APPROACH 2: Sign-Texture-Derived Importance
    # =========================================================================
    print("\n" + "=" * 60)
    print("APPROACH 2: Sign-Texture-Derived Importance")
    print("Local sign agreement determines importance (0 storage)")
    print("=" * 60)

    def compute_sign_agreement(W_binary, kernel_size=3):
        """Compute local sign agreement using convolution."""
        W = W_binary.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Average filter (excludes center)
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2 - 1)
        kernel[0, 0, kernel_size//2, kernel_size//2] = 0

        # Pad and convolve
        pad = kernel_size // 2
        W_padded = F.pad(W, (pad, pad, pad, pad), mode='reflect')
        neighbor_avg = F.conv2d(W_padded, kernel)

        # Agreement: sign matches neighbor direction
        agreement = (W_binary * neighbor_avg.squeeze()).abs()
        return agreement

    def sign_texture_forward(x, W_binary, temperature=2.0):
        """Forward with importance from sign texture."""
        agreement = compute_sign_agreement(W_binary)
        # Normalize to importance
        importance = agreement / (agreement.mean() + 1e-8)
        importance = torch.clamp(importance, 0.1, 3.0)

        W_effective = W_binary * importance
        return x @ W_effective.T

    # Train with sign-texture importance
    W_latent = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
    scale = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([W_latent, scale], lr=0.01)

    for epoch in range(1500):
        optimizer.zero_grad()

        W_binary = ste_sign(W_latent)
        W_binary = torch.where(W_binary == 0, torch.ones_like(W_binary), W_binary)

        Y_pred = sign_texture_forward(X_train, W_binary * scale)
        loss = F.mse_loss(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_binary = torch.sign(W_latent)
        W_binary = torch.where(W_binary == 0, torch.ones_like(W_binary), W_binary)
        Y_pred = sign_texture_forward(X_test, W_binary * scale)

    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    results['SignTexture'] = (corr, 1.0)
    print(f"SignTexture Importance: corr={corr:.4f}, BPP=1.00")

    # =========================================================================
    # APPROACH 3: Output-Weighted Binary
    # =========================================================================
    print("\n" + "=" * 60)
    print("APPROACH 3: Output-Weighted Binary")
    print("Weight outputs by their magnitude (emergent importance)")
    print("=" * 60)

    def output_weighted_forward(x, W_binary, scale):
        """Weight outputs by their own magnitude."""
        y_raw = x @ (W_binary * scale).T

        # Per-output importance from batch statistics
        output_mag = y_raw.abs().mean(dim=0)  # [d_out]
        importance = output_mag / (output_mag.mean() + 1e-8)
        importance = torch.clamp(importance, 0.3, 2.0)

        return y_raw * importance

    W_latent = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
    scale = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([W_latent, scale], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()

        W_binary = ste_sign(W_latent)
        W_binary = torch.where(W_binary == 0, torch.ones_like(W_binary), W_binary)

        Y_pred = output_weighted_forward(X_train, W_binary, scale)
        loss = F.mse_loss(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_binary = torch.sign(W_latent)
        W_binary = torch.where(W_binary == 0, torch.ones_like(W_binary), W_binary)
        Y_pred = output_weighted_forward(X_test, W_binary, scale)

    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    results['OutputWeighted'] = (corr, 1.0)
    print(f"Output-Weighted Binary: corr={corr:.4f}, BPP=1.00")

    # =========================================================================
    # APPROACH 4: Binary Superposition
    # =========================================================================
    print("\n" + "=" * 60)
    print("APPROACH 4: Binary Superposition")
    print("One binary matrix, multiple views via deterministic masks")
    print("=" * 60)

    class BinarySuperposition(nn.Module):
        def __init__(self, out_f, in_f, n_views=4):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.n_views = n_views

            # Deterministic view masks (regenerated, not stored)
            masks = []
            for v in range(n_views):
                torch.manual_seed(1000 + v)
                mask = torch.sign(torch.randn(out_f, in_f))
                masks.append(mask)
            self.register_buffer('masks', torch.stack(masks))

            # Learnable view weights
            self.view_weights = nn.Parameter(torch.ones(n_views) / n_views)

        def forward(self, x):
            W_binary = ste_sign(self.weight)
            W_binary = torch.where(W_binary == 0, torch.ones_like(W_binary), W_binary)

            # Combine views
            outputs = []
            for v in range(self.n_views):
                W_view = W_binary * self.masks[v]
                y_v = x @ (W_view * self.scale).T
                outputs.append(y_v)

            # Weighted sum of views
            view_w = F.softmax(self.view_weights, dim=0)
            y = sum(view_w[v] * outputs[v] for v in range(self.n_views))
            return y

    model = BinarySuperposition(d_out, d_in, n_views=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = F.mse_loss(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Y_pred = model(X_test)

    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    # BPP: 1 bit per weight + view weights (negligible)
    results['Superposition 4-view'] = (corr, 1.0)
    print(f"Superposition (4 views): corr={corr:.4f}, BPP=1.00")

    # =========================================================================
    # APPROACH 5: Learned Binary Basis
    # =========================================================================
    print("\n" + "=" * 60)
    print("APPROACH 5: Learned Binary Basis")
    print("Shared basis vectors + 1-bit coefficients")
    print("=" * 60)

    class LearnedBinaryBasis(nn.Module):
        def __init__(self, out_f, in_f, n_basis=4):
            super().__init__()
            # Shared basis (small overhead)
            self.basis = nn.Parameter(torch.randn(n_basis, in_f) * 0.1)
            # Per-output coefficients (binary: which basis, with sign)
            self.coefficients = nn.Parameter(torch.randn(out_f, n_basis) * 0.1)
            self.n_basis = n_basis

        def forward(self, x):
            # Binarize coefficients
            coef_binary = ste_sign(self.coefficients)
            coef_binary = torch.where(coef_binary == 0, torch.ones_like(coef_binary), coef_binary)

            # Construct weight matrix: W[i] = sum_k coef[i,k] * basis[k]
            W = coef_binary @ self.basis  # [out, in]

            return x @ W.T

        def bits_per_param(self, out_f, in_f):
            # 1 bit per coefficient + 32 bits per basis element
            n_coef_bits = out_f * self.n_basis
            n_basis_bits = self.n_basis * in_f * 32
            total_bits = n_coef_bits + n_basis_bits
            n_effective_weights = out_f * in_f
            return total_bits / n_effective_weights

    for n_basis in [2, 4, 8]:
        model = LearnedBinaryBasis(d_out, d_in, n_basis=n_basis)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1000):
            optimizer.zero_grad()
            Y_pred = model(X_train)
            loss = F.mse_loss(Y_pred, Y_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            Y_pred = model(X_test)

        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        bpp = model.bits_per_param(d_out, d_in)
        results[f'BinaryBasis k={n_basis}'] = (corr, bpp)
        print(f"Binary Basis (k={n_basis}): corr={corr:.4f}, BPP={bpp:.3f}")

    # =========================================================================
    # Baselines for comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)

    # Binary baseline
    W_latent = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
    scale = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([W_latent, scale], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        W_binary = ste_sign(W_latent)
        W_binary = torch.where(W_binary == 0, torch.ones_like(W_binary), W_binary)
        Y_pred = X_train @ (W_binary * scale).T
        loss = F.mse_loss(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_binary = torch.sign(W_latent)
        W_binary = torch.where(W_binary == 0, torch.ones_like(W_binary), W_binary)
        Y_pred = X_test @ (W_binary * scale).T

    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    results['Binary baseline'] = (corr, 1.0)
    print(f"Binary baseline: corr={corr:.4f}, BPP=1.00")

    # Ternary baseline
    W_latent = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
    scale = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([W_latent, scale], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()

        W_abs = torch.abs(W_latent)
        thresh = 0.3 * W_abs.mean()
        W_sign = ste_sign(W_latent)
        W_mask = (W_abs > thresh).float()
        W_tern = W_sign * W_mask

        Y_pred = X_train @ (W_tern * scale).T
        loss = F.mse_loss(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_abs = torch.abs(W_latent)
        thresh = 0.3 * W_abs.mean()
        W_sign = torch.sign(W_latent)
        W_mask = (W_abs > thresh).float()
        W_tern = W_sign * W_mask
        Y_pred = X_test @ (W_tern * scale).T

    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    results['Ternary baseline'] = (corr, 1.58)
    print(f"Ternary baseline: corr={corr:.4f}, BPP=1.58")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: First Principles Approaches")
    print("=" * 80)

    tern_corr = results['Ternary baseline'][0]
    bin_corr = results['Binary baseline'][0]

    print(f"\n{'Method':<30} {'Corr':>8} {'BPP':>8} {'vs Tern':>10} {'vs Bin':>10}")
    print("-" * 75)

    for method in sorted(results.keys(), key=lambda m: -results[m][0]):
        corr, bpp = results[method]
        vs_tern = ((corr / tern_corr) - 1) * 100
        vs_bin = ((corr / bin_corr) - 1) * 100

        # Stars for methods that beat baselines at ≤1.0 bpp
        if bpp <= 1.0 and corr > bin_corr:
            marker = "★"
            if corr > tern_corr:
                marker = "★★★ BEATS TERNARY!"
        else:
            marker = ""

        print(f"{method:<30} {corr:>8.4f} {bpp:>8.3f} {vs_tern:>+9.1f}% {vs_bin:>+9.1f}% {marker}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    best_sub1bpp = max([(m, c, b) for m, (c, b) in results.items() if b < 1.0],
                       key=lambda x: x[1], default=None)
    best_1bpp = max([(m, c, b) for m, (c, b) in results.items() if b <= 1.01 and 'baseline' not in m],
                    key=lambda x: x[1], default=None)

    if best_sub1bpp:
        print(f"\nBest sub-1.0 bpp: {best_sub1bpp[0]} at {best_sub1bpp[2]:.3f} bpp, corr={best_sub1bpp[1]:.4f}")

    if best_1bpp:
        print(f"Best at ~1.0 bpp: {best_1bpp[0]} at {best_1bpp[2]:.3f} bpp, corr={best_1bpp[1]:.4f}")
        if best_1bpp[1] > tern_corr:
            print("\n🎉 BREAKTHROUGH: Beats ternary at 1.0 bpp!")

    return results


if __name__ == "__main__":
    test_first_principles_approaches()

