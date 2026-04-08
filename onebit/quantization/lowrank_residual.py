"""Binary + quantised low-rank residual correction.

The idea: W ≈ sign(W) * scale  +  U @ V^T
where U, V are low-rank factors that can optionally be stored in INT8
instead of FP32, drastically reducing the bpp overhead while keeping
most of the correction quality.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any


class LowRankResidual:
    """Binary base + quantised low-rank residual.

    At rank *r* with FP32 factors on a (d_out, d_in) matrix the overhead
    is  r*(d_out + d_in)*32  bits.  Quantising factors to INT8 cuts that
    by 4x, allowing higher rank at the same bpp budget.
    """

    def __init__(
        self,
        rank: int = 4,
        factor_bits: int = 8,
        hessian_rank_alloc: bool = True,
    ):
        self.rank = rank
        self.factor_bits = factor_bits
        self.hessian_rank_alloc = hessian_rank_alloc

        self.signs: Optional[np.ndarray] = None
        self.scale: float = 0.0
        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self._d_out = 0
        self._d_in = 0

    def _quantize_factors(self, M: np.ndarray) -> np.ndarray:
        """Symmetric INT-N quantization of low-rank factors."""
        if self.factor_bits >= 32:
            return M.astype(np.float32)
        max_val = np.max(np.abs(M)) + 1e-10
        n_levels = (1 << (self.factor_bits - 1)) - 1
        scaled = np.clip(np.round(M / max_val * n_levels), -n_levels, n_levels)
        return (scaled / n_levels * max_val).astype(np.float32)

    def quantize(
        self,
        W: np.ndarray,
        H_diag: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Quantize W = sign*scale + U@V^T.

        Args:
            W: (d_out, d_in) weight matrix.
            H_diag: (d_in,) Hessian diagonal for optional weighting.

        Returns:
            W_recon of same shape.
        """
        self._d_out, self._d_in = W.shape

        S = np.sign(W)
        S[S == 0] = 1.0
        self.signs = S
        self.scale = float(np.mean(np.abs(W)))

        W_binary = S * self.scale
        residual = W - W_binary

        if H_diag is not None and self.hessian_rank_alloc:
            h_weight = np.sqrt(np.clip(H_diag, 1e-12, None))
            h_weight = h_weight / (h_weight.mean() + 1e-10)
            residual_weighted = residual * h_weight[None, :]
        else:
            residual_weighted = residual
            h_weight = None

        U_full, sigma, Vt_full = np.linalg.svd(residual_weighted, full_matrices=False)
        r = min(self.rank, len(sigma))

        U_r = U_full[:, :r] * sigma[:r][None, :]
        V_r = Vt_full[:r, :]

        if h_weight is not None:
            V_r = V_r / h_weight[None, :]

        self.U = self._quantize_factors(U_r)
        self.V = self._quantize_factors(V_r)

        W_recon = W_binary + self.U @ self.V
        return W_recon

    def effective_bpp(self) -> float:
        """Strict BPP accounting."""
        n_weights = self._d_out * self._d_in
        sign_bits = n_weights
        scale_bits = 32
        r = self.U.shape[1] if self.U is not None else 0
        factor_bits = (self._d_out * r + r * self._d_in) * self.factor_bits
        return (sign_bits + scale_bits + factor_bits) / n_weights

    def get_components(self) -> Dict[str, Any]:
        return {
            "signs": self.signs,
            "scale": self.scale,
            "U": self.U,
            "V": self.V,
            "rank": self.rank,
            "factor_bits": self.factor_bits,
            "bpp": self.effective_bpp(),
        }


class ResidualHessianVQ:
    """Two-stage VQ: coarse HessianVQ + residual HessianVQ.

    Stage 1 captures the bulk of the magnitude distribution.
    Stage 2 fits the reconstruction error from stage 1.
    The residual can be negative, so we VQ the *signed* residual
    (using sign + magnitude decomposition internally for each stage).
    """

    def __init__(
        self,
        n_codes_coarse: int = 64,
        n_codes_fine: int = 32,
        block_size: int = 4,
        max_iter: int = 25,
    ):
        self.n_codes_coarse = n_codes_coarse
        self.n_codes_fine = n_codes_fine
        self.block_size = block_size
        self.max_iter = max_iter

        from .hessian_vq import HessianVQ
        self.vq1 = HessianVQ(
            n_codes=n_codes_coarse,
            block_size=block_size,
            max_iter=max_iter,
            use_hessian_weight=True,
        )
        self.vq2 = HessianVQ(
            n_codes=n_codes_fine,
            block_size=block_size,
            max_iter=max_iter,
            use_hessian_weight=True,
        )
        self._d_out = 0
        self._d_in = 0

    def quantize(self, W: np.ndarray, H_diag: np.ndarray) -> np.ndarray:
        """Two-stage quantize.

        Args:
            W: (d_out, d_in) weight matrix.
            H_diag: (d_in,) Hessian diagonal.

        Returns:
            W_recon: (d_out, d_in) reconstructed weights.
        """
        self._d_out, self._d_in = W.shape

        W_stage1 = self.vq1.quantize(W, H_diag)

        residual = W - W_stage1
        W_stage2 = self.vq2.quantize(residual, H_diag)

        return W_stage1 + W_stage2

    def effective_bpp(self) -> float:
        n_weights = self._d_out * self._d_in
        bpp1 = self.vq1.effective_bpp(n_weights)
        bpp2 = self.vq2.effective_bpp(n_weights)
        return bpp1 + bpp2 - 1.0  # signs shared between stages
