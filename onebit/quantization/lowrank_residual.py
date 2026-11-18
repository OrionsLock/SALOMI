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

    Factor quantization (``factor_bits < 32``) is symmetric INT-N with one
    fp16 scale per rank component: per column of ``U`` (d_out x r) and per
    row of ``V`` (r x d_in), i.e. ``2 * r * 16`` extra bits, counted in
    :meth:`effective_bpp`.  (Earlier versions used a single global max-abs
    scale per factor matrix.)

    Binary base scale: by default one fp32 scale ``mean|W|`` for the whole
    tensor (what the historical numbers used).  With ``per_row_scale=True``
    one fp16 ``mean|W_row|`` per output row is used instead (``d_out * 16``
    bits).
    """

    def __init__(
        self,
        rank: int = 4,
        factor_bits: int = 8,
        hessian_rank_alloc: bool = True,
        per_row_scale: bool = False,
    ):
        self.rank = rank
        self.factor_bits = factor_bits
        self.hessian_rank_alloc = hessian_rank_alloc
        self.per_row_scale = per_row_scale

        self.signs: Optional[np.ndarray] = None
        self.scale: Any = 0.0  # float, or (d_out, 1) array when per_row_scale
        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self._d_out = 0
        self._d_in = 0

    def _quantize_factors(self, M: np.ndarray, axis: int = 0) -> np.ndarray:
        """Symmetric INT-N quantization of a low-rank factor.

        One scale per rank component: ``axis`` is the axis reduced to compute
        the max-abs scale (``axis=0`` -> per column, used for U (d_out, r);
        ``axis=1`` -> per row, used for V (r, d_in)).
        """
        if self.factor_bits >= 32:
            return M.astype(np.float32)
        max_val = np.max(np.abs(M), axis=axis, keepdims=True) + 1e-10
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
        if self.per_row_scale:
            self.scale = np.mean(np.abs(W), axis=1, keepdims=True)
        else:
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

        self.U = self._quantize_factors(U_r, axis=0)  # one scale per rank column of U
        self.V = self._quantize_factors(V_r, axis=1)  # one scale per rank row of V

        W_recon = W_binary + self.U @ self.V
        return W_recon

    def effective_bpp(self) -> float:
        """Strict BPP accounting.

        sign bits (1/weight)
        + binary scale (32 bits per-tensor, or d_out * 16 with ``per_row_scale``)
        + factor values (r * (d_out + d_in) * factor_bits)
        + factor scales (2 * r * 16 bits when factor_bits < 32).
        """
        n_weights = self._d_out * self._d_in
        sign_bits = n_weights
        scale_bits = self._d_out * 16 if self.per_row_scale else 32
        r = self.U.shape[1] if self.U is not None else 0
        factor_bits = (self._d_out * r + r * self._d_in) * self.factor_bits
        factor_scale_bits = 2 * r * 16 if self.factor_bits < 32 else 0
        return (sign_bits + scale_bits + factor_bits + factor_scale_bits) / n_weights

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
        # Signs are NOT shared: stage 2 quantizes sign(residual), which is a new,
        # independent sign plane, so both stages' sign bits are counted.
        return bpp1 + bpp2
