"""Ternary + sparse-residual quantizer for Proxy-SR-VQ.

Used as the "low-R" fallback: blocks that are highly redundant (easy to
compress) get this cheaper quantiser instead of the heavier HessianVQ.

    W_q = ternary_base(W, sparsity)  +  sparse_topk_correction(W - W_tern)

The ternary base uses {-1, 0, +1} * scale with a magnitude threshold
that zeros out the smallest ``sparsity`` fraction.  The sparse top-k
residual stores the ``residual_top_k`` largest absolute errors at
``residual_bits`` precision.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class TernarySparse:
    """Ternary base + sparse residual quantizer.

    Matches the ``HessianVQ.quantize(W, H_diag)`` interface so the
    DynamicAllocator can swap them transparently.
    """

    def __init__(
        self,
        sparsity: float = 0.3,
        residual_top_k: float = 0.01,
        residual_bits: int = 8,
    ):
        self.sparsity = sparsity
        self.residual_top_k = residual_top_k
        self.residual_bits = residual_bits

        self._ternary_scale: float = 0.0
        self._n_sparse: int = 0
        self._n_weights: int = 0

    def quantize(self, W: np.ndarray, H_diag: Optional[np.ndarray] = None) -> np.ndarray:
        """Quantize *W* with ternary + sparse residual.

        Args:
            W: (d_out, d_in) weight matrix.
            H_diag: ignored (accepted for interface compatibility).

        Returns:
            W_recon: (d_out, d_in) reconstructed weight matrix.
        """
        self._n_weights = W.size
        S = np.sign(W)
        S[S == 0] = 1.0

        thresh = np.percentile(np.abs(W), self.sparsity * 100)
        mask = np.abs(W) > thresh
        active = W[mask]
        self._ternary_scale = float(np.mean(np.abs(active))) if active.size else 1.0
        W_tern = S * mask.astype(np.float32) * self._ternary_scale

        residual = W - W_tern
        n_topk = max(1, int(self.residual_top_k * W.size))
        self._n_sparse = n_topk
        flat_res = residual.flatten()
        topk_idx = np.argpartition(np.abs(flat_res), -n_topk)[-n_topk:]

        n_levels = (1 << (self.residual_bits - 1)) - 1
        max_val = np.max(np.abs(flat_res[topk_idx])) + 1e-10
        quant_vals = np.round(flat_res[topk_idx] / max_val * n_levels) / n_levels * max_val

        sparse_correction = np.zeros_like(flat_res)
        sparse_correction[topk_idx] = quant_vals

        W_recon = W_tern + sparse_correction.reshape(W.shape)
        return W_recon

    def effective_bpp(self, n_weights: Optional[int] = None) -> float:
        """Strict BPP accounting for ternary + sparse residual."""
        n = n_weights or self._n_weights
        if n == 0:
            return 0.0
        tern_bits = np.log2(3) * n
        sparse_index_bits = self._n_sparse * np.log2(max(n, 2))
        sparse_val_bits = self._n_sparse * self.residual_bits
        scale_bits = 32
        return (tern_bits + sparse_index_bits + sparse_val_bits + scale_bits) / n
