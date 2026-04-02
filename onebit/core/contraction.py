from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .packbits import pack_signs_rowmajor


def _unpack_signs_rowmajor(bits: np.ndarray, K: int) -> np.ndarray:
    """Unpack [M, Kw] uint32 into float32 signs [+1,-1] of shape [M, K]."""
    M, Kw = bits.shape
    out = np.empty((M, K), dtype=np.float32)
    for w in range(Kw):
        start = w * 32
        end = min(start + 32, K)
        word = bits[:, w][:, None].astype(np.uint32)
        for b in range(end - start):
            mask = ((word >> np.uint32(b)) & np.uint32(1)).astype(bool)[:, 0]
            out[:, start + b] = np.where(mask, 1.0, -1.0)
    return out


def hutch_pp_norm_estimator(
    W_bits: np.ndarray,
    sample_fn: Callable[[int, int], np.ndarray],
    probes: int = 32,
    blocks: int = 1,
) -> float:
    """Estimate a spectral norm upper bound using max over random sign probes.

    This is a lightweight surrogate for Hutch++: for probes s_i in {±1}^K,
    take max_i ||W s_i||_2. Monotone in `probes` by construction.
    """
    M, Kw = W_bits.shape
    K = Kw * 32
    W = _unpack_signs_rowmajor(W_bits, K)
    best = 0.0
    for i in range(probes):
        s = sample_fn(K, i).astype(np.float32)
        y = W @ s
        val = float(np.linalg.norm(y))
        if val > best:
            best = val
    # Optional block scaling (placeholder): expand bound by sqrt(blocks)
    return best * (blocks ** 0.5)


def choose_gamma(kappa_hat: float, L: int, delta_global: float = 0.05) -> float:
    """Choose a contraction gamma in (0,1) that shrinks with L.

    We use a simple heuristic: gamma = 1 / (1 + kappa_hat * log(1+L)).
    Clipped to [0.01, 0.99].
    """
    if L <= 0:
        return 0.99
    gamma = 1.0 / (1.0 + float(kappa_hat) * math.log1p(float(L)))
    gamma = float(min(0.99, max(0.01, gamma)))
    return gamma


def apply_block_rescale(y_block: np.ndarray, gamma: float) -> np.ndarray:
    """Apply runtime rescaling to a block output (ephemeral)."""
    return (np.asarray(y_block, dtype=np.float32) * float(gamma)).astype(np.float32)

