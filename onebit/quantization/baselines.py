"""Reference (correctly implemented) baseline weight quantizers.

All functions take a float ``(out_features, in_features)`` weight and return
``(dequantized_weight, total_storage_bits)``.  Storage bits include every scale,
zero-point and other side information, counted at 16 bits per FP value.

These are the bars any SALOMI method has to beat at equal bits.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

FP_META_BITS = 16  # scales / zero-points stored as fp16


def _grouped(w: torch.Tensor, group: Optional[int]) -> Tuple[torch.Tensor, int]:
    out, inn = w.shape
    g = inn if not group else group
    if inn % g != 0:
        raise ValueError(f"in_features={inn} not divisible by group={g}")
    return w.reshape(out, inn // g, g), g


def rtn(w: torch.Tensor, bits: int, group: Optional[int] = 128, sym: bool = False) -> Tuple[torch.Tensor, float]:
    """Round-to-nearest uniform quantization with per-row (per-group) min-max scales.

    ``group=None`` means one scale (and zero-point) per output channel.
    Asymmetric: 2**bits levels, fp16 scale + fp16 zero-point per group.
    Symmetric: levels -(2**(b-1)-1) .. 2**(b-1)-1, fp16 scale per group.
    """
    wg, g = _grouped(w.float(), group)
    if sym:
        qmax = 2 ** (bits - 1) - 1
        scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        wq = torch.clamp(torch.round(wg / scale), -qmax, qmax) * scale
        meta_per_group = FP_META_BITS
    else:
        qmax = 2 ** bits - 1
        lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
        scale = (hi - lo).clamp(min=1e-8) / qmax
        zero = torch.round(-lo / scale)
        wq = (torch.clamp(torch.round(wg / scale) + zero, 0, qmax) - zero) * scale
        meta_per_group = 2 * FP_META_BITS
    n_groups = wg.shape[0] * wg.shape[1]
    total_bits = w.numel() * bits + n_groups * meta_per_group
    return wq.reshape_as(w), float(total_bits)


def ternary_absmean(w: torch.Tensor, group: Optional[int] = None) -> Tuple[torch.Tensor, float]:
    """BitNet-b1.58-style ternary: scale = mean|w| per row/group, q = clip(round(w/scale), -1, 1).

    Storage: 1.6 bits/weight (5 trits packed per byte, the practical packing for
    log2(3)=1.585) + one fp16 scale per group.
    """
    wg, g = _grouped(w.float(), group)
    scale = wg.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    wq = torch.clamp(torch.round(wg / scale), -1, 1) * scale
    n_groups = wg.shape[0] * wg.shape[1]
    total_bits = w.numel() * 1.6 + n_groups * FP_META_BITS
    return wq.reshape_as(w), float(total_bits)


def ternary_twn(w: torch.Tensor, group: Optional[int] = None) -> Tuple[torch.Tensor, float]:
    """Ternary Weight Networks (Li et al., 2016) post-training ternarization.

    threshold = 0.7 * mean|w|; scale = mean of |w| over the kept (non-zero) weights, per row/group.
    Unlike ``ternary_absmean`` (the BitNet b1.58 *training* rule, which assigns every non-zero the
    overall mean|w| and so shrinks large weights), this is the approximately MSE-optimal ternary
    and therefore the fair post-training ternary baseline. Same storage: 1.6 bits/weight + fp16 scale.
    """
    wg, g = _grouped(w.float(), group)
    a = wg.abs()
    delta = 0.7 * a.mean(-1, keepdim=True)
    mask = (a > delta).float()
    scale = (a * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1.0)
    wq = torch.sign(wg) * mask * scale
    n_groups = wg.shape[0] * wg.shape[1]
    total_bits = w.numel() * 1.6 + n_groups * FP_META_BITS
    return wq.reshape_as(w), float(total_bits)


def binary_absmean(w: torch.Tensor, group: Optional[int] = None) -> Tuple[torch.Tensor, float]:
    """sign(w) * mean|w| per output row (or per group). 1 bit/weight + fp16 scale per group."""
    wg, g = _grouped(w.float(), group)
    scale = wg.abs().mean(-1, keepdim=True)
    s = torch.sign(wg)
    s[s == 0] = 1
    n_groups = wg.shape[0] * wg.shape[1]
    total_bits = w.numel() * 1.0 + n_groups * FP_META_BITS
    return (s * scale).reshape_as(w), float(total_bits)


def binary_svid(w: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """OneBit-style post-training init: W ~= diag(a) sign(W) diag(b), (a, b) = rank-1 SVD of |W|.

    1 bit/weight + fp16 row vector + fp16 column vector. This is the *starting point*
    for KD-QAT; on its own (no training) it is not expected to be usable.
    """
    w = w.float()
    u, s, vh = torch.linalg.svd(w.abs(), full_matrices=False)
    a = (u[:, 0] * s[0].sqrt()).abs()
    b = (vh[0] * s[0].sqrt()).abs()
    sign = torch.sign(w)
    sign[sign == 0] = 1
    out, inn = w.shape
    total_bits = w.numel() + FP_META_BITS * (out + inn)
    return a[:, None] * sign * b[None, :], float(total_bits)


def ideal_bits_note() -> str:
    return (
        "Ternary is stored at 1.6 bits (5 trits/byte); the information-theoretic "
        f"minimum is log2(3)={math.log2(3):.4f}."
    )
