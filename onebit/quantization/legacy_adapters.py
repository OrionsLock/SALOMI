"""Adapters that run SALOMI's legacy numpy quantizers through the shared harness.

Configurations mirror ``tests/test_improvements.py`` PART 2 (the source of the
historical headline numbers), but receive a *real* calibration Hessian diagonal
(``mean(x**2)`` over 128x512 WikiText-2 train tokens) instead of all-ones, unless
the harness is run with ``use_calibration_hessian=False``.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Tuple

import numpy as np
import torch

from onebit.quantization.hessian_vq import HessianVQ
from onebit.quantization.lowrank_residual import LowRankResidual

_LAYER_RE = re.compile(r"\.(\d+)\.")


def _np(w: torch.Tensor) -> np.ndarray:
    return w.detach().cpu().numpy().astype(np.float64)


def _h(ctx: Dict, d_in: int) -> np.ndarray:
    h = ctx.get("h_diag")
    if h is None:
        return np.ones(d_in, dtype=np.float32)
    return h.detach().cpu().numpy().astype(np.float32)


def _hvq(w: torch.Tensor, ctx: Dict, n_codes: int = 64) -> Tuple[torch.Tensor, float]:
    np.random.seed(ctx.get("seed", 0))
    q = HessianVQ(n_codes=n_codes, block_size=4, max_iter=15, use_hessian_weight=True)
    out = q.quantize(_np(w), _h(ctx, w.shape[1]))
    return torch.tensor(out, dtype=torch.float32), q.effective_bpp(w.numel()) * w.numel()


def _lowrank(w: torch.Tensor, ctx: Dict, rank: int) -> Tuple[torch.Tensor, float]:
    q = LowRankResidual(rank=rank, factor_bits=8)
    out = q.quantize(_np(w), _h(ctx, w.shape[1]))
    return torch.tensor(out, dtype=torch.float32), q.effective_bpp() * w.numel()


def _mixed(w: torch.Tensor, ctx: Dict) -> Tuple[torch.Tensor, float]:
    """The 'Mixed (L0/11 protected)' config from tests/test_improvements.py."""
    name = ctx["name"]
    layer = int(_LAYER_RE.search(name).group(1))
    is_mlp = ".mlp." in name
    if layer in (0, 11):
        return _lowrank(w, ctx, 12 if is_mlp else 8)
    if is_mlp:
        return _lowrank(w, ctx, 6)
    return _hvq(w, ctx)


_REGISTRY: Dict[str, Callable] = {
    "legacy_hvq_k64": _hvq,
    "legacy_lowrank_r8": lambda w, c: _lowrank(w, c, 8),
    "legacy_mixed": _mixed,
}


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown legacy method {name}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name]
