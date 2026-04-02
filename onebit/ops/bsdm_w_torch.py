from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    from torch.autograd import Function
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Function = object  # type: ignore

from .bsdm_w import SDConfig, bsdm_w_dot


class BSDMWDotFn(Function):  # type: ignore[misc]
    """Torch autograd wrapper: forward-only (no gradients)."""
    @staticmethod
    def forward(ctx, a_bits_t, b_bits_t, k: int, cfg: SDConfig, seed: int):  # type: ignore[override]
        if torch is None:
            raise ImportError("PyTorch not available for BSDMWDotFn")
        a_np = a_bits_t.detach().cpu().numpy()
        b_np = b_bits_t.detach().cpu().numpy()
        est, _ = bsdm_w_dot(a_np, b_np, int(k), cfg, int(seed))
        return torch.tensor(est, dtype=torch.float32, device=a_bits_t.device)


def bsdm_w_dot_torch(a_bits_t, b_bits_t, k: int, cfg: Optional[SDConfig] = None, seed: int = 0):
    if torch is None:
        raise ImportError("PyTorch not available for bsdm_w_dot_torch")
    if cfg is None:
        cfg = SDConfig()
    return BSDMWDotFn.apply(a_bits_t, b_bits_t, int(k), cfg, int(seed))

