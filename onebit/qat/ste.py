"""Straight-Through Estimator (STE) primitives for KD-QAT.

All default ops avoid custom autograd graph breaks where possible so they work
cleanly under ``torch.compile`` and ``bf16``/``fp16`` autocast.
"""
from __future__ import annotations

from typing import Optional

import torch


class _SignClippedSTE(torch.autograd.Function):
    """Sign forward (0 -> +1) with gradient clipped to |x| <= grad_clip."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, grad_clip: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.grad_clip = grad_clip
        s = torch.sign(x)
        return torch.where(s == 0, torch.ones_like(s), s)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        mask = (x.abs() <= ctx.grad_clip).to(grad_output.dtype)
        return grad_output * mask, None


def sign_ste(x: torch.Tensor, grad_clip: Optional[float] = None) -> torch.Tensor:
    """Binarize to {-1, +1} (mapping 0 -> +1) with straight-through gradient.

    Args:
        x: Latent float tensor.
        grad_clip: If provided, zero gradients where ``|x| > grad_clip``
            (HardTanh-style STE). If ``None``, pass full identity gradient.
    """
    if grad_clip is not None:
        return _SignClippedSTE.apply(x, float(grad_clip))
    s = torch.sign(x)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return x + (s - x).detach()


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest integer with identity straight-through gradient."""
    return x + (torch.round(x) - x).detach()


def clamp_round_ste(x: torch.Tensor, qmin: float, qmax: float) -> torch.Tensor:
    """Clamp to ``[qmin, qmax]`` and round with LSQ/BitNet-style clamped STE.

    Forward produces integers in ``[qmin, qmax]``. Backward passes gradient
    through inside ``[qmin, qmax]`` and zeroes it strictly outside.
    """
    xc = torch.clamp(x, qmin, qmax)
    return xc + (torch.round(xc) - xc).detach()


def lsq_grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Rescale gradients of ``x`` by ``scale`` while keeping forward value identical.

    Used in LSQ / ParetoQ to balance learnable step-size gradients against
    weight gradients: ``scale = 1 / sqrt(n_elements_per_scale * qmax)``.
    """
    return x * scale + (x - x * scale).detach()
