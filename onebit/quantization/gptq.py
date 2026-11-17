"""GPTQ (Frantar et al., 2022, arXiv:2210.17323) for HF causal LMs.

Faithful to the reference implementation:
  * Full input Hessian H = 2/n * X^T X per linear layer, collected from calibration data
    flowing through the *already-quantized* preceding blocks (sequential quantization).
  * Dampening lambda = percdamp * mean(diag(H)).
  * Upper Cholesky factor of H^-1; columns quantized in order, quantization error
    propagated to the not-yet-quantized columns with the H^-1 rows (lazy batch updates).
  * Per-group asymmetric min-max scales recomputed on the error-compensated weights.

Replaces the earlier ``HessianVQ._gptq_refine`` which was not GPTQ.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from onebit.eval.harness import (
    BitReport,
    block_linear_modules,
    get_weight,
    set_weight,
    transformer_blocks,
    unique_param_count,
)
from onebit.quantization.baselines import FP_META_BITS


class _MinMaxQuantizer:
    def __init__(self, bits: int):
        self.qmax = 2 ** bits - 1

    def fit(self, w: torch.Tensor) -> None:  # w: (out, cols)
        lo = torch.clamp(w.amin(1), max=0)
        hi = torch.clamp(w.amax(1), min=0)
        same = (hi - lo) < 1e-12
        lo[same], hi[same] = -1, 1
        self.scale = ((hi - lo) / self.qmax).unsqueeze(1)
        self.zero = torch.round(-lo / self.scale.squeeze(1)).unsqueeze(1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # x: (out, 1)
        q = torch.clamp(torch.round(x / self.scale) + self.zero, 0, self.qmax)
        return self.scale * (q - self.zero)


def gptq_layer(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    group_size: Optional[int] = 128,
    blocksize: int = 128,
    percdamp: float = 0.01,
    act_order: bool = False,
) -> torch.Tensor:
    """Quantize one ``(out, in)`` weight with GPTQ. Returns the dequantized weight."""
    W = W.clone().float()
    H = H.clone().float()
    out, cols = W.shape
    quant = _MinMaxQuantizer(bits)
    if not group_size:
        quant.fit(W)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]

    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(cols, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            if group_size and (i1 + i) % group_size == 0:
                quant.fit(W[:, i1 + i: i1 + i + group_size])
            q = quant(w.unsqueeze(1)).flatten()
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q


class _StopForward(Exception):
    pass


@torch.no_grad()
def gptq_quantize_model(
    model: nn.Module,
    calib: torch.Tensor,
    bits: int = 4,
    group_size: Optional[int] = 128,
    batch_size: int = 8,
    percdamp: float = 0.01,
    act_order: bool = False,
) -> BitReport:
    """Sequential (block-by-block) GPTQ over all transformer-block linears, in place."""
    device = next(model.parameters()).device
    report = BitReport(total_params=unique_param_count(model))
    blocks = transformer_blocks(model)
    all_linears = dict(block_linear_modules(model))

    for bi, block in enumerate(blocks):
        in_block = {id(m) for m in block.modules()}
        layer_names = [n for n, m in all_linears.items() if id(m) in in_block]
        H: Dict[str, torch.Tensor] = {}
        nsamp: Dict[str, int] = {}
        hooks = []
        for name in layer_names:
            mod = all_linears[name]
            cols = get_weight(mod).shape[1]
            H[name] = torch.zeros(cols, cols, device=device)
            nsamp[name] = 0

            def hook(m, inp, out, name=name):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                n = x.shape[0]
                H[name] *= nsamp[name] / (nsamp[name] + n)
                nsamp[name] += n
                x = math.sqrt(2 / nsamp[name]) * x
                H[name] += x.t() @ x

            hooks.append(mod.register_forward_hook(hook))

        def stop(m, inp, out):
            raise _StopForward

        hooks.append(block.register_forward_hook(stop))
        try:
            for i in range(0, calib.shape[0], batch_size):
                try:
                    model(calib[i: i + batch_size].to(device))
                except _StopForward:
                    pass
        finally:
            for h in hooks:
                h.remove()

        for name in layer_names:
            mod = all_linears[name]
            W = get_weight(mod).to(device)
            Q = gptq_layer(W, H[name], bits, group_size, percdamp=percdamp, act_order=act_order)
            set_weight(mod, Q)
            out, inn = W.shape
            n_groups = out * (inn // group_size if group_size else 1)
            report.add(name, W.numel(), W.numel() * bits + n_groups * 2 * FP_META_BITS)
        del H
        print(f"[gptq] block {bi + 1}/{len(blocks)} done ({len(layer_names)} linears)", flush=True)
    return report
