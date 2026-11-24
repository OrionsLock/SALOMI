"""Quantization-Aware Training (QAT) modules for ~1-bit, ternary, and low-bit embeddings.

Every quantized module exposes exact storage-bit accounting matching
``onebit/quantization/baselines.py`` and ``onebit/eval/harness.py``:
  * Ternary weights: 1.6 bits/weight (5 trits packed per byte) + FP16 scales.
  * Binary weights: 1.0 bit/weight + FP16 scale/shift vectors.
  * INTb embeddings: b bits/weight + FP16 per-row/group scale (and zero-point if asymmetric).
  * Original weights are stored in ``(out_features, in_features)`` layout even when
    converted from Hugging Face GPT-2 ``Conv1D`` (which stores ``(in, out)``), so
    standard ``F.linear`` cuBLAS GEMMs are used on GPU.
"""
from __future__ import annotations

import math
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from onebit.eval.harness import get_weight
from onebit.qat.ste import clamp_round_ste, lsq_grad_scale, sign_ste

FP_META_BITS = 16
TERNARY_BITS = 1.6  # 5 trits/byte practical packing for log2(3) ~= 1.585


def _resolve_groups(in_features: int, group_size: Optional[int]) -> Tuple[int, int]:
    """Return ``(n_groups, effective_group_size)`` and validate divisibility."""
    g = in_features if not group_size or group_size <= 0 else int(group_size)
    if in_features % g != 0:
        raise ValueError(f"in_features={in_features} is not divisible by group_size={g}")
    return in_features // g, g


class QuantizedModule(nn.Module):
    """Base interface for all Phase 1 QAT modules with exact bit accounting."""

    quant_kind: str = "linear"  # "linear" or "embedding"

    def orig_num_params(self) -> int:
        raise NotImplementedError

    def storage_bits(self) -> float:
        raise NotImplementedError

    def latent_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError

    def scale_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError

    def meta_num_params(self) -> int:
        """Number of extra quantizer metadata parameters (scales/shifts) not in the base FP model."""
        return sum(p.numel() for p in self.scale_parameters())


# ---------------------------------------------------------------------------
# Ternary Linear (BitNet b1.58 / ParetoQ-style STE with learnable scales)
# ---------------------------------------------------------------------------


class TernaryLinear(QuantizedModule):
    """Ternary {-1, 0, +1} linear layer with learnable per-channel or per-group scales.

    Forward weight quantization::

        W_g = W_lat.reshape(out, n_groups, g)
        W_q = clamp(round((W_g - shift) / s), -1, 1) * s + shift

    By default ``shift`` is disabled (symmetric ternary, BitNet b1.58 / ParetoQ)
    and ``s`` is initialized using the MSE-optimal TWN rule (threshold 0.7 * mean|W|,
    scale = mean of |W| over kept entries) so initial PPL is ~50x better than naive
    absmean initialization.
    """

    quant_kind = "linear"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: Optional[int] = None,
        learnable_scale: bool = True,
        use_shift: bool = False,
        lsq_scale_grad: bool = False,
        train_weights: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_groups, self.group_size_eff = _resolve_groups(self.in_features, group_size)
        self.group_size = None if (not group_size or group_size <= 0) else int(group_size)
        self.learnable_scale = bool(learnable_scale)
        self.use_shift = bool(use_shift)
        self.lsq_scale_grad = bool(lsq_scale_grad)
        self.eps = float(eps)

        self.w_lat = nn.Parameter(torch.empty(self.out_features, self.in_features), requires_grad=train_weights)
        if self.learnable_scale:
            self.scale = nn.Parameter(torch.ones(self.out_features, self.n_groups, 1))
        else:
            self.register_parameter("scale", None)

        if self.use_shift:
            self.shift = nn.Parameter(torch.zeros(self.out_features, self.n_groups, 1))
        else:
            self.register_parameter("shift", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.w_lat, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        self.init_scales_from_weight(method="twn")

    @torch.no_grad()
    def init_scales_from_weight(self, method: str = "twn") -> None:
        """Initialize learnable scale (and optional shift) from ``self.w_lat``."""
        wg = self.w_lat.detach().float().reshape(self.out_features, self.n_groups, self.group_size_eff)
        if self.use_shift and self.shift is not None:
            sh = wg.mean(dim=-1, keepdim=True)
            self.shift.data.copy_(sh.to(self.shift.dtype))
            wg = wg - sh

        if self.scale is None:
            return

        a = wg.abs()
        if method == "twn":
            delta = 0.7 * a.mean(dim=-1, keepdim=True)
            mask = (a > delta).float()
            s = (a * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            # Fall back to absmean if a group had zero active entries.
            fallback = a.mean(dim=-1, keepdim=True)
            s = torch.where(mask.sum(dim=-1, keepdim=True) > 0, s, fallback)
        elif method == "absmean":
            s = a.mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown ternary init_method: {method!r} (expected 'twn' or 'absmean')")

        self.scale.data.copy_(s.clamp(min=self.eps).to(self.scale.dtype))

    @classmethod
    def from_module(
        cls,
        mod: nn.Module,
        group_size: Optional[int] = None,
        init_method: str = "twn",
        learnable_scale: bool = True,
        use_shift: bool = False,
        lsq_scale_grad: bool = False,
        train_weights: bool = True,
    ) -> "TernaryLinear":
        """Convert an ``nn.Linear`` or HF GPT-2 ``Conv1D`` into a ``TernaryLinear``."""
        w = get_weight(mod)  # float32 (out_features, in_features)
        out_f, in_f = w.shape
        has_bias = getattr(mod, "bias", None) is not None
        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=has_bias,
            group_size=group_size,
            learnable_scale=learnable_scale,
            use_shift=use_shift,
            lsq_scale_grad=lsq_scale_grad,
            train_weights=train_weights,
        )
        with torch.no_grad():
            layer.w_lat.copy_(w)
            if has_bias and layer.bias is not None:
                layer.bias.copy_(mod.bias.detach().float())
        layer.init_scales_from_weight(method=init_method)
        return layer

    def quantized_weight(self) -> torch.Tensor:
        """Return differentiable fake-quantized ``(out_features, in_features)`` weight in fp32."""
        wg = self.w_lat.float().reshape(self.out_features, self.n_groups, self.group_size_eff)
        if self.shift is not None:
            sh = self.shift.float()
            if self.lsq_scale_grad:
                sh = lsq_grad_scale(sh, 1.0 / math.sqrt(self.group_size_eff))
            wg = wg - sh
        else:
            sh = None

        if self.scale is not None:
            s = self.scale.float().clamp(min=self.eps)
            if self.lsq_scale_grad:
                s = lsq_grad_scale(s, 1.0 / math.sqrt(self.group_size_eff))
        else:
            s = wg.detach().abs().mean(dim=-1, keepdim=True).clamp(min=self.eps)

        q = clamp_round_ste(wg / s, -1.0, 1.0)
        wq = q * s
        if sh is not None:
            wq = wq + sh
        return wq.reshape(self.out_features, self.in_features)

    @torch.no_grad()
    def dequantized_weight(self) -> torch.Tensor:
        return self.quantized_weight().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = self.quantized_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, wq, b)

    def orig_num_params(self) -> int:
        return self.out_features * self.in_features

    def storage_bits(self) -> float:
        n_groups_total = self.out_features * self.n_groups
        meta_per_group = FP_META_BITS * (2 if self.use_shift else 1)
        return float(self.orig_num_params() * TERNARY_BITS + n_groups_total * meta_per_group)

    def latent_parameters(self) -> Iterator[nn.Parameter]:
        yield self.w_lat

    def scale_parameters(self) -> Iterator[nn.Parameter]:
        if self.scale is not None:
            yield self.scale
        if self.shift is not None:
            yield self.shift


# ---------------------------------------------------------------------------
# Binary Linear (OneBit SVID + optional ParetoQ scale/shift)
# ---------------------------------------------------------------------------


class BinaryLinear(QuantizedModule):
    """~1-bit binary linear layer supporting OneBit SVID and ParetoQ modes.

    Modes:
      * ``"onebit"`` (default, ~1.028 bpp linear on GPT-2 124M):
        ``W_q = diag(row_scale) · sign(W_lat) · diag(col_scale) [+ shift[:, None]]``
        Initialized via SVID (rank-1 SVD of ``|W|``, OneBit arXiv:2402.11295).
      * ``"paretoq"`` (~1.016–1.031 bpp per-channel, or grouped):
        ``W_q = scale * sign(W_lat - shift) + shift``
        Per-output-channel or per-group learnable scale and optional shift.
    """

    quant_kind = "linear"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "onebit",
        group_size: Optional[int] = None,
        use_shift: bool = False,
        grad_clip: Optional[float] = None,
        lsq_scale_grad: bool = False,
        train_signs: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in ("onebit", "paretoq"):
            raise ValueError(f"Unknown binary mode: {mode!r} (expected 'onebit' or 'paretoq')")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.mode = mode
        self.use_shift = bool(use_shift)
        self.grad_clip = grad_clip
        self.lsq_scale_grad = bool(lsq_scale_grad)
        self.eps = float(eps)

        self.w_lat = nn.Parameter(torch.empty(self.out_features, self.in_features), requires_grad=train_signs)

        if self.mode == "onebit":
            self.n_groups = 1
            self.group_size_eff = self.in_features
            self.group_size = None
            self.row_scale = nn.Parameter(torch.ones(self.out_features))
            self.col_scale = nn.Parameter(torch.ones(self.in_features))
            self.register_parameter("scale", None)
            if self.use_shift:
                self.shift = nn.Parameter(torch.zeros(self.out_features))
            else:
                self.register_parameter("shift", None)
        else:
            self.n_groups, self.group_size_eff = _resolve_groups(self.in_features, group_size)
            self.group_size = None if (not group_size or group_size <= 0) else int(group_size)
            self.register_parameter("row_scale", None)
            self.register_parameter("col_scale", None)
            self.scale = nn.Parameter(torch.ones(self.out_features, self.n_groups, 1))
            if self.use_shift:
                self.shift = nn.Parameter(torch.zeros(self.out_features, self.n_groups, 1))
            else:
                self.register_parameter("shift", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.w_lat, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        self.init_scales_from_weight()

    @torch.no_grad()
    def init_scales_from_weight(self) -> None:
        """Initialize binary scales (SVID for ``onebit``, absmean for ``paretoq``)."""
        w = self.w_lat.detach().float()
        if self.mode == "onebit":
            u, s, vh = torch.linalg.svd(w.abs(), full_matrices=False)
            sqrt_s0 = s[0].clamp(min=0.0).sqrt()
            a = (u[:, 0] * sqrt_s0).abs().clamp(min=self.eps)
            b = (vh[0, :] * sqrt_s0).abs().clamp(min=self.eps)
            self.row_scale.data.copy_(a.to(self.row_scale.dtype))
            self.col_scale.data.copy_(b.to(self.col_scale.dtype))
            if self.use_shift and self.shift is not None:
                sgn = torch.sign(w)
                sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
                residual_mean = (w - a[:, None] * sgn * b[None, :]).mean(dim=-1)
                self.shift.data.copy_(residual_mean.to(self.shift.dtype))
        else:
            wg = w.reshape(self.out_features, self.n_groups, self.group_size_eff)
            if self.use_shift and self.shift is not None:
                sh = wg.mean(dim=-1, keepdim=True)
                self.shift.data.copy_(sh.to(self.shift.dtype))
                wg = wg - sh
            sc = wg.abs().mean(dim=-1, keepdim=True).clamp(min=self.eps)
            self.scale.data.copy_(sc.to(self.scale.dtype))

    @classmethod
    def from_module(
        cls,
        mod: nn.Module,
        mode: str = "onebit",
        group_size: Optional[int] = None,
        use_shift: bool = False,
        grad_clip: Optional[float] = None,
        lsq_scale_grad: bool = False,
        train_signs: bool = True,
    ) -> "BinaryLinear":
        """Convert an ``nn.Linear`` or HF GPT-2 ``Conv1D`` into a ``BinaryLinear``."""
        w = get_weight(mod)  # float32 (out_features, in_features)
        out_f, in_f = w.shape
        has_bias = getattr(mod, "bias", None) is not None
        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=has_bias,
            mode=mode,
            group_size=group_size,
            use_shift=use_shift,
            grad_clip=grad_clip,
            lsq_scale_grad=lsq_scale_grad,
            train_signs=train_signs,
        )
        with torch.no_grad():
            layer.w_lat.copy_(w)
            if has_bias and layer.bias is not None:
                layer.bias.copy_(mod.bias.detach().float())
        layer.init_scales_from_weight()
        return layer

    def quantized_weight(self) -> torch.Tensor:
        """Return differentiable fake-quantized ``(out_features, in_features)`` weight in fp32."""
        if self.mode == "onebit":
            s = sign_ste(self.w_lat.float(), grad_clip=self.grad_clip)
            a = self.row_scale.float()
            b = self.col_scale.float()
            if self.lsq_scale_grad:
                a = lsq_grad_scale(a, 1.0 / math.sqrt(self.in_features))
                b = lsq_grad_scale(b, 1.0 / math.sqrt(self.out_features))
            wq = a[:, None] * s * b[None, :]
            if self.shift is not None:
                sh = self.shift.float()
                if self.lsq_scale_grad:
                    sh = lsq_grad_scale(sh, 1.0 / math.sqrt(self.in_features))
                wq = wq + sh[:, None]
            return wq

        wg = self.w_lat.float().reshape(self.out_features, self.n_groups, self.group_size_eff)
        sh = None
        if self.shift is not None:
            sh = self.shift.float()
            if self.lsq_scale_grad:
                sh = lsq_grad_scale(sh, 1.0 / math.sqrt(self.group_size_eff))
            wg = wg - sh
        sc = self.scale.float().clamp(min=self.eps)
        if self.lsq_scale_grad:
            sc = lsq_grad_scale(sc, 1.0 / math.sqrt(self.group_size_eff))
        s = sign_ste(wg, grad_clip=self.grad_clip)
        wq = s * sc
        if sh is not None:
            wq = wq + sh
        return wq.reshape(self.out_features, self.in_features)

    @torch.no_grad()
    def dequantized_weight(self) -> torch.Tensor:
        return self.quantized_weight().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "onebit":
            # Factored forward avoids materializing the scaled (out, in) matrix:
            # y = ((x * b) @ s.T) * a + x.sum(-1, keepdim=True) * shift + bias
            dt = x.dtype
            s = sign_ste(self.w_lat.float(), grad_clip=self.grad_clip).to(dt)
            a = self.row_scale.float()
            b = self.col_scale.float()
            if self.lsq_scale_grad:
                a = lsq_grad_scale(a, 1.0 / math.sqrt(self.in_features))
                b = lsq_grad_scale(b, 1.0 / math.sqrt(self.out_features))
            y = F.linear(x * b.to(dt), s, bias=None) * a.to(dt)
            if self.shift is not None:
                sh = self.shift.float()
                if self.lsq_scale_grad:
                    sh = lsq_grad_scale(sh, 1.0 / math.sqrt(self.in_features))
                y = y + x.sum(dim=-1, keepdim=True) * sh.to(dt)
            if self.bias is not None:
                y = y + self.bias.to(dt)
            return y

        wq = self.quantized_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, wq, b)

    def orig_num_params(self) -> int:
        return self.out_features * self.in_features

    def storage_bits(self) -> float:
        n = self.orig_num_params()
        if self.mode == "onebit":
            meta = FP_META_BITS * (self.out_features + self.in_features)
            if self.use_shift:
                meta += FP_META_BITS * self.out_features
            return float(n * 1.0 + meta)
        n_groups_total = self.out_features * self.n_groups
        meta_per_group = FP_META_BITS * (2 if self.use_shift else 1)
        return float(n * 1.0 + n_groups_total * meta_per_group)

    def latent_parameters(self) -> Iterator[nn.Parameter]:
        yield self.w_lat

    def scale_parameters(self) -> Iterator[nn.Parameter]:
        if self.mode == "onebit":
            yield self.row_scale
            yield self.col_scale
        else:
            yield self.scale
        if self.shift is not None:
            yield self.shift


# ---------------------------------------------------------------------------
# Quantized Embeddings (per-row / per-group INT8 or INT4 with STE) + Tied LM Head
# ---------------------------------------------------------------------------


def _fake_quant_int_ste(
    w: torch.Tensor,
    bits: int,
    n_groups: int,
    group_size_eff: int,
    symmetric: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fake-quantize the last dimension of ``w`` in groups using STE."""
    orig_shape = w.shape
    wg = w.float().reshape(*orig_shape[:-1], n_groups, group_size_eff)
    if symmetric:
        qmax = float(2 ** (bits - 1) - 1)
        qmin = -qmax
        scale = wg.detach().abs().amax(dim=-1, keepdim=True).clamp(min=eps) / qmax
        wq = clamp_round_ste(wg / scale, qmin, qmax) * scale
    else:
        qmax = float(2 ** bits - 1)
        lo = wg.detach().amin(dim=-1, keepdim=True)
        hi = wg.detach().amax(dim=-1, keepdim=True)
        scale = (hi - lo).clamp(min=eps) / qmax
        zero = torch.round(-lo / scale)
        wq = (clamp_round_ste(wg / scale + zero, 0.0, qmax) - zero) * scale
    return wq.reshape(orig_shape)


class QuantizedEmbedding(QuantizedModule):
    """Per-row (or per-group) INT8/INT4 fake-quantized embedding with STE.

    When tied with the LM head via ``TiedQuantizedLMHead``, the exact same
    quantized weight matrix is used for both input token lookup and output
    logit projection, and its bits are counted exactly once in ``BitReport``.
    """

    quant_kind = "embedding"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int = 8,
        symmetric: bool = True,
        group_size: Optional[int] = None,
        padding_idx: Optional[int] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if bits < 2 or bits > 8:
            raise ValueError(f"QuantizedEmbedding supports 2 <= bits <= 8, got {bits}")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.n_groups, self.group_size_eff = _resolve_groups(self.embedding_dim, group_size)
        self.group_size = None if (not group_size or group_size <= 0) else int(group_size)
        self.padding_idx = padding_idx
        self.eps = float(eps)

        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        nn.init.normal_(self.weight)

    @classmethod
    def from_embedding(
        cls,
        emb: nn.Embedding,
        bits: int = 8,
        symmetric: bool = True,
        group_size: Optional[int] = None,
    ) -> "QuantizedEmbedding":
        mod = cls(
            num_embeddings=emb.num_embeddings,
            embedding_dim=emb.embedding_dim,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            padding_idx=emb.padding_idx,
        )
        with torch.no_grad():
            mod.weight.copy_(emb.weight.detach().float())
        return mod

    def quantized_weight(self) -> torch.Tensor:
        """Full ``(num_embeddings, embedding_dim)`` fake-quantized weight (fp32, STE)."""
        return _fake_quant_int_ste(
            self.weight,
            bits=self.bits,
            n_groups=self.n_groups,
            group_size_eff=self.group_size_eff,
            symmetric=self.symmetric,
            eps=self.eps,
        )

    @torch.no_grad()
    def dequantized_weight(self) -> torch.Tensor:
        return self.quantized_weight().detach()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Per-row/group quantization along the last dim is independent across rows,
        # so quantizing only the looked-up rows is exact and much faster than
        # quantizing all vocab rows on every forward pass.
        w_sel = F.embedding(input_ids, self.weight, padding_idx=self.padding_idx)
        wq_sel = _fake_quant_int_ste(
            w_sel,
            bits=self.bits,
            n_groups=self.n_groups,
            group_size_eff=self.group_size_eff,
            symmetric=self.symmetric,
            eps=self.eps,
        )
        return wq_sel.to(w_sel.dtype)

    def orig_num_params(self) -> int:
        return self.num_embeddings * self.embedding_dim

    def storage_bits(self) -> float:
        n_groups_total = self.num_embeddings * self.n_groups
        meta_per_group = FP_META_BITS if self.symmetric else 2 * FP_META_BITS
        return float(self.orig_num_params() * self.bits + n_groups_total * meta_per_group)

    def latent_parameters(self) -> Iterator[nn.Parameter]:
        yield self.weight

    def scale_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())


class TiedQuantizedLMHead(nn.Module):
    """LM head sharing weights with a ``QuantizedEmbedding`` without duplicating parameters.

    Stores the reference to ``embedding`` outside ``nn.Module._modules`` so
    ``state_dict()``, ``named_modules()``, and ``BitReport`` never double-count
    the tied weight matrix.
    """

    def __init__(self, embedding: QuantizedEmbedding, bias: Optional[nn.Parameter] = None) -> None:
        super().__init__()
        object.__setattr__(self, "_tied_embedding", embedding)
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().float().clone())
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> nn.Parameter:
        return self._tied_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = self._tied_embedding.quantized_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, wq, b)


class QuantizedLinearInt(QuantizedModule):
    """Per-row/group INT8 or INT4 linear layer with STE (for untied LM heads)."""

    quant_kind = "embedding"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        symmetric: bool = True,
        group_size: Optional[int] = None,
        bias: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if bits < 2 or bits > 8:
            raise ValueError(f"QuantizedLinearInt supports 2 <= bits <= 8, got {bits}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.n_groups, self.group_size_eff = _resolve_groups(self.in_features, group_size)
        self.group_size = None if (not group_size or group_size <= 0) else int(group_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        lin: nn.Module,
        bits: int = 8,
        symmetric: bool = True,
        group_size: Optional[int] = None,
    ) -> "QuantizedLinearInt":
        w = get_weight(lin)
        out_f, in_f = w.shape
        has_bias = getattr(lin, "bias", None) is not None
        mod = cls(
            in_features=in_f,
            out_features=out_f,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            bias=has_bias,
        )
        with torch.no_grad():
            mod.weight.copy_(w)
            if has_bias and mod.bias is not None:
                mod.bias.copy_(lin.bias.detach().float())
        return mod

    def quantized_weight(self) -> torch.Tensor:
        return _fake_quant_int_ste(
            self.weight,
            bits=self.bits,
            n_groups=self.n_groups,
            group_size_eff=self.group_size_eff,
            symmetric=self.symmetric,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = self.quantized_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, wq, b)

    def orig_num_params(self) -> int:
        return self.out_features * self.in_features

    def storage_bits(self) -> float:
        n_groups_total = self.out_features * self.n_groups
        meta_per_group = FP_META_BITS if self.symmetric else 2 * FP_META_BITS
        return float(self.orig_num_params() * self.bits + n_groups_total * meta_per_group)

    def latent_parameters(self) -> Iterator[nn.Parameter]:
        yield self.weight

    def scale_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())
