"""Two-phase QAT loop for Proxy-SR-VQ.

Phase 1 -- Block-wise calibration
    For each transformer block, freeze everything else, run calibration
    samples through the block, and optimise quantisation parameters
    (codebook entries, scales) to minimise block-output MSE with the
    assigned quantiser.

Phase 2 -- Lightweight QAT with STE
    Standard forward-pass training loop where the quantised weights are
    used in the forward path (STE for gradients).  Every N steps the
    Redun Scores are recomputed and the DynamicAllocator can reallocate
    blocks whose scores drifted beyond a threshold.

    NOTE (Phase 0): Phase 2 is currently DISABLED and raises
    ``NotImplementedError``.  The previous implementation could not train:
    the STE was applied under ``torch.no_grad()`` inside a forward pre-hook
    and written into ``weight.data`` (so no gradient ever reached the latent
    weights or the scales, and the FP weights were destroyed on the first
    step), all model parameters were frozen, it would also binarize the tied
    LM head / embeddings, and the per-row scale used the wrong axis for HF
    ``Conv1D`` (stored as (in, out)).  A proper KD-QAT path (teacher logits,
    latent FP weights, STE inside autograd, block linears only) is planned
    as a separate module.
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ======================================================================
# Phase 1: Block-wise Calibration
# ======================================================================

@dataclass
class CalibrationResult:
    """Result of calibrating one block."""
    layer_idx: int
    method: str
    mse_before: float
    mse_after: float
    bpp: float


class BlockCalibrator:
    """Calibrate quantisation parameters block-by-block.

    For each transformer block:
      1. Collect (input, output) pairs from unquantised forward passes.
      2. Quantise every weight matrix in the block using the allocator's
         assigned method.
      3. Report per-block MSE and BPP.
    """

    def __init__(
        self,
        n_calib_samples: int = 256,
        max_len: int = 64,
    ):
        self.n_calib_samples = n_calib_samples
        self.max_len = max_len

    def calibrate_model(
        self,
        model,
        tokenizer,
        config,
        calib_texts: List[str],
        device: str = "cpu",
        adapter=None,
    ) -> List[CalibrationResult]:
        """Run block-wise calibration for the entire model.

        Args:
            model: HuggingFace model (will be modified in-place).
            tokenizer: matching tokenizer.
            config: ``MixedPrecisionConfig`` from the DynamicAllocator.
            calib_texts: list of calibration strings.
            device: torch device.
            adapter: optional ``ModelAdapter``.  Created automatically if None.

        Returns:
            List of CalibrationResult, one per layer.
        """
        if adapter is None:
            from .model_factory import ModelAdapter
            adapter = ModelAdapter(model)

        n_layers = adapter.n_layers
        results: List[CalibrationResult] = []

        for layer_idx in range(n_layers):
            budget = config.layer_budgets[layer_idx]
            layer_targets = adapter.get_layer_targets(layer_idx)

            attn_kwargs = {"n_codes": budget.attn_n_codes, "rank": budget.attn_rank, "factor_bits": budget.attn_factor_bits}
            mlp_kwargs = {"n_codes": budget.mlp_n_codes, "rank": budget.mlp_rank, "factor_bits": budget.mlp_factor_bits}

            layer_mse_before = 0.0
            layer_mse_after = 0.0
            layer_bits = 0.0
            layer_params = 0

            for comp_name, submod in layer_targets:
                is_attn = comp_name.startswith("attn")
                method_name = budget.attn_method if is_attn else budget.mlp_method
                method_kwargs = attn_kwargs if is_attn else mlp_kwargs

                X_inputs, Y_outputs = self._collect_block_io(
                    model, tokenizer, submod, calib_texts,
                    self.n_calib_samples, self.max_len,
                )
                if X_inputs is None:
                    continue

                W_orig = adapter.get_weight_numpy(submod)
                H_diag = np.mean(X_inputs ** 2, axis=0).astype(np.float32)

                # The hooked outputs include the bias; add it to the linear
                # reconstructions so mse_before measures only numerical error.
                bias = getattr(submod, "bias", None)
                b = bias.detach().cpu().float().numpy() if bias is not None else 0.0

                Y_before = X_inputs @ W_orig.T + b
                mse_before = float(np.mean((Y_before - Y_outputs) ** 2))
                layer_mse_before += mse_before

                quantizer = self._make_quantizer(method_name, method_kwargs)
                W_q = quantizer.quantize(W_orig, H_diag)

                Y_after = X_inputs @ W_q.T + b
                mse_after = float(np.mean((Y_after - Y_outputs) ** 2))
                layer_mse_after += mse_after

                n_w = int(W_orig.size)
                layer_bits += self._quantizer_bpp(quantizer, n_w) * n_w
                layer_params += n_w

                adapter.set_weight_numpy(submod, W_q)

            bpp = layer_bits / layer_params if layer_params else float("nan")
            results.append(CalibrationResult(
                layer_idx=layer_idx,
                method=f"attn:{budget.attn_method}/mlp:{budget.mlp_method}",
                mse_before=layer_mse_before,
                mse_after=layer_mse_after,
                bpp=bpp,
            ))
            print(f"  Layer {layer_idx}: MSE {layer_mse_before:.6f} -> {layer_mse_after:.6f} | bpp {bpp:.3f}")

        return results

    def _collect_block_io(
        self, model, tokenizer, submod, calib_texts, n_samples, max_len,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Hook into *submod* and collect (input, output) numpy arrays."""
        inputs_list: list = []
        outputs_list: list = []

        def _hook(mod, inp, out):
            x = inp[0].detach().cpu().numpy().reshape(-1, inp[0].shape[-1])
            y = out.detach().cpu().numpy().reshape(-1, out.shape[-1])
            inputs_list.append(x)
            outputs_list.append(y)

        handle = submod.register_forward_hook(_hook)
        device = next(model.parameters()).device
        try:
            with torch.no_grad():
                for text in calib_texts[:n_samples]:
                    ids = tokenizer(
                        text, return_tensors="pt",
                        truncation=True, max_length=max_len,
                    ).to(device)
                    model(**ids)  # errors propagate (previously swallowed silently)
        finally:
            handle.remove()

        if not inputs_list:
            return None, None

        return (
            np.concatenate(inputs_list, axis=0),
            np.concatenate(outputs_list, axis=0),
        )

    @staticmethod
    def _make_quantizer(method: str, kwargs: dict):
        """Instantiate a quantizer from its name."""
        from ..quantization.hessian_vq import HessianVQ
        from ..quantization.ternary_sparse import TernarySparse
        from ..quantization.lowrank_residual import LowRankResidual

        if method == "hessian_vq":
            # gptq_refine was a fake refinement and now raises; real GPTQ lives in
            # onebit/quantization/gptq.py.
            return HessianVQ(
                n_codes=kwargs.get("n_codes", 64),
                block_size=4,
                max_iter=15,
                use_hessian_weight=True,
            )
        elif method == "ternary_sparse":
            return TernarySparse(sparsity=0.3, residual_top_k=0.01, residual_bits=8)
        elif method == "lowrank":
            return LowRankResidual(
                rank=kwargs.get("rank", 8),
                factor_bits=kwargs.get("factor_bits", 8),
            )
        else:
            return HessianVQ(n_codes=64, block_size=4, max_iter=15)

    @staticmethod
    def _quantizer_bpp(quantizer, n_weights: int) -> float:
        """Actual bits/weight reported by a fitted quantizer (incl. its side info)."""
        from ..quantization.lowrank_residual import LowRankResidual

        if isinstance(quantizer, LowRankResidual):
            return float(quantizer.effective_bpp())
        return float(quantizer.effective_bpp(n_weights))


# ======================================================================
# STE primitives (proper PyTorch autograd)
# ======================================================================

class STESign(torch.autograd.Function):
    """Straight-Through sign: forward = sign(x), backward = identity."""
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STERound(torch.autograd.Function):
    """Straight-Through round for ternary quantisation."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return STESign.apply(x)


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return STERound.apply(x)


# ======================================================================
# Phase 2: Lightweight QAT with dynamic reallocation
# ======================================================================

@dataclass
class QATConfig:
    """Configuration for the QAT phase."""
    n_steps: int = 1000
    batch_size: int = 4
    max_len: int = 128
    lr: float = 1e-4
    reprobe_every: int = 100
    drift_threshold: float = 0.15
    log_every: int = 50


class QuantizedLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that applies STE quantisation.

    Stores the latent FP32 weight and applies STE-sign + per-row scale
    in the forward pass.
    """

    def __init__(self, in_features: int, out_features: int, method: str = "hessian_vq"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.scale = nn.Parameter(torch.ones(out_features, 1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method in ("hessian_vq", "ternary_sparse"):
            W_q = ste_sign(self.weight) * self.scale
        else:
            W_q = self.weight
        return F.linear(x, W_q, self.bias)


_PHASE2_DISABLED_MSG = (
    "QAT Phase 2 (STE) is disabled: the previous implementation could not train "
    "(STE applied under torch.no_grad() and written into weight.data, so no gradients "
    "reached the latent weights/scales and FP weights were overwritten; all parameters "
    "frozen; tied LM head/embeddings would be binarized; wrong scale axis for HF Conv1D). "
    "A proper KD-QAT path (teacher logits, latent FP weights, STE inside autograd, "
    "transformer-block linears only) is planned as a separate module. Use Phase 1 "
    "(BlockCalibrator) or onebit/quantization/gptq.py for post-training quantization."
)


class QATLoop:
    """Lightweight QAT with periodic Redun Score reprobing.

    Token budget: ~10% of original training data.  Uses STE for the
    quantised forward pass and periodically recomputes Redun Scores
    to adapt allocation.

    Currently disabled (raises ``NotImplementedError``); see module docstring.
    """

    def __init__(self, cfg: QATConfig):
        self.cfg = cfg

    def run(
        self,
        model,
        tokenizer,
        train_texts: List[str],
        allocator=None,
        redun_computer=None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Execute Phase 2 QAT (disabled).

        Args:
            model: the *already calibrated* model (from Phase 1).
            tokenizer: matching tokenizer.
            train_texts: text samples (10% token budget).
            allocator: DynamicAllocator for reallocation.
            redun_computer: RedunScoreComputer for reprobing.
            device: torch device.

        Raises:
            NotImplementedError: always; see ``_PHASE2_DISABLED_MSG``.
        """
        raise NotImplementedError(_PHASE2_DISABLED_MSG)

    @staticmethod
    def _inject_ste_scales(model) -> List[nn.Parameter]:
        """Removed: the hook binarized ``weight.data`` under ``no_grad`` (no gradient path)."""
        raise NotImplementedError(_PHASE2_DISABLED_MSG)


# ======================================================================
# Convenience: run both phases
# ======================================================================

def run_full_pipeline(
    model,
    tokenizer,
    config,
    calib_texts: List[str],
    train_texts: Optional[List[str]] = None,
    allocator=None,
    redun_computer=None,
    qat_cfg: Optional[QATConfig] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run Phase 1 (+ Phase 2 if explicitly requested).

    Args:
        model: HuggingFace GPT-2 model.
        tokenizer: matching tokenizer.
        config: MixedPrecisionConfig from the DynamicAllocator.
        calib_texts: calibration strings (Phase 1).
        train_texts: training strings for Phase 2.  Phase 2 only runs when
            ``train_texts`` or ``qat_cfg`` is given, and it currently raises
            ``NotImplementedError`` (see module docstring).
        allocator: optional DynamicAllocator for Phase 2 reallocation.
        redun_computer: optional RedunScoreComputer for Phase 2 reprobing.
        qat_cfg: optional QATConfig overrides.
        device: torch device.

    Returns:
        Dict with phase1_results, phase2_results (None unless Phase 2 requested).
    """
    print("=== Phase 1: Block-wise Calibration ===")
    calibrator = BlockCalibrator(n_calib_samples=min(256, len(calib_texts)))
    phase1 = calibrator.calibrate_model(model, tokenizer, config, calib_texts, device)

    phase2 = None
    if train_texts or qat_cfg is not None:
        texts = train_texts or calib_texts
        print("\n=== Phase 2: Lightweight QAT ===")
        qat = QATLoop(qat_cfg or QATConfig(n_steps=200))
        phase2 = qat.run(model, tokenizer, texts, allocator, redun_computer, device)
    else:
        print("(Phase 2 QAT skipped: disabled pending the KD-QAT implementation)")

    return {"phase1_results": phase1, "phase2_results": phase2}
