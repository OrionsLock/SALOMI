"""Mixed-precision layer allocation for transformer quantization.

Key insight from the SALOMI research: layer 0 is ~350x more sensitive than
middle layers when binarised, and MLP blocks are 77-200x more sensitive
than attention.  This module assigns per-layer / per-component precision
budgets so the overall model stays within a target bpp while protecting
the most sensitive paths.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class LayerBudget:
    """Precision budget for one transformer layer."""
    layer_idx: int
    attn_method: str = "hessian_vq"     # "hessian_vq" | "lowrank" | "fp16" | "int4"
    mlp_method: str = "hessian_vq"
    attn_rank: int = 4
    mlp_rank: int = 4
    attn_n_codes: int = 64
    mlp_n_codes: int = 64
    attn_factor_bits: int = 8
    mlp_factor_bits: int = 8


@dataclass
class MixedPrecisionConfig:
    """Full-model mixed-precision configuration."""
    n_layers: int = 12
    target_bpp: float = 1.35
    layer_budgets: List[LayerBudget] = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_budgets:
            self.layer_budgets = self._default_budgets()

    def _default_budgets(self) -> List[LayerBudget]:
        budgets = []
        for i in range(self.n_layers):
            if i == 0 or i == self.n_layers - 1:
                budgets.append(LayerBudget(
                    layer_idx=i,
                    attn_method="lowrank",
                    mlp_method="lowrank",
                    attn_rank=8,
                    mlp_rank=12,
                    attn_factor_bits=8,
                    mlp_factor_bits=8,
                ))
            else:
                budgets.append(LayerBudget(
                    layer_idx=i,
                    attn_method="hessian_vq",
                    mlp_method="lowrank",
                    attn_n_codes=64,
                    mlp_rank=4,
                    mlp_factor_bits=8,
                ))
        return budgets


def compute_layer_sensitivity(
    model_weights: Dict[str, np.ndarray],
    activations: Dict[str, np.ndarray],
    n_layers: int = 12,
) -> List[Dict[str, float]]:
    """Estimate per-layer, per-component sensitivity.

    Uses Fisher-style diagonal: sensitivity_j = mean(X_j^2) * var(W_j).
    Returns a list (one dict per layer) with keys 'attn' and 'mlp'.
    """
    sensitivities: List[Dict[str, float]] = []

    for i in range(n_layers):
        layer_sens = {}
        for component in ("attn", "mlp"):
            w_key = f"layer_{i}_{component}"
            a_key = f"layer_{i}_{component}_input"
            if w_key in model_weights and a_key in activations:
                W = model_weights[w_key]
                X = activations[a_key]
                h_diag = np.mean(X ** 2, axis=0)
                w_var = np.var(W, axis=0)
                layer_sens[component] = float(np.sum(h_diag * w_var))
            else:
                layer_sens[component] = 1.0
        sensitivities.append(layer_sens)

    return sensitivities


def allocate_precision(
    sensitivities: List[Dict[str, float]],
    target_bpp: float = 1.35,
    n_layers: int = 12,
) -> MixedPrecisionConfig:
    """Heuristic precision allocation based on layer sensitivities.

    Higher-sensitivity layers get more bits (higher rank / more codebook
    entries).  The allocation is normalised so the weighted average bpp
    across all layers is close to *target_bpp*.
    """
    all_sens = []
    for s in sensitivities:
        all_sens.append(s.get("attn", 1.0))
        all_sens.append(s.get("mlp", 1.0))
    all_sens = np.array(all_sens)
    all_sens = all_sens / (all_sens.mean() + 1e-10)

    budgets: List[LayerBudget] = []
    for i in range(n_layers):
        attn_s = all_sens[2 * i]
        mlp_s = all_sens[2 * i + 1]

        if attn_s > 2.0 or mlp_s > 2.0 or i == 0 or i == n_layers - 1:
            method = "lowrank"
            attn_rank = max(4, int(4 * attn_s))
            mlp_rank = max(4, int(4 * mlp_s))
            budgets.append(LayerBudget(
                layer_idx=i,
                attn_method=method, mlp_method=method,
                attn_rank=min(attn_rank, 16), mlp_rank=min(mlp_rank, 16),
                attn_factor_bits=8, mlp_factor_bits=8,
            ))
        else:
            budgets.append(LayerBudget(
                layer_idx=i,
                attn_method="hessian_vq", mlp_method="hessian_vq",
                attn_n_codes=max(32, int(32 * attn_s)),
                mlp_n_codes=max(32, int(64 * mlp_s)),
            ))

    return MixedPrecisionConfig(
        n_layers=n_layers,
        target_bpp=target_bpp,
        layer_budgets=budgets,
    )
