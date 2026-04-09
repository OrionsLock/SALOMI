"""Dynamic allocator for Proxy-SR-VQ.

Maps per-block Redun Scores to quantisation method assignments, producing
a ``MixedPrecisionConfig``.  Supports reallocation during QAT when
scores drift beyond a threshold.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .mixed_precision import LayerBudget, MixedPrecisionConfig
from .redun_score import RedunResult


@dataclass
class AllocatorConfig:
    """Knobs for the high-R and low-R quantiser branches."""
    hessian_vq_n_codes: int = 64
    hessian_vq_block_size: int = 4
    hessian_vq_gptq: bool = True
    ternary_sparsity: float = 0.3
    ternary_topk: float = 0.01
    ternary_bits: int = 8
    lowrank_rank: int = 8
    lowrank_factor_bits: int = 8


class DynamicAllocator:
    """Score-driven method selector.

    Blocks with ``redun_score >= redun_threshold`` are considered highly
    redundant and assigned the cheaper ternary+sparse quantiser.  Blocks
    below the threshold are sensitive and get HessianVQ (or lowrank for
    edge layers).

    Parameters
    ----------
    redun_threshold : float
        Boundary between high-R and low-R blocks.  Tuned during the
        proxy stage.
    high_r_cfg, low_r_cfg : AllocatorConfig
        Configuration structs for each branch.
    protect_edge_layers : bool
        Always assign lowrank to layer 0 and last layer regardless of
        Redun Score.
    """

    def __init__(
        self,
        redun_threshold: float = 1.0,
        high_r_cfg: Optional[AllocatorConfig] = None,
        low_r_cfg: Optional[AllocatorConfig] = None,
        protect_edge_layers: bool = True,
    ):
        self.redun_threshold = redun_threshold
        self.high_r = high_r_cfg or AllocatorConfig()
        self.low_r = low_r_cfg or AllocatorConfig()
        self.protect_edge_layers = protect_edge_layers
        self._prev_scores: Optional[Dict[str, Dict[str, float]]] = None

    def allocate(
        self,
        redun_scores: Dict[str, Dict[str, RedunResult]],
        n_layers: int = 12,
        target_bpp: float = 1.2,
    ) -> MixedPrecisionConfig:
        """Produce a ``MixedPrecisionConfig`` from Redun Scores.

        Args:
            redun_scores: ``{f"h.{i}": {"mlp_fc": RedunResult, ...}}``.
            n_layers: number of transformer layers.
            target_bpp: advisory (used in the returned config metadata).

        Returns:
            MixedPrecisionConfig with per-layer budgets.
        """
        self._prev_scores = {
            layer_key: {comp: r.redun_score for comp, r in comps.items()}
            for layer_key, comps in redun_scores.items()
        }

        budgets: List[LayerBudget] = []
        for i in range(n_layers):
            layer_key = f"h.{i}"
            comps = redun_scores.get(layer_key, {})

            attn_r = self._mean_component_r(comps, "attn")
            mlp_r = self._mean_component_r(comps, "mlp")

            is_edge = self.protect_edge_layers and (i == 0 or i == n_layers - 1)

            if is_edge:
                budgets.append(LayerBudget(
                    layer_idx=i,
                    attn_method="lowrank", mlp_method="lowrank",
                    attn_rank=self.low_r.lowrank_rank,
                    mlp_rank=self.low_r.lowrank_rank + 4,
                    attn_factor_bits=self.low_r.lowrank_factor_bits,
                    mlp_factor_bits=self.low_r.lowrank_factor_bits,
                ))
            else:
                attn_method = self._pick_method(attn_r)
                mlp_method = self._pick_method(mlp_r)
                budgets.append(LayerBudget(
                    layer_idx=i,
                    attn_method=attn_method,
                    mlp_method=mlp_method,
                    attn_n_codes=self.high_r.hessian_vq_n_codes,
                    mlp_n_codes=self.high_r.hessian_vq_n_codes,
                    attn_rank=self.low_r.lowrank_rank,
                    mlp_rank=self.low_r.lowrank_rank,
                    attn_factor_bits=self.low_r.lowrank_factor_bits,
                    mlp_factor_bits=self.low_r.lowrank_factor_bits,
                ))

        return MixedPrecisionConfig(
            n_layers=n_layers,
            target_bpp=target_bpp,
            layer_budgets=budgets,
        )

    def reallocate(
        self,
        prev_config: MixedPrecisionConfig,
        new_scores: Dict[str, Dict[str, RedunResult]],
        drift_threshold: float = 0.15,
    ) -> MixedPrecisionConfig:
        """Re-allocate only blocks whose Redun Score drifted significantly.

        Used during Phase 2 QAT to avoid thrashing while still adapting
        to emerging redundancy patterns.
        """
        if self._prev_scores is None:
            return self.allocate(new_scores, prev_config.n_layers, prev_config.target_bpp)

        n_layers = prev_config.n_layers
        budgets = list(prev_config.layer_budgets)

        changed = 0
        for i in range(n_layers):
            layer_key = f"h.{i}"
            old_comps = self._prev_scores.get(layer_key, {})
            new_comps = new_scores.get(layer_key, {})

            for comp_prefix in ("attn", "mlp"):
                old_r = self._mean_from_dict(old_comps, comp_prefix)
                new_r = self._mean_component_r(new_comps, comp_prefix)
                if abs(new_r - old_r) / (abs(old_r) + 1e-10) > drift_threshold:
                    new_method = self._pick_method(new_r)
                    if comp_prefix == "attn":
                        budgets[i].attn_method = new_method
                    else:
                        budgets[i].mlp_method = new_method
                    changed += 1

        self._prev_scores = {
            lk: {c: r.redun_score for c, r in cs.items()}
            for lk, cs in new_scores.items()
        }

        return MixedPrecisionConfig(
            n_layers=n_layers,
            target_bpp=prev_config.target_bpp,
            layer_budgets=budgets,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pick_method(self, r: float) -> str:
        if r >= self.redun_threshold:
            return "ternary_sparse"
        return "hessian_vq"

    @staticmethod
    def _mean_component_r(
        comps: Dict[str, RedunResult], prefix: str,
    ) -> float:
        vals = [r.redun_score for k, r in comps.items() if k.startswith(prefix)]
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _mean_from_dict(d: Dict[str, float], prefix: str) -> float:
        vals = [v for k, v in d.items() if k.startswith(prefix)]
        return float(np.mean(vals)) if vals else 0.0
