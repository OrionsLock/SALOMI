"""Policy export / import for Proxy-SR-VQ.

Packages per-block quantisation assignments, Redun Scores, and scaling
law coefficients into a compact JSON file.  A downstream consumer can
load the policy, optionally extrapolate to a different model size, and
apply the quantisation plan without re-running calibration.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List
from pathlib import Path


@dataclass
class BlockPolicy:
    """Quantisation plan for a single transformer block."""
    layer_idx: int
    attn_method: str
    mlp_method: str
    attn_redun_score: float = 0.0
    mlp_redun_score: float = 0.0
    attn_n_codes: int = 64
    mlp_n_codes: int = 64
    attn_rank: int = 4
    mlp_rank: int = 4
    attn_factor_bits: int = 8
    mlp_factor_bits: int = 8


@dataclass
class PolicyMetadata:
    """Top-level metadata for the exported policy."""
    model_name: str
    n_params: int
    n_layers: int
    target_bpp: float
    proxy_sizes_used: List[str]
    redun_threshold: float
    timestamp: str = ""


def export_policy(
    config,
    redun_scores: Dict[str, Dict[str, Any]],
    scaling_params: Optional[Dict[str, Any]] = None,
    metadata: Optional[PolicyMetadata] = None,
    output_path: str = "proxy_sr_vq_policy.json",
) -> str:
    """Save the full allocation policy to JSON.

    Args:
        config: MixedPrecisionConfig from the DynamicAllocator.
        redun_scores: ``{layer_key: {comp: RedunResult or score}}``.
        scaling_params: dict from ``ScalingLawFitter.to_dict()``.
        metadata: optional PolicyMetadata.
        output_path: file path.

    Returns:
        The path written to.
    """
    import datetime

    blocks: List[Dict[str, Any]] = []
    for budget in config.layer_budgets:
        layer_key = f"h.{budget.layer_idx}"
        comps = redun_scores.get(layer_key, {})

        def _score(comp_prefix: str) -> float:
            for k, v in comps.items():
                if k.startswith(comp_prefix):
                    return float(v.redun_score if hasattr(v, "redun_score") else v)
            return 0.0

        blocks.append(asdict(BlockPolicy(
            layer_idx=budget.layer_idx,
            attn_method=budget.attn_method,
            mlp_method=budget.mlp_method,
            attn_redun_score=_score("attn"),
            mlp_redun_score=_score("mlp"),
            attn_n_codes=budget.attn_n_codes,
            mlp_n_codes=budget.mlp_n_codes,
            attn_rank=budget.attn_rank,
            mlp_rank=budget.mlp_rank,
            attn_factor_bits=budget.attn_factor_bits,
            mlp_factor_bits=budget.mlp_factor_bits,
        )))

    policy = {
        "version": "1.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "metadata": asdict(metadata) if metadata else {},
        "target_bpp": config.target_bpp,
        "n_layers": config.n_layers,
        "blocks": blocks,
        "scaling_law": scaling_params or {},
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(policy, f, indent=2)

    return output_path


def import_policy(
    policy_path: str,
    target_model_info: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a policy and optionally extrapolate for a different model size.

    Args:
        policy_path: path to the JSON policy file.
        target_model_info: optional ModelInfo for extrapolation.

    Returns:
        Dict with ``config``, ``scaling_law``, ``metadata``, and
        optionally ``extrapolated_config``.
    """
    from ..quantization.mixed_precision import LayerBudget, MixedPrecisionConfig

    with open(policy_path) as f:
        policy = json.load(f)

    budgets = []
    for blk in policy["blocks"]:
        budgets.append(LayerBudget(
            layer_idx=blk["layer_idx"],
            attn_method=blk["attn_method"],
            mlp_method=blk["mlp_method"],
            attn_n_codes=blk.get("attn_n_codes", 64),
            mlp_n_codes=blk.get("mlp_n_codes", 64),
            attn_rank=blk.get("attn_rank", 4),
            mlp_rank=blk.get("mlp_rank", 4),
            attn_factor_bits=blk.get("attn_factor_bits", 8),
            mlp_factor_bits=blk.get("mlp_factor_bits", 8),
        ))

    config = MixedPrecisionConfig(
        n_layers=policy["n_layers"],
        target_bpp=policy["target_bpp"],
        layer_budgets=budgets,
    )

    result: Dict[str, Any] = {
        "config": config,
        "scaling_law": policy.get("scaling_law", {}),
        "metadata": policy.get("metadata", {}),
        "blocks_raw": policy["blocks"],
    }

    if target_model_info is not None and policy.get("scaling_law"):
        from .scaling_law import ScalingLawFitter
        from ..quantization.dynamic_allocator import DynamicAllocator

        fitter = ScalingLawFitter.from_dict(policy["scaling_law"])
        threshold = policy["metadata"].get("redun_threshold", 1.0) if policy.get("metadata") else 1.0
        allocator = DynamicAllocator(redun_threshold=threshold)

        extrapolated = fitter.predict_allocation(
            target_n_params=target_model_info.n_params,
            allocator=allocator,
            n_layers=target_model_info.n_layers,
            target_bpp=policy["target_bpp"],
        )
        result["extrapolated_config"] = extrapolated

    return result
