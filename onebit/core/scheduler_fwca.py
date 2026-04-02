"""Fisher-Weighted Compute Allocation (FWCA).

This module implements adaptive compute allocation based on Fisher information.
Layers/blocks with higher Fisher information receive more compute budget (higher T),
improving model performance without increasing total compute.

Key features:
- Estimate Fisher information per layer from calibration data
- Allocate compute budget proportional to Fisher scores
- Support multiple allocation strategies (proportional, threshold, hybrid)
- Maintain total compute budget constraint
- Zero storage overhead (ephemeral Fisher scores)

Typical usage:
    # Estimate Fisher information from calibration data
    fisher_scores = estimate_fisher_scores(calibration_stream, n_tokens=50000)

    # Create allocation scheduler
    scheduler = FWCAScheduler(fisher_scores, total_budget=192)

    # Get T allocation for each layer
    T_alloc = scheduler.allocate_T(layer_id)
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional, Iterator, Literal
from dataclasses import dataclass
import numpy as np


def estimate_fisher_scores(
    gradient_stream: Iterator[Tuple[int, np.ndarray]],
    n_samples: int = 50000,
    method: Literal["gradient_variance", "gradient_norm", "hessian_diag"] = "gradient_variance",
) -> FisherScores:
    """Estimate Fisher information scores from gradient stream.

    Fisher information measures the sensitivity of the model to parameter changes.
    Higher Fisher = more important layer = deserves more compute.

    Args:
        gradient_stream: Iterator yielding (layer_id, gradient) tuples
        n_samples: Number of samples to process
        method: Estimation method:
            - "gradient_variance": Var(∇L) (default, fast)
            - "gradient_norm": E[||∇L||²] (similar to variance)
            - "hessian_diag": E[∇²L] (more accurate, slower)

    Returns:
        FisherScores object with per-layer scores
    """
    # Accumulate statistics per layer
    layer_stats = {}

    samples_processed = 0
    for layer_id, grad in gradient_stream:
        if layer_id not in layer_stats:
            layer_stats[layer_id] = {
                "sum_grad": 0.0,
                "sum_grad_sq": 0.0,
                "n": 0,
            }

        if method == "gradient_variance":
            # Var(∇L) = E[∇L²] - E[∇L]²
            layer_stats[layer_id]["sum_grad"] += np.sum(grad)
            layer_stats[layer_id]["sum_grad_sq"] += np.sum(grad ** 2)
            layer_stats[layer_id]["n"] += grad.size

        elif method == "gradient_norm":
            # E[||∇L||²]
            layer_stats[layer_id]["sum_grad_sq"] += np.sum(grad ** 2)
            layer_stats[layer_id]["n"] += 1

        elif method == "hessian_diag":
            # Approximate diagonal Hessian with gradient outer product
            # E[∇L ⊗ ∇L] ≈ Fisher
            layer_stats[layer_id]["sum_grad_sq"] += np.sum(grad ** 2)
            layer_stats[layer_id]["n"] += grad.size

        samples_processed += 1
        if samples_processed >= n_samples:
            break

    # Compute Fisher scores
    scores = {}
    for layer_id, stats in layer_stats.items():
        if method == "gradient_variance":
            mean_grad = stats["sum_grad"] / stats["n"]
            mean_grad_sq = stats["sum_grad_sq"] / stats["n"]
            variance = mean_grad_sq - mean_grad ** 2
            scores[layer_id] = max(0.0, variance)  # Ensure non-negative

        elif method == "gradient_norm":
            scores[layer_id] = stats["sum_grad_sq"] / stats["n"]

        elif method == "hessian_diag":
            scores[layer_id] = stats["sum_grad_sq"] / stats["n"]

    return FisherScores(
        scores=scores,
        n_samples=samples_processed,
        method=method,
    )


@dataclass
class FisherScores:
    """Fisher information scores for layers.
    
    Attributes:
        scores: Per-layer Fisher scores (higher = more important)
        n_samples: Number of samples used for estimation
        method: Estimation method used
    """
    scores: Dict[int, float]
    n_samples: int
    method: str


@dataclass
class FWCAConfig:
    """FWCA scheduler configuration.
    
    Attributes:
        total_budget: Total compute budget (sum of all T allocations)
        T_min: Minimum T per layer (default: 8)
        T_max: Maximum T per layer (default: 32)
        T_quantize: Quantization levels for T (default: [8, 12, 16, 24, 32])
        strategy: Allocation strategy ("proportional", "threshold", "hybrid")
        alpha: Smoothing parameter for hybrid strategy (default: 0.5)
    """
    total_budget: int
    T_min: int = 8
    T_max: int = 32
    T_quantize: tuple[int, ...] = (8, 12, 16, 24, 32)
    strategy: Literal["proportional", "threshold", "hybrid"] = "proportional"
    alpha: float = 0.5  # For hybrid strategy


class FWCAScheduler:
    """Fisher-Weighted Compute Allocation scheduler.
    
    Allocates compute budget (T) to layers based on Fisher information scores.
    Higher Fisher scores receive more compute.
    """
    
    def __init__(self, fisher_scores: FisherScores, cfg: FWCAConfig):
        """Initialize FWCA scheduler.
        
        Args:
            fisher_scores: Per-layer Fisher information scores
            cfg: FWCA configuration
        """
        self.fisher_scores = fisher_scores
        self.cfg = cfg
        
        # Compute T allocation for all layers
        self.T_alloc = self._compute_allocation()
    
    def _compute_allocation(self) -> Dict[int, int]:
        """Compute T allocation for all layers.
        
        Returns:
            Dictionary mapping layer_id to T allocation
        """
        if self.cfg.strategy == "proportional":
            return self._allocate_proportional()
        elif self.cfg.strategy == "threshold":
            return self._allocate_threshold()
        elif self.cfg.strategy == "hybrid":
            return self._allocate_hybrid()
        else:
            raise ValueError(f"Unknown strategy: {self.cfg.strategy}")
    
    def _allocate_proportional(self) -> Dict[int, int]:
        """Proportional allocation: T ∝ Fisher score.
        
        Returns:
            Dictionary mapping layer_id to T allocation
        """
        scores = self.fisher_scores.scores
        layer_ids = sorted(scores.keys())
        n_layers = len(layer_ids)
        
        # Normalize Fisher scores
        total_fisher = sum(scores.values())
        if total_fisher == 0:
            # Uniform allocation if all scores are zero
            T_uniform = self.cfg.total_budget // n_layers
            return {lid: T_uniform for lid in layer_ids}
        
        # Allocate proportionally
        T_alloc_raw = {}
        for lid in layer_ids:
            T_raw = (scores[lid] / total_fisher) * self.cfg.total_budget
            T_alloc_raw[lid] = T_raw
        
        # Quantize and adjust to meet budget
        return self._quantize_and_adjust(T_alloc_raw, layer_ids)
    
    def _allocate_threshold(self) -> Dict[int, int]:
        """Threshold allocation: High-Fisher layers get T_max, others get T_min.
        
        Returns:
            Dictionary mapping layer_id to T allocation
        """
        scores = self.fisher_scores.scores
        layer_ids = sorted(scores.keys())
        n_layers = len(layer_ids)
        
        # Compute threshold to meet budget
        # Let k layers get T_max, rest get T_min
        # k * T_max + (n - k) * T_min = total_budget
        # k = (total_budget - n * T_min) / (T_max - T_min)
        
        k_high = (self.cfg.total_budget - n_layers * self.cfg.T_min) / (self.cfg.T_max - self.cfg.T_min)
        k_high = int(round(k_high))
        k_high = max(0, min(n_layers, k_high))
        
        # Sort layers by Fisher score
        sorted_layers = sorted(layer_ids, key=lambda lid: scores[lid], reverse=True)
        
        # Allocate
        T_alloc = {}
        for i, lid in enumerate(sorted_layers):
            if i < k_high:
                T_alloc[lid] = self.cfg.T_max
            else:
                T_alloc[lid] = self.cfg.T_min
        
        return T_alloc
    
    def _allocate_hybrid(self) -> Dict[int, int]:
        """Hybrid allocation: Blend proportional and threshold.
        
        Returns:
            Dictionary mapping layer_id to T allocation
        """
        # Get both allocations
        T_prop = self._allocate_proportional()
        T_thresh = self._allocate_threshold()
        
        # Blend with alpha
        layer_ids = sorted(self.fisher_scores.scores.keys())
        T_alloc_raw = {}
        for lid in layer_ids:
            T_raw = self.cfg.alpha * T_prop[lid] + (1 - self.cfg.alpha) * T_thresh[lid]
            T_alloc_raw[lid] = T_raw
        
        # Quantize and adjust
        return self._quantize_and_adjust(T_alloc_raw, layer_ids)
    
    def _quantize_and_adjust(
        self,
        T_alloc_raw: Dict[int, float],
        layer_ids: list[int],
    ) -> Dict[int, int]:
        """Quantize T allocations and adjust to meet budget constraint.
        
        Args:
            T_alloc_raw: Raw (float) T allocations
            layer_ids: List of layer IDs
        
        Returns:
            Quantized T allocations that sum to total_budget
        """
        # Quantize to nearest level
        T_alloc = {}
        for lid in layer_ids:
            T_raw = T_alloc_raw[lid]
            T_quant = self._quantize_T(T_raw)
            T_alloc[lid] = T_quant
        
        # Adjust to meet budget
        current_budget = sum(T_alloc.values())
        deficit = self.cfg.total_budget - current_budget
        
        if deficit == 0:
            return T_alloc
        
        # Adjust by adding/removing from layers with largest residuals
        residuals = {}
        for lid in layer_ids:
            T_raw = T_alloc_raw[lid]
            T_quant = T_alloc[lid]
            residuals[lid] = abs(T_raw - T_quant)

        # Sort by residual (descending)
        sorted_layers = sorted(layer_ids, key=lambda lid: residuals[lid], reverse=True)

        # Adjust iteratively until budget is met
        max_iterations = len(layer_ids) * 10  # Prevent infinite loop
        iteration = 0

        while deficit != 0 and iteration < max_iterations:
            iteration += 1
            made_change = False

            for lid in sorted_layers:
                if deficit == 0:
                    break

                if deficit > 0:
                    # Need to add budget
                    next_T = self._next_higher_T(T_alloc[lid])
                    if next_T is not None and next_T <= self.cfg.T_max:
                        delta = next_T - T_alloc[lid]
                        if delta <= deficit:
                            T_alloc[lid] = next_T
                            deficit -= delta
                            made_change = True
                else:
                    # Need to remove budget
                    next_T = self._next_lower_T(T_alloc[lid])
                    if next_T is not None and next_T >= self.cfg.T_min:
                        delta = T_alloc[lid] - next_T
                        if delta <= abs(deficit):
                            T_alloc[lid] = next_T
                            deficit += delta
                            made_change = True

            # If no changes were made, we can't meet the budget exactly
            if not made_change:
                break

        return T_alloc
    
    def _quantize_T(self, T_raw: float) -> int:
        """Quantize T to nearest level.
        
        Args:
            T_raw: Raw T value
        
        Returns:
            Quantized T
        """
        T_levels = self.cfg.T_quantize
        T_clamped = max(self.cfg.T_min, min(self.cfg.T_max, T_raw))
        
        # Find nearest level
        best_T = T_levels[0]
        best_dist = abs(T_clamped - best_T)
        
        for T in T_levels:
            dist = abs(T_clamped - T)
            if dist < best_dist:
                best_dist = dist
                best_T = T
        
        return best_T
    
    def _next_higher_T(self, T_current: int) -> Optional[int]:
        """Get next higher T level.
        
        Args:
            T_current: Current T
        
        Returns:
            Next higher T, or None if at max
        """
        T_levels = sorted(self.cfg.T_quantize)
        for T in T_levels:
            if T > T_current:
                return T
        return None
    
    def _next_lower_T(self, T_current: int) -> Optional[int]:
        """Get next lower T level.
        
        Args:
            T_current: Current T
        
        Returns:
            Next lower T, or None if at min
        """
        T_levels = sorted(self.cfg.T_quantize, reverse=True)
        for T in T_levels:
            if T < T_current:
                return T
        return None
    
    def get_T(self, layer_id: int) -> int:
        """Get T allocation for a layer.
        
        Args:
            layer_id: Layer ID
        
        Returns:
            T allocation
        """
        return self.T_alloc.get(layer_id, self.cfg.T_min)
    
    def get_allocation_summary(self) -> Dict:
        """Get summary of allocation.
        
        Returns:
            Dictionary with allocation statistics
        """
        T_values = list(self.T_alloc.values())
        fisher_values = [self.fisher_scores.scores[lid] for lid in self.T_alloc.keys()]
        
        return {
            "total_budget": sum(T_values),
            "target_budget": self.cfg.total_budget,
            "n_layers": len(T_values),
            "T_mean": np.mean(T_values),
            "T_std": np.std(T_values),
            "T_min": min(T_values),
            "T_max": max(T_values),
            "fisher_mean": np.mean(fisher_values),
            "fisher_std": np.std(fisher_values),
            "strategy": self.cfg.strategy,
        }

