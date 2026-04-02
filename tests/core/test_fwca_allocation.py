"""Tests for Fisher-Weighted Compute Allocation (FWCA).

Tests cover:
1. Fisher score estimation
2. Proportional allocation
3. Threshold allocation
4. Hybrid allocation
5. Budget constraints
6. Quantization
"""
import numpy as np
import pytest
from typing import Optional
from onebit.core.scheduler_fwca import (
    estimate_fisher_scores,
    FWCAScheduler,
    FWCAConfig,
    FisherScores,
)


def mock_gradient_stream(
    n_samples: int,
    n_layers: int = 6,
    layer_importance: Optional[dict] = None,
    seed: int = 42,
):
    """Generate mock gradient stream for testing.
    
    Args:
        n_samples: Number of gradient samples
        n_layers: Number of layers
        layer_importance: Dict mapping layer_id to importance (default: uniform)
        seed: Random seed
    
    Yields:
        (layer_id, gradient) tuples
    """
    rng = np.random.RandomState(seed)
    
    # Default: uniform importance
    if layer_importance is None:
        layer_importance = {i: 1.0 for i in range(n_layers)}
    
    for _ in range(n_samples):
        for layer_id in range(n_layers):
            # Generate gradient with variance proportional to importance
            importance = layer_importance.get(layer_id, 1.0)
            grad = rng.randn(128) * np.sqrt(importance)
            yield layer_id, grad.astype(np.float32)


class TestFisherEstimation:
    """Test Fisher score estimation."""
    
    def test_gradient_variance_method(self):
        """Test Fisher estimation with gradient variance."""
        # Create stream with known importance
        importance = {0: 1.0, 1: 2.0, 2: 4.0}
        stream = mock_gradient_stream(1000, n_layers=3, layer_importance=importance)
        
        fisher = estimate_fisher_scores(stream, n_samples=1000, method="gradient_variance")
        
        assert len(fisher.scores) == 3
        assert fisher.n_samples == 1000
        assert fisher.method == "gradient_variance"
        
        # Higher importance should give higher Fisher scores
        assert fisher.scores[2] > fisher.scores[1] > fisher.scores[0]
    
    def test_gradient_norm_method(self):
        """Test Fisher estimation with gradient norm."""
        importance = {0: 1.0, 1: 3.0, 2: 5.0}
        stream = mock_gradient_stream(500, n_layers=3, layer_importance=importance)
        
        fisher = estimate_fisher_scores(stream, n_samples=500, method="gradient_norm")
        
        assert len(fisher.scores) == 3
        assert fisher.method == "gradient_norm"
        
        # Scores should be ordered by importance
        assert fisher.scores[2] > fisher.scores[1] > fisher.scores[0]
    
    def test_uniform_importance(self):
        """Test with uniform importance (all layers equal)."""
        stream = mock_gradient_stream(500, n_layers=4, layer_importance=None)
        
        fisher = estimate_fisher_scores(stream, n_samples=500)
        
        # All scores should be similar (within 20% due to random variation)
        scores = list(fisher.scores.values())
        mean_score = np.mean(scores)
        for score in scores:
            assert abs(score - mean_score) / mean_score < 0.2


class TestProportionalAllocation:
    """Test proportional allocation strategy."""
    
    def test_proportional_basic(self):
        """Test basic proportional allocation."""
        # Create Fisher scores
        fisher = FisherScores(
            scores={0: 1.0, 1: 2.0, 2: 3.0},
            n_samples=1000,
            method="gradient_variance",
        )
        
        # Total budget: 72 (should allocate 12, 24, 36)
        cfg = FWCAConfig(
            total_budget=72,
            T_min=8,
            T_max=32,
            T_quantize=(8, 12, 16, 24, 32),
            strategy="proportional",
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        
        # Check allocations
        T0 = scheduler.get_T(0)
        T1 = scheduler.get_T(1)
        T2 = scheduler.get_T(2)
        
        # Higher Fisher should get more T
        assert T2 > T1 > T0
        
        # Total should match budget (within quantization error)
        total = T0 + T1 + T2
        assert abs(total - 72) <= 8  # Allow some slack for quantization
    
    def test_proportional_meets_budget(self):
        """Test that proportional allocation meets budget constraint."""
        fisher = FisherScores(
            scores={i: float(i + 1) for i in range(6)},
            n_samples=1000,
            method="gradient_variance",
        )
        
        cfg = FWCAConfig(
            total_budget=96,
            strategy="proportional",
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        
        # Total should exactly match budget
        total = sum(scheduler.T_alloc.values())
        assert total == 96
    
    def test_zero_fisher_uniform_allocation(self):
        """Test that zero Fisher scores result in uniform allocation."""
        fisher = FisherScores(
            scores={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            n_samples=1000,
            method="gradient_variance",
        )
        
        cfg = FWCAConfig(
            total_budget=64,
            strategy="proportional",
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        
        # All layers should get equal allocation
        T_values = list(scheduler.T_alloc.values())
        assert len(set(T_values)) == 1  # All same
        assert T_values[0] == 16  # 64 / 4 = 16


class TestThresholdAllocation:
    """Test threshold allocation strategy."""
    
    def test_threshold_basic(self):
        """Test basic threshold allocation."""
        fisher = FisherScores(
            scores={0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
            n_samples=1000,
            method="gradient_variance",
        )
        
        # Budget: 80 = 2*32 + 2*8 (2 high, 2 low)
        cfg = FWCAConfig(
            total_budget=80,
            T_min=8,
            T_max=32,
            strategy="threshold",
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        
        # Top 2 layers (2, 3) should get T_max
        assert scheduler.get_T(3) == 32
        assert scheduler.get_T(2) == 32
        
        # Bottom 2 layers (0, 1) should get T_min
        assert scheduler.get_T(0) == 8
        assert scheduler.get_T(1) == 8
    
    def test_threshold_meets_budget(self):
        """Test that threshold allocation meets budget."""
        fisher = FisherScores(
            scores={i: float(i + 1) for i in range(8)},
            n_samples=1000,
            method="gradient_variance",
        )
        
        cfg = FWCAConfig(
            total_budget=128,
            T_min=8,
            T_max=24,
            strategy="threshold",
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        
        total = sum(scheduler.T_alloc.values())
        assert total == 128


class TestHybridAllocation:
    """Test hybrid allocation strategy."""
    
    def test_hybrid_basic(self):
        """Test basic hybrid allocation."""
        fisher = FisherScores(
            scores={0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
            n_samples=1000,
            method="gradient_variance",
        )
        
        cfg = FWCAConfig(
            total_budget=80,
            T_min=8,
            T_max=32,
            strategy="hybrid",
            alpha=0.5,
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        
        # Should be between proportional and threshold
        # Higher Fisher should still get more T
        T_values = [scheduler.get_T(i) for i in range(4)]
        assert T_values[3] >= T_values[2] >= T_values[1] >= T_values[0]
    
    def test_hybrid_alpha_extremes(self):
        """Test hybrid with alpha=0 and alpha=1."""
        fisher = FisherScores(
            scores={0: 1.0, 1: 2.0, 2: 3.0},
            n_samples=1000,
            method="gradient_variance",
        )
        
        # alpha=1 should be pure proportional
        cfg_prop = FWCAConfig(
            total_budget=48,
            strategy="hybrid",
            alpha=1.0,
        )
        scheduler_prop = FWCAScheduler(fisher, cfg_prop)
        
        # alpha=0 should be pure threshold
        cfg_thresh = FWCAConfig(
            total_budget=48,
            strategy="hybrid",
            alpha=0.0,
        )
        scheduler_thresh = FWCAScheduler(fisher, cfg_thresh)
        
        # Both should meet budget
        assert sum(scheduler_prop.T_alloc.values()) == 48
        assert sum(scheduler_thresh.T_alloc.values()) == 48


class TestBudgetConstraints:
    """Test budget constraint enforcement."""
    
    def test_budget_exactly_met(self):
        """Test that budget is exactly met after quantization (when possible)."""
        # Use budgets that are achievable with 12 layers and T_min=8
        # 12 * 8 = 96 (minimum possible)
        for total_budget in [96, 128, 192]:
            fisher = FisherScores(
                scores={i: float(i + 1) for i in range(12)},
                n_samples=1000,
                method="gradient_variance",
            )

            cfg = FWCAConfig(
                total_budget=total_budget,
                strategy="proportional",
            )

            scheduler = FWCAScheduler(fisher, cfg)

            total = sum(scheduler.T_alloc.values())
            assert total == total_budget, f"Budget mismatch: {total} != {total_budget}"

    def test_budget_infeasible(self):
        """Test that infeasible budgets are handled gracefully."""
        # Budget too low: 64 < 12 * 8 = 96
        fisher = FisherScores(
            scores={i: float(i + 1) for i in range(12)},
            n_samples=1000,
            method="gradient_variance",
        )

        cfg = FWCAConfig(
            total_budget=64,  # Infeasible
            T_min=8,
            strategy="proportional",
        )

        scheduler = FWCAScheduler(fisher, cfg)

        # Should allocate minimum to all layers (best effort)
        total = sum(scheduler.T_alloc.values())
        assert total >= 96  # At least T_min for all layers
    
    def test_T_within_bounds(self):
        """Test that all T allocations are within [T_min, T_max]."""
        fisher = FisherScores(
            scores={i: float(i + 1) ** 2 for i in range(10)},
            n_samples=1000,
            method="gradient_variance",
        )
        
        cfg = FWCAConfig(
            total_budget=160,
            T_min=8,
            T_max=32,
            strategy="proportional",
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        
        for T in scheduler.T_alloc.values():
            assert T >= cfg.T_min
            assert T <= cfg.T_max
    
    def test_T_quantized(self):
        """Test that all T allocations are from quantization levels."""
        fisher = FisherScores(
            scores={i: float(i + 1) for i in range(8)},
            n_samples=1000,
            method="gradient_variance",
        )
        
        cfg = FWCAConfig(
            total_budget=128,
            T_quantize=(8, 12, 16, 24, 32),
            strategy="proportional",
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        
        for T in scheduler.T_alloc.values():
            assert T in cfg.T_quantize


class TestAllocationSummary:
    """Test allocation summary statistics."""
    
    def test_summary_statistics(self):
        """Test that summary provides correct statistics."""
        fisher = FisherScores(
            scores={0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
            n_samples=1000,
            method="gradient_variance",
        )
        
        cfg = FWCAConfig(
            total_budget=80,
            strategy="proportional",
        )
        
        scheduler = FWCAScheduler(fisher, cfg)
        summary = scheduler.get_allocation_summary()
        
        assert summary["total_budget"] == 80
        assert summary["target_budget"] == 80
        assert summary["n_layers"] == 4
        assert summary["strategy"] == "proportional"
        assert "T_mean" in summary
        assert "T_std" in summary
        assert "fisher_mean" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

