"""Tests for quality polish (MoM-SC + PTE + Richardson).

Tests cover:
1. Richardson extrapolation
2. MoM-SC (Method-of-Moments Self-Correction)
3. PTE (Probability-Transformed Ensemble)
4. Combined polish pipeline
5. Variance reduction validation
"""
import pytest
import numpy as np
from onebit.core.mom_sc import (
    MoMSelfCorrector,
    MoMSCConfig,
    richardson_extrapolation,
    compute_variance_reduction,
)
from onebit.core.pte import (
    PTEnsemble,
    PTEConfig,
    calibrate_temperature,
    compute_calibration_error,
)


class TestRichardsonExtrapolation:
    """Test Richardson extrapolation."""
    
    def test_basic_extrapolation(self):
        """Test basic Richardson extrapolation."""
        # Simulate bias: y_T = y_true + 10/T
        y_true = 5.0
        y_8 = y_true + 10/8  # 6.25
        y_16 = y_true + 10/16  # 5.625
        
        # Extrapolate
        y_extrap = richardson_extrapolation(y_8, y_16, T_coarse=8, T_fine=16, order=1)
        
        # Should be close to y_true
        assert abs(y_extrap - y_true) < 0.1
    
    def test_same_T(self):
        """Test extrapolation with same T (should average)."""
        y_extrap = richardson_extrapolation(5.0, 6.0, T_coarse=16, T_fine=16)
        assert y_extrap == 5.5
    
    def test_higher_order(self):
        """Test higher-order extrapolation."""
        # Simulate quadratic bias: y_T = y_true + 10/T^2
        y_true = 5.0
        y_8 = y_true + 10/(8**2)  # 5.15625
        y_16 = y_true + 10/(16**2)  # 5.0390625
        
        # Extrapolate with order=2
        y_extrap = richardson_extrapolation(y_8, y_16, T_coarse=8, T_fine=16, order=2)
        
        # Should be closer to y_true than order=1
        assert abs(y_extrap - y_true) < 0.2


class TestMoMSC:
    """Test Method-of-Moments Self-Correction."""
    
    def test_power_law_correction(self):
        """Test power law bias correction."""
        # Simulate bias: y_T = y_true + 10/T
        y_true = 5.0
        # Use only 2 estimates for exact Richardson extrapolation
        estimates = {
            8: y_true + 10/8,   # 6.25
            16: y_true + 10/16,  # 5.625
        }

        cfg = MoMSCConfig(bias_model="power_law")
        corrector = MoMSelfCorrector(cfg)

        y_corrected = corrector.correct(estimates)

        # Should be very close to y_true (Richardson extrapolation is exact for this bias model)
        error_corrected = abs(y_corrected - y_true)

        assert error_corrected < 0.1
    
    def test_exponential_correction(self):
        """Test exponential bias correction."""
        # Simulate exponential bias
        y_true = 5.0
        estimates = {
            8: y_true + 2.0 * np.exp(-0.1 * 8),
            16: y_true + 2.0 * np.exp(-0.1 * 16),
            32: y_true + 2.0 * np.exp(-0.1 * 32),
        }
        
        cfg = MoMSCConfig(bias_model="exponential")
        corrector = MoMSelfCorrector(cfg)
        
        y_corrected = corrector.correct(estimates)
        
        # Should be reasonable
        assert 4.0 < y_corrected < 6.0
    
    def test_linear_correction(self):
        """Test linear bias correction."""
        # Simulate linear bias
        y_true = 5.0
        estimates = {
            8: y_true + 2.0 - 0.1 * 8,
            16: y_true + 2.0 - 0.1 * 16,
            32: y_true + 2.0 - 0.1 * 32,
        }
        
        cfg = MoMSCConfig(bias_model="linear")
        corrector = MoMSelfCorrector(cfg)
        
        y_corrected = corrector.correct(estimates)
        
        # Should be reasonable
        assert 3.0 < y_corrected < 7.0
    
    def test_single_estimate(self):
        """Test with single estimate (should return as-is)."""
        estimates = {16: 5.0}
        
        corrector = MoMSelfCorrector()
        y_corrected = corrector.correct(estimates)
        
        assert y_corrected == 5.0
    
    def test_weighted_correction(self):
        """Test weighted correction."""
        estimates = {
            8: 5.0,
            16: 5.5,
            32: 5.8,
        }
        
        # Weight higher T more
        weights = {8: 1.0, 16: 2.0, 32: 3.0}
        
        corrector = MoMSelfCorrector()
        y_corrected = corrector.correct(estimates, weights)
        
        # Should be closer to higher-T estimates
        assert y_corrected > 5.4
    
    def test_batch_correction(self):
        """Test batch correction."""
        estimates_batch = [
            {8: 5.0, 16: 5.5},
            {8: 3.0, 16: 3.3},
            {8: 7.0, 16: 7.2},
        ]
        
        corrector = MoMSelfCorrector()
        y_corrected_batch = corrector.correct_batch(estimates_batch)
        
        assert len(y_corrected_batch) == 3
        assert all(isinstance(y, (int, float, np.number)) for y in y_corrected_batch)


class TestVarianceReduction:
    """Test variance reduction statistics."""
    
    def test_variance_reduction(self):
        """Test variance reduction computation."""
        # Original estimates (high variance)
        original = np.array([4.0, 5.0, 6.0, 4.5, 5.5])
        
        # Corrected estimates (low variance)
        corrected = np.array([4.9, 5.0, 5.1, 4.95, 5.05])
        
        stats = compute_variance_reduction(original, corrected)
        
        assert stats["var_original"] > stats["var_corrected"]
        assert stats["reduction_ratio"] > 1.0
        assert stats["reduction_pct"] > 0.0


class TestPTE:
    """Test Probability-Transformed Ensemble."""
    
    def test_basic_ensemble(self):
        """Test basic ensemble."""
        # Create logits for 3 runs
        vocab_size = 10
        logits_list = [
            np.random.randn(vocab_size),
            np.random.randn(vocab_size),
            np.random.randn(vocab_size),
        ]
        
        pte = PTEnsemble()
        probs = pte.ensemble(logits_list)
        
        # Should be valid probability distribution
        assert probs.shape == (vocab_size,)
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
    
    def test_single_logits(self):
        """Test with single logits (should just softmax)."""
        logits = np.array([1.0, 2.0, 3.0])
        
        pte = PTEnsemble()
        probs = pte.ensemble([logits])
        
        # Should be valid probability distribution
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
    
    def test_temperature_scaling(self):
        """Test temperature scaling."""
        logits = np.array([1.0, 2.0, 3.0])
        
        # High temperature (smoother)
        pte_smooth = PTEnsemble(PTEConfig(temperature=2.0))
        probs_smooth = pte_smooth.ensemble([logits])
        
        # Low temperature (sharper)
        pte_sharp = PTEnsemble(PTEConfig(temperature=0.5))
        probs_sharp = pte_sharp.ensemble([logits])
        
        # Sharp should have higher max probability
        assert probs_sharp.max() > probs_smooth.max()
    
    def test_probability_sharpening(self):
        """Test probability sharpening."""
        logits = np.array([1.0, 2.0, 3.0])
        
        # No sharpening
        pte_normal = PTEnsemble(PTEConfig(alpha=1.0))
        probs_normal = pte_normal.ensemble([logits])
        
        # Sharpening
        pte_sharp = PTEnsemble(PTEConfig(alpha=2.0))
        probs_sharp = pte_sharp.ensemble([logits])
        
        # Sharp should have higher max probability
        assert probs_sharp.max() > probs_normal.max()
    
    def test_geometric_mean_ensemble(self):
        """Test geometric mean ensemble."""
        logits_list = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5, 3.5]),
            np.array([0.5, 1.5, 2.5]),
        ]
        
        cfg = PTEConfig(ensemble_method="geometric_mean")
        pte = PTEnsemble(cfg)
        probs = pte.ensemble(logits_list)
        
        # Should be valid probability distribution
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
    
    def test_median_ensemble(self):
        """Test median ensemble."""
        logits_list = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5, 3.5]),
            np.array([0.5, 1.5, 2.5]),
        ]
        
        cfg = PTEConfig(ensemble_method="median")
        pte = PTEnsemble(cfg)
        probs = pte.ensemble(logits_list)
        
        # Should be valid probability distribution
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
    
    def test_weighted_ensemble(self):
        """Test weighted ensemble."""
        logits_list = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5, 3.5]),
        ]
        
        # Weight second more
        weights = [1.0, 3.0]
        
        pte = PTEnsemble()
        probs = pte.ensemble(logits_list, weights)
        
        # Should be valid probability distribution
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)


class TestCalibration:
    """Test calibration utilities."""
    
    def test_temperature_calibration(self):
        """Test temperature calibration."""
        # Create synthetic data
        n_samples = 100
        vocab_size = 10
        
        # Logits (slightly overconfident)
        logits = np.random.randn(n_samples, vocab_size) * 2.0
        
        # Targets
        targets = np.random.randint(0, vocab_size, n_samples)
        
        # Calibrate temperature
        T_opt = calibrate_temperature(logits, targets)
        
        # Should be reasonable
        assert 0.5 <= T_opt <= 2.0
    
    def test_calibration_error(self):
        """Test calibration error computation."""
        # Create synthetic data
        n_samples = 100
        vocab_size = 10
        
        # Perfect calibration: probs = one-hot
        probs = np.zeros((n_samples, vocab_size))
        targets = np.random.randint(0, vocab_size, n_samples)
        for i, t in enumerate(targets):
            probs[i, t] = 1.0
        
        stats = compute_calibration_error(probs, targets)
        
        # Should have perfect accuracy
        assert stats["accuracy"] == 1.0
        assert stats["ece"] >= 0.0


class TestCombinedPolish:
    """Test combined polish pipeline."""
    
    def test_mom_sc_plus_pte(self):
        """Test MoM-SC + PTE pipeline."""
        # Simulate multiple runs at different T
        vocab_size = 10
        n_runs = 3
        
        # For each T, run multiple times
        logits_T8 = [np.random.randn(vocab_size) + 0.5 for _ in range(n_runs)]
        logits_T16 = [np.random.randn(vocab_size) + 0.3 for _ in range(n_runs)]
        logits_T32 = [np.random.randn(vocab_size) + 0.1 for _ in range(n_runs)]
        
        # Step 1: PTE ensemble for each T
        pte = PTEnsemble()
        probs_T8 = pte.ensemble(logits_T8)
        probs_T16 = pte.ensemble(logits_T16)
        probs_T32 = pte.ensemble(logits_T32)
        
        # Step 2: MoM-SC correction (on log-probs for numerical stability)
        log_probs_estimates = {
            8: np.log(probs_T8 + 1e-10),
            16: np.log(probs_T16 + 1e-10),
            32: np.log(probs_T32 + 1e-10),
        }
        
        corrector = MoMSelfCorrector()
        log_probs_corrected = corrector.correct_batch([
            {T: log_probs_estimates[T][i] for T in [8, 16, 32]}
            for i in range(vocab_size)
        ])
        
        # Convert back to probs
        probs_final = np.exp(log_probs_corrected)
        probs_final = probs_final / probs_final.sum()
        
        # Should be valid probability distribution
        assert np.allclose(probs_final.sum(), 1.0)
        assert np.all(probs_final >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

