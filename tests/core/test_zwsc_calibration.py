"""Tests for Zero-Write Self-Calibration (ZWSC) module.

Tests cover:
1. Online statistics computation (Welford's algorithm)
2. Statistics collection from activation streams
3. Affine parameter computation
4. Affine transformation application
5. Calibration profile creation
6. Domain-specific calibration
"""
import numpy as np
import pytest
from onebit.core.calib_zwsc import (
    OnlineStats,
    collect_stats,
    compute_affine_params,
    apply_affine,
    apply_affine_inplace,
    create_profile,
    LayerStats,
    AffineParams,
    CalibrationProfile,
)


class TestOnlineStats:
    """Test online statistics computation."""
    
    def test_single_sample(self):
        """Test with a single sample."""
        stats = OnlineStats(dim=3)
        x = np.array([1.0, 2.0, 3.0])
        stats.update(x)
        
        mean, std = stats.get_stats()
        np.testing.assert_allclose(mean, x, rtol=1e-5)
        np.testing.assert_allclose(std, np.ones(3), rtol=1e-5)  # Default to 1.0 for n<2
    
    def test_multiple_samples(self):
        """Test with multiple samples."""
        stats = OnlineStats(dim=2)
        samples = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        
        for sample in samples:
            stats.update(sample)
        
        mean, std = stats.get_stats()
        expected_mean = np.mean(samples, axis=0)
        expected_std = np.std(samples, axis=0)
        
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(std, expected_std, rtol=1e-5)
    
    def test_constant_samples(self):
        """Test with constant samples (zero variance)."""
        stats = OnlineStats(dim=2)
        constant = np.array([5.0, 10.0])
        
        for _ in range(10):
            stats.update(constant)
        
        mean, std = stats.get_stats()
        np.testing.assert_allclose(mean, constant, rtol=1e-5)
        # Std should be near zero (with epsilon for stability)
        assert np.all(std < 1e-3)


class TestCollectStats:
    """Test statistics collection from activation streams."""
    
    def test_single_layer(self):
        """Test collecting stats for a single layer."""
        def activation_stream():
            layer_id = 0
            for i in range(100):
                x_in = np.random.randn(64).astype(np.float32)
                x_out = x_in * 2.0 + 1.0  # Simple transformation
                yield layer_id, x_in, x_out
        
        stats = collect_stats(activation_stream(), n_tokens=100)
        
        assert 0 in stats
        assert stats[0].n_samples == 100
        assert stats[0].mu_in.shape == (64,)
        assert stats[0].sigma_in.shape == (64,)
        assert stats[0].mu_out.shape == (64,)
        assert stats[0].sigma_out.shape == (64,)
    
    def test_multiple_layers(self):
        """Test collecting stats for multiple layers."""
        def activation_stream():
            for i in range(50):
                for layer_id in [0, 1, 2]:
                    x_in = np.random.randn(32).astype(np.float32)
                    x_out = x_in * 1.5
                    yield layer_id, x_in, x_out
        
        stats = collect_stats(activation_stream(), n_tokens=150)
        
        assert len(stats) == 3
        for layer_id in [0, 1, 2]:
            assert layer_id in stats
            assert stats[layer_id].n_samples == 50
    
    def test_layer_filtering(self):
        """Test filtering by layer IDs."""
        def activation_stream():
            for i in range(30):
                for layer_id in [0, 1, 2, 3]:
                    x_in = np.random.randn(16).astype(np.float32)
                    x_out = x_in
                    yield layer_id, x_in, x_out
        
        stats = collect_stats(activation_stream(), n_tokens=60, layer_ids=[1, 3])
        
        assert len(stats) == 2
        assert 1 in stats
        assert 3 in stats
        assert 0 not in stats
        assert 2 not in stats
    
    def test_early_termination(self):
        """Test that collection stops after n_tokens."""
        def activation_stream():
            for i in range(1000):  # Generate many tokens
                x_in = np.random.randn(8).astype(np.float32)
                x_out = x_in
                yield 0, x_in, x_out
        
        stats = collect_stats(activation_stream(), n_tokens=50)
        
        assert stats[0].n_samples == 50  # Should stop at 50


class TestComputeAffineParams:
    """Test affine parameter computation."""
    
    def test_output_mode(self):
        """Test computing affine params from output statistics."""
        # Create mock statistics
        stats = {
            0: LayerStats(
                mu_in=np.array([0.0, 0.0]),
                sigma_in=np.array([1.0, 1.0]),
                mu_out=np.array([5.0, 10.0]),
                sigma_out=np.array([2.0, 4.0]),
                n_samples=100,
            )
        }
        
        params = compute_affine_params(stats, target_mean=0.0, target_std=1.0, mode="output")
        
        assert 0 in params
        # a = target_std / sigma_out = 1.0 / [2.0, 4.0] = [0.5, 0.25]
        np.testing.assert_allclose(params[0].a, np.array([0.5, 0.25]), rtol=1e-5)
        # b = target_mean - a * mu_out = 0.0 - [0.5, 0.25] * [5.0, 10.0] = [-2.5, -2.5]
        np.testing.assert_allclose(params[0].b, np.array([-2.5, -2.5]), rtol=1e-5)
    
    def test_input_mode(self):
        """Test computing affine params from input statistics."""
        stats = {
            0: LayerStats(
                mu_in=np.array([2.0, 4.0]),
                sigma_in=np.array([1.0, 2.0]),
                mu_out=np.array([0.0, 0.0]),
                sigma_out=np.array([1.0, 1.0]),
                n_samples=100,
            )
        }
        
        params = compute_affine_params(stats, target_mean=0.0, target_std=1.0, mode="input")
        
        # a = 1.0 / [1.0, 2.0] = [1.0, 0.5]
        np.testing.assert_allclose(params[0].a, np.array([1.0, 0.5]), rtol=1e-5)
        # b = 0.0 - [1.0, 0.5] * [2.0, 4.0] = [-2.0, -2.0]
        np.testing.assert_allclose(params[0].b, np.array([-2.0, -2.0]), rtol=1e-5)
    
    def test_both_mode(self):
        """Test computing affine params from averaged statistics."""
        stats = {
            0: LayerStats(
                mu_in=np.array([1.0, 2.0]),
                sigma_in=np.array([1.0, 1.0]),
                mu_out=np.array([3.0, 4.0]),
                sigma_out=np.array([1.0, 1.0]),
                n_samples=100,
            )
        }
        
        params = compute_affine_params(stats, target_mean=0.0, target_std=1.0, mode="both")
        
        # Average: mu = [2.0, 3.0], sigma = [1.0, 1.0]
        # a = 1.0 / 1.0 = 1.0
        np.testing.assert_allclose(params[0].a, np.array([1.0, 1.0]), rtol=1e-5)
        # b = 0.0 - 1.0 * [2.0, 3.0] = [-2.0, -3.0]
        np.testing.assert_allclose(params[0].b, np.array([-2.0, -3.0]), rtol=1e-5)
    
    def test_custom_target(self):
        """Test with custom target mean and std."""
        stats = {
            0: LayerStats(
                mu_in=np.array([0.0]),
                sigma_in=np.array([1.0]),
                mu_out=np.array([10.0]),
                sigma_out=np.array([5.0]),
                n_samples=100,
            )
        }
        
        params = compute_affine_params(
            stats, target_mean=5.0, target_std=2.0, mode="output"
        )
        
        # a = 2.0 / 5.0 = 0.4
        np.testing.assert_allclose(params[0].a, np.array([0.4]), rtol=1e-5)
        # b = 5.0 - 0.4 * 10.0 = 1.0
        np.testing.assert_allclose(params[0].b, np.array([1.0]), rtol=1e-5)


class TestApplyAffine:
    """Test affine transformation application."""
    
    def test_apply_affine_1d(self):
        """Test applying affine transformation to 1D array."""
        y = np.array([1.0, 2.0, 3.0])
        params = AffineParams(
            a=np.array([2.0, 2.0, 2.0]),
            b=np.array([1.0, 1.0, 1.0]),
        )
        
        result = apply_affine(y, params)
        expected = np.array([3.0, 5.0, 7.0])  # 2*y + 1
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        # Original should be unchanged
        np.testing.assert_allclose(y, np.array([1.0, 2.0, 3.0]), rtol=1e-5)
    
    def test_apply_affine_2d(self):
        """Test applying affine transformation to 2D array."""
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        params = AffineParams(
            a=np.array([0.5, 0.5]),
            b=np.array([0.0, 1.0]),
        )
        
        result = apply_affine(y, params)
        expected = np.array([[0.5, 2.0], [1.5, 3.0]])  # [0.5*y[0], 0.5*y[1]+1]
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_apply_affine_inplace(self):
        """Test in-place affine transformation."""
        y = np.array([1.0, 2.0, 3.0])
        params = AffineParams(
            a=np.array([3.0, 3.0, 3.0]),
            b=np.array([-1.0, -1.0, -1.0]),
        )
        
        result = apply_affine_inplace(y, params)
        expected = np.array([2.0, 5.0, 8.0])  # 3*y - 1
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        # Should modify in-place
        np.testing.assert_allclose(y, expected, rtol=1e-5)
        assert result is y  # Same object


class TestCalibrationProfile:
    """Test calibration profile creation and usage."""
    
    def test_create_profile(self):
        """Test creating a calibration profile."""
        def activation_stream():
            for i in range(100):
                x_in = np.random.randn(32).astype(np.float32)
                x_out = x_in * 2.0 + 3.0
                yield 0, x_in, x_out
        
        profile = create_profile(
            name="test_profile",
            activation_stream=activation_stream(),
            n_tokens=100,
            target_mean=0.0,
            target_std=1.0,
            mode="output",
            domain_tags=["test", "synthetic"],
        )
        
        assert profile.name == "test_profile"
        assert profile.n_tokens == 100
        assert profile.domain_tags == ["test", "synthetic"]
        assert 0 in profile.affine_params
        assert isinstance(profile.affine_params[0], AffineParams)
    
    def test_profile_normalization(self):
        """Test that profile correctly normalizes activations."""
        # Create a stream with known statistics
        def activation_stream():
            np.random.seed(42)
            for i in range(1000):
                # Output has mean=10, std=5
                x_in = np.random.randn(16).astype(np.float32)
                x_out = x_in * 5.0 + 10.0
                yield 0, x_in, x_out
        
        profile = create_profile(
            name="normalize_test",
            activation_stream=activation_stream(),
            n_tokens=1000,
            target_mean=0.0,
            target_std=1.0,
            mode="output",
        )
        
        # Test on new data with same distribution
        np.random.seed(123)
        test_data = np.random.randn(100, 16).astype(np.float32) * 5.0 + 10.0
        
        # Apply calibration
        calibrated = apply_affine(test_data, profile.affine_params[0])
        
        # Check that calibrated data has target statistics
        assert abs(np.mean(calibrated)) < 0.2  # Close to 0
        assert abs(np.std(calibrated) - 1.0) < 0.2  # Close to 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

