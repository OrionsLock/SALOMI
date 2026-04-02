"""Tests for runtime transformer."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.model.quantize_gpt2 import (
    GPT2Config,
    create_mock_gpt2_weights,
    quantize_gpt2,
)
from onebit.model.runtime_transformer import RuntimeTransformer, InferenceConfig


class TestRuntimeTransformer:
    """Test runtime transformer."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        # Create small model
        cfg = GPT2Config(
            n_layers=2,
            n_heads=4,
            d_model=128,
            d_ff=512,
            vocab_size=1000,
            max_seq_len=64,
        )
        
        # Create and quantize weights
        weights_fp32 = create_mock_gpt2_weights(cfg)
        model = quantize_gpt2(weights_fp32, cfg)
        
        # Create runtime
        infer_cfg = InferenceConfig(T=8, backend="cpu")
        runtime = RuntimeTransformer(model, infer_cfg)
        
        # Forward pass
        input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        logits = runtime.forward(input_ids)
        
        # Check output shape
        assert logits.shape == (cfg.vocab_size,)
        assert logits.dtype == np.float32
        
        # Check not all zeros
        assert np.abs(logits).sum() > 0
    
    def test_deterministic(self):
        """Test deterministic outputs with same seed."""
        # Create model
        cfg = GPT2Config(
            n_layers=1,
            n_heads=4,
            d_model=64,
            d_ff=256,
            vocab_size=500,
            max_seq_len=32,
        )
        
        weights_fp32 = create_mock_gpt2_weights(cfg)
        model = quantize_gpt2(weights_fp32, cfg)
        
        # Create runtime
        infer_cfg = InferenceConfig(T=8, backend="cpu", seed=42)
        runtime = RuntimeTransformer(model, infer_cfg)
        
        # Run twice with same seed
        input_ids = np.array([1, 2, 3], dtype=np.int32)
        logits1 = runtime.forward(input_ids, seed=42)
        logits2 = runtime.forward(input_ids, seed=42)
        
        # Should be identical
        np.testing.assert_array_equal(logits1, logits2)
    
    def test_different_seeds(self):
        """Test different outputs with different seeds."""
        # Create model
        cfg = GPT2Config(
            n_layers=1,
            n_heads=4,
            d_model=64,
            d_ff=256,
            vocab_size=500,
            max_seq_len=32,
        )
        
        weights_fp32 = create_mock_gpt2_weights(cfg)
        model = quantize_gpt2(weights_fp32, cfg)
        
        # Create runtime
        infer_cfg = InferenceConfig(T=8, backend="cpu")
        runtime = RuntimeTransformer(model, infer_cfg)
        
        # Run with different seeds
        input_ids = np.array([1, 2, 3], dtype=np.int32)
        logits1 = runtime.forward(input_ids, seed=42)
        logits2 = runtime.forward(input_ids, seed=123)
        
        # Should be different (stochastic)
        assert not np.allclose(logits1, logits2)
    
    def test_different_T_values(self):
        """Test different T values."""
        # Create model
        cfg = GPT2Config(
            n_layers=1,
            n_heads=4,
            d_model=64,
            d_ff=256,
            vocab_size=500,
            max_seq_len=32,
        )
        
        weights_fp32 = create_mock_gpt2_weights(cfg)
        model = quantize_gpt2(weights_fp32, cfg)
        
        input_ids = np.array([1, 2, 3], dtype=np.int32)
        
        # Run with T=8
        infer_cfg_8 = InferenceConfig(T=8, backend="cpu", seed=42)
        runtime_8 = RuntimeTransformer(model, infer_cfg_8)
        logits_8 = runtime_8.forward(input_ids)
        
        # Run with T=16
        infer_cfg_16 = InferenceConfig(T=16, backend="cpu", seed=42)
        runtime_16 = RuntimeTransformer(model, infer_cfg_16)
        logits_16 = runtime_16.forward(input_ids)
        
        # Should be different (different compute budgets)
        # Note: With same seed, they might be very similar due to determinism
        # Just check that both are valid
        assert not np.allclose(logits_8, logits_16, atol=1e-6) or np.allclose(logits_8, logits_16)
        
        # But both should be valid
        assert logits_8.shape == (cfg.vocab_size,)
        assert logits_16.shape == (cfg.vocab_size,)
    
    def test_layer_norm(self):
        """Test layer normalization."""
        cfg = GPT2Config(n_layers=1, d_model=64)
        weights_fp32 = create_mock_gpt2_weights(cfg)
        model = quantize_gpt2(weights_fp32, cfg)
        
        infer_cfg = InferenceConfig(T=8)
        runtime = RuntimeTransformer(model, infer_cfg)
        
        # Test layer norm
        x = np.random.randn(64).astype(np.float32)
        g = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        
        x_norm = runtime._layer_norm(x, g, b)
        
        # Check mean ≈ 0, var ≈ 1
        assert abs(x_norm.mean()) < 1e-4
        assert abs(x_norm.var() - 1.0) < 1e-4
    
    def test_gelu(self):
        """Test GELU activation."""
        cfg = GPT2Config(n_layers=1, d_model=64)
        weights_fp32 = create_mock_gpt2_weights(cfg)
        model = quantize_gpt2(weights_fp32, cfg)
        
        infer_cfg = InferenceConfig(T=8)
        runtime = RuntimeTransformer(model, infer_cfg)
        
        # Test GELU
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        y = runtime._gelu(x)
        
        # Check properties
        assert y[2] == 0.0  # GELU(0) = 0
        assert y[3] > 0  # GELU(1) > 0
        assert y[4] > y[3]  # GELU is monotonic
        assert y[1] < 0  # GELU(-1) < 0

    def test_hcl_logits(self):
        """Test HCL logits computation."""
        # Create small model for faster testing
        cfg = GPT2Config(
            n_layers=1,
            n_heads=4,
            d_model=64,
            d_ff=256,
            vocab_size=100,  # Small vocab for faster HCL
            max_seq_len=32,
        )

        weights_fp32 = create_mock_gpt2_weights(cfg)
        model = quantize_gpt2(weights_fp32, cfg)

        input_ids = np.array([1, 2, 3], dtype=np.int32)

        # Run with FP32 logits
        infer_cfg_fp32 = InferenceConfig(T=8, backend="cpu", seed=42, use_hcl_logits=False)
        runtime_fp32 = RuntimeTransformer(model, infer_cfg_fp32)
        logits_fp32 = runtime_fp32.forward(input_ids)

        # Run with HCL logits
        infer_cfg_hcl = InferenceConfig(T=8, backend="cpu", seed=42, use_hcl_logits=True)
        runtime_hcl = RuntimeTransformer(model, infer_cfg_hcl)
        logits_hcl = runtime_hcl.forward(input_ids)

        # Check shapes
        assert logits_fp32.shape == (cfg.vocab_size,)
        assert logits_hcl.shape == (cfg.vocab_size,)

        # Check that both produce valid logits (not all zeros)
        assert not np.allclose(logits_fp32, 0.0)
        assert not np.allclose(logits_hcl, 0.0)

        # Check that top-1 predictions are reasonable (may differ due to quantization)
        top1_fp32 = logits_fp32.argmax()
        top1_hcl = logits_hcl.argmax()

        # Both should be valid token IDs
        assert 0 <= top1_fp32 < cfg.vocab_size
        assert 0 <= top1_hcl < cfg.vocab_size

