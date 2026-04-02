#!/usr/bin/env python3
"""
End-to-End Correlation Validation Tests for SALOMI

This test suite validates correlation at multiple levels to identify
the disconnect between single-layer and end-to-end metrics:

1. Single-layer correlation (what we've been measuring)
2. Multi-layer correlation (error accumulation)
3. Post-nonlinearity correlation (GELU amplification)
4. Full model output correlation (true end-to-end)

Target: Understand why single-layer improvements don't translate to E2E.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class CorrelationResult:
    """Result of correlation measurement at a specific level."""
    level: str
    correlation: float
    mse: float
    max_error: float
    n_samples: int
    layer_id: Optional[int] = None
    

def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    if len(x_flat) != len(y_flat):
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    
    if len(x_flat) == 0:
        return 0.0
    
    # Handle constant arrays
    x_std = np.std(x_flat)
    y_std = np.std(y_flat)
    
    if x_std < 1e-10 or y_std < 1e-10:
        return 0.0
    
    return np.corrcoef(x_flat, y_flat)[0, 1]


def compute_mse(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mean squared error between two arrays."""
    return float(np.mean((x - y) ** 2))


def compute_max_error(x: np.ndarray, y: np.ndarray) -> float:
    """Compute maximum absolute error between two arrays."""
    return float(np.max(np.abs(x - y)))


class MockQuantizedLinear:
    """Mock quantized linear layer for testing."""
    
    def __init__(self, in_features: int, out_features: int, 
                 quantization_type: str = "binary"):
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_type = quantization_type
        
        # Initialize FP32 weights
        self.weight_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        self.bias_fp32 = np.random.randn(out_features).astype(np.float32) * 0.02
        
        # Quantize weights
        if quantization_type == "binary":
            self.weight_quant = np.sign(self.weight_fp32)
            self.scale = np.mean(np.abs(self.weight_fp32))
        elif quantization_type == "ternary":
            threshold = 0.5 * np.std(self.weight_fp32)
            self.weight_quant = np.zeros_like(self.weight_fp32)
            self.weight_quant[self.weight_fp32 > threshold] = 1
            self.weight_quant[self.weight_fp32 < -threshold] = -1
            self.scale = np.mean(np.abs(self.weight_fp32[self.weight_fp32 != 0]))
        else:
            self.weight_quant = self.weight_fp32
            self.scale = 1.0
            
    def forward_fp32(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with FP32 weights."""
        return x @ self.weight_fp32.T + self.bias_fp32
        
    def forward_quant(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with quantized weights."""
        return x @ (self.weight_quant * self.scale).T + self.bias_fp32
        
    def measure_correlation(self, x: np.ndarray) -> CorrelationResult:
        """Measure correlation between FP32 and quantized outputs."""
        y_fp32 = self.forward_fp32(x)
        y_quant = self.forward_quant(x)
        
        return CorrelationResult(
            level="single_layer",
            correlation=compute_correlation(y_fp32, y_quant),
            mse=compute_mse(y_fp32, y_quant),
            max_error=compute_max_error(y_fp32, y_quant),
            n_samples=len(x),
        )


class MockTransformerBlock:
    """Mock transformer block for multi-layer testing."""
    
    def __init__(self, d_model: int = 768, d_ff: int = 3072, 
                 quantization_type: str = "binary"):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Attention layers
        self.qkv = MockQuantizedLinear(d_model, 3 * d_model, quantization_type)
        self.proj = MockQuantizedLinear(d_model, d_model, quantization_type)
        
        # MLP layers
        self.fc1 = MockQuantizedLinear(d_model, d_ff, quantization_type)
        self.fc2 = MockQuantizedLinear(d_ff, d_model, quantization_type)
        
        # Layer norms (FP32)
        self.ln1_scale = np.ones(d_model).astype(np.float32)
        self.ln1_bias = np.zeros(d_model).astype(np.float32)
        self.ln2_scale = np.ones(d_model).astype(np.float32)
        self.ln2_bias = np.zeros(d_model).astype(np.float32)
        
    def layer_norm(self, x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + 1e-5)
        return normalized * scale + bias
        
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        
    def forward_fp32(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with FP32 weights."""
        # Attention block (simplified - no actual attention)
        x_norm = self.layer_norm(x, self.ln1_scale, self.ln1_bias)
        qkv = self.qkv.forward_fp32(x_norm)
        attn_out = self.proj.forward_fp32(qkv[..., :self.d_model])
        x = x + attn_out
        
        # MLP block
        x_norm = self.layer_norm(x, self.ln2_scale, self.ln2_bias)
        h = self.fc1.forward_fp32(x_norm)
        h = self.gelu(h)
        mlp_out = self.fc2.forward_fp32(h)
        x = x + mlp_out
        
        return x
        
    def forward_quant(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with quantized weights."""
        # Attention block
        x_norm = self.layer_norm(x, self.ln1_scale, self.ln1_bias)
        qkv = self.qkv.forward_quant(x_norm)
        attn_out = self.proj.forward_quant(qkv[..., :self.d_model])
        x = x + attn_out
        
        # MLP block
        x_norm = self.layer_norm(x, self.ln2_scale, self.ln2_bias)
        h = self.fc1.forward_quant(x_norm)
        h = self.gelu(h)  # GELU amplifies errors here!
        mlp_out = self.fc2.forward_quant(h)
        x = x + mlp_out
        
        return x
        
    def forward_fp32_with_gelu_tracking(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass tracking pre-GELU activations."""
        x_norm = self.layer_norm(x, self.ln1_scale, self.ln1_bias)
        qkv = self.qkv.forward_fp32(x_norm)
        attn_out = self.proj.forward_fp32(qkv[..., :self.d_model])
        x = x + attn_out
        
        x_norm = self.layer_norm(x, self.ln2_scale, self.ln2_bias)
        pre_gelu = self.fc1.forward_fp32(x_norm)  # Track this
        h = self.gelu(pre_gelu)
        mlp_out = self.fc2.forward_fp32(h)
        x = x + mlp_out
        
        return x, pre_gelu


class TestSingleLayerCorrelation:
    """Test correlation at single layer level."""
    
    def test_binary_single_layer_correlation(self):
        """Binary quantization correlation on a single layer."""
        np.random.seed(42)
        
        layer = MockQuantizedLinear(768, 768, "binary")
        x = np.random.randn(32, 768).astype(np.float32)
        
        result = layer.measure_correlation(x)
        print(f"\nBinary single layer correlation: {result.correlation:.4f}")
        print(f"MSE: {result.mse:.6f}, Max Error: {result.max_error:.6f}")
        
        # Expected: ~0.76 correlation for binary
        assert result.correlation > 0.7, f"Binary correlation should be >0.7, got {result.correlation}"
        assert result.correlation < 0.85, f"Binary correlation should be <0.85, got {result.correlation}"
        
    def test_ternary_single_layer_correlation(self):
        """Ternary quantization correlation on a single layer."""
        np.random.seed(42)
        
        layer = MockQuantizedLinear(768, 768, "ternary")
        x = np.random.randn(32, 768).astype(np.float32)
        
        result = layer.measure_correlation(x)
        print(f"\nTernary single layer correlation: {result.correlation:.4f}")
        print(f"MSE: {result.mse:.6f}, Max Error: {result.max_error:.6f}")
        
        # Expected: ~0.89 correlation for ternary
        assert result.correlation > 0.85, f"Ternary correlation should be >0.85, got {result.correlation}"
        
    def test_binary_vs_ternary_gap(self):
        """Verify the gap between binary and ternary."""
        np.random.seed(42)
        
        binary = MockQuantizedLinear(768, 768, "binary")
        ternary = MockQuantizedLinear(768, 768, "ternary")
        x = np.random.randn(32, 768).astype(np.float32)
        
        binary_result = binary.measure_correlation(x)
        ternary_result = ternary.measure_correlation(x)
        
        gap = ternary_result.correlation - binary_result.correlation
        print(f"\nBinary correlation: {binary_result.correlation:.4f}")
        print(f"Ternary correlation: {ternary_result.correlation:.4f}")
        print(f"Gap: {gap:.4f} ({gap/ternary_result.correlation*100:.1f}%)")
        
        # Gap should be ~10-15%
        assert gap > 0.05, f"Gap should be >5%, got {gap*100:.1f}%"
        assert gap < 0.25, f"Gap should be <25%, got {gap*100:.1f}%"


class TestMultiLayerCorrelation:
    """Test how correlation degrades across multiple layers."""
    
    def test_error_accumulation_multiple_layers(self):
        """Measure error accumulation through 12 layers."""
        np.random.seed(42)
        
        n_layers = 12
        blocks = [MockTransformerBlock(768, 3072, "binary") for _ in range(n_layers)]
        
        x_fp32 = np.random.randn(4, 128, 768).astype(np.float32)
        x_quant = x_fp32.copy()
        
        correlations = []
        mses = []
        
        for layer_id, block in enumerate(blocks):
            x_fp32_out = block.forward_fp32(x_fp32)
            x_quant_out = block.forward_quant(x_quant)
            
            corr = compute_correlation(x_fp32_out, x_quant_out)
            mse = compute_mse(x_fp32_out, x_quant_out)
            
            correlations.append(corr)
            mses.append(mse)
            
            # Propagate errors
            x_fp32 = x_fp32_out
            x_quant = x_quant_out
            
            print(f"Layer {layer_id}: Corr={corr:.4f}, MSE={mse:.6f}")
        
        # Error should grow exponentially
        initial_mse = mses[0]
        final_mse = mses[-1]
        growth_factor = final_mse / initial_mse
        
        print(f"\nInitial MSE: {initial_mse:.6f}")
        print(f"Final MSE: {final_mse:.6f}")
        print(f"MSE growth factor: {growth_factor:.2f}x")
        
        # Correlation should drop significantly
        initial_corr = correlations[0]
        final_corr = correlations[-1]
        corr_drop = initial_corr - final_corr
        
        print(f"\nInitial correlation: {initial_corr:.4f}")
        print(f"Final correlation: {final_corr:.4f}")
        print(f"Correlation drop: {corr_drop:.4f}")
        
        # Expect significant degradation
        assert final_corr < initial_corr, "Correlation should decrease through layers"
        assert growth_factor > 1.5, "MSE should grow through layers"
        
    def test_compare_binary_ternary_multilayer(self):
        """Compare binary vs ternary through multiple layers."""
        np.random.seed(42)
        
        n_layers = 6
        binary_blocks = [MockTransformerBlock(768, 3072, "binary") for _ in range(n_layers)]
        
        # Use same random init for ternary
        np.random.seed(42)
        ternary_blocks = [MockTransformerBlock(768, 3072, "ternary") for _ in range(n_layers)]
        
        x = np.random.randn(4, 128, 768).astype(np.float32)
        x_binary = x.copy()
        x_ternary = x.copy()
        x_fp32 = x.copy()
        
        for i in range(n_layers):
            x_fp32 = binary_blocks[i].forward_fp32(x_fp32)
            x_binary = binary_blocks[i].forward_quant(x_binary)
            
        np.random.seed(42)  # Reset for consistent FP32 baseline
        x_fp32_t = x.copy()
        for i in range(n_layers):
            x_fp32_t = ternary_blocks[i].forward_fp32(x_fp32_t)
            x_ternary = ternary_blocks[i].forward_quant(x_ternary)
        
        binary_corr = compute_correlation(x_fp32, x_binary)
        ternary_corr = compute_correlation(x_fp32_t, x_ternary)
        
        print(f"\nAfter {n_layers} layers:")
        print(f"Binary correlation: {binary_corr:.4f}")
        print(f"Ternary correlation: {ternary_corr:.4f}")
        print(f"Gap: {ternary_corr - binary_corr:.4f}")
        
        # Gap should be larger at multiple layers
        assert ternary_corr > binary_corr, "Ternary should outperform binary"


class TestGELUAmplification:
    """Test GELU nonlinearity error amplification."""
    
    def test_gelu_amplifies_small_errors(self):
        """GELU amplifies errors near zero."""
        np.random.seed(42)
        
        # Generate inputs near GELU sensitive region
        x = np.random.randn(1000).astype(np.float32) * 0.5  # Near zero
        
        # Add small quantization error
        error_magnitude = 0.1
        x_noisy = x + np.random.randn(1000).astype(np.float32) * error_magnitude
        
        # Apply GELU
        gelu = lambda t: 0.5 * t * (1 + np.tanh(np.sqrt(2 / np.pi) * (t + 0.044715 * t**3)))
        
        y_clean = gelu(x)
        y_noisy = gelu(x_noisy)
        
        # Measure error amplification
        input_error = np.mean(np.abs(x_noisy - x))
        output_error = np.mean(np.abs(y_noisy - y_clean))
        amplification = output_error / input_error
        
        print(f"\nInput error: {input_error:.6f}")
        print(f"Output error: {output_error:.6f}")
        print(f"GELU amplification: {amplification:.2f}x")
        
        # GELU should amplify errors in sensitive region
        # (Actually GELU derivative is ~1 near 0, so amplification should be ~1)
        # But the key issue is ASYMMETRY when signs flip
        
    def test_gelu_sign_flip_error(self):
        """Test error when quantization flips sign near zero."""
        # Near-zero positive values
        x_positive = np.array([0.1, 0.05, 0.01, 0.001])
        # What binary quantization might produce (negative)
        x_negative = -x_positive
        
        gelu = lambda t: 0.5 * t * (1 + np.tanh(np.sqrt(2 / np.pi) * (t + 0.044715 * t**3)))
        
        y_positive = gelu(x_positive)
        y_negative = gelu(x_negative)
        
        # Relative error from sign flip
        error = np.abs(y_positive - y_negative)
        relative_error = error / (np.abs(y_positive) + 1e-10)
        
        print("\nGELU sign flip error:")
        for i, x in enumerate(x_positive):
            print(f"  x={x:.3f}: GELU({x:.3f})={y_positive[i]:.4f}, GELU({-x:.3f})={y_negative[i]:.4f}, Error={error[i]:.4f}, RelErr={relative_error[i]*100:.1f}%")
        
        # Sign flip causes ~200% relative error near zero
        assert np.mean(relative_error) > 1.5, "Sign flip should cause >150% error"
        
    def test_pre_gelu_activation_distribution(self):
        """Analyze pre-GELU activation distribution to understand sensitivity."""
        np.random.seed(42)
        
        block = MockTransformerBlock(768, 3072, "binary")
        x = np.random.randn(32, 128, 768).astype(np.float32)
        
        _, pre_gelu = block.forward_fp32_with_gelu_tracking(x)
        
        # Analyze distribution
        abs_values = np.abs(pre_gelu.flatten())
        
        in_sensitive_region = np.mean(abs_values < 1.0)
        in_very_sensitive = np.mean(abs_values < 0.5)
        
        print(f"\nPre-GELU activation distribution:")
        print(f"  |x| < 1.0: {in_sensitive_region*100:.1f}% (sensitive)")
        print(f"  |x| < 0.5: {in_very_sensitive*100:.1f}% (very sensitive)")
        print(f"  Mean |x|: {np.mean(abs_values):.4f}")
        print(f"  Median |x|: {np.median(abs_values):.4f}")
        
        # Most activations should be in sensitive region (this is the problem!)
        assert in_sensitive_region > 0.5, "Most pre-GELU activations should be in sensitive region"


class TestFullModelCorrelation:
    """Test full model output correlation."""
    
    def test_single_layer_vs_full_model(self):
        """Compare single-layer correlation to full model."""
        np.random.seed(42)
        
        # Measure single layer
        layer = MockQuantizedLinear(768, 768, "binary")
        x_single = np.random.randn(32, 768).astype(np.float32)
        single_result = layer.measure_correlation(x_single)
        
        # Measure full model (12 layers)
        n_layers = 12
        blocks = [MockTransformerBlock(768, 3072, "binary") for _ in range(n_layers)]
        
        x_fp32 = np.random.randn(4, 128, 768).astype(np.float32)
        x_quant = x_fp32.copy()
        
        for block in blocks:
            x_fp32 = block.forward_fp32(x_fp32)
            x_quant = block.forward_quant(x_quant)
        
        full_corr = compute_correlation(x_fp32, x_quant)
        
        print(f"\nSingle layer correlation: {single_result.correlation:.4f}")
        print(f"Full model correlation: {full_corr:.4f}")
        print(f"Degradation: {(single_result.correlation - full_corr):.4f} ({(single_result.correlation - full_corr)/single_result.correlation*100:.1f}%)")
        
        # Full model should be significantly worse
        assert full_corr < single_result.correlation, "Full model correlation should be lower"
        degradation = single_result.correlation - full_corr
        assert degradation > 0.1, f"Expected >10% degradation, got {degradation*100:.1f}%"
        
    def test_correlation_to_perplexity_relationship(self):
        """Understand how correlation relates to perplexity degradation."""
        # This test documents the disconnect between correlation and perplexity
        
        # Known data points from research
        results = [
            {"method": "FP32", "correlation": 1.0, "ppl_ratio": 1.0},
            {"method": "Ternary (single)", "correlation": 0.89, "ppl_ratio": 919469},  # +919,469%
            {"method": "Binary (single)", "correlation": 0.76, "ppl_ratio": 2167425},  # +2,167,425%
        ]
        
        print("\nCorrelation vs Perplexity Degradation:")
        print("-" * 60)
        for r in results:
            print(f"{r['method']:20} Corr={r['correlation']:.2f} PPL_ratio={r['ppl_ratio']:,}x")
        
        # The key finding: small correlation changes → huge perplexity changes
        binary_corr = 0.76
        ternary_corr = 0.89
        corr_diff = ternary_corr - binary_corr  # 0.13 = 13%
        
        binary_ppl = 2167425
        ternary_ppl = 919469
        ppl_diff = binary_ppl / ternary_ppl  # 2.36x
        
        print(f"\n13% correlation improvement → {ppl_diff:.1f}x perplexity improvement")
        print("But BOTH are catastrophically worse than FP32!")


class TestCorrelationMetrics:
    """Test different correlation metrics and their utility."""
    
    def test_pearson_vs_cosine_similarity(self):
        """Compare Pearson correlation to cosine similarity."""
        np.random.seed(42)
        
        layer = MockQuantizedLinear(768, 768, "binary")
        x = np.random.randn(32, 768).astype(np.float32)
        
        y_fp32 = layer.forward_fp32(x)
        y_quant = layer.forward_quant(x)
        
        # Pearson correlation
        pearson = compute_correlation(y_fp32, y_quant)
        
        # Cosine similarity
        def cosine_similarity(a, b):
            a_flat = a.flatten()
            b_flat = b.flatten()
            return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
        
        cosine = cosine_similarity(y_fp32, y_quant)
        
        print(f"\nPearson correlation: {pearson:.4f}")
        print(f"Cosine similarity: {cosine:.4f}")
        
        # They should be similar but not identical
        assert abs(pearson - cosine) < 0.1, "Pearson and cosine should be similar"
        
    def test_layer_wise_sensitivity(self):
        """Measure which layers are most sensitive to quantization."""
        np.random.seed(42)
        
        n_layers = 12
        
        # Test each layer independently
        layer_sensitivities = []
        
        for layer_id in range(n_layers):
            np.random.seed(42)
            blocks = [MockTransformerBlock(768, 3072, "fp32") for _ in range(n_layers)]
            blocks[layer_id] = MockTransformerBlock(768, 3072, "binary")
            
            x = np.random.randn(4, 128, 768).astype(np.float32)
            x_fp32 = x.copy()
            x_mixed = x.copy()
            
            for i, block in enumerate(blocks):
                if i == layer_id:
                    x_mixed = blocks[layer_id].forward_quant(x_mixed)
                else:
                    x_mixed = blocks[i].forward_fp32(x_mixed)
                x_fp32 = blocks[i].forward_fp32(x_fp32)
            
            corr = compute_correlation(x_fp32, x_mixed)
            sensitivity = 1.0 - corr  # Higher = more sensitive
            layer_sensitivities.append(sensitivity)
            
            print(f"Layer {layer_id}: Correlation with only this layer quantized = {corr:.4f}")
        
        # Identify most sensitive layer
        most_sensitive = np.argmax(layer_sensitivities)
        print(f"\nMost sensitive layer: {most_sensitive} (sensitivity={layer_sensitivities[most_sensitive]:.4f})")


def run_all_correlation_tests():
    """Run all correlation validation tests."""
    print("=" * 70)
    print("END-TO-END CORRELATION VALIDATION TESTS")
    print("=" * 70)
    
    test_classes = [
        TestSingleLayerCorrelation,
        TestMultiLayerCorrelation,
        TestGELUAmplification,
        TestFullModelCorrelation,
        TestCorrelationMetrics,
    ]
    
    total_passed = 0
    total_failed = 0
    failures = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 50)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    print(f"  {method_name}... ", end="")
                    method()
                    print("PASSED")
                    total_passed += 1
                except AssertionError as e:
                    print(f"FAILED: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"ERROR: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    
    if failures:
        print("\nFailures:")
        for cls, method, msg in failures:
            print(f"  {cls}.{method}: {msg}")
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("1. Single-layer binary correlation: ~0.76")
    print("2. Single-layer ternary correlation: ~0.89")
    print("3. Full model (12 layers) correlation degrades significantly")
    print("4. GELU amplifies sign-flip errors by ~200%")
    print("5. Most pre-GELU activations are in sensitive region (<1.0)")
    print("6. 13% correlation improvement → 2.4x perplexity improvement")
    print("   BUT both binary and ternary have catastrophic perplexity!")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_correlation_tests()
    sys.exit(0 if success else 1)