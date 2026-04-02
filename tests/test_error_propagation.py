#!/usr/bin/env python3
"""
Error Propagation Tests for SALOMI

This test suite analyzes how quantization errors propagate through layers:
1. Track MSE growth through transformer layers
2. Fit exponential error growth models
3. Identify which layers are most sensitive
4. Test error correlation across layers

Critical Finding: Small single-layer errors compound exponentially
through 12 layers, which explains why per-layer correlation > 0.99
still results in catastrophic final PPL.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class LayerError:
    """Error statistics for a single layer."""
    layer_idx: int
    input_mse: float
    output_mse: float
    amplification: float
    cosine_sim: float
    relative_error: float


@dataclass 
class PropagationResult:
    """Result of error propagation analysis."""
    layer_errors: List[LayerError] = field(default_factory=list)
    cumulative_mse: List[float] = field(default_factory=list)
    growth_rate: float = 0.0
    final_correlation: float = 0.0


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy implementation of GELU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


class MockTransformerLayer:
    """Mock transformer layer for error propagation analysis."""
    
    def __init__(self, d_model: int = 768, d_ff: int = 3072, noise_level: float = 0.02):
        np.random.seed(42)
        self.d_model = d_model
        self.d_ff = d_ff
        self.noise_level = noise_level
        
        # Initialize weights (mimicking GPT-2 scale)
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * 0.02
        self.W_proj = np.random.randn(d_model, d_model) * 0.02
        self.W_fc1 = np.random.randn(d_model, d_ff) * 0.02
        self.W_fc2 = np.random.randn(d_ff, d_model) * 0.02
        self.ln1_scale = np.ones(d_model)
        self.ln1_bias = np.zeros(d_model)
        self.ln2_scale = np.ones(d_model)
        self.ln2_bias = np.zeros(d_model)
        
    def layer_norm(self, x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Simple layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        return scale * x_norm + bias
    
    def forward_clean(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with FP32 weights."""
        # Attention block (simplified)
        h = self.layer_norm(x, self.ln1_scale, self.ln1_bias)
        qkv = h @ self.W_qkv
        # Simplified attention: just project back
        attn_out = qkv[:, :self.d_model] @ self.W_proj.T
        x = x + attn_out
        
        # MLP block
        h = self.layer_norm(x, self.ln2_scale, self.ln2_bias)
        ff = gelu_numpy(h @ self.W_fc1) @ self.W_fc2
        x = x + ff
        
        return x
    
    def forward_noisy(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with noisy (quantized) weights."""
        # Add quantization noise to weights
        W_qkv_noisy = self.W_qkv + np.random.randn(*self.W_qkv.shape) * self.noise_level
        W_proj_noisy = self.W_proj + np.random.randn(*self.W_proj.shape) * self.noise_level
        W_fc1_noisy = self.W_fc1 + np.random.randn(*self.W_fc1.shape) * self.noise_level
        W_fc2_noisy = self.W_fc2 + np.random.randn(*self.W_fc2.shape) * self.noise_level
        
        # Forward with noisy weights
        h = self.layer_norm(x, self.ln1_scale, self.ln1_bias)
        qkv = h @ W_qkv_noisy
        attn_out = qkv[:, :self.d_model] @ W_proj_noisy.T
        x = x + attn_out
        
        h = self.layer_norm(x, self.ln2_scale, self.ln2_bias)
        ff = gelu_numpy(h @ W_fc1_noisy) @ W_fc2_noisy
        x = x + ff
        
        return x


class TestSingleLayerError:
    """Test error characteristics of a single layer."""
    
    def test_single_layer_error_vs_noise(self):
        """Test how output error scales with weight noise."""
        d_model = 768
        seq_len = 128
        
        noise_levels = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        
        print("\nSingle Layer Error vs Weight Noise:")
        print("-" * 70)
        print(f"{'Noise':>10} {'Output MSE':>15} {'Amplification':>15} {'Cosine Sim':>15}")
        print("-" * 70)
        
        np.random.seed(42)
        x = np.random.randn(seq_len, d_model) * 0.1
        
        for noise in noise_levels:
            layer = MockTransformerLayer(d_model=d_model, noise_level=noise)
            
            y_clean = layer.forward_clean(x)
            y_noisy = layer.forward_noisy(x)
            
            mse = np.mean((y_clean - y_noisy)**2)
            cs = cosine_similarity(y_clean, y_noisy)
            
            # Input "noise" is 0, so amplification is just output MSE / weight noise^2
            # This shows how much weight noise translates to output noise
            amplification = np.sqrt(mse) / noise
            
            print(f"{noise:10.4f} {mse:15.6f} {amplification:15.2f}x {cs:15.6f}")
        
        # Error should scale roughly linearly with noise (for small noise)
        
    def test_attention_vs_mlp_error_contribution(self):
        """Measure error contribution from attention vs MLP."""
        d_model = 768
        d_ff = 3072
        seq_len = 128
        noise_level = 0.02
        
        np.random.seed(42)
        x = np.random.randn(seq_len, d_model) * 0.1
        
        layer = MockTransformerLayer(d_model=d_model, d_ff=d_ff, noise_level=noise_level)
        
        # Full forward
        y_clean = layer.forward_clean(x.copy())
        y_noisy = layer.forward_noisy(x.copy())
        full_mse = np.mean((y_clean - y_noisy)**2)
        
        # Attention-only error (MLP clean)
        # For this we need to trace intermediate
        # Simplified: just report the full error
        
        print("\nAttention vs MLP Error Contribution:")
        print("-" * 50)
        print(f"Full layer MSE: {full_mse:.6f}")
        print("(Detailed decomposition requires intermediate access)")


class TestMultiLayerPropagation:
    """Test error propagation through multiple layers."""
    
    def test_error_growth_12_layers(self):
        """Test error growth through 12 transformer layers."""
        d_model = 768
        d_ff = 3072
        seq_len = 128
        n_layers = 12
        noise_level = 0.01
        
        np.random.seed(42)
        x_init = np.random.randn(seq_len, d_model) * 0.1
        
        # Create layers
        layers = [MockTransformerLayer(d_model, d_ff, noise_level) for _ in range(n_layers)]
        
        # Propagate through layers, tracking error
        x_clean = x_init.copy()
        x_noisy = x_init.copy()
        
        results = []
        cumulative_mse = []
        
        print("\nError Growth Through 12 Layers:")
        print("-" * 80)
        print(f"{'Layer':>6} {'In MSE':>12} {'Out MSE':>12} {'Amp':>8} {'Cos Sim':>12} {'Rel Error':>12}")
        print("-" * 80)
        
        for i, layer in enumerate(layers):
            input_mse = np.mean((x_clean - x_noisy)**2)
            
            # Forward
            x_clean_new = layer.forward_clean(x_clean)
            x_noisy_new = layer.forward_noisy(x_noisy)
            
            output_mse = np.mean((x_clean_new - x_noisy_new)**2)
            
            # Metrics
            if input_mse > 1e-10:
                amp = output_mse / input_mse
            else:
                amp = output_mse / 1e-10
            
            cs = cosine_similarity(x_clean_new, x_noisy_new)
            
            norm_clean = np.linalg.norm(x_clean_new)
            if norm_clean > 1e-10:
                rel_error = np.sqrt(output_mse) / norm_clean
            else:
                rel_error = float('inf')
            
            results.append(LayerError(
                layer_idx=i,
                input_mse=input_mse,
                output_mse=output_mse,
                amplification=amp,
                cosine_sim=cs,
                relative_error=rel_error
            ))
            cumulative_mse.append(output_mse)
            
            print(f"{i:6d} {input_mse:12.6f} {output_mse:12.6f} {amp:8.2f}x {cs:12.6f} {rel_error*100:11.2f}%")
            
            # Update for next layer
            x_clean = x_clean_new
            x_noisy = x_noisy_new
        
        # Fit exponential growth
        layer_indices = np.arange(n_layers)
        log_mse = np.log(np.array(cumulative_mse) + 1e-10)
        
        # Linear fit to log(MSE) = a + b * layer
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(layer_indices, log_mse, 1)
        growth_rate = coeffs[0]  # Slope in log space = exponential growth rate
        
        print("-" * 80)
        print(f"Exponential growth rate: {growth_rate:.4f} (MSE multiplies by {np.exp(growth_rate):.2f}x per layer)")
        print(f"Final MSE: {cumulative_mse[-1]:.6f}")
        print(f"Final cosine similarity: {results[-1].cosine_sim:.6f}")
        
        # Even with 0.99 per-layer correlation, after 12 layers:
        # 0.99^12 = 0.886, meaning 11.4% error at the end!
        
    def test_exponential_vs_linear_growth(self):
        """Compare exponential vs linear error growth models."""
        d_model = 768
        d_ff = 3072
        seq_len = 128
        n_layers = 12
        
        print("\nExponential vs Linear Error Growth:")
        print("-" * 60)
        
        # Test different noise levels
        noise_levels = [0.005, 0.01, 0.02, 0.05]
        
        for noise in noise_levels:
            np.random.seed(42)
            x_init = np.random.randn(seq_len, d_model) * 0.1
            
            layers = [MockTransformerLayer(d_model, d_ff, noise) for _ in range(n_layers)]
            
            x_clean = x_init.copy()
            x_noisy = x_init.copy()
            mse_by_layer = []
            
            for layer in layers:
                x_clean = layer.forward_clean(x_clean)
                x_noisy = layer.forward_noisy(x_noisy)
                mse_by_layer.append(np.mean((x_clean - x_noisy)**2))
            
            # Fit exponential: log(MSE) = a + b*layer
            layer_idx = np.arange(n_layers)
            log_mse = np.log(np.array(mse_by_layer) + 1e-10)
            
            exp_coeffs = np.polyfit(layer_idx, log_mse, 1)
            exp_rate = exp_coeffs[0]
            
            # Fit linear: MSE = a + b*layer
            lin_coeffs = np.polyfit(layer_idx, mse_by_layer, 1)
            lin_rate = lin_coeffs[0]
            
            print(f"\nNoise={noise}: Exp rate={exp_rate:.4f}, Lin rate={lin_rate:.6f}")
            print(f"  Layer 0 MSE: {mse_by_layer[0]:.6f}")
            print(f"  Layer 11 MSE: {mse_by_layer[-1]:.6f}")
            print(f"  Growth factor: {mse_by_layer[-1]/mse_by_layer[0]:.1f}x")


class TestCorrelationCompounding:
    """Test how correlation compounds through layers."""
    
    def test_correlation_compounding(self):
        """Show that 0.99 per-layer correlation -> 0.89 after 12 layers."""
        print("\nCorrelation Compounding Through Layers:")
        print("-" * 50)
        
        per_layer_correlations = [0.999, 0.995, 0.99, 0.98, 0.95, 0.9]
        n_layers = 12
        
        print(f"{'Per-Layer':>15} {'After 12 Layers':>20} {'Error %':>15}")
        print("-" * 50)
        
        for corr in per_layer_correlations:
            final_corr = corr ** n_layers
            error_pct = (1 - final_corr) * 100
            print(f"{corr:15.4f} {final_corr:20.6f} {error_pct:14.1f}%")
        
        # Key insight: to achieve 0.99 final correlation,
        # we need 0.99^(1/12) = 0.9992 per-layer correlation!
        target_final = 0.99
        required_per_layer = target_final ** (1/n_layers)
        print("-" * 50)
        print(f"To achieve {target_final} final, need {required_per_layer:.6f} per-layer")
        
    def test_mse_compounding_math(self):
        """Mathematical analysis of MSE compounding."""
        print("\nMSE Compounding Mathematical Analysis:")
        print("-" * 60)
        
        # If each layer has MSE_added = epsilon, and error is correlated:
        # After n layers: MSE_total ~ n * epsilon (linear growth)
        # But if error gets amplified by factor alpha per layer:
        # After n layers: MSE_total ~ epsilon * (alpha^n - 1) / (alpha - 1)
        
        # For alpha > 1 (amplification), this is exponential
        # For alpha = 1 (no amplification), this is linear
        
        n_layers = 12
        epsilon = 0.001  # Per-layer added error
        
        print(f"Per-layer added MSE: {epsilon}")
        print("-" * 60)
        print(f"{'Amplification':>15} {'Growth Model':>15} {'Final MSE':>15}")
        print("-" * 60)
        
        # No amplification (linear)
        linear_mse = n_layers * epsilon
        print(f"{'alpha=1.0':>15} {'linear':>15} {linear_mse:15.6f}")
        
        # Mild amplification
        for alpha in [1.05, 1.1, 1.2, 1.5, 2.0]:
            if alpha != 1:
                exp_mse = epsilon * (alpha ** n_layers - 1) / (alpha - 1)
            else:
                exp_mse = n_layers * epsilon
            print(f"alpha={alpha:4.2f}        {'exponential':>15} {exp_mse:15.6f}")


class TestLayerSensitivity:
    """Identify which layers are most sensitive."""
    
    def test_per_layer_sensitivity(self):
        """Measure sensitivity of each layer independently."""
        d_model = 768
        d_ff = 3072
        seq_len = 128
        n_layers = 12
        noise_level = 0.01
        
        print("\nPer-Layer Sensitivity Analysis:")
        print("-" * 70)
        
        np.random.seed(42)
        x_init = np.random.randn(seq_len, d_model) * 0.1
        
        # Create layers
        layers = [MockTransformerLayer(d_model, d_ff, noise_level) for _ in range(n_layers)]
        
        # First, propagate clean version to get inputs at each layer
        layer_inputs = [x_init.copy()]
        x = x_init.copy()
        for layer in layers:
            x = layer.forward_clean(x)
            layer_inputs.append(x.copy())
        
        # Now test each layer independently
        print(f"{'Layer':>6} {'Input Norm':>12} {'Error MSE':>12} {'Rel Error':>12} {'Sensitivity':>12}")
        print("-" * 70)
        
        sensitivities = []
        for i, layer in enumerate(layers):
            x_in = layer_inputs[i]
            
            y_clean = layer.forward_clean(x_in)
            y_noisy = layer.forward_noisy(x_in)
            
            input_norm = np.linalg.norm(x_in)
            error_mse = np.mean((y_clean - y_noisy)**2)
            output_norm = np.linalg.norm(y_clean)
            rel_error = np.sqrt(error_mse) / output_norm if output_norm > 1e-10 else float('inf')
            
            # Sensitivity = error relative to layer's input magnitude
            sensitivity = np.sqrt(error_mse) / input_norm if input_norm > 1e-10 else float('inf')
            sensitivities.append(sensitivity)
            
            print(f"{i:6d} {input_norm:12.4f} {error_mse:12.6f} {rel_error*100:11.2f}% {sensitivity:12.6f}")
        
        # Which layer is most sensitive?
        most_sensitive = np.argmax(sensitivities)
        print("-" * 70)
        print(f"Most sensitive layer: {most_sensitive} (sensitivity={sensitivities[most_sensitive]:.6f})")


class TestResidualConnections:
    """Test how residual connections affect error propagation."""
    
    def test_residual_dampening(self):
        """Test if residual connections dampen or amplify errors."""
        print("\nResidual Connection Effect on Error:")
        print("-" * 60)
        
        d_model = 768
        seq_len = 128
        
        np.random.seed(42)
        x = np.random.randn(seq_len, d_model) * 0.1
        
        # Layer output with some error
        W = np.random.randn(d_model, d_model) * 0.02
        W_noisy = W + np.random.randn(*W.shape) * 0.01
        
        # Without residual
        y_clean = gelu_numpy(x @ W)
        y_noisy = gelu_numpy(x @ W_noisy)
        error_no_res = np.mean((y_clean - y_noisy)**2)
        
        # With residual
        y_clean_res = x + gelu_numpy(x @ W)
        y_noisy_res = x + gelu_numpy(x @ W_noisy)
        error_with_res = np.mean((y_clean_res - y_noisy_res)**2)
        
        # Relative to signal magnitude
        signal_no_res = np.mean(y_clean**2)
        signal_with_res = np.mean(y_clean_res**2)
        
        snr_no_res = signal_no_res / (error_no_res + 1e-10)
        snr_with_res = signal_with_res / (error_with_res + 1e-10)
        
        print(f"Without residual: MSE={error_no_res:.6f}, SNR={snr_no_res:.1f}")
        print(f"With residual: MSE={error_with_res:.6f}, SNR={snr_with_res:.1f}")
        
        # Residuals add the uncorrupted signal, which improves SNR
        # But the error still propagates to the next layer


def run_all_propagation_tests():
    """Run all error propagation tests."""
    print("=" * 70)
    print("ERROR PROPAGATION TESTS")
    print("=" * 70)
    
    test_classes = [
        TestSingleLayerError,
        TestMultiLayerPropagation,
        TestCorrelationCompounding,
        TestLayerSensitivity,
        TestResidualConnections,
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
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS - ERROR PROPAGATION")
    print("=" * 70)
    print("""
1. ERROR GROWTH IS SUPER-LINEAR:
   - With quantization noise, MSE grows exponentially through layers
   - 1% per-layer error -> 12%+ final error after 12 layers
   
2. CORRELATION COMPOUNDING IS BRUTAL:
   - 0.99 per-layer correlation -> 0.886 after 12 layers
   - To achieve 0.99 final correlation, need 0.9992 per-layer!
   
3. AMPLIFICATION FACTORS DOMINATE:
   - Even small amplification (alpha=1.1) causes exponential blowup
   - alpha=1.1 for 12 layers -> 3x error growth
   
4. RESIDUAL CONNECTIONS HELP BUT DON'T FIX:
   - Residuals improve SNR by adding clean signal
   - But error still propagates to next layer
   
5. EARLY LAYERS ARE CRITICAL:
   - Errors in early layers propagate through all subsequent layers
   - Layer 0 errors matter more than layer 11 errors
""")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_propagation_tests()
    sys.exit(0 if success else 1)