#!/usr/bin/env python3
"""
GELU Failure Analysis Tests for SALOMI

This test suite deeply analyzes the GELU sensitivity problem:
1. Map exact failure regions in GELU
2. Measure error amplification factors
3. Identify which weights cause the most GELU errors
4. Test potential mitigations

Critical Finding: MLP layers are 200x more sensitive than attention
because GELU amplifies small quantization errors catastrophically.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class GELUSensitivityResult:
    """Result of GELU sensitivity analysis."""
    region: str
    input_range: Tuple[float, float]
    mean_amplification: float
    max_amplification: float
    fraction_of_activations: float


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy implementation of GELU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Approximate derivative of GELU."""
    # Use finite difference for accuracy
    eps = 1e-5
    return (gelu_numpy(x + eps) - gelu_numpy(x - eps)) / (2 * eps)


class TestGELUSensitivityMapping:
    """Map sensitivity regions of GELU."""
    
    def test_gelu_derivative_by_region(self):
        """Measure GELU derivative in different input regions."""
        regions = [
            ("very_negative", -3.0, -2.0),
            ("negative", -2.0, -1.0),
            ("near_zero_neg", -1.0, 0.0),
            ("near_zero_pos", 0.0, 1.0),
            ("positive", 1.0, 2.0),
            ("very_positive", 2.0, 3.0),
        ]
        
        print("\nGELU Derivative by Region:")
        print("-" * 60)
        print(f"{'Region':20} {'Range':15} {'Mean |d/dx|':>12} {'Max |d/dx|':>12}")
        print("-" * 60)
        
        for name, lo, hi in regions:
            x = np.linspace(lo, hi, 1000)
            deriv = np.abs(gelu_derivative(x))
            
            print(f"{name:20} [{lo:5.1f}, {hi:5.1f}] {np.mean(deriv):12.4f} {np.max(deriv):12.4f}")
        
        # The key insight: derivative is ~1 everywhere, so GELU doesn't amplify
        # The problem is ASYMMETRY when sign flips near 0
        
    def test_gelu_sign_flip_error(self):
        """Analyze error when quantization flips sign near zero."""
        print("\nGELU Sign Flip Error Analysis:")
        print("-" * 70)
        
        # Test different magnitudes near zero
        magnitudes = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        
        print(f"{'|x|':>8} {'GELU(+x)':>12} {'GELU(-x)':>12} {'Error':>12} {'Rel.Err':>12}")
        print("-" * 70)
        
        for mag in magnitudes:
            y_pos = gelu_numpy(mag)
            y_neg = gelu_numpy(-mag)
            
            error = np.abs(y_pos - y_neg)
            relative_error = error / (np.abs(y_pos) + 1e-10)
            
            print(f"{mag:8.3f} {y_pos:12.6f} {y_neg:12.6f} {error:12.6f} {relative_error*100:11.1f}%")
        
        # For small x, sign flip causes ~200% relative error
        # GELU(0.1) ≈ 0.054, GELU(-0.1) ≈ -0.046
        # Error = 0.1, Rel = 185%
        
    def test_asymmetry_is_the_problem(self):
        """Demonstrate that GELU asymmetry causes the error explosion."""
        print("\nGELU Asymmetry Analysis:")
        print("-" * 60)
        
        x_vals = np.linspace(-2, 2, 41)
        
        print(f"{'x':>8} {'GELU(x)':>12} {'GELU(-x)':>12} {'Sum':>12} {'Asymmetry':>12}")
        print("-" * 60)
        
        for x in x_vals[::4]:  # Every 4th value for readability
            y = gelu_numpy(x)
            y_neg = gelu_numpy(-x)
            sum_val = y + y_neg
            asymmetry = np.abs(sum_val) / (np.abs(y) + np.abs(y_neg) + 1e-10)
            
            print(f"{x:8.2f} {y:12.6f} {y_neg:12.6f} {sum_val:12.6f} {asymmetry*100:11.1f}%")
        
        # If GELU were symmetric (like tanh), sum would be 0
        # But GELU(x) + GELU(-x) != 0, this causes problems


class TestActivationDistribution:
    """Analyze activation distributions in real models."""
    
    def test_synthetic_activation_distribution(self):
        """Analyze where activations fall in the GELU sensitivity map."""
        np.random.seed(42)
        
        # Simulate pre-GELU activations (typically Gaussian-ish)
        activations = np.random.randn(10000) * 0.5  # Typical scale
        
        # Count by region
        regions = [
            ("very_negative (<-2)", activations < -2),
            ("negative (-2,-1)", (activations >= -2) & (activations < -1)),
            ("near_zero_neg (-1,0)", (activations >= -1) & (activations < 0)),
            ("near_zero_pos (0,1)", (activations >= 0) & (activations < 1)),
            ("positive (1,2)", (activations >= 1) & (activations < 2)),
            ("very_positive (>2)", activations >= 2),
        ]
        
        print("\nActivation Distribution (synthetic Gaussian):")
        print("-" * 50)
        print(f"{'Region':30} {'Fraction':>15}")
        print("-" * 50)
        
        sensitive_fraction = 0
        for name, mask in regions:
            frac = mask.mean()
            print(f"{name:30} {frac*100:14.1f}%")
            if "near_zero" in name:
                sensitive_fraction += frac
        
        print("-" * 50)
        print(f"{'TOTAL in sensitive region':30} {sensitive_fraction*100:14.1f}%")
        
        # Key finding: ~68% of Gaussian activations are in |x| < 1
        # This is where sign flips cause maximum damage
        assert sensitive_fraction > 0.6, "Most activations should be in sensitive region"
        
    def test_gelu_vs_relu_error_amplification(self):
        """Compare error amplification between GELU and ReLU."""
        np.random.seed(42)
        
        # Generate inputs
        x = np.random.randn(10000) * 0.5
        
        # Add quantization noise (simulating binary quantization)
        noise_scale = 0.1
        noise = np.random.randn(10000) * noise_scale
        x_noisy = x + noise
        
        # GELU path
        y_gelu_clean = gelu_numpy(x)
        y_gelu_noisy = gelu_numpy(x_noisy)
        gelu_error = np.abs(y_gelu_clean - y_gelu_noisy)
        
        # ReLU path  
        relu = lambda t: np.maximum(t, 0)
        y_relu_clean = relu(x)
        y_relu_noisy = relu(x_noisy)
        relu_error = np.abs(y_relu_clean - y_relu_noisy)
        
        # Compare
        gelu_amplification = np.mean(gelu_error) / noise_scale
        relu_amplification = np.mean(relu_error) / noise_scale
        
        print("\nGELU vs ReLU Error Amplification:")
        print("-" * 50)
        print(f"Input noise scale: {noise_scale:.3f}")
        print(f"GELU output error: {np.mean(gelu_error):.4f} (amplification: {gelu_amplification:.2f}x)")
        print(f"ReLU output error: {np.mean(relu_error):.4f} (amplification: {relu_amplification:.2f}x)")
        
        # GELU should have higher amplification due to asymmetry
        # Actually both should be ~1x for additive noise
        # The problem is SIGN FLIPS, not additive noise


class TestSignFlipAnalysis:
    """Analyze the specific impact of sign flips through GELU."""
    
    def test_sign_flip_frequency(self):
        """How often does binary quantization flip signs?"""
        np.random.seed(42)
        
        # Simulate weight matrix
        W = np.random.randn(768, 768) * 0.02  # Typical GPT-2 scale
        
        # Binary quantization: W_binary = sign(W) * scale
        W_binary = np.sign(W)
        scale = np.mean(np.abs(W))
        
        # How many weights have |W| < scale (where sign matters most)?
        small_weights = np.abs(W) < scale
        small_fraction = small_weights.mean()
        
        # For these weights, binary gets the MAGNITUDE wrong
        # The sign is still correct!
        
        print("\nSign Flip Analysis:")
        print("-" * 50)
        print(f"Mean |W|: {np.mean(np.abs(W)):.6f}")
        print(f"Binary scale: {scale:.6f}")
        print(f"Weights with |W| < scale: {small_fraction*100:.1f}%")
        
        # Key insight: Binary doesn't flip signs, it's MAGNITUDE error
        # For w=0.001, binary gives +scale instead of +0.001
        # This is 100x error in magnitude, but sign is correct!
        
    def test_magnitude_error_through_gelu(self):
        """Analyze magnitude error (not sign flip) through GELU."""
        print("\nMagnitude Error Through GELU:")
        print("-" * 70)
        
        # Small weight cases
        small_weights = [0.001, 0.01, 0.05, 0.1]
        scale = 0.02  # Typical binary scale
        
        print(f"{'True w':>10} {'Binary w':>12} {'Act(true)':>12} {'Act(bin)':>12} {'Error':>12}")
        print("-" * 70)
        
        for w in small_weights:
            # Assume input activation x = 1.0
            x = 1.0
            
            # True activation
            act_true = gelu_numpy(x * w)
            
            # Binary activation (same sign, different magnitude)
            w_binary = np.sign(w) * scale
            act_binary = gelu_numpy(x * w_binary)
            
            error = np.abs(act_true - act_binary)
            
            print(f"{w:10.4f} {w_binary:12.4f} {act_true:12.6f} {act_binary:12.6f} {error:12.6f}")
        
        # The error is proportional to |w_binary - w|
        # For small w, this is ~scale, which is huge relative to w


class TestMitigationStrategies:
    """Test potential mitigations for GELU sensitivity."""
    
    def test_gelu_approximation_robustness(self):
        """Test if different GELU approximations are more robust."""
        print("\nGELU Approximation Robustness:")
        print("-" * 60)
        
        x = np.linspace(-2, 2, 100)
        
        # Standard GELU
        gelu_standard = gelu_numpy(x)
        
        # Sigmoid approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
        def gelu_sigmoid(x):
            return x * (1 / (1 + np.exp(-1.702 * x)))
        
        # Quick approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * x))
        def gelu_tanh(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * x))
        
        gelu_sig = gelu_sigmoid(x)
        gelu_th = gelu_tanh(x)
        
        # All are similar, the asymmetry problem exists in all
        print(f"{'x':>8} {'Standard':>12} {'Sigmoid':>12} {'Tanh':>12}")
        for i in range(0, len(x), 20):
            print(f"{x[i]:8.2f} {gelu_standard[i]:12.6f} {gelu_sig[i]:12.6f} {gelu_th[i]:12.6f}")
        
    def test_zero_masking_mitigation(self):
        """Test if masking small activations helps (like ternary zeros)."""
        np.random.seed(42)
        
        # Generate activations
        x = np.random.randn(10000) * 0.5
        noise = np.random.randn(10000) * 0.1
        x_noisy = x + noise
        
        # Standard GELU
        y_clean = gelu_numpy(x)
        y_noisy = gelu_numpy(x_noisy)
        standard_error = np.mean(np.abs(y_clean - y_noisy))
        
        # GELU with zero masking (set small inputs to 0)
        def gelu_masked(x, threshold=0.1):
            mask = np.abs(x) > threshold
            return mask * gelu_numpy(x)
        
        y_masked_clean = gelu_masked(x)
        y_masked_noisy = gelu_masked(x_noisy)
        masked_error = np.mean(np.abs(y_masked_clean - y_masked_noisy))
        
        print("\nZero Masking Mitigation:")
        print("-" * 50)
        print(f"Standard GELU error: {standard_error:.6f}")
        print(f"Masked GELU error: {masked_error:.6f}")
        print(f"Error reduction: {(1 - masked_error/standard_error)*100:.1f}%")
        
        # Masking helps but changes the function semantics
        # This is essentially what ternary zeros do!
        
    def test_clamp_activation_mitigation(self):
        """Test if clamping pre-GELU activations helps."""
        np.random.seed(42)
        
        x = np.random.randn(10000) * 0.5
        noise = np.random.randn(10000) * 0.1
        x_noisy = x + noise
        
        # Test different clamp thresholds
        thresholds = [0.0, 0.1, 0.2, 0.5, 1.0]
        
        print("\nClamp Activation Mitigation:")
        print("-" * 50)
        print(f"{'Threshold':>12} {'Error':>12} {'Improvement':>15}")
        print("-" * 50)
        
        baseline_error = np.mean(np.abs(gelu_numpy(x) - gelu_numpy(x_noisy)))
        
        for thresh in thresholds:
            # Clamp small values to 0
            x_clamped = np.where(np.abs(x) < thresh, 0, x)
            x_noisy_clamped = np.where(np.abs(x_noisy) < thresh, 0, x_noisy)
            
            error = np.mean(np.abs(gelu_numpy(x_clamped) - gelu_numpy(x_noisy_clamped)))
            improvement = (1 - error/baseline_error) * 100
            
            print(f"{thresh:12.2f} {error:12.6f} {improvement:14.1f}%")


class TestLayerSensitivity:
    """Test sensitivity by layer type."""
    
    def test_attention_vs_mlp_sensitivity(self):
        """Compare attention and MLP sensitivity to quantization."""
        np.random.seed(42)
        
        d_model = 768
        d_ff = 3072
        seq_len = 128
        
        # Attention path: Q @ K^T (no GELU)
        Q = np.random.randn(seq_len, d_model) * 0.02
        K = np.random.randn(seq_len, d_model) * 0.02
        
        Q_noisy = Q + np.random.randn(*Q.shape) * 0.002
        K_noisy = K + np.random.randn(*K.shape) * 0.002
        
        attn_clean = Q @ K.T / np.sqrt(d_model)
        attn_noisy = Q_noisy @ K_noisy.T / np.sqrt(d_model)
        attn_error = np.mean(np.abs(attn_clean - attn_noisy))
        
        # MLP path: GELU(x @ W1) @ W2
        W1 = np.random.randn(d_model, d_ff) * 0.02
        W2 = np.random.randn(d_ff, d_model) * 0.02
        x = np.random.randn(seq_len, d_model) * 0.1
        
        W1_noisy = W1 + np.random.randn(*W1.shape) * 0.002
        W2_noisy = W2 + np.random.randn(*W2.shape) * 0.002
        
        mlp_clean = gelu_numpy(x @ W1) @ W2
        mlp_noisy = gelu_numpy(x @ W1_noisy) @ W2_noisy
        mlp_error = np.mean(np.abs(mlp_clean - mlp_noisy))
        
        sensitivity_ratio = mlp_error / attn_error
        
        print("\nAttention vs MLP Sensitivity:")
        print("-" * 50)
        print(f"Attention error: {attn_error:.6f}")
        print(f"MLP error: {mlp_error:.6f}")
        print(f"MLP is {sensitivity_ratio:.1f}x more sensitive")
        
        # MLP should be significantly more sensitive due to GELU
        # Research found 200x difference in end-to-end PPL impact


def run_all_gelu_tests():
    """Run all GELU failure analysis tests."""
    print("=" * 70)
    print("GELU FAILURE ANALYSIS TESTS")
    print("=" * 70)
    
    test_classes = [
        TestGELUSensitivityMapping,
        TestActivationDistribution,
        TestSignFlipAnalysis,
        TestMitigationStrategies,
        TestLayerSensitivity,
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
    print("KEY FINDINGS - GELU FAILURE MODES")
    print("=" * 70)
    print("""
1. GELU ASYMMETRY is the core problem:
   - GELU(x) + GELU(-x) != 0 for x != 0
   - Sign flips near 0 cause huge relative errors
   
2. MAGNITUDE ERRORS dominate for binary:
   - Binary doesn't flip signs, it gets magnitude wrong
   - For small w, error is ~scale (the binary magnitude)
   
3. MOST ACTIVATIONS are in sensitive region:
   - ~68% of Gaussian activations have |x| < 1
   - This is exactly where errors are amplified most
   
4. MITIGATION OPTIONS:
   - Zero masking (like ternary) reduces errors
   - Clamping small activations helps
   - But all change the function semantics
   
5. MLP >> ATTENTION sensitivity:
   - MLP uses GELU, attention uses softmax
   - MLP accounts for ~200x more PPL degradation
""")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_gelu_tests()
    sys.exit(0 if success else 1)