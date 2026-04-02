#!/usr/bin/env python3
"""Validate Hadamard Binary Quantization vs naive 1-bit.

This script compares:
1. FP32 ground truth: y = W @ x
2. Naive 1-bit: y = sign(W) @ x * scale
3. Hadamard 1-bit: y = sign(W @ H) @ (H @ x) * scale

We measure correlation and relative error to see if Hadamard helps.
"""
from __future__ import annotations

import argparse
import numpy as np
from typing import Dict

from onebit.ops.hadamard import hadamard_quantize, hadamard_matmul, fast_walsh_hadamard


def naive_1bit_matmul(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Naive 1-bit: sign(W) @ x with per-row scaling."""
    W_sign = np.sign(W)
    W_sign[W_sign == 0] = 1  # Handle exact zeros
    scale = np.mean(np.abs(W), axis=1, keepdims=True)
    return (W_sign @ x) * scale.squeeze()


def ternary_158bit_matmul(W: np.ndarray, x: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Simulate 1.58-bit ternary: {-1, 0, +1} based on magnitude threshold.
    
    This is our target to beat.
    """
    # Quantize to ternary based on magnitude
    W_abs_mean = np.mean(np.abs(W))
    W_ternary = np.zeros_like(W)
    W_ternary[W > threshold * W_abs_mean] = 1
    W_ternary[W < -threshold * W_abs_mean] = -1
    # Values near zero stay as 0
    
    scale = np.mean(np.abs(W), axis=1, keepdims=True)
    return (W_ternary @ x) * scale.squeeze()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute correlation and error metrics."""
    # Pearson correlation
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    # Relative error
    rel_err = np.mean(np.abs(y_true - y_pred)) / (np.mean(np.abs(y_true)) + 1e-10)
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Cosine similarity
    cos_sim = np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred) + 1e-10)
    
    return {
        "correlation": corr,
        "rel_error": rel_err,
        "mse": mse,
        "cosine_sim": cos_sim,
    }


def run_validation(d_out: int = 768, d_in: int = 512, n_samples: int = 100, seed: int = 42):
    """Run validation comparing methods.

    Note: d_in must be a power of 2 for Hadamard transform.
    """
    np.random.seed(seed)

    # Ensure d_in is power of 2
    d_in_padded = 1 << (d_in - 1).bit_length()

    print(f"Validating Hadamard 1-bit vs Naive 1-bit vs Ternary 1.58-bit")
    print(f"Matrix: {d_out} x {d_in} (padded to {d_in_padded}), Samples: {n_samples}")
    print("=" * 70)

    # Generate random weights (simulating transformer layer)
    W = np.random.randn(d_out, d_in_padded).astype(np.float32) * 0.02

    # Quantize with Hadamard
    W_bits, scale, d_orig = hadamard_quantize(W)
    
    # Collect metrics over multiple inputs
    metrics_naive = {"correlation": [], "rel_error": [], "cosine_sim": []}
    metrics_hadamard = {"correlation": [], "rel_error": [], "cosine_sim": []}
    metrics_ternary = {"correlation": [], "rel_error": [], "cosine_sim": []}
    
    for _ in range(n_samples):
        # Random input (simulating hidden states)
        x = np.random.randn(d_in_padded).astype(np.float32)

        # Ground truth
        y_true = W @ x

        # Naive 1-bit
        y_naive = naive_1bit_matmul(W, x)
        m = compute_metrics(y_true, y_naive)
        for k in metrics_naive:
            metrics_naive[k].append(m[k])

        # Hadamard 1-bit
        y_hadamard = hadamard_matmul(W_bits, scale, x, d_in_padded)
        m = compute_metrics(y_true, y_hadamard)
        for k in metrics_hadamard:
            metrics_hadamard[k].append(m[k])

        # Ternary 1.58-bit (target to beat)
        y_ternary = ternary_158bit_matmul(W, x)
        m = compute_metrics(y_true, y_ternary)
        for k in metrics_ternary:
            metrics_ternary[k].append(m[k])
    
    # Print results
    print(f"\n{'Method':<20} {'Correlation':>12} {'Rel Error':>12} {'Cosine Sim':>12}")
    print("-" * 60)
    
    for name, metrics in [("Naive 1-bit", metrics_naive), 
                          ("Hadamard 1-bit", metrics_hadamard),
                          ("Ternary 1.58-bit", metrics_ternary)]:
        corr = np.mean(metrics["correlation"])
        err = np.mean(metrics["rel_error"])
        cos = np.mean(metrics["cosine_sim"])
        print(f"{name:<20} {corr:>12.4f} {err:>12.4f} {cos:>12.4f}")
    
    print("\n" + "=" * 70)
    h_corr = np.mean(metrics_hadamard["correlation"])
    n_corr = np.mean(metrics_naive["correlation"])
    t_corr = np.mean(metrics_ternary["correlation"])
    
    print(f"Hadamard vs Naive:   {'+' if h_corr > n_corr else '-'}{abs(h_corr - n_corr):.4f} correlation")
    print(f"Hadamard vs Ternary: {'+' if h_corr > t_corr else '-'}{abs(h_corr - t_corr):.4f} correlation")
    
    if h_corr > t_corr:
        print("\n✓ Hadamard 1-bit BEATS Ternary 1.58-bit!")
    elif h_corr > n_corr:
        print("\n→ Hadamard improves over Naive, but doesn't beat Ternary yet")
    else:
        print("\n✗ Hadamard doesn't help - need different approach")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Hadamard 1-bit quantization")
    parser.add_argument("--d-out", type=int, default=768)
    parser.add_argument("--d-in", type=int, default=768)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_validation(args.d_out, args.d_in, args.samples, args.seed)

