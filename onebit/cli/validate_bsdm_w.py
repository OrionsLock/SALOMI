#!/usr/bin/env python3
"""BSDM-W Validation Harness.

This script definitively tests whether BSDM-W stochastic estimation converges
to correct dot-product values when given 1-bit quantized weights.

Key questions answered:
1. Does BSDM-W converge to ground truth as ticks increase?
2. How does it compare to naive 1-bit (sign-only)?
3. What's the convergence rate (variance vs ticks)?
4. How well do calibration factors (a, b) fit?

Usage:
    python -m onebit.cli.validate_bsdm_w
    python -m onebit.cli.validate_bsdm_w --ticks 16,32,64,128,256
    python -m onebit.cli.validate_bsdm_w --layer "h.0.mlp.c_fc.w" --samples 100
"""
from __future__ import annotations

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface, GPT2Config
from onebit.core.packbits import pack_signs_rowmajor, pack_float_to_stream
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig


@dataclass
class ValidationResult:
    """Results from a single validation run."""
    ticks: int
    correlation: float
    relative_error: float
    mse: float
    calibration_a: float
    calibration_b: float
    naive_correlation: float  # Baseline: sign-only, no ΣΔ
    time_ms: float


def compute_naive_1bit(W_fp32: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Naive 1-bit baseline: just sign quantization, no ΣΔ."""
    # Sign quantize weights
    W_sign = np.sign(W_fp32)  # +1 or -1
    W_sign[W_sign == 0] = 1    # Handle exact zeros
    
    # Compute dot product with sign-quantized weights
    return W_sign @ x


def validate_layer(
    W_fp32: np.ndarray,
    layer_name: str,
    ticks_list: List[int],
    n_samples: int = 50,
    seed: int = 42,
) -> List[ValidationResult]:
    """Validate BSDM-W on a single layer with multiple tick counts.
    
    Args:
        W_fp32: Weight matrix [d_out, d_in]
        layer_name: Name for logging
        ticks_list: List of tick counts to test
        n_samples: Number of random inputs to average over
        seed: Random seed
    """
    print(f"\n{'='*60}")
    print(f"Layer: {layer_name}")
    print(f"Shape: {W_fp32.shape}")
    print(f"Weight stats: mean={W_fp32.mean():.4f}, std={W_fp32.std():.4f}")
    print(f"{'='*60}")
    
    d_out, d_in = W_fp32.shape
    rng = np.random.default_rng(seed)
    
    # Pre-quantize weights (done once)
    mean_w = np.mean(W_fp32, axis=1, keepdims=True)  # [d_out, 1]
    W_centered = W_fp32 - mean_w
    scale_w = np.mean(np.abs(W_centered), axis=1)     # [d_out]
    scale_w[scale_w < 1e-9] = 1e-9
    W_bits = pack_signs_rowmajor(W_centered)
    Kw = W_bits.shape[1]
    Kbits = Kw * 32
    
    # SD config (locked params from project spec)
    sd_cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    
    results = []
    
    for T in ticks_list:
        print(f"\n  Testing T={T} ticks...")
        
        # Accumulators across samples
        all_y_fp32 = []
        all_y_bsdm = []
        all_y_naive = []
        total_time = 0.0
        
        for i in range(n_samples):
            # Generate random input (simulate hidden states)
            # Use realistic scale: layer-norm outputs are roughly unit variance
            x = rng.standard_normal(d_in).astype(np.float32) * 0.5
            
            # Occasionally add bias (some activations are biased)
            if i % 3 == 0:
                x += rng.uniform(-0.5, 0.5)
            
            # Ground truth
            y_fp32 = W_fp32 @ x
            
            # Naive 1-bit baseline
            y_naive = compute_naive_1bit(W_fp32, x)
            
            # BSDM-W estimation
            max_x = np.max(np.abs(x))
            if max_x < 1e-9:
                max_x = 1e-9
            x_norm = x / max_x
            
            # Pad if needed
            padded_d_in = Kw * 32
            if d_in < padded_d_in:
                x_padded = np.zeros(padded_d_in, dtype=np.float32)
                x_padded[:d_in] = x_norm
                x_norm = x_padded
            
            x_stream = pack_float_to_stream(x_norm, k=T)
            
            t0 = time.perf_counter()
            y_raw = bsdm_w_matmul(W_bits, x_stream, k=T, cfg=sd_cfg, seed=seed+i, scale=1.0)
            total_time += (time.perf_counter() - t0) * 1000
            
            # Reconstruct with scaling
            y_est = y_raw * scale_w * max_x * Kbits
            
            # Mean correction
            correction = mean_w.flatten() * np.sum(x)
            y_bsdm = y_est + correction
            
            all_y_fp32.append(y_fp32)
            all_y_bsdm.append(y_bsdm)
            all_y_naive.append(y_naive)
        
        # Flatten all results
        y_fp32_all = np.concatenate(all_y_fp32)
        y_bsdm_all = np.concatenate(all_y_bsdm)
        y_naive_all = np.concatenate(all_y_naive)
        
        # Compute metrics
        corr_bsdm = np.corrcoef(y_fp32_all, y_bsdm_all)[0, 1]
        corr_naive = np.corrcoef(y_fp32_all, y_naive_all)[0, 1]
        
        # Fit calibration: y_fp32 ≈ a * y_bsdm + b
        # Linear regression: y = a*x + b
        X = np.vstack([y_bsdm_all, np.ones_like(y_bsdm_all)]).T
        a, b = np.linalg.lstsq(X, y_fp32_all, rcond=None)[0]
        
        y_calibrated = a * y_bsdm_all + b
        rel_err = np.linalg.norm(y_fp32_all - y_calibrated) / np.linalg.norm(y_fp32_all)
        mse = np.mean((y_fp32_all - y_calibrated) ** 2)
        
        result = ValidationResult(
            ticks=T,
            correlation=float(corr_bsdm),
            relative_error=float(rel_err),
            mse=float(mse),
            calibration_a=float(a),
            calibration_b=float(b),
            naive_correlation=float(corr_naive),
            time_ms=total_time / n_samples,
        )
        results.append(result)
        
        # Print results
        print(f"    BSDM-W Corr:   {result.correlation:.4f}")
        print(f"    Naive Corr:    {result.naive_correlation:.4f}")
        print(f"    Improvement:   {result.correlation - result.naive_correlation:+.4f}")
        print(f"    Rel Error:     {result.relative_error:.4f}")
        print(f"    Calibration:   a={result.calibration_a:.4f}, b={result.calibration_b:.4f}")
        print(f"    Time/sample:   {result.time_ms:.2f} ms")

    return results


def print_summary(all_results: Dict[str, List[ValidationResult]]) -> None:
    """Print summary table of all results."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    # Header
    print(f"\n{'Layer':<30} {'T':>6} {'BSDM Corr':>10} {'Naive Corr':>10} {'Δ':>8} {'RelErr':>8}")
    print("-" * 80)

    for layer_name, results in all_results.items():
        for r in results:
            delta = r.correlation - r.naive_correlation
            print(f"{layer_name:<30} {r.ticks:>6} {r.correlation:>10.4f} {r.naive_correlation:>10.4f} {delta:>+8.4f} {r.relative_error:>8.4f}")

    # Conclusions
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)

    # Check if BSDM-W helps
    improvements = []
    for layer_name, results in all_results.items():
        for r in results:
            improvements.append(r.correlation - r.naive_correlation)

    avg_improvement = np.mean(improvements)
    min_improvement = np.min(improvements)
    max_improvement = np.max(improvements)

    print(f"\n1. BSDM-W vs Naive 1-bit:")
    print(f"   Average improvement: {avg_improvement:+.4f}")
    print(f"   Range: [{min_improvement:+.4f}, {max_improvement:+.4f}]")

    if avg_improvement > 0.05:
        print("   ✅ BSDM-W SIGNIFICANTLY IMPROVES over naive 1-bit")
    elif avg_improvement > 0.01:
        print("   ⚠️  BSDM-W shows MODEST improvement over naive 1-bit")
    else:
        print("   ❌ BSDM-W shows NO significant improvement over naive 1-bit")

    # Check convergence
    print(f"\n2. Convergence with ticks:")
    for layer_name, results in all_results.items():
        if len(results) > 1:
            first_corr = results[0].correlation
            last_corr = results[-1].correlation
            improvement = last_corr - first_corr
            print(f"   {layer_name}: {first_corr:.4f} → {last_corr:.4f} (Δ={improvement:+.4f})")

    # Check calibration stability
    print(f"\n3. Calibration factors (a, b):")
    for layer_name, results in all_results.items():
        a_vals = [r.calibration_a for r in results]
        b_vals = [r.calibration_b for r in results]
        print(f"   {layer_name}:")
        print(f"     a: mean={np.mean(a_vals):.4f}, std={np.std(a_vals):.4f}")
        print(f"     b: mean={np.mean(b_vals):.4f}, std={np.std(b_vals):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Validate BSDM-W convergence")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--ticks", type=str, default="16,32,64,128", help="Comma-separated tick counts")
    parser.add_argument("--samples", type=int, default=50, help="Number of random inputs per tick count")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer names (default: test key layers)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ticks_list = [int(t.strip()) for t in args.ticks.split(",")]

    print("="*80)
    print("BSDM-W VALIDATION HARNESS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Ticks: {ticks_list}")
    print(f"Samples per tick: {args.samples}")
    print(f"Seed: {args.seed}")

    # Load weights
    print("\nLoading model weights...")
    try:
        weights, cfg = load_gpt2_from_huggingface(args.model)
    except ImportError as e:
        print(f"Error: {e}")
        print("Install transformers: pip install transformers torch")
        return

    # Select layers to test
    if args.layers:
        layer_names = [l.strip() for l in args.layers.split(",")]
    else:
        # Default: test key layer types
        layer_names = [
            "h.0.attn.c_attn.w",   # Attention QKV (early layer)
            "h.0.mlp.c_fc.w",      # FFN up (early layer)
            "h.5.attn.c_attn.w",   # Attention QKV (middle layer)
            "h.11.mlp.c_proj.w",   # FFN down (last layer)
        ]

    all_results = {}

    for layer_name in layer_names:
        if layer_name not in weights:
            print(f"Warning: Layer {layer_name} not found, skipping")
            continue

        W_fp32 = weights[layer_name]

        # GPT-2 uses Conv1D: [d_in, d_out], we need [d_out, d_in]
        if W_fp32.ndim == 2:
            W_fp32 = W_fp32.T

        results = validate_layer(
            W_fp32=W_fp32,
            layer_name=layer_name,
            ticks_list=ticks_list,
            n_samples=args.samples,
            seed=args.seed,
        )
        all_results[layer_name] = results

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()

