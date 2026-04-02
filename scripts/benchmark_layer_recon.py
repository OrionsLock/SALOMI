#!/usr/bin/env python3
\"\"\"
SALOMI Layer Reconstruction Benchmark (Phase 1.1)

Tests whether BSDM-W 1-bit matmul can approximate float matmul with high correlation.
Target: Pearson correlation > 0.95 on at least one realistic configuration.

Usage:
    python scripts/benchmark_layer_recon.py
\"\"\"
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

from onebit.core.packbits import pack_signs_rowmajor, pack_float_to_stream
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig


@dataclass
class BenchmarkConfig:
    \"\"\"Configuration for a single benchmark run.\"\"\"
    name: str
    d_in: int
    d_out: int
    T: int  # Number of ticks
    order: int  # Sigma-Delta order (1 or 2)
    beta: float  # SD-2 beta parameter
    lambd: float  # Leak parameter
    walsh_N: int  # Walsh carriers
    antithetic: bool  # Use antithetic pairs
    n_samples: int  # Number of random inputs to test


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    \"\"\"Compute Pearson correlation coefficient.\"\"\"
    x_flat = x.flatten()
    y_flat = y.flatten()
    if len(x_flat) != len(y_flat):
        raise ValueError("Arrays must have same length")
    
    x_mean = np.mean(x_flat)
    y_mean = np.mean(y_flat)
    
    num = np.sum((x_flat - x_mean) * (y_flat - y_mean))
    den = np.sqrt(np.sum((x_flat - x_mean)**2) * np.sum((y_flat - y_mean)**2))
    
    if den < 1e-12:
        return 0.0
    return float(num / den)


def run_layer_benchmark(cfg: BenchmarkConfig, seed: int = 42) -> Dict:
    \"\"\"Run a single layer reconstruction benchmark.
    
    Creates a random float weight matrix W, quantizes it to 1-bit,
    then compares float matmul vs BSDM-W matmul on random inputs.
    \"\"\"
    rng = np.random.default_rng(seed)
    
    # Create random float weight matrix
    W_float = rng.standard_normal((cfg.d_out, cfg.d_in)).astype(np.float32)
    
    # Scale to have reasonable magnitude
    W_float = W_float / np.sqrt(cfg.d_in)
    
    # Compute per-row scale (magnitude recovery)
    row_scale = np.max(np.abs(W_float), axis=1, keepdims=True)
    row_scale = np.maximum(row_scale, 1e-9)
    
    # Normalize and pack to 1-bit
    W_norm = W_float / row_scale
    W_bits = pack_signs_rowmajor(W_norm)  # [d_out, Kw]
    
    # SD config
    sd_cfg = SDConfig(
        order=cfg.order,
        beta=cfg.beta,
        lambd=cfg.lambd,
        walsh_N=cfg.walsh_N,
        antithetic=cfg.antithetic,
    )
    
    # Collect results
    y_float_all = []
    y_1bit_all = []
    
    t0 = time.perf_counter()
    
    for i in range(cfg.n_samples):
        # Random input
        x = rng.standard_normal(cfg.d_in).astype(np.float32)
        
        # Float matmul (ground truth)
        y_float = W_float @ x
        
        # 1-bit matmul via BSDM-W
        # Normalize input to [-1, 1]
        x_scale = np.max(np.abs(x))
        x_scale = max(x_scale, 1e-9)
        x_norm = x / x_scale
        
        # Pad if needed
        Kw = W_bits.shape[1]
        padded_d_in = Kw * 32
        if cfg.d_in < padded_d_in:
            x_padded = np.zeros(padded_d_in, dtype=np.float32)
            x_padded[:cfg.d_in] = x_norm
            x_norm = x_padded
        
        # Pack input to stream
        x_stream = pack_float_to_stream(x_norm, k=cfg.T)  # [T, Kw]
        
        # Compute effective scale
        Kbits = Kw * 32
        effective_scale = row_scale.flatten() * Kbits * x_scale
        
        # BSDM-W matmul
        y_1bit = bsdm_w_matmul(
            W_bits, x_stream, k=cfg.T, cfg=sd_cfg,
            seed=seed + i, scale=effective_scale
        )
        
        y_float_all.append(y_float)
        y_1bit_all.append(y_1bit)
    
    elapsed = time.perf_counter() - t0
    
    # Stack results
    y_float_all = np.stack(y_float_all)  # [n_samples, d_out]
    y_1bit_all = np.stack(y_1bit_all)    # [n_samples, d_out]
    
    # Compute metrics
    corr = pearson_correlation(y_float_all, y_1bit_all)
    mse = float(np.mean((y_float_all - y_1bit_all)**2))
    rmse = float(np.sqrt(mse))
    
    # Per-output correlations (to see distribution)
    per_output_corr = []
    for j in range(cfg.d_out):
        c = pearson_correlation(y_float_all[:, j], y_1bit_all[:, j])
        per_output_corr.append(c)
    
    return {
        "config": cfg.name,
        "correlation": corr,
        "mse": mse,
        "rmse": rmse,
        "per_output_corr_mean": float(np.mean(per_output_corr)),
        "per_output_corr_min": float(np.min(per_output_corr)),
        "per_output_corr_max": float(np.max(per_output_corr)),
        "elapsed_sec": elapsed,
        "samples": cfg.n_samples,
    }


# ============ BENCHMARK CONFIGURATIONS ============

CONFIGS = [
    # Small layer, baseline SD-2
    BenchmarkConfig(
        name="small_baseline",
        d_in=256, d_out=256, T=64, order=2,
        beta=0.30, lambd=1/256, walsh_N=2, antithetic=True,
        n_samples=100
    ),
    # Medium layer (GPT-2 scale), baseline
    BenchmarkConfig(
        name="medium_baseline",
        d_in=768, d_out=768, T=64, order=2,
        beta=0.30, lambd=1/256, walsh_N=2, antithetic=True,
        n_samples=50
    ),
    # Medium layer, more ticks
    BenchmarkConfig(
        name="medium_T128",
        d_in=768, d_out=768, T=128, order=2,
        beta=0.30, lambd=1/256, walsh_N=2, antithetic=True,
        n_samples=50
    ),
    # Medium layer, more Walsh carriers
    BenchmarkConfig(
        name="medium_walsh4",
        d_in=768, d_out=768, T=64, order=2,
        beta=0.30, lambd=1/256, walsh_N=4, antithetic=True,
        n_samples=50
    ),
    # MLP expansion (768 -> 3072)
    BenchmarkConfig(
        name="mlp_expand",
        d_in=768, d_out=3072, T=64, order=2,
        beta=0.30, lambd=1/256, walsh_N=2, antithetic=True,
        n_samples=30
    ),
]


def main():
    print("=" * 70)
    print("SALOMI Layer Reconstruction Benchmark")
    print("Target: Pearson correlation > 0.95")
    print("=" * 70)
    print()
    
    results = []
    best_corr = 0.0
    best_cfg = None
    
    for cfg in CONFIGS:
        print(f"Running: {cfg.name} ({cfg.d_in}x{cfg.d_out}, T={cfg.T})...")
        result = run_layer_benchmark(cfg)
        results.append(result)
        
        status = "PASS" if result["correlation"] >= 0.95 else "FAIL"
        print(f"  Correlation: {result['correlation']:.4f} [{status}]")
        print(f"  RMSE: {result['rmse']:.6f}")
        print(f"  Per-output corr: min={result['per_output_corr_min']:.4f}, "
              f"mean={result['per_output_corr_mean']:.4f}, "
              f"max={result['per_output_corr_max']:.4f}")
        print(f"  Time: {result['elapsed_sec']:.2f}s")
        print()
        
        if result["correlation"] > best_corr:
            best_corr = result["correlation"]
            best_cfg = cfg.name
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = [r for r in results if r["correlation"] >= 0.95]
    print(f"Passed: {len(passed)}/{len(results)}")
    print(f"Best correlation: {best_corr:.4f} ({best_cfg})")
    
    if best_corr >= 0.95:
        print("\n>>> VALIDATION PASSED: SALOMI numerics meet the 0.95 threshold <<<")
        return 0
    else:
        print(f"\n>>> VALIDATION FAILED: Best correlation {best_corr:.4f} < 0.95 <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
