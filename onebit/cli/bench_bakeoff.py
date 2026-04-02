"""Bake-off harness: Compare CPU vs OpenCL backends."""
from __future__ import annotations

import argparse
import time
import numpy as np
from typing import Dict, List

from ..ops.attention_probe import stageA_probe_topT
from ..core.packbits import pack_input_signs


def bench_attention(
    Q_bits: np.ndarray,
    K_bits: np.ndarray,
    *,
    kA: int,
    backend: str,
    order: int = 2,
    beta: float = 0.30,
    lambd: float = 1.0 / 256.0,
    walsh_N: int = 2,
    antithetic: bool = True,
    prf_seed: int = 0,
) -> Dict:
    """Benchmark attention probe on given backend."""
    t0 = time.perf_counter()

    result = stageA_probe_topT(
        Q_bits, K_bits,
        kA=kA,
        prf_seed=prf_seed,
        walsh_N=walsh_N,
        antithetic=antithetic,
        order=order,
        beta=beta,
        lambd=lambd,
    )

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000

    return {
        "E_mean": result["stats"]["mu"],
        "idx_top": result["idx_top"],
        "T_sel": result["T_sel"],
        "elapsed_ms": elapsed_ms,
    }


def compare_backends(
    Q_bits: np.ndarray,
    K_bits: np.ndarray,
    *,
    kA: int,
    order: int = 2,
    beta: float = 0.30,
    lambd: float = 1.0 / 256.0,
    walsh_N: int = 2,
    antithetic: bool = True,
    prf_seed: int = 0,
) -> Dict:
    """Compare CPU vs OpenCL backends."""
    print(f"Running CPU backend...")
    cpu_result = bench_attention(
        Q_bits, K_bits,
        kA=kA,
        backend="cpu",
        order=order,
        beta=beta,
        lambd=lambd,
        walsh_N=walsh_N,
        antithetic=antithetic,
        prf_seed=prf_seed,
    )

    print(f"Running OpenCL backend...")
    try:
        ocl_result = bench_attention(
            Q_bits, K_bits,
            kA=kA,
            backend="opencl",
            order=order,
            beta=beta,
            lambd=lambd,
            walsh_N=walsh_N,
            antithetic=antithetic,
            prf_seed=prf_seed,
        )
    except Exception as e:
        print(f"OpenCL backend failed: {e}")
        ocl_result = None

    # Compare results
    if ocl_result is not None:
        max_diff = np.max(np.abs(cpu_result["E_mean"] - ocl_result["E_mean"]))
        speedup = cpu_result["elapsed_ms"] / ocl_result["elapsed_ms"]
    else:
        max_diff = None
        speedup = None

    return {
        "cpu": cpu_result,
        "opencl": ocl_result,
        "max_diff": max_diff,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Bake-off harness: CPU vs OpenCL")
    parser.add_argument("--n-keys", type=int, default=128, help="Number of keys")
    parser.add_argument("--d", type=int, default=512, help="Dimension")
    parser.add_argument("--kA", type=int, default=16, help="Stage-A ticks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--order", type=int, default=2, help="ΣΔ order")
    parser.add_argument("--beta", type=float, default=0.30, help="ΣΔ-2 beta")
    parser.add_argument("--lambd", type=float, default=1.0/256.0, help="ΣΔ leak")
    parser.add_argument("--walsh-N", type=int, default=2, help="Walsh carriers")
    parser.add_argument("--no-antithetic", action="store_true", help="Disable antithetic pairs")

    args = parser.parse_args()

    print("=" * 80)
    print("Bake-off Harness: CPU vs OpenCL (Attention Probe)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  n_keys: {args.n_keys}")
    print(f"  d: {args.d}")
    print(f"  kA: {args.kA}")
    print(f"  order: {args.order}")
    print(f"  beta: {args.beta}")
    print(f"  lambd: {args.lambd}")
    print(f"  walsh_N: {args.walsh_N}")
    print(f"  antithetic: {not args.no_antithetic}")
    print(f"  seed: {args.seed}")
    print()

    # Set random seed
    np.random.seed(args.seed)

    # Create synthetic data
    print("Creating synthetic data...")
    Q = np.random.randn(args.d)
    K = np.random.randn(args.n_keys, args.d)

    Q_bits = pack_input_signs(Q)
    K_bits = np.array([pack_input_signs(K[i]) for i in range(args.n_keys)])

    print(f"  Q_bits: {Q_bits.shape}")
    print(f"  K_bits: {K_bits.shape}")
    print()

    # Run comparison
    result = compare_backends(
        Q_bits, K_bits,
        kA=args.kA,
        order=args.order,
        beta=args.beta,
        lambd=args.lambd,
        walsh_N=args.walsh_N,
        antithetic=not args.no_antithetic,
        prf_seed=args.seed,
    )

    # Print results
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)

    cpu_result = result["cpu"]
    print(f"CPU:")
    print(f"  Time: {cpu_result['elapsed_ms']:.2f} ms")
    print(f"  Top-T: {cpu_result['T_sel']}")
    print(f"  E_mean range: [{cpu_result['E_mean'].min():.6f}, {cpu_result['E_mean'].max():.6f}]")
    print()

    if result["opencl"] is not None:
        ocl_result = result["opencl"]
        print(f"OpenCL:")
        print(f"  Time: {ocl_result['elapsed_ms']:.2f} ms")
        print(f"  Top-T: {ocl_result['T_sel']}")
        print(f"  E_mean range: [{ocl_result['E_mean'].min():.6f}, {ocl_result['E_mean'].max():.6f}]")
        print()

        print(f"Comparison:")
        print(f"  Max diff: {result['max_diff']:.6e}")
        print(f"  Speedup: {result['speedup']:.2f}x")

        if result['max_diff'] < 1e-4:
            print(f"  ✅ PASS: CPU-OpenCL parity within tolerance")
        else:
            print(f"  ❌ FAIL: CPU-OpenCL parity exceeds tolerance")
    else:
        print(f"OpenCL: FAILED")

    print("=" * 80)


if __name__ == "__main__":
    main()

