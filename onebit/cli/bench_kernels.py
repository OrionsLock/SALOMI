"""Benchmark naive vs tiled BSDM-W kernels with parity checks.

Usage:
    python -m onebit.cli.bench_kernels --M 128 --Kw 2048 --k 16 --order 2 --runs 50
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from onebit.backends.opencl.host_opencl import OpenCLBinGemm
from onebit.core.packbits import pack_input_signs


def check_parity(r_naive, r_tiled, tol=1e-6):
    """Check byte-parity between naive and tiled results.

    Returns:
        (bool, str): (is_match, error_message)
    """
    # Check Y_mean
    if not np.allclose(r_naive['Y'], r_tiled['Y'], atol=tol):
        return False, "Y_mean mismatch"

    # Check T_eff
    if not np.array_equal(r_naive['T_eff'], r_tiled['T_eff']):
        return False, "T_eff mismatch"

    # Check y_bits_main
    if not np.array_equal(r_naive['y_bits_main'], r_tiled['y_bits_main']):
        return False, "y_bits_main mismatch"

    # Check y_bits_twin
    if not np.array_equal(r_naive['y_bits_twin'], r_tiled['y_bits_twin']):
        return False, "y_bits_twin mismatch"

    # Check pc32 if present
    if 'pc32_main' in r_naive and 'pc32_main' in r_tiled:
        if not np.array_equal(r_naive['pc32_main'], r_tiled['pc32_main']):
            return False, "pc32_main mismatch"

    if 'pc32_twin' in r_naive and 'pc32_twin' in r_tiled:
        if not np.array_equal(r_naive['pc32_twin'], r_tiled['pc32_twin']):
            return False, "pc32_twin mismatch"

    return True, ""


def run_parity_check(backend, M, Kw, T, order, early_exit, use_ctg, seed=42):
    """Run parity check between naive and tiled kernels.

    Returns:
        (bool, str): (is_match, error_message)
    """
    # Generate test data
    np.random.seed(seed)
    d = Kw * 32
    W = np.random.randn(M, d).astype(np.float32)
    X = np.random.randn(T, d).astype(np.float32)

    W_bits = np.array([pack_input_signs(W[i]) for i in range(M)])
    X_bits = np.array([pack_input_signs(X[t]) for t in range(T)])

    # Run naive
    r_naive = backend.run_bsdm_w_naive_norm(
        W_bits=W_bits,
        X_bits=X_bits,
        T=T,
        eps=0.0,
        delta=1e-3,
        order=order,
        early_exit_enable=early_exit,
        use_ctg=use_ctg,
        prf_seed=seed,
        want_y_pack=True,
        want_pc32=False,
        kernel="naive"
    )

    # Run tiled
    r_tiled = backend.run_bsdm_w_naive_norm(
        W_bits=W_bits,
        X_bits=X_bits,
        T=T,
        eps=0.0,
        delta=1e-3,
        order=order,
        early_exit_enable=early_exit,
        use_ctg=use_ctg,
        prf_seed=seed,
        want_y_pack=True,
        want_pc32=False,
        kernel="tiled"
    )

    return check_parity(r_naive, r_tiled)


def bench_kernel(backend, W_bits, X_bits, T, order, early_exit, use_ctg, seed, kernel, runs, warmup=3):
    """Benchmark a single kernel configuration.

    Returns:
        List of run times in milliseconds
    """
    times = []

    # Warmup
    for _ in range(warmup):
        backend.run_bsdm_w_naive_norm(
            W_bits=W_bits,
            X_bits=X_bits,
            T=T,
            eps=0.0,
            delta=1e-3,
            order=order,
            early_exit_enable=early_exit,
            use_ctg=use_ctg,
            prf_seed=seed,
            want_y_pack=False,
            want_pc32=False,
            kernel=kernel
        )

    # Timed runs
    for _ in range(runs):
        t0 = time.perf_counter()
        backend.run_bsdm_w_naive_norm(
            W_bits=W_bits,
            X_bits=X_bits,
            T=T,
            eps=0.0,
            delta=1e-3,
            order=order,
            early_exit_enable=early_exit,
            use_ctg=use_ctg,
            prf_seed=seed,
            want_y_pack=False,
            want_pc32=False,
            kernel=kernel
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # Convert to ms

    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark BSDM-W kernels")
    parser.add_argument("--M", type=int, default=128, help="Number of rows")
    parser.add_argument("--Kw", type=int, default=64, help="Number of columns (words)")
    parser.add_argument("--T", type=int, default=16, help="Number of ticks")
    parser.add_argument("--order", type=int, choices=[1, 2], default=2, help="SD order")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--backend", type=str, default="opencl", choices=["opencl"], help="Backend")
    parser.add_argument("--kernel", type=str, default="auto", choices=["naive", "tiled", "auto"], help="Kernel to benchmark")
    parser.add_argument("--early-exit", type=int, default=0, choices=[0, 1], help="Enable early exit")
    parser.add_argument("--ctg", type=int, default=0, choices=[0, 1], help="Enable CTG")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-parity", action="store_true", help="Skip parity check")
    
    args = parser.parse_args()
    
    # Initialize backend
    print(f"Initializing {args.backend} backend...")
    backend = OpenCLBinGemm()
    
    # Get device info
    device = backend.ctx.devices[0]
    device_vendor = device.vendor.strip()
    device_name = device.name.strip()
    device_driver = device.driver_version.strip()
    
    print(f"Device: {device_vendor} {device_name}")
    print(f"Driver: {device_driver}")
    print(f"Config: M={args.M}, Kw={args.Kw}, T={args.T}, order={args.order}")
    print(f"Flags: early_exit={args.early_exit}, ctg={args.ctg}")
    print()

    # Run parity check if not skipped
    if not args.skip_parity and args.kernel in ["auto", "tiled"]:
        print("Running parity check (naive vs tiled)...")
        is_match, error_msg = run_parity_check(
            backend, args.M, args.Kw, args.T, args.order,
            args.early_exit, args.ctg, args.seed
        )
        
        if not is_match:
            print(f"❌ PARITY CHECK FAILED: {error_msg}")
            sys.exit(1)
        else:
            print("✅ Parity check passed")
            print()
    
    # Generate test data
    np.random.seed(args.seed)
    d = args.Kw * 32
    W = np.random.randn(args.M, d).astype(np.float32)
    X = np.random.randn(args.T, d).astype(np.float32)

    W_bits = np.array([pack_input_signs(W[i]) for i in range(args.M)])
    X_bits = np.array([pack_input_signs(X[t]) for t in range(args.T)])

    # Determine which kernels to benchmark
    kernels_to_bench = []
    if args.kernel == "auto":
        kernels_to_bench = ["naive", "tiled"]
    else:
        kernels_to_bench = [args.kernel]

    # Benchmark each kernel
    results = []
    for kernel in kernels_to_bench:
        print(f"Benchmarking {kernel} kernel ({args.runs} runs)...")
        times = bench_kernel(
            backend, W_bits, X_bits, args.T, args.order,
            args.early_exit, args.ctg, args.seed,
            kernel, args.runs
        )
        
        mean_ms = np.mean(times)
        p50_ms = np.percentile(times, 50)
        p95_ms = np.percentile(times, 95)
        
        print(f"  Mean: {mean_ms:.3f} ms")
        print(f"  P50:  {p50_ms:.3f} ms")
        print(f"  P95:  {p95_ms:.3f} ms")
        print()
        
        # Store results
        for run_idx, run_ms in enumerate(times):
            results.append({
                "device": device_name,
                "vendor": device_vendor,
                "driver": device_driver,
                "order": args.order,
                "kernel": kernel,
                "M": args.M,
                "Kw": args.Kw,
                "T": args.T,
                "early_exit": args.early_exit,
                "ctg": args.ctg,
                "run_idx": run_idx,
                "run_ms": run_ms,
            })
    
    # Write CSV
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ["device", "vendor", "driver", "order", "kernel",
                         "M", "Kw", "T", "early_exit", "ctg", "run_idx", "run_ms"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results written to {output_path}")
    
    # Compute speedup if both kernels were benchmarked
    if len(kernels_to_bench) == 2:
        naive_times = [r["run_ms"] for r in results if r["kernel"] == "naive"]
        tiled_times = [r["run_ms"] for r in results if r["kernel"] == "tiled"]
        
        speedup = np.mean(naive_times) / np.mean(tiled_times)
        print(f"Speedup (naive/tiled): {speedup:.2f}x")


if __name__ == "__main__":
    main()

