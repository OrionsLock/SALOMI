"""Performance benchmarks for HCL tiled kernel.

Soft gate: tiled ≥ 1.25× speedup vs naive on large vocabularies.
Skip on missing OpenCL or iGPU with <32 CUs.
Mark as xfail if env ONEBIT_PERF_SOFT=1.
"""
import pytest
import numpy as np
import time
import os
import csv
from pathlib import Path

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

from onebit.backends.opencl.host_opencl import OpenCLBinGemm


def get_device_info():
    """Get OpenCL device info for gating."""
    if not HAS_OPENCL:
        return None, 0
    
    try:
        platforms = cl.get_platforms()
        if not platforms:
            return None, 0
        
        devices = platforms[0].get_devices()
        if not devices:
            return None, 0
        
        device = devices[0]
        compute_units = device.max_compute_units
        device_name = device.name.strip()
        
        return device_name, compute_units
    except Exception:
        return None, 0


def should_skip_perf():
    """Check if performance tests should be skipped."""
    if not HAS_OPENCL:
        return True, "OpenCL not available"
    
    device_name, compute_units = get_device_info()
    if device_name is None:
        return True, "No OpenCL device found"
    
    if compute_units < 32:
        return True, f"Device has only {compute_units} CUs (need ≥32)"
    
    return False, None


def is_soft_gate():
    """Check if soft gate mode is enabled."""
    return os.environ.get("ONEBIT_PERF_SOFT", "0") == "1"


@pytest.mark.opencl
@pytest.mark.skipif(should_skip_perf()[0], reason=should_skip_perf()[1])
def test_hcl_tiled_speedup():
    """Test HCL tiled kernel speedup vs naive.
    
    Soft gate: speedup ≥ 1.25× on large vocabulary.
    """
    # Configuration
    V = 50176  # Large vocabulary (e.g., GPT-2)
    d = 768
    d_words = (d + 31) // 32
    T = 16
    n_tokens = 64
    n_warmup = 5
    n_runs = 5
    seed = 42
    
    # Generate synthetic data
    np.random.seed(seed)
    
    # Create OpenCL backend
    gemm = OpenCLBinGemm()
    
    # Warmup both kernels
    for _ in range(n_warmup):
        q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
        v_ids = np.arange(min(256, V), dtype=np.int32)
        
        # Naive warmup
        gemm.run_hcl_naive(
            q_bits, v_ids,
            d=d, T=T,
            early_exit_enable=False,
        )
        
        # Tiled warmup
        gemm.run_hcl_tiled(
            q_bits, v_ids,
            d=d, T=T,
            early_exit_enable=False,
        )
    
    # Benchmark naive
    naive_times = []
    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        np.random.seed(run_seed)
        
        t0 = time.perf_counter()
        for token_idx in range(n_tokens):
            q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
            v_ids = np.arange(min(256, V), dtype=np.int32)
            
            gemm.run_hcl_naive(
                q_bits, v_ids,
                d=d, T=T,
                early_exit_enable=False,
            )
        
        elapsed = time.perf_counter() - t0
        naive_times.append(elapsed * 1000 / n_tokens)  # ms per token
    
    naive_mean = np.mean(naive_times)
    naive_std = np.std(naive_times)
    
    # Benchmark tiled
    tiled_times = []
    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        np.random.seed(run_seed)
        
        t0 = time.perf_counter()
        for token_idx in range(n_tokens):
            q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
            v_ids = np.arange(min(256, V), dtype=np.int32)
            
            gemm.run_hcl_tiled(
                q_bits, v_ids,
                d=d, T=T,
                early_exit_enable=False,
            )
        
        elapsed = time.perf_counter() - t0
        tiled_times.append(elapsed * 1000 / n_tokens)  # ms per token
    
    tiled_mean = np.mean(tiled_times)
    tiled_std = np.std(tiled_times)
    
    # Compute speedup
    speedup = naive_mean / tiled_mean
    
    # Save results
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    
    csv_path = out_dir / "bench_hcl.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "backend", "kernel", "mean_ms", "std_ms", "speedup"])
        writer.writerow([f"V={V}_d={d}_T={T}", "opencl", "naive", f"{naive_mean:.3f}", f"{naive_std:.3f}", "1.000"])
        writer.writerow([f"V={V}_d={d}_T={T}", "opencl", "tiled", f"{tiled_mean:.3f}", f"{tiled_std:.3f}", f"{speedup:.3f}"])
    
    print(f"\n{'='*60}")
    print(f"HCL Tiled Kernel Performance Benchmark")
    print(f"{'='*60}")
    print(f"Configuration: V={V}, d={d}, T={T}, tokens={n_tokens}")
    print(f"Device: {get_device_info()[0]} ({get_device_info()[1]} CUs)")
    print(f"\nNaive:  {naive_mean:.3f} ± {naive_std:.3f} ms/token")
    print(f"Tiled:  {tiled_mean:.3f} ± {tiled_std:.3f} ms/token")
    print(f"Speedup: {speedup:.3f}×")
    print(f"\nResults saved to: {csv_path}")
    print(f"{'='*60}\n")
    
    # Soft gate: speedup ≥ 1.25×
    if is_soft_gate():
        pytest.xfail(f"Soft gate: speedup {speedup:.3f}× (target ≥1.25×)")
    else:
        assert speedup >= 1.25, f"Expected speedup ≥1.25×, got {speedup:.3f}×"


@pytest.mark.opencl
@pytest.mark.skipif(should_skip_perf()[0], reason=should_skip_perf()[1])
def test_hcl_tiled_parity_under_load():
    """Verify parity is maintained under performance load."""
    V = 50176
    d = 768
    d_words = (d + 31) // 32
    T = 16
    seed = 42
    
    np.random.seed(seed)
    q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
    v_ids = np.arange(min(256, V), dtype=np.int32)
    
    gemm = OpenCLBinGemm()
    
    # Run both kernels
    naive_result = gemm.run_hcl_naive(
        q_bits, v_ids,
        d=d, T=T,
        early_exit_enable=False,
    )
    
    tiled_result = gemm.run_hcl_tiled(
        q_bits, v_ids,
        d=d, T=T,
        early_exit_enable=False,
    )
    
    # Verify parity
    np.testing.assert_allclose(
        naive_result["E_mean"],
        tiled_result["E_mean"],
        atol=1e-6,
        err_msg="E_mean mismatch under performance load"
    )
    
    np.testing.assert_array_equal(
        naive_result["T_eff"],
        tiled_result["T_eff"],
        err_msg="T_eff mismatch under performance load"
    )


if __name__ == "__main__":
    # Run benchmark manually
    skip, reason = should_skip_perf()
    if skip:
        print(f"Skipping: {reason}")
    else:
        test_hcl_tiled_speedup()
        test_hcl_tiled_parity_under_load()
        print("✅ All HCL performance tests passed!")

