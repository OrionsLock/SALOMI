"""Throughput benchmarks for LDP-KV Stage-2 OpenCL kernel.

Soft gate: OpenCL ≥ 1.6× speedup vs CPU on medium/large cases.
Skip on missing OpenCL or iGPU with <32 CUs.
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

from onebit.ops.ldpkv import encode_kv_ldp, decode_kv_ldp_stage2
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


def benchmark_ldpkv_stage2(case_name, n_pos, d_kv, group_size, n_winners, n_runs=5):
    """Benchmark LDP-KV Stage-2 CPU vs OpenCL.
    
    Args:
        case_name: Test case name
        n_pos: Number of positions
        d_kv: KV dimension
        group_size: Group size
        n_winners: Number of winners to decode
        n_runs: Number of benchmark runs
    
    Returns:
        Dict with cpu_mean, opencl_mean, speedup
    """
    seed = 42
    np.random.seed(seed)
    
    d_kv_words = (d_kv + 31) // 32
    K_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    V_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    
    # Encode
    d_left = 6
    d_right = 3
    prf_seed = seed
    
    enc_result = encode_kv_ldp(
        K_bits, V_bits,
        d_kv=d_kv,
        d_left=d_left,
        d_right=d_right,
        prf_seed=prf_seed,
    )
    
    V_enc = enc_result["V_enc"]
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    
    winner_positions = np.arange(n_winners, dtype=np.int32)
    
    # Warmup CPU
    for _ in range(3):
        decode_kv_ldp_stage2(
            None, V_enc,
            d_kv=d_kv,
            winner_positions=winner_positions,
            row_ptr=row_ptr,
            col_idx=col_idx,
            edge_weights=edge_weights,
            k_ticks=16,
            prf_seed=prf_seed,
        )
    
    # Benchmark CPU
    cpu_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        decode_kv_ldp_stage2(
            None, V_enc,
            d_kv=d_kv,
            winner_positions=winner_positions,
            row_ptr=row_ptr,
            col_idx=col_idx,
            edge_weights=edge_weights,
            k_ticks=16,
            prf_seed=prf_seed,
        )
        cpu_times.append((time.perf_counter() - t0) * 1000)
    
    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    
    # Warmup OpenCL
    gemm = OpenCLBinGemm()
    for _ in range(3):
        gemm.run_ldpkv_decode_stage2(
            V_enc, row_ptr, col_idx, edge_weights,
            winner_positions, d_kv=d_kv
        )
    
    # Benchmark OpenCL
    opencl_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        gemm.run_ldpkv_decode_stage2(
            V_enc, row_ptr, col_idx, edge_weights,
            winner_positions, d_kv=d_kv
        )
        opencl_times.append((time.perf_counter() - t0) * 1000)
    
    opencl_mean = np.mean(opencl_times)
    opencl_std = np.std(opencl_times)
    
    speedup = cpu_mean / opencl_mean
    
    return {
        "case": case_name,
        "cpu_mean": cpu_mean,
        "cpu_std": cpu_std,
        "opencl_mean": opencl_mean,
        "opencl_std": opencl_std,
        "speedup": speedup,
    }


@pytest.mark.opencl
@pytest.mark.skipif(should_skip_perf()[0], reason=should_skip_perf()[1])
def test_ldpkv_stage2_throughput():
    """Test LDP-KV Stage-2 throughput across multiple cases.
    
    Soft gate: speedup ≥ 1.6× on medium and large cases.
    """
    # Test cases: (name, n_pos, d_kv, group_size, n_winners)
    cases = [
        ("small", 1024, 512, 64, 16),
        ("medium", 8192, 2048, 64, 32),
        ("large", 65536, 4096, 128, 64),
    ]
    
    results = []
    for case_name, n_pos, d_kv, group_size, n_winners in cases:
        print(f"\nBenchmarking {case_name}: n_pos={n_pos}, d_kv={d_kv}, winners={n_winners}")
        result = benchmark_ldpkv_stage2(case_name, n_pos, d_kv, group_size, n_winners)
        results.append(result)
        
        print(f"  CPU:    {result['cpu_mean']:.3f} ± {result['cpu_std']:.3f} ms")
        print(f"  OpenCL: {result['opencl_mean']:.3f} ± {result['opencl_std']:.3f} ms")
        print(f"  Speedup: {result['speedup']:.3f}×")
    
    # Save results
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    
    csv_path = out_dir / "bench_ldpkv.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "backend", "kernel", "mean_ms", "std_ms", "speedup"])
        
        for result in results:
            writer.writerow([
                result["case"], "cpu", "reference",
                f"{result['cpu_mean']:.3f}", f"{result['cpu_std']:.3f}", "1.000"
            ])
            writer.writerow([
                result["case"], "opencl", "stage2",
                f"{result['opencl_mean']:.3f}", f"{result['opencl_std']:.3f}",
                f"{result['speedup']:.3f}"
            ])
    
    print(f"\n{'='*60}")
    print(f"LDP-KV Stage-2 Throughput Benchmark")
    print(f"{'='*60}")
    print(f"Device: {get_device_info()[0]} ({get_device_info()[1]} CUs)")
    print(f"\nResults saved to: {csv_path}")
    print(f"{'='*60}\n")
    
    # Check soft gates
    medium_result = results[1]  # medium case
    large_result = results[2]   # large case
    
    if is_soft_gate():
        if medium_result["speedup"] < 1.6:
            pytest.xfail(f"Medium speedup {medium_result['speedup']:.3f}× (target ≥1.6×)")
        if large_result["speedup"] < 1.6:
            pytest.xfail(f"Large speedup {large_result['speedup']:.3f}× (target ≥1.6×)")
    else:
        assert medium_result["speedup"] >= 1.6, \
            f"Medium case: expected speedup ≥1.6×, got {medium_result['speedup']:.3f}×"
        assert large_result["speedup"] >= 1.6, \
            f"Large case: expected speedup ≥1.6×, got {large_result['speedup']:.3f}×"


@pytest.mark.opencl
@pytest.mark.skipif(should_skip_perf()[0], reason=should_skip_perf()[1])
def test_ldpkv_stage2_parity_under_load():
    """Verify parity is maintained under throughput load."""
    seed = 42
    np.random.seed(seed)
    
    n_pos = 8192
    d_kv = 2048
    d_kv_words = (d_kv + 31) // 32
    
    K_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    V_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    
    # Encode
    enc_result = encode_kv_ldp(
        K_bits, V_bits,
        d_kv=d_kv,
        d_left=6,
        d_right=3,
        prf_seed=seed,
    )
    
    V_enc = enc_result["V_enc"]
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    
    winner_positions = np.arange(32, dtype=np.int32)
    
    # CPU
    cpu_result = decode_kv_ldp_stage2(
        None, V_enc,
        d_kv=d_kv,
        winner_positions=winner_positions,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=16,
        prf_seed=seed,
    )
    
    # OpenCL
    gemm = OpenCLBinGemm()
    opencl_result = gemm.run_ldpkv_decode_stage2(
        V_enc, row_ptr, col_idx, edge_weights,
        winner_positions, d_kv=d_kv
    )
    
    # Verify parity
    np.testing.assert_array_equal(
        cpu_result["V_decoded"],
        opencl_result["V_decoded"],
        err_msg="V_decoded mismatch under throughput load"
    )


if __name__ == "__main__":
    # Run benchmark manually
    skip, reason = should_skip_perf()
    if skip:
        print(f"Skipping: {reason}")
    else:
        test_ldpkv_stage2_throughput()
        test_ldpkv_stage2_parity_under_load()
        print("✅ All LDP-KV Stage-2 throughput tests passed!")

