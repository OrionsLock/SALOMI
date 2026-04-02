"""CTG (Constant-Time Grammar) parity and correctness tests.

Tests:
1. CTG off parity: with use_ctg=0, CPU==OpenCL logs bit-exact (already true).
2. CTG on parity: with use_ctg=1, identical seeds => identical y_bits_*, pc32_*, ctg_digest across CPU/OpenCL.
3. Overhead budget: CTG on increases kernel time <=15% vs CTG off at k=16, N=2.
4. Distribution sanity: with CTG on, empirical mean/variance of {y_bar_t} unchanged within 1e-3 on iid synthetic.
5. Storage guard: export size unchanged; no new buffers.
"""
from __future__ import annotations

import numpy as np
import pytest
import time

from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig
from onebit.core.packbits import pack_input_signs


@pytest.mark.opencl
def test_ctg_off_parity():
    """Test 1: CTG off parity - CPU==OpenCL logs bit-exact."""
    from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    
    np.random.seed(42)
    a = pack_input_signs(np.random.randn(1024))
    b = pack_input_signs(np.random.randn(1024))
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    
    # CPU
    est_cpu, diags_cpu = bsdm_w_dot(
        a, b, 16, cfg, seed=12345, 
        use_ctg=False, early_exit_enable=False,
        want_pc32=True
    )
    
    # OpenCL
    gemm = OpenCLBinGemm()
    W_bits = np.tile(a[None, :], (1, 1))
    X_bits = np.tile(b[None, :], (16, 1))
    
    out_ocl = gemm.run_bsdm_w_naive_norm(
        W_bits, X_bits, T=16,
        eps=0.05, delta=0.001,
        order=2, beta=0.30, lambd=1.0/256.0,
        walsh_N=2, antithetic=True,
        use_ctg=False, prf_seed=12345,
        early_exit_enable=False,
        local_size=256,
        want_y_pack=True, want_pc32=True,
    )
    
    # Check parity
    assert diags_cpu["k_used"] == out_ocl["T_eff"][0], "k_used mismatch"
    assert np.allclose(est_cpu, out_ocl["Y"][0], atol=1e-6), "Estimate mismatch"
    assert np.array_equal(diags_cpu["y_bits_main"], out_ocl["y_bits_main"][0]), "y_bits_main mismatch"
    assert np.array_equal(diags_cpu["y_bits_twin"], out_ocl["y_bits_twin"][0]), "y_bits_twin mismatch"
    assert np.array_equal(diags_cpu["pc32_main"], out_ocl["pc32_main"][0]), "pc32_main mismatch"
    assert np.array_equal(diags_cpu["pc32_twin"], out_ocl["pc32_twin"][0]), "pc32_twin mismatch"
    assert diags_cpu["ctg_digest"] == out_ocl["ctg_digest"][0], "ctg_digest mismatch (should be 0)"


@pytest.mark.opencl
def test_ctg_on_parity():
    """Test 2: CTG on parity - identical seeds => identical outputs across CPU/OpenCL."""
    from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    
    np.random.seed(99)
    a = pack_input_signs(np.random.randn(1024))
    b = pack_input_signs(np.random.randn(1024))
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    
    # CPU
    est_cpu, diags_cpu = bsdm_w_dot(
        a, b, 16, cfg, seed=54321, 
        use_ctg=True, early_exit_enable=False,
        want_pc32=True
    )
    
    # OpenCL
    gemm = OpenCLBinGemm()
    W_bits = np.tile(a[None, :], (1, 1))
    X_bits = np.tile(b[None, :], (16, 1))
    
    out_ocl = gemm.run_bsdm_w_naive_norm(
        W_bits, X_bits, T=16,
        eps=0.05, delta=0.001,
        order=2, beta=0.30, lambd=1.0/256.0,
        walsh_N=2, antithetic=True,
        use_ctg=True, prf_seed=54321,
        early_exit_enable=False,
        local_size=256,
        want_y_pack=True, want_pc32=True,
    )
    
    # Check parity
    assert diags_cpu["k_used"] == out_ocl["T_eff"][0], "k_used mismatch"
    assert np.allclose(est_cpu, out_ocl["Y"][0], atol=1e-6), "Estimate mismatch"
    assert np.array_equal(diags_cpu["y_bits_main"], out_ocl["y_bits_main"][0]), "y_bits_main mismatch"
    assert np.array_equal(diags_cpu["y_bits_twin"], out_ocl["y_bits_twin"][0]), "y_bits_twin mismatch"
    assert np.array_equal(diags_cpu["pc32_main"], out_ocl["pc32_main"][0]), "pc32_main mismatch"
    assert np.array_equal(diags_cpu["pc32_twin"], out_ocl["pc32_twin"][0]), "pc32_twin mismatch"
    assert diags_cpu["ctg_digest"] == out_ocl["ctg_digest"][0], "ctg_digest mismatch"
    assert diags_cpu["ctg_digest"] != 0, "ctg_digest should be non-zero with CTG on"


@pytest.mark.opencl
@pytest.mark.slow
def test_ctg_overhead_budget():
    """Test 3: CTG on increases kernel time <=15% vs CTG off at k=16, N=2."""
    from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    
    np.random.seed(123)
    M = 256  # Enough rows to measure timing
    Kw = 32  # 1024 bits
    
    W_bits = np.random.randint(0, 2**32, (M, Kw), dtype=np.uint32)
    X_bits = np.random.randint(0, 2**32, (16, Kw), dtype=np.uint32)
    
    gemm = OpenCLBinGemm()
    
    # Warmup
    for _ in range(3):
        gemm.run_bsdm_w_naive_norm(
            W_bits, X_bits, T=16,
            eps=0.05, delta=0.001,
            order=2, beta=0.30, lambd=1.0/256.0,
            walsh_N=2, antithetic=True,
            use_ctg=False, prf_seed=12345,
            early_exit_enable=False,
            local_size=256,
            want_y_pack=False, want_pc32=False,
        )
    
    # Measure CTG off
    n_runs = 10
    t0 = time.perf_counter()
    for _ in range(n_runs):
        gemm.run_bsdm_w_naive_norm(
            W_bits, X_bits, T=16,
            eps=0.05, delta=0.001,
            order=2, beta=0.30, lambd=1.0/256.0,
            walsh_N=2, antithetic=True,
            use_ctg=False, prf_seed=12345,
            early_exit_enable=False,
            local_size=256,
            want_y_pack=False, want_pc32=False,
        )
    t_off = (time.perf_counter() - t0) / n_runs
    
    # Measure CTG on
    t0 = time.perf_counter()
    for _ in range(n_runs):
        gemm.run_bsdm_w_naive_norm(
            W_bits, X_bits, T=16,
            eps=0.05, delta=0.001,
            order=2, beta=0.30, lambd=1.0/256.0,
            walsh_N=2, antithetic=True,
            use_ctg=True, prf_seed=12345,
            early_exit_enable=False,
            local_size=256,
            want_y_pack=False, want_pc32=False,
        )
    t_on = (time.perf_counter() - t0) / n_runs
    
    overhead = (t_on - t_off) / t_off
    print(f"\nCTG overhead: {overhead*100:.1f}% (t_off={t_off*1000:.2f}ms, t_on={t_on*1000:.2f}ms)")
    
    assert overhead <= 0.15, f"CTG overhead {overhead*100:.1f}% exceeds 15% budget"


def test_ctg_distribution_sanity():
    """Test 4: CTG on preserves mean/variance within 1e-3 on iid synthetic."""
    np.random.seed(777)
    
    # Generate iid synthetic data
    n_trials = 100
    k = 16
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    
    estimates_off = []
    estimates_on = []
    
    for i in range(n_trials):
        a = pack_input_signs(np.random.randn(1024))
        b = pack_input_signs(np.random.randn(1024))
        
        # CTG off
        est_off, _ = bsdm_w_dot(
            a, b, k, cfg, seed=i, 
            use_ctg=False, early_exit_enable=False
        )
        estimates_off.append(est_off)
        
        # CTG on
        est_on, _ = bsdm_w_dot(
            a, b, k, cfg, seed=i, 
            use_ctg=True, early_exit_enable=False
        )
        estimates_on.append(est_on)
    
    mean_off = np.mean(estimates_off)
    mean_on = np.mean(estimates_on)
    var_off = np.var(estimates_off)
    var_on = np.var(estimates_on)
    
    print(f"\nCTG off: mean={mean_off:.6f}, var={var_off:.6f}")
    print(f"CTG on:  mean={mean_on:.6f}, var={var_on:.6f}")
    print(f"Mean diff: {abs(mean_on - mean_off):.6f}")
    print(f"Var diff:  {abs(var_on - var_off):.6f}")
    
    # CTG should preserve mean/variance (contrast-preserving, not biasing)
    assert abs(mean_on - mean_off) < 1e-3, f"Mean changed by {abs(mean_on - mean_off):.6f}"
    assert abs(var_on - var_off) < 1e-3, f"Variance changed by {abs(var_on - var_off):.6f}"


def test_ctg_storage_guard():
    """Test 5: Export size unchanged; no new buffers."""
    np.random.seed(888)
    a = pack_input_signs(np.random.randn(1024))
    b = pack_input_signs(np.random.randn(1024))
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    
    # CTG off
    _, diags_off = bsdm_w_dot(
        a, b, 16, cfg, seed=12345, 
        use_ctg=False, early_exit_enable=False,
        want_pc32=True
    )
    
    # CTG on
    _, diags_on = bsdm_w_dot(
        a, b, 16, cfg, seed=12345, 
        use_ctg=True, early_exit_enable=False,
        want_pc32=True
    )
    
    # Check that y_bits arrays have same size
    assert diags_off["y_bits_main"].shape == diags_on["y_bits_main"].shape, "y_bits_main size changed"
    assert diags_off["y_bits_twin"].shape == diags_on["y_bits_twin"].shape, "y_bits_twin size changed"
    
    # Check that no new arrays were added (except ctg_digest which is a scalar)
    assert set(diags_off.keys()) == set(diags_on.keys()), "Diagnostic keys changed"
    
    # Verify ctg_digest is a scalar (int), not an array
    assert isinstance(diags_on["ctg_digest"], int), "ctg_digest should be a scalar int"

