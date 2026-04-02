"""Tests for HCL tiled kernel byte-parity with naive kernel."""
import pytest
import numpy as np

from onebit.backends.opencl.host_opencl import OpenCLBinGemm


@pytest.mark.parametrize("seed", [42, 7, 99999])
@pytest.mark.parametrize("Kc,d", [(16, 512), (32, 1024), (64, 2048)])
@pytest.mark.parametrize("order", [1, 2])
def test_hcl_tiled_parity_vs_naive(seed, Kc, d, order):
    """Test that HCL tiled kernel matches naive kernel exactly."""
    np.random.seed(seed)
    
    # Generate test data
    d_words = (d + 31) // 32
    Q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
    v_ids = np.random.randint(0, 50000, size=Kc, dtype=np.int32)
    
    T = 16
    prf_seed = seed
    use_ctg = False
    beta = 0.30
    lambd = 1.0 / 256.0
    
    gemm = OpenCLBinGemm()
    
    # Run naive kernel
    naive_result = gemm.run_hcl_naive(
        Q_bits, v_ids,
        d=d, T=T,
        use_ctg=use_ctg,
        prf_seed=prf_seed,
        early_exit_enable=False,
        eps=0.0,
        delta=1e-3,
        order=order,
        beta=beta,
        lambd=lambd,
        want_bits=False,
    )
    
    # Run tiled kernel
    tiled_result = gemm.run_hcl_tiled(
        Q_bits, v_ids,
        d=d, T=T,
        use_ctg=use_ctg,
        prf_seed=prf_seed,
        early_exit_enable=False,
        eps=0.0,
        delta=1e-3,
        order=order,
        beta=beta,
        lambd=lambd,
        want_bits=False,
    )
    
    # Check byte-parity
    np.testing.assert_allclose(
        naive_result["E_mean"],
        tiled_result["E_mean"],
        atol=1e-6,
        err_msg="E_mean should match between naive and tiled"
    )
    
    np.testing.assert_array_equal(
        naive_result["T_eff"],
        tiled_result["T_eff"],
        err_msg="T_eff should match exactly"
    )
    
    np.testing.assert_array_equal(
        naive_result["ctg_digest"],
        tiled_result["ctg_digest"],
        err_msg="CTG digest should match exactly"
    )


@pytest.mark.parametrize("use_ctg", [False, True])
def test_hcl_tiled_with_ctg(use_ctg):
    """Test HCL tiled kernel with CTG enabled/disabled."""
    seed = 12345
    np.random.seed(seed)
    
    Kc = 32
    d = 1024
    d_words = (d + 31) // 32
    
    Q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
    v_ids = np.random.randint(0, 50000, size=Kc, dtype=np.int32)
    
    T = 16
    prf_seed = seed
    
    gemm = OpenCLBinGemm()
    
    # Run naive kernel
    naive_result = gemm.run_hcl_naive(
        Q_bits, v_ids,
        d=d, T=T,
        use_ctg=use_ctg,
        prf_seed=prf_seed,
        early_exit_enable=False,
    )
    
    # Run tiled kernel
    tiled_result = gemm.run_hcl_tiled(
        Q_bits, v_ids,
        d=d, T=T,
        use_ctg=use_ctg,
        prf_seed=prf_seed,
        early_exit_enable=False,
    )
    
    # Check parity
    np.testing.assert_allclose(
        naive_result["E_mean"],
        tiled_result["E_mean"],
        atol=1e-6,
    )
    
    if use_ctg:
        # CTG digest should match
        np.testing.assert_array_equal(
            naive_result["ctg_digest"],
            tiled_result["ctg_digest"],
        )


def test_hcl_tiled_determinism():
    """Test that HCL tiled kernel is deterministic."""
    seed = 7777
    np.random.seed(seed)
    
    Kc = 24
    d = 768
    d_words = (d + 31) // 32
    
    Q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
    v_ids = np.random.randint(0, 50000, size=Kc, dtype=np.int32)
    
    T = 16
    prf_seed = seed
    
    gemm = OpenCLBinGemm()
    
    # Run twice with same seed
    result1 = gemm.run_hcl_tiled(
        Q_bits, v_ids,
        d=d, T=T,
        use_ctg=False,
        prf_seed=prf_seed,
        early_exit_enable=False,
    )
    
    result2 = gemm.run_hcl_tiled(
        Q_bits, v_ids,
        d=d, T=T,
        use_ctg=False,
        prf_seed=prf_seed,
        early_exit_enable=False,
    )
    
    # Should be identical
    np.testing.assert_array_equal(
        result1["E_mean"],
        result2["E_mean"],
        err_msg="Results should be deterministic"
    )
    
    np.testing.assert_array_equal(
        result1["T_eff"],
        result2["T_eff"],
    )


@pytest.mark.skipif(
    True,  # Skip by default, run manually for performance testing
    reason="Performance test, run manually"
)
def test_hcl_tiled_speedup():
    """Test that HCL tiled kernel is faster than naive on large shapes."""
    import time
    
    seed = 42
    np.random.seed(seed)
    
    Kc = 256
    d = 4096
    d_words = (d + 31) // 32
    
    Q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
    v_ids = np.random.randint(0, 50000, size=Kc, dtype=np.int32)
    
    T = 32
    prf_seed = seed
    
    gemm = OpenCLBinGemm()
    
    # Warmup
    for _ in range(3):
        gemm.run_hcl_naive(Q_bits, v_ids, d=d, T=T, prf_seed=prf_seed, early_exit_enable=False)
        gemm.run_hcl_tiled(Q_bits, v_ids, d=d, T=T, prf_seed=prf_seed, early_exit_enable=False)
    
    # Benchmark naive
    runs = 10
    t0 = time.perf_counter()
    for _ in range(runs):
        gemm.run_hcl_naive(Q_bits, v_ids, d=d, T=T, prf_seed=prf_seed, early_exit_enable=False)
    t_naive = (time.perf_counter() - t0) / runs
    
    # Benchmark tiled
    t0 = time.perf_counter()
    for _ in range(runs):
        gemm.run_hcl_tiled(Q_bits, v_ids, d=d, T=T, prf_seed=prf_seed, early_exit_enable=False)
    t_tiled = (time.perf_counter() - t0) / runs
    
    speedup = t_naive / t_tiled
    print(f"\nNaive: {t_naive*1000:.2f}ms, Tiled: {t_tiled*1000:.2f}ms, Speedup: {speedup:.2f}x")
    
    # Soft gate: expect at least 1.25x speedup
    assert speedup >= 1.25, f"Expected speedup >= 1.25x, got {speedup:.2f}x"


if __name__ == "__main__":
    # Run parity tests
    test_hcl_tiled_parity_vs_naive(42, 32, 1024, 2)
    test_hcl_tiled_with_ctg(False)
    test_hcl_tiled_with_ctg(True)
    test_hcl_tiled_determinism()
    print("✅ All HCL tiled parity tests passed!")

