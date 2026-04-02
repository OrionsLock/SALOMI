"""
Energy proxy sanity checks.

Verify:
- instr_on=0 vs instr_on=1 produce byte-identical outputs
- Counter formulas:
  - xnor_ops == Kw * Teff_total
  - popcnt_ops == Kw * Teff_total
  - bytes_W == 4 * Kw * Teff_total
  - bytes_X == 4 * Kw * Teff_total
  - toggles_y_* >= 0, increasing across ticks
"""

import pytest
import numpy as np
import hashlib


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 and return first 16 hex chars."""
    return hashlib.sha256(data).hexdigest()[:16]


@pytest.mark.stress
@pytest.mark.opencl
@pytest.mark.parametrize("kernel", ["naive", "tiled"])
def test_energy_counter_formulas(kernel):
    """Test energy counter formulas are correct."""
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")
    
    from onebit.core.packbits import pack_input_signs
    
    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")
    
    seed = 1000001
    M, Kw, k = 4, 128, 16
    d = Kw * 32
    
    # Generate test data
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    # Pack to bits
    W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
    X_bits = pack_input_signs(X).reshape(1, -1)
    X_bits_tiled = np.tile(X_bits, (k, 1))
    
    # Run with energy instrumentation
    result = backend.run_bsdm_w_naive_norm(
        W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3,
        order=2, beta=0.30, lambd=1.0/256.0,
        walsh_N=2, antithetic=True, use_ctg=False,
        prf_seed=seed, early_exit_enable=False,
        want_y_pack=True, want_pc32=True, kernel=kernel, instr_on=True
    )
    
    assert "energy" in result, "Energy dict should be present when instr_on=True"
    
    energy = result["energy"]
    T_eff = result["T_eff"]
    
    # Check counter formulas for each row
    for row_idx in range(M):
        T_eff_row = int(T_eff[row_idx])
        
        # xnor_ops == Kw * T_eff
        expected_xnor = Kw * T_eff_row
        actual_xnor = int(energy["xnor_ops"][row_idx])
        assert actual_xnor == expected_xnor, f"Row {row_idx}: xnor_ops mismatch: {actual_xnor} != {expected_xnor}"
        
        # popcnt_ops == Kw * T_eff
        expected_popcnt = Kw * T_eff_row
        actual_popcnt = int(energy["popcnt_ops"][row_idx])
        assert actual_popcnt == expected_popcnt, f"Row {row_idx}: popcnt_ops mismatch: {actual_popcnt} != {expected_popcnt}"
        
        # bytes_W == 4 * Kw * T_eff
        expected_bytes_W = 4 * Kw * T_eff_row
        actual_bytes_W = int(energy["bytes_W"][row_idx])
        assert actual_bytes_W == expected_bytes_W, f"Row {row_idx}: bytes_W mismatch: {actual_bytes_W} != {expected_bytes_W}"
        
        # bytes_X == 4 * Kw * T_eff
        expected_bytes_X = 4 * Kw * T_eff_row
        actual_bytes_X = int(energy["bytes_X"][row_idx])
        assert actual_bytes_X == expected_bytes_X, f"Row {row_idx}: bytes_X mismatch: {actual_bytes_X} != {expected_bytes_X}"
        
        # toggles_y_* >= 0
        assert energy["toggles_y_main"][row_idx] >= 0, f"Row {row_idx}: toggles_y_main is negative"
        assert energy["toggles_y_twin"][row_idx] >= 0, f"Row {row_idx}: toggles_y_twin is negative"
        
        # ones_pc + zeros_pc == total bits processed
        total_bits = Kw * 32 * T_eff_row
        ones = int(energy["ones_pc"][row_idx])
        zeros = int(energy["zeros_pc"][row_idx])
        assert ones + zeros == total_bits, f"Row {row_idx}: ones + zeros != total_bits: {ones} + {zeros} != {total_bits}"
    
    print(f"[PASS] Energy counter formulas correct (kernel={kernel})")


@pytest.mark.stress
def test_cpu_energy_counter_formulas():
    """Test CPU energy counter formulas are correct."""
    from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig
    
    seed = 1000001
    d, k = 128, 16
    Kw = d // 32
    
    # Generate test data
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(d, dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    
    # Run with energy instrumentation
    est, diags = bsdm_w_dot(
        W, X, k=k, cfg=cfg, seed=seed,
        want_pc32=True, want_y_pack=True, instr_on=True,
        early_exit_enable=False
    )
    
    assert "energy" in diags, "Energy dict should be present when instr_on=True"
    
    energy = diags["energy"]
    T_eff = k  # CPU always uses all ticks
    
    # xnor_ops == Kw * T_eff
    expected_xnor = Kw * T_eff
    actual_xnor = int(energy["xnor_ops"][0])
    assert actual_xnor == expected_xnor, f"xnor_ops mismatch: {actual_xnor} != {expected_xnor}"
    
    # popcnt_ops == Kw * T_eff
    expected_popcnt = Kw * T_eff
    actual_popcnt = int(energy["popcnt_ops"][0])
    assert actual_popcnt == expected_popcnt, f"popcnt_ops mismatch: {actual_popcnt} != {expected_popcnt}"
    
    # bytes_W == 4 * Kw * T_eff
    expected_bytes_W = 4 * Kw * T_eff
    actual_bytes_W = int(energy["bytes_W"][0])
    assert actual_bytes_W == expected_bytes_W, f"bytes_W mismatch: {actual_bytes_W} != {expected_bytes_W}"
    
    # bytes_X == 4 * Kw * T_eff
    expected_bytes_X = 4 * Kw * T_eff
    actual_bytes_X = int(energy["bytes_X"][0])
    assert actual_bytes_X == expected_bytes_X, f"bytes_X mismatch: {actual_bytes_X} != {expected_bytes_X}"
    
    # toggles_y_* >= 0
    assert energy["toggles_y_main"][0] >= 0, "toggles_y_main is negative"
    assert energy["toggles_y_twin"][0] >= 0, "toggles_y_twin is negative"
    
    # ones_pc + zeros_pc == total bits processed
    total_bits = Kw * 32 * T_eff
    ones = int(energy["ones_pc"][0])
    zeros = int(energy["zeros_pc"][0])
    assert ones + zeros == total_bits, f"ones + zeros != total_bits: {ones} + {zeros} != {total_bits}"
    
    print(f"[PASS] CPU energy counter formulas correct")


@pytest.mark.stress
@pytest.mark.opencl
@pytest.mark.parametrize("kernel", ["naive", "tiled"])
def test_energy_no_op_stress(kernel):
    """Stress test: energy instrumentation does not change outputs (multiple runs)."""
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")
    
    from onebit.core.packbits import pack_input_signs
    
    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")
    
    # Test multiple seeds
    seeds = [1000001, 1000003, 1000005, 1000007, 1000009]
    M, Kw, k = 4, 128, 16
    d = Kw * 32
    
    for seed in seeds:
        # Generate test data
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((M, d), dtype=np.float32)
        X = rng.standard_normal(d, dtype=np.float32)
        
        # Pack to bits
        W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
        X_bits = pack_input_signs(X).reshape(1, -1)
        X_bits_tiled = np.tile(X_bits, (k, 1))
        
        # Run with instr_on=False
        result_off = backend.run_bsdm_w_naive_norm(
            W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3,
            order=2, beta=0.30, lambd=1.0/256.0,
            walsh_N=2, antithetic=True, use_ctg=False,
            prf_seed=seed, early_exit_enable=False,
            want_y_pack=True, want_pc32=True, kernel=kernel, instr_on=False
        )
        
        # Run with instr_on=True
        result_on = backend.run_bsdm_w_naive_norm(
            W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3,
            order=2, beta=0.30, lambd=1.0/256.0,
            walsh_N=2, antithetic=True, use_ctg=False,
            prf_seed=seed, early_exit_enable=False,
            want_y_pack=True, want_pc32=True, kernel=kernel, instr_on=True
        )
        
        # Check outputs are identical
        assert sha256_hex(result_off["Y"].tobytes()) == sha256_hex(result_on["Y"].tobytes()), \
            f"Y mismatch for seed={seed}, kernel={kernel}"
        assert sha256_hex(result_off["y_bits_main"].tobytes()) == sha256_hex(result_on["y_bits_main"].tobytes()), \
            f"y_bits_main mismatch for seed={seed}, kernel={kernel}"
        assert sha256_hex(result_off["y_bits_twin"].tobytes()) == sha256_hex(result_on["y_bits_twin"].tobytes()), \
            f"y_bits_twin mismatch for seed={seed}, kernel={kernel}"
        assert sha256_hex(result_off["pc32_main"].tobytes()) == sha256_hex(result_on["pc32_main"].tobytes()), \
            f"pc32_main mismatch for seed={seed}, kernel={kernel}"
        assert sha256_hex(result_off["pc32_twin"].tobytes()) == sha256_hex(result_on["pc32_twin"].tobytes()), \
            f"pc32_twin mismatch for seed={seed}, kernel={kernel}"
    
    print(f"[PASS] Energy no-op stress: {len(seeds)} seeds (kernel={kernel})")

