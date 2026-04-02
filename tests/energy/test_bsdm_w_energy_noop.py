"""Test that energy instrumentation does not change BSDM-W outputs (PR-3.7).

Acceptance gate: counters off vs on produce identical outputs (hard).
"""
import hashlib
import numpy as np
import pytest

from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig


def sha16_u32(a: np.ndarray) -> str:
    """SHA256 digest of uint32 array, truncated to 16 hex chars."""
    a = np.ascontiguousarray(a.astype(np.uint32))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()[:16]


def sha16_i32(a: np.ndarray) -> str:
    """SHA256 digest of int32 array, truncated to 16 hex chars."""
    a = np.ascontiguousarray(a.astype(np.int32))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()[:16]


def sha16_f32(a: np.ndarray) -> str:
    """SHA256 digest of float32 array, truncated to 16 hex chars."""
    a = np.ascontiguousarray(a.astype(np.float32))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()[:16]


def test_bsdm_w_energy_noop_cpu():
    """Test CPU BSDM-W: instr_on=False vs instr_on=True produce identical outputs."""
    seed = 42
    d, k = 128, 8

    rng = np.random.default_rng(seed)
    W = rng.standard_normal(d, dtype=np.float32)  # Single row
    X = rng.standard_normal(d, dtype=np.float32)

    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)

    # Run with instr_on=False
    est_off, diags_off = bsdm_w_dot(
        W, X, k=k, cfg=cfg, seed=seed, want_y_pack=True, want_pc32=True, instr_on=False
    )

    # Run with instr_on=True
    est_on, diags_on = bsdm_w_dot(
        W, X, k=k, cfg=cfg, seed=seed, want_y_pack=True, want_pc32=True, instr_on=True
    )

    # Check outputs are identical
    assert abs(est_off - est_on) < 1e-9, "Estimate mismatch"
    assert diags_off["k_used"] == diags_on["k_used"], "k_used mismatch"
    assert sha16_u32(diags_off["y_bits_main"]) == sha16_u32(diags_on["y_bits_main"]), "y_bits_main mismatch"
    assert sha16_u32(diags_off["y_bits_twin"]) == sha16_u32(diags_on["y_bits_twin"]), "y_bits_twin mismatch"

    # pc32 lists - convert to arrays for hashing
    pc32_main_off = np.array(diags_off["pc32_main"], dtype=np.int32)
    pc32_main_on = np.array(diags_on["pc32_main"], dtype=np.int32)
    assert sha16_i32(pc32_main_off) == sha16_i32(pc32_main_on), "pc32_main mismatch"

    pc32_twin_off = np.array(diags_off["pc32_twin"], dtype=np.int32)
    pc32_twin_on = np.array(diags_on["pc32_twin"], dtype=np.int32)
    assert sha16_i32(pc32_twin_off) == sha16_i32(pc32_twin_on), "pc32_twin mismatch"

    # Check energy dict is present only when instr_on=True
    assert "energy" not in diags_off, "energy dict should not be present when instr_on=False"
    assert "energy" in diags_on, "energy dict should be present when instr_on=True"

    # Check energy dict has all required keys
    energy = diags_on["energy"]
    required_keys = [
        "toggles_y_main", "toggles_y_twin", "ones_pc", "zeros_pc",
        "xnor_ops", "popcnt_ops", "bytes_W", "bytes_X", "bytes_out"
    ]
    for key in required_keys:
        assert key in energy, f"Missing energy key: {key}"
        assert energy[key].shape == (1,), f"Wrong shape for {key} (expected (1,) for single-row CPU)"
        assert energy[key].dtype == np.uint64, f"Wrong dtype for {key}"

    print(f"[PASS] CPU energy instrumentation does not change outputs")
    print(f"  Estimate: {est_on:.6f}")
    print(f"  Energy keys present: {list(energy.keys())}")


@pytest.mark.opencl
def test_bsdm_w_energy_noop_opencl():
    """Test OpenCL BSDM-W: instr_on=False vs instr_on=True produce identical outputs."""
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")
    
    seed = 42
    M, d, k = 4, 128, 8
    Kw = d // 32
    
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    # Pack to bits
    from onebit.core.packbits import pack_input_signs
    W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
    X_bits = pack_input_signs(X).reshape(1, -1)
    X_bits_tiled = np.tile(X_bits, (k, 1))  # Replicate for k ticks
    
    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")
    
    # Run with instr_on=False (tiled kernel)
    result_off = backend.run_bsdm_w_naive_norm(
        W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3, order=2,
        beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True,
        use_ctg=False, prf_seed=seed, early_exit_enable=False,
        want_y_pack=True, want_pc32=True, kernel="tiled", instr_on=False
    )
    
    # Run with instr_on=True (tiled kernel)
    result_on = backend.run_bsdm_w_naive_norm(
        W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3, order=2,
        beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True,
        use_ctg=False, prf_seed=seed, early_exit_enable=False,
        want_y_pack=True, want_pc32=True, kernel="tiled", instr_on=True
    )
    
    # Check outputs are identical
    assert sha16_f32(result_off["Y"]) == sha16_f32(result_on["Y"]), "Y mismatch"
    assert sha16_i32(result_off["T_eff"]) == sha16_i32(result_on["T_eff"]), "T_eff mismatch"
    assert sha16_u32(result_off["y_bits_main"]) == sha16_u32(result_on["y_bits_main"]), "y_bits_main mismatch"
    assert sha16_u32(result_off["y_bits_twin"]) == sha16_u32(result_on["y_bits_twin"]), "y_bits_twin mismatch"
    assert sha16_i32(result_off["pc32_main"]) == sha16_i32(result_on["pc32_main"]), "pc32_main mismatch"
    assert sha16_i32(result_off["pc32_twin"]) == sha16_i32(result_on["pc32_twin"]), "pc32_twin mismatch"
    assert sha16_u32(result_off["ctg_digest"]) == sha16_u32(result_on["ctg_digest"]), "ctg_digest mismatch"
    
    # Check energy dict is present only when instr_on=True
    assert "energy" not in result_off, "energy dict should not be present when instr_on=False"
    assert "energy" in result_on, "energy dict should be present when instr_on=True"
    
    # Check energy dict has all required keys
    energy = result_on["energy"]
    required_keys = [
        "toggles_y_main", "toggles_y_twin", "ones_pc", "zeros_pc",
        "xnor_ops", "popcnt_ops", "bytes_W", "bytes_X", "bytes_out"
    ]
    for key in required_keys:
        assert key in energy, f"Missing energy key: {key}"
        assert energy[key].shape == (M,), f"Wrong shape for {key}"
        assert energy[key].dtype == np.uint64, f"Wrong dtype for {key}"
    
    print(f"[PASS] OpenCL tiled energy instrumentation does not change outputs")
    print(f"  Y digest: {sha16_f32(result_on['Y'])}")
    print(f"  Energy keys present: {list(energy.keys())}")


@pytest.mark.opencl
def test_bsdm_w_energy_noop_opencl_naive():
    """Test OpenCL BSDM-W naive kernel: instr_on=False vs instr_on=True produce identical outputs."""
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")

    seed = 42
    M, d, k = 4, 128, 8
    Kw = d // 32

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)

    # Pack to bits
    from onebit.core.packbits import pack_input_signs
    W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
    X_bits = pack_input_signs(X).reshape(1, -1)
    X_bits_tiled = np.tile(X_bits, (k, 1))  # Replicate for k ticks

    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")

    # Run with instr_on=False (naive kernel)
    result_off = backend.run_bsdm_w_naive_norm(
        W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3, order=2,
        beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True,
        use_ctg=False, prf_seed=seed, early_exit_enable=False,
        want_y_pack=True, want_pc32=True, kernel="naive", instr_on=False
    )

    # Run with instr_on=True (naive kernel)
    result_on = backend.run_bsdm_w_naive_norm(
        W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3, order=2,
        beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True,
        use_ctg=False, prf_seed=seed, early_exit_enable=False,
        want_y_pack=True, want_pc32=True, kernel="naive", instr_on=True
    )

    # Check outputs are identical
    assert sha16_f32(result_off["Y"]) == sha16_f32(result_on["Y"]), "Y mismatch"
    assert sha16_i32(result_off["T_eff"]) == sha16_i32(result_on["T_eff"]), "T_eff mismatch"
    assert sha16_u32(result_off["y_bits_main"]) == sha16_u32(result_on["y_bits_main"]), "y_bits_main mismatch"
    assert sha16_u32(result_off["y_bits_twin"]) == sha16_u32(result_on["y_bits_twin"]), "y_bits_twin mismatch"
    assert sha16_i32(result_off["pc32_main"]) == sha16_i32(result_on["pc32_main"]), "pc32_main mismatch"
    assert sha16_i32(result_off["pc32_twin"]) == sha16_i32(result_on["pc32_twin"]), "pc32_twin mismatch"
    assert sha16_u32(result_off["ctg_digest"]) == sha16_u32(result_on["ctg_digest"]), "ctg_digest mismatch"

    # Check energy dict is present only when instr_on=True
    assert "energy" not in result_off, "energy dict should not be present when instr_on=False"
    assert "energy" in result_on, "energy dict should be present when instr_on=True"

    # Check energy dict has all required keys
    energy = result_on["energy"]
    required_keys = [
        "toggles_y_main", "toggles_y_twin", "ones_pc", "zeros_pc",
        "xnor_ops", "popcnt_ops", "bytes_W", "bytes_X", "bytes_out"
    ]
    for key in required_keys:
        assert key in energy, f"Missing energy key: {key}"
        assert energy[key].shape == (M,), f"Wrong shape for {key}"
        assert energy[key].dtype == np.uint64, f"Wrong dtype for {key}"

    print(f"[PASS] OpenCL naive energy instrumentation does not change outputs")
    print(f"  Y digest: {sha16_f32(result_on['Y'])}")
    print(f"  Energy keys present: {list(energy.keys())}")


if __name__ == "__main__":
    test_bsdm_w_energy_noop_cpu()
    print()
    test_bsdm_w_energy_noop_opencl()

