"""Golden log parity tests for BSDM-W (CPU vs OpenCL)."""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from onebit.backends.opencl.host_opencl import OpenCLBinGemm
from onebit.core.golden_bits import pack_y_bits_to_hex
from onebit.core.packbits import pack_input_signs
from onebit.core.prf import derive_seed
from onebit.ops.bsdm_w import SDConfig, bsdm_w_dot


def _rand_pm1(K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=K, dtype=np.int8)
    return (x * 2 - 1).astype(np.int8)


def _pack_pm1(x: np.ndarray) -> np.ndarray:
    return pack_input_signs(x.astype(np.float32))


@pytest.mark.opencl
def test_golden_parity_single_row():
    """Test CPU vs OpenCL parity for a single row (simplest case)."""
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")
    
    K = 1024
    M = 1
    k_max = 16
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    eps = 0.05
    delta = 0.001
    run_id = str(uuid.uuid4())
    layer, token = 0, 0
    
    # Generate data
    rng = np.random.default_rng(42)
    W_row = _rand_pm1(K, int(rng.integers(0, 1<<31)))
    X_vec = _rand_pm1(K, int(rng.integers(0, 1<<31)))
    W_bits = _pack_pm1(W_row)[None, :]
    X_bits = _pack_pm1(X_vec)[None, :]
    
    # Derive seed
    run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
    seed = derive_seed(layer, 0, token, run_id_int)
    
    # CPU
    est_cpu, diags_cpu = bsdm_w_dot(W_bits[0], X_bits[0], k_max, cfg, seed=int(seed), want_pc32=True, eps=eps, delta=delta)
    k_used_cpu = diags_cpu["k_used"]
    y_main_cpu = diags_cpu["y_bits_main"]
    y_twin_cpu = diags_cpu["y_bits_twin"]
    pc32_main_cpu = diags_cpu["pc32_main"]
    pc32_twin_cpu = diags_cpu["pc32_twin"]
    
    # OpenCL
    X_ticks = np.tile(X_bits, (k_max, 1))
    out_cl = gemm.run_bsdm_w_naive_norm(
        W_bits, X_ticks, T=k_max, eps=eps, delta=delta,
        order=cfg.order, beta=cfg.beta, lambd=cfg.lambd,
        walsh_N=cfg.walsh_N, antithetic=cfg.antithetic,
        use_ctg=False, prf_seed=int(seed),
        local_size=256, want_y_pack=True, want_pc32=True
    )
    est_cl = float(out_cl["Y"][0])
    k_used_cl = int(out_cl["T_eff"][0])
    y_main_cl = out_cl["y_bits_main"][0]
    y_twin_cl = out_cl["y_bits_twin"][0]
    pc32_main_cl = out_cl["pc32_main"][0]
    pc32_twin_cl = out_cl["pc32_twin"][0]
    
    # Parity checks
    assert k_used_cpu == k_used_cl, f"T_eff mismatch: CPU={k_used_cpu}, OpenCL={k_used_cl}"
    assert np.allclose(est_cpu, est_cl, atol=1e-6, rtol=0.0), f"Y_mean mismatch: CPU={est_cpu}, OpenCL={est_cl}"
    
    # Bit parity
    samples_per_channel = k_used_cpu * cfg.walsh_N
    y_main_hex_cpu = pack_y_bits_to_hex(y_main_cpu, samples_per_channel)
    y_main_hex_cl = pack_y_bits_to_hex(y_main_cl, samples_per_channel)
    assert y_main_hex_cpu == y_main_hex_cl, f"y_bits_main mismatch"
    
    y_twin_hex_cpu = pack_y_bits_to_hex(y_twin_cpu, samples_per_channel)
    y_twin_hex_cl = pack_y_bits_to_hex(y_twin_cl, samples_per_channel)
    assert y_twin_hex_cpu == y_twin_hex_cl, f"y_bits_twin mismatch"
    
    # pc32 parity
    assert pc32_main_cpu == pc32_main_cl[:k_used_cpu].tolist(), f"pc32_main mismatch"
    assert pc32_twin_cpu == pc32_twin_cl[:k_used_cpu].tolist(), f"pc32_twin mismatch"


@pytest.mark.opencl
@pytest.mark.slow
def test_golden_parity_64_rows_3_configs():
    """Test CPU vs OpenCL parity for 64 rows × 3 configs (acceptance gate for PR-1.3)."""
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")
    
    K = 2048
    M = 64
    k_max = 32
    eps = 0.05
    delta = 0.001
    
    configs = [
        SDConfig(order=1, walsh_N=2, antithetic=True),
        SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True),
        SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=1, antithetic=False),
    ]
    
    for cfg_idx, cfg in enumerate(configs):
        run_id = str(uuid.uuid4())
        layer, token = 0, cfg_idx
        
        # Generate data
        rng = np.random.default_rng(100 + cfg_idx)
        W_rows = [_rand_pm1(K, int(rng.integers(0, 1<<31))) for _ in range(M)]
        X_vec = _rand_pm1(K, int(rng.integers(0, 1<<31)))
        W_bits = np.stack([_pack_pm1(w) for w in W_rows], axis=0)
        X_bits = _pack_pm1(X_vec)[None, :]
        
        # Derive seeds
        run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
        seeds = [derive_seed(layer, row, token, run_id_int) for row in range(M)]
        
        # CPU (per-row)
        cpu_results = []
        for row in range(M):
            est, diags = bsdm_w_dot(W_bits[row], X_bits[0], k_max, cfg, seed=int(seeds[row]), want_pc32=False)
            cpu_results.append({
                "Y_mean": est,
                "T_eff": diags["k_used"],
                "y_bits_main": diags["y_bits_main"],
                "y_bits_twin": diags.get("y_bits_twin"),
            })
        
        # OpenCL (batch)
        # Note: Current kernel uses single prf_seed; for parity we run row-by-row
        opencl_results = []
        for row in range(M):
            W_single = W_bits[row:row+1]
            X_ticks = np.tile(X_bits, (k_max, 1))
            out_cl = gemm.run_bsdm_w_naive_norm(
                W_single, X_ticks, T=k_max, eps=eps, delta=delta,
                order=cfg.order, beta=cfg.beta, lambd=cfg.lambd,
                walsh_N=cfg.walsh_N, antithetic=cfg.antithetic,
                use_ctg=False, prf_seed=int(seeds[row]),
                local_size=256, want_y_pack=True, want_pc32=False
            )
            opencl_results.append({
                "Y_mean": float(out_cl["Y"][0]),
                "T_eff": int(out_cl["T_eff"][0]),
                "y_bits_main": out_cl["y_bits_main"][0],
                "y_bits_twin": out_cl.get("y_bits_twin")[0] if out_cl.get("y_bits_twin") is not None else None,
            })
        
        # Parity checks
        for row in range(M):
            cpu_r = cpu_results[row]
            cl_r = opencl_results[row]
            
            assert cpu_r["T_eff"] == cl_r["T_eff"], f"Config {cfg_idx}, row {row}: T_eff mismatch"
            assert np.allclose(cpu_r["Y_mean"], cl_r["Y_mean"], atol=1e-6, rtol=0.0), \
                f"Config {cfg_idx}, row {row}: Y_mean mismatch"
            
            samples_per_channel = cpu_r["T_eff"] * cfg.walsh_N
            y_main_hex_cpu = pack_y_bits_to_hex(cpu_r["y_bits_main"], samples_per_channel)
            y_main_hex_cl = pack_y_bits_to_hex(cl_r["y_bits_main"], samples_per_channel)
            assert y_main_hex_cpu == y_main_hex_cl, f"Config {cfg_idx}, row {row}: y_bits_main mismatch"
            
            if cpu_r["y_bits_twin"] is not None and cl_r["y_bits_twin"] is not None:
                y_twin_hex_cpu = pack_y_bits_to_hex(cpu_r["y_bits_twin"], samples_per_channel)
                y_twin_hex_cl = pack_y_bits_to_hex(cl_r["y_bits_twin"], samples_per_channel)
                assert y_twin_hex_cpu == y_twin_hex_cl, f"Config {cfg_idx}, row {row}: y_bits_twin mismatch"


@pytest.mark.opencl
def test_golden_parity_walsh_N1_no_antithetic():
    """Test parity with Walsh N=1, no antithetic (wiring test)."""
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")
    
    K = 1024
    M = 4
    k_max = 16
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=1, antithetic=False)
    eps = 0.05
    delta = 0.001
    run_id = str(uuid.uuid4())
    layer, token = 0, 0
    
    # Generate data
    rng = np.random.default_rng(77)
    W_rows = [_rand_pm1(K, int(rng.integers(0, 1<<31))) for _ in range(M)]
    X_vec = _rand_pm1(K, int(rng.integers(0, 1<<31)))
    W_bits = np.stack([_pack_pm1(w) for w in W_rows], axis=0)
    X_bits = _pack_pm1(X_vec)[None, :]
    
    # Derive seeds
    run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
    seeds = [derive_seed(layer, row, token, run_id_int) for row in range(M)]
    
    # CPU
    cpu_results = []
    for row in range(M):
        est, diags = bsdm_w_dot(W_bits[row], X_bits[0], k_max, cfg, seed=int(seeds[row]), want_pc32=False)
        cpu_results.append({
            "Y_mean": est,
            "T_eff": diags["k_used"],
            "y_bits_main": diags["y_bits_main"],
        })
    
    # OpenCL
    opencl_results = []
    for row in range(M):
        W_single = W_bits[row:row+1]
        X_ticks = np.tile(X_bits, (k_max, 1))
        out_cl = gemm.run_bsdm_w_naive_norm(
            W_single, X_ticks, T=k_max, eps=eps, delta=delta,
            order=cfg.order, beta=cfg.beta, lambd=cfg.lambd,
            walsh_N=cfg.walsh_N, antithetic=cfg.antithetic,
            use_ctg=False, prf_seed=int(seeds[row]),
            local_size=256, want_y_pack=True, want_pc32=False
        )
        opencl_results.append({
            "Y_mean": float(out_cl["Y"][0]),
            "T_eff": int(out_cl["T_eff"][0]),
            "y_bits_main": out_cl["y_bits_main"][0],
        })
    
    # Parity checks
    for row in range(M):
        cpu_r = cpu_results[row]
        cl_r = opencl_results[row]
        
        assert cpu_r["T_eff"] == cl_r["T_eff"], f"Row {row}: T_eff mismatch"
        assert np.allclose(cpu_r["Y_mean"], cl_r["Y_mean"], atol=1e-6, rtol=0.0), f"Row {row}: Y_mean mismatch"
        
        samples_per_channel = cpu_r["T_eff"] * cfg.walsh_N
        y_main_hex_cpu = pack_y_bits_to_hex(cpu_r["y_bits_main"], samples_per_channel)
        y_main_hex_cl = pack_y_bits_to_hex(cl_r["y_bits_main"], samples_per_channel)
        assert y_main_hex_cpu == y_main_hex_cl, f"Row {row}: y_bits_main mismatch"

