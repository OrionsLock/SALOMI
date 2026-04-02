"""Bound correctness tests for BSDM-W golden logs."""
from __future__ import annotations

import math
import uuid

import numpy as np
import pytest

from onebit.backends.opencl.host_opencl import OpenCLBinGemm
from onebit.core.packbits import pack_input_signs
from onebit.core.prf import derive_seed
from onebit.ops.bsdm_w import SDConfig, bsdm_w_dot


def _rand_pm1(K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=K, dtype=np.int8)
    return (x * 2 - 1).astype(np.int8)


def _pack_pm1(x: np.ndarray) -> np.ndarray:
    return pack_input_signs(x.astype(np.float32))


def test_bound_correctness_cpu():
    """Test Hoeffding bound correctness on CPU backend."""
    K = 2048
    M = 32
    k_max = 64
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    eps = 0.05
    delta = 0.001
    run_id = str(uuid.uuid4())
    layer, token = 0, 0
    
    # Generate random data (near-zero means to trigger early exit)
    rng = np.random.default_rng(123)
    W_rows = [_rand_pm1(K, int(rng.integers(0, 1<<31))) for _ in range(M)]
    X_vec = _rand_pm1(K, int(rng.integers(0, 1<<31)))
    W_bits = np.stack([_pack_pm1(w) for w in W_rows], axis=0)
    X_bits = _pack_pm1(X_vec)[None, :]
    
    # Derive seeds
    run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
    seeds = [derive_seed(layer, row, token, run_id_int) for row in range(M)]
    
    violations = 0
    for row in range(M):
        est, diags = bsdm_w_dot(W_bits[row], X_bits[0], k_max, cfg, seed=int(seeds[row]), want_pc32=False, eps=eps, delta=delta)
        k_used = diags["k_used"]

        # Empirical bound check: |Y_mean| <= eps + sqrt(log(2/delta)/(2*T_eff)) + tolerance
        # The early-exit logic ensures this bound is satisfied
        thr = math.sqrt(math.log(2.0 / delta) / (2.0 * k_used))
        if abs(est) > eps + thr + 1e-5:
            violations += 1

    assert violations == 0, f"Bound violations: {violations}/{M}"


@pytest.mark.opencl
def test_bound_correctness_opencl():
    """Test Hoeffding bound correctness on OpenCL backend."""
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")
    
    K = 2048
    M = 32
    k_max = 64
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    eps = 0.05
    delta = 0.001
    run_id = str(uuid.uuid4())
    layer, token = 0, 0
    
    # Generate random data
    rng = np.random.default_rng(456)
    W_rows = [_rand_pm1(K, int(rng.integers(0, 1<<31))) for _ in range(M)]
    X_vec = _rand_pm1(K, int(rng.integers(0, 1<<31)))
    W_bits = np.stack([_pack_pm1(w) for w in W_rows], axis=0)
    X_bits = _pack_pm1(X_vec)[None, :]
    
    # Derive seeds
    run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
    seeds = [derive_seed(layer, row, token, run_id_int) for row in range(M)]
    
    violations = 0
    for row in range(M):
        W_single = W_bits[row:row+1]
        X_ticks = np.tile(X_bits, (k_max, 1))
        out_cl = gemm.run_bsdm_w_naive_norm(
            W_single, X_ticks, T=k_max, eps=eps, delta=delta,
            order=cfg.order, beta=cfg.beta, lambd=cfg.lambd,
            walsh_N=cfg.walsh_N, antithetic=cfg.antithetic,
            use_ctg=False, prf_seed=int(seeds[row]),
            local_size=256, want_y_pack=False, want_pc32=False
        )
        est = float(out_cl["Y"][0])
        k_used = int(out_cl["T_eff"][0])

        # Empirical bound check
        thr = math.sqrt(math.log(2.0 / delta) / (2.0 * k_used))
        if abs(est) > eps + thr + 1e-5:
            violations += 1
    
    assert violations == 0, f"Bound violations: {violations}/{M}"


def test_bound_never_violated_cpu_stress():
    """Stress test: run 1000 rows, verify bound never violated."""
    K = 1024
    M = 1000
    k_max = 32
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    eps = 0.05
    delta = 0.001
    run_id = str(uuid.uuid4())
    layer, token = 0, 0
    
    # Generate random data
    rng = np.random.default_rng(789)
    run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
    
    violations = 0
    for row in range(M):
        W_row = _rand_pm1(K, int(rng.integers(0, 1<<31)))
        X_vec = _rand_pm1(K, int(rng.integers(0, 1<<31)))
        W_bits = _pack_pm1(W_row)
        X_bits = _pack_pm1(X_vec)
        
        seed = derive_seed(layer, row, token, run_id_int)
        est, diags = bsdm_w_dot(W_bits, X_bits, k_max, cfg, seed=int(seed), want_pc32=False, eps=eps, delta=delta)
        k_used = diags["k_used"]

        # Empirical bound check
        thr = math.sqrt(math.log(2.0 / delta) / (2.0 * k_used))
        if abs(est) > eps + thr + 1e-5:
            violations += 1
    
    assert violations == 0, f"Bound violations: {violations}/{M}"


def test_early_exit_reduces_T_eff():
    """Test that early-exit actually reduces T_eff on near-zero means."""
    K = 2048
    M = 64
    k_max = 64
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    eps = 0.05
    delta = 0.001
    run_id = str(uuid.uuid4())
    layer, token = 0, 0
    
    # Generate random data (near-zero means)
    rng = np.random.default_rng(999)
    W_rows = [_rand_pm1(K, int(rng.integers(0, 1<<31))) for _ in range(M)]
    X_vec = _rand_pm1(K, int(rng.integers(0, 1<<31)))
    W_bits = np.stack([_pack_pm1(w) for w in W_rows], axis=0)
    X_bits = _pack_pm1(X_vec)[None, :]
    
    # Derive seeds
    run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
    seeds = [derive_seed(layer, row, token, run_id_int) for row in range(M)]
    
    T_eff_values = []
    for row in range(M):
        est, diags = bsdm_w_dot(W_bits[row], X_bits[0], k_max, cfg, seed=int(seeds[row]), want_pc32=False, eps=eps, delta=delta)
        T_eff_values.append(diags["k_used"])
    
    # At least some rows should early-exit (T_eff < k_max)
    early_exits = sum(1 for t in T_eff_values if t < k_max)
    assert early_exits > 0, f"No early exits observed (all T_eff == k_max)"
    
    # Average T_eff should be significantly less than k_max
    avg_T_eff = np.mean(T_eff_values)
    assert avg_T_eff < k_max * 0.8, f"Average T_eff={avg_T_eff} not significantly less than k_max={k_max}"

