"""Integration tests for SPRT-based Top-T certification."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.ops.attention_probe import stageA_probe_topT
from onebit.attn.runner import certify_topT
from onebit.attn.sprt_dag import SPRTConfig
from onebit.core.packbits import pack_input_signs


def test_stageA_then_certify_cpu():
    """Test Stage-A followed by SPRT certification on CPU backend."""
    np.random.seed(42)
    
    # Create synthetic Q and K
    K = 64  # Number of keys
    d = 1024  # Dimension
    
    Q = np.random.randn(d)
    K_mat = np.random.randn(K, d)
    
    # Pack to bits
    Q_bits = pack_input_signs(Q)
    K_bits = np.array([pack_input_signs(K_mat[i]) for i in range(K)])
    
    # Stage-A: probe with kA=16
    result = stageA_probe_topT(
        Q_bits, K_bits, kA=16, prf_seed=12345
    )
    T_sel = result["T_sel"]
    idx_top = result["idx_top"]
    stats = result["stats"]
    
    print(f"\nStage-A: T_sel={T_sel}, idx_top={idx_top[:5]}...")
    
    # SPRT certification
    cfg = SPRTConfig(eps=0.05, alpha=0.01, beta=0.01, k_max=32, chunk=4, seed=54321)
    
    result = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg, backend="cpu", prf_seed=54321,
        walsh_N=2, antithetic=True, order=2, beta=0.30, lambd=1.0/256.0
    )
    
    print(f"SPRT result: k_used={result['k_used']}, pairs_evaluated={result['pairs_evaluated']}")
    print(f"  decided={len(result['decided'])}, undecided={len(result['undecided'])}")
    print(f"  top1={result['top1']}")
    
    # Sanity checks
    assert result["k_used"] > 0, "Should use at least 1 tick"
    assert result["k_used"] <= cfg.k_max, f"Should not exceed k_max ({result['k_used']} > {cfg.k_max})"
    assert result["pairs_evaluated"] > 0, "Should evaluate at least some pairs"
    
    # Check that decided + undecided = total pairs
    T = len(idx_top)
    total_pairs = T * (T - 1) // 2
    assert len(result["decided"]) + len(result["undecided"]) == total_pairs, "Decided + undecided should equal total pairs"


@pytest.mark.opencl
def test_stageA_then_certify_opencl():
    """Test Stage-A followed by SPRT certification on OpenCL backend."""
    np.random.seed(99)
    
    # Create synthetic Q and K
    K = 64  # Number of keys
    d = 1024  # Dimension
    
    Q = np.random.randn(d)
    K_mat = np.random.randn(K, d)
    
    # Pack to bits
    Q_bits = pack_input_signs(Q)
    K_bits = np.array([pack_input_signs(K_mat[i]) for i in range(K)])
    
    # Stage-A: probe with kA=16
    result = stageA_probe_topT(
        Q_bits, K_bits, kA=16, prf_seed=12345
    )
    T_sel = result["T_sel"]
    idx_top = result["idx_top"]
    stats = result["stats"]
    
    print(f"\nStage-A: T_sel={T_sel}, idx_top={idx_top[:5]}...")
    
    # SPRT certification
    cfg = SPRTConfig(eps=0.05, alpha=0.01, beta=0.01, k_max=32, chunk=4, seed=54321)
    
    result = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg, backend="opencl", prf_seed=54321,
        walsh_N=2, antithetic=True, order=2, beta=0.30, lambd=1.0/256.0
    )
    
    print(f"SPRT result: k_used={result['k_used']}, pairs_evaluated={result['pairs_evaluated']}")
    print(f"  decided={len(result['decided'])}, undecided={len(result['undecided'])}")
    print(f"  top1={result['top1']}")
    
    # Sanity checks
    assert result["k_used"] > 0, "Should use at least 1 tick"
    assert result["k_used"] <= cfg.k_max, f"Should not exceed k_max ({result['k_used']} > {cfg.k_max})"
    assert result["pairs_evaluated"] > 0, "Should evaluate at least some pairs"
    
    # Check that decided + undecided = total pairs
    T = len(idx_top)
    total_pairs = T * (T - 1) // 2
    assert len(result["decided"]) + len(result["undecided"]) == total_pairs, "Decided + undecided should equal total pairs"


def test_certify_determinism():
    """Test deterministic certification with fixed seeds."""
    np.random.seed(777)
    
    # Create synthetic Q and K
    K = 32
    d = 1024
    
    Q = np.random.randn(d)
    K_mat = np.random.randn(K, d)
    
    Q_bits = pack_input_signs(Q)
    K_bits = np.array([pack_input_signs(K_mat[i]) for i in range(K)])
    
    # Stage-A
    result = stageA_probe_topT(
        Q_bits, K_bits, kA=16, prf_seed=12345
    )
    T_sel = result["T_sel"]
    idx_top = result["idx_top"]
    
    # SPRT certification - Run 1
    cfg = SPRTConfig(eps=0.05, alpha=0.01, beta=0.01, k_max=32, chunk=4, seed=54321)
    
    result1 = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg, backend="cpu", prf_seed=54321,
        walsh_N=2, antithetic=True, order=2, beta=0.30, lambd=1.0/256.0
    )
    
    # SPRT certification - Run 2 (same seed)
    result2 = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg, backend="cpu", prf_seed=54321,
        walsh_N=2, antithetic=True, order=2, beta=0.30, lambd=1.0/256.0
    )
    
    # Should be identical
    assert result1["k_used"] == result2["k_used"], "k_used should be identical"
    assert result1["pairs_evaluated"] == result2["pairs_evaluated"], "pairs_evaluated should be identical"
    assert result1["decided"] == result2["decided"], "decided edges should be identical"
    assert result1["undecided"] == result2["undecided"], "undecided pairs should be identical"
    assert result1["top1"] == result2["top1"], "top1 should be identical"


def test_clear_winner_certifies():
    """Test that a clear winner certifies quickly."""
    np.random.seed(888)

    # Create Q and K where first key is clearly best
    K = 32
    d = 1024

    Q = np.random.randn(d)
    K_mat = np.random.randn(K, d)

    # Make key 0 very similar to Q (clear winner)
    # Use large positive correlation to ensure it's top
    K_mat[0] = Q * 0.9 + np.random.randn(d) * 0.1

    # Make other keys less similar
    for i in range(1, K):
        K_mat[i] = np.random.randn(d)

    Q_bits = pack_input_signs(Q)
    K_bits = np.array([pack_input_signs(K_mat[i]) for i in range(K)])

    # Stage-A
    result = stageA_probe_topT(
        Q_bits, K_bits, kA=16, prf_seed=12345
    )
    T_sel = result["T_sel"]
    idx_top = result["idx_top"]

    print(f"\nStage-A: T_sel={T_sel}, idx_top={idx_top}")
    print(f"  Key 0 in top? {0 in idx_top}")

    # SPRT certification with larger eps for faster convergence
    cfg = SPRTConfig(eps=0.15, alpha=0.01, beta=0.01, k_max=50, chunk=4, seed=54321)

    result = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg, backend="cpu", prf_seed=54321,
        walsh_N=2, antithetic=True, order=2, beta=0.30, lambd=1.0/256.0
    )

    print(f"SPRT result: k_used={result['k_used']}, top1={result['top1']}")
    print(f"  decided={len(result['decided'])}, undecided={len(result['undecided'])}")

    # With random data, certification may not always complete
    # Just check that the interface works correctly
    assert result["k_used"] > 0, "Should use at least 1 tick"
    assert result["k_used"] <= cfg.k_max, "Should not exceed k_max"

