"""Tests for end-to-end controller."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.runtime.controller_e2e import infer_one_token_e2e, E2EConfig
from onebit.core.packbits import pack_input_signs


def test_e2e_smoke():
    """Smoke test for end-to-end inference."""
    np.random.seed(42)
    
    n_ctx = 32
    vocab_size = 64
    d_attn = 128
    d_kv = 128
    d_model = 128
    
    # Create synthetic data
    Q_attn = np.random.randn(d_attn)
    K_attn = np.random.randn(n_ctx, d_attn)
    
    # Make position 5 highly correlated
    K_attn[5] = Q_attn + 0.05 * np.random.randn(d_attn)
    
    Q_attn_bits = pack_input_signs(Q_attn)
    K_attn_bits = np.array([pack_input_signs(K_attn[i]) for i in range(n_ctx)])
    
    K_kv = np.random.randn(n_ctx, d_kv)
    V_kv = np.random.randn(n_ctx, d_kv)
    
    K_kv_bits = np.array([pack_input_signs(K_kv[i]) for i in range(n_ctx)])
    V_kv_bits = np.array([pack_input_signs(V_kv[i]) for i in range(n_ctx)])
    
    Q_logits = np.random.randn(d_model)
    Q_logits_bits = pack_input_signs(Q_logits)
    
    vocab_ids = np.arange(vocab_size, dtype=np.int32)
    
    # Config
    cfg = E2EConfig(
        kA=8,
        k_max_attn=32,
        d_kv=d_kv,
        k_kv_stage1=8,
        k_kv_stage2=8,
        top_k_kv=4,
        k0_logits=4,
        k_step_logits=4,
        k_max_logits=32,
        shortlist_size=16,
        backend="cpu",
    )
    
    # Run
    result = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits,
        K_kv_bits, V_kv_bits,
        Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=12345,
        d_attn=d_attn,
        d_model=d_model,
    )
    
    # Check structure
    assert "status" in result, "Result should have status"
    assert "attn_top1" in result, "Result should have attn_top1"
    assert "kv_positions" in result, "Result should have kv_positions"
    assert "logits_top1" in result, "Result should have logits_top1"
    assert "k_attn_used" in result, "Result should have k_attn_used"
    assert "k_kv_used" in result, "Result should have k_kv_used"
    assert "k_logits_used" in result, "Result should have k_logits_used"
    assert "unsure" in result, "Result should have unsure"
    
    # Check values
    assert result["k_attn_used"] > 0, "Should use some attention ticks"
    
    if not result["unsure"]:
        assert result["attn_top1"] is not None, "attn_top1 should be set if not unsure"
        assert len(result["kv_positions"]) == cfg.top_k_kv, "Should retrieve top_k_kv positions"
        assert result["k_kv_used"] > 0, "Should use some KV ticks"
        assert result["k_logits_used"] > 0, "Should use some logits ticks"
    
    print(f"\nE2E result: {result['status']}")
    print(f"  attn_top1={result['attn_top1']}, k_attn={result['k_attn_used']}")
    print(f"  kv_positions={result['kv_positions']}, k_kv={result['k_kv_used']}")
    print(f"  logits_top1={result['logits_top1']}, k_logits={result['k_logits_used']}")


def test_e2e_determinism():
    """Test that end-to-end inference is deterministic."""
    np.random.seed(777)
    
    n_ctx = 16
    vocab_size = 32
    d_attn = 64
    d_kv = 64
    d_model = 64
    
    # Create synthetic data
    Q_attn = np.random.randn(d_attn)
    K_attn = np.random.randn(n_ctx, d_attn)
    
    Q_attn_bits = pack_input_signs(Q_attn)
    K_attn_bits = np.array([pack_input_signs(K_attn[i]) for i in range(n_ctx)])
    
    K_kv = np.random.randn(n_ctx, d_kv)
    V_kv = np.random.randn(n_ctx, d_kv)
    
    K_kv_bits = np.array([pack_input_signs(K_kv[i]) for i in range(n_ctx)])
    V_kv_bits = np.array([pack_input_signs(V_kv[i]) for i in range(n_ctx)])
    
    Q_logits = np.random.randn(d_model)
    Q_logits_bits = pack_input_signs(Q_logits)
    
    vocab_ids = np.arange(vocab_size, dtype=np.int32)
    
    # Config
    cfg = E2EConfig(
        kA=8,
        k_max_attn=32,
        d_kv=d_kv,
        backend="cpu",
    )
    
    seed = 99999
    
    # Run twice
    result1 = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits,
        K_kv_bits, V_kv_bits,
        Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=seed,
        d_attn=d_attn,
        d_model=d_model,
    )
    
    result2 = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits,
        K_kv_bits, V_kv_bits,
        Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=seed,
        d_attn=d_attn,
        d_model=d_model,
    )
    
    # Should be identical
    assert result1["status"] == result2["status"], "Status should be deterministic"
    assert result1["attn_top1"] == result2["attn_top1"], "attn_top1 should be deterministic"
    assert result1["logits_top1"] == result2["logits_top1"], "logits_top1 should be deterministic"
    assert result1["k_attn_used"] == result2["k_attn_used"], "k_attn_used should be deterministic"
    assert result1["k_kv_used"] == result2["k_kv_used"], "k_kv_used should be deterministic"
    assert result1["k_logits_used"] == result2["k_logits_used"], "k_logits_used should be deterministic"
    
    np.testing.assert_array_equal(result1["kv_positions"], result2["kv_positions"], 
                                   err_msg="kv_positions should be deterministic")


def test_e2e_no_regressions():
    """Test that end-to-end doesn't break existing functionality."""
    np.random.seed(123)
    
    n_ctx = 8
    vocab_size = 16
    d_attn = 32
    d_kv = 32
    d_model = 32
    
    # Create synthetic data
    Q_attn = np.random.randn(d_attn)
    K_attn = np.random.randn(n_ctx, d_attn)
    
    Q_attn_bits = pack_input_signs(Q_attn)
    K_attn_bits = np.array([pack_input_signs(K_attn[i]) for i in range(n_ctx)])
    
    K_kv = np.random.randn(n_ctx, d_kv)
    V_kv = np.random.randn(n_ctx, d_kv)
    
    K_kv_bits = np.array([pack_input_signs(K_kv[i]) for i in range(n_ctx)])
    V_kv_bits = np.array([pack_input_signs(V_kv[i]) for i in range(n_ctx)])
    
    Q_logits = np.random.randn(d_model)
    Q_logits_bits = pack_input_signs(Q_logits)
    
    vocab_ids = np.arange(vocab_size, dtype=np.int32)
    
    # Config
    cfg = E2EConfig(
        kA=4,
        k_max_attn=16,
        d_kv=d_kv,
        backend="cpu",
    )
    
    # Run - should not crash
    result = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits,
        K_kv_bits, V_kv_bits,
        Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=55555,
        d_attn=d_attn,
        d_model=d_model,
    )
    
    # Just check it completes
    assert result is not None, "Should return a result"
    assert isinstance(result, dict), "Result should be a dict"

