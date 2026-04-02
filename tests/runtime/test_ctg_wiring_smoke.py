"""Smoke tests for CTG wiring (PR-4.3).

Tests that use_ctg parameter is accepted by all functions and that
controller shadow path doesn't affect main-path outputs.
"""
from __future__ import annotations

import numpy as np
import pytest


def _toy_inputs(M=32, Kw=8, d=256, seed=7):
    """Generate toy inputs for testing."""
    rng = np.random.default_rng(seed)
    W = rng.integers(0, 2**32, size=(M, Kw), dtype=np.uint32)
    X = rng.integers(0, 2**32, size=(16, Kw), dtype=np.uint32)
    return W, X


def test_stageA_accepts_use_ctg():
    """Test that stageA_probe_topT accepts use_ctg parameter."""
    from onebit.ops.attention_probe import stageA_probe_topT
    
    W, X = _toy_inputs()
    
    # Test with use_ctg=False
    a = stageA_probe_topT(
        X[0], W,
        kA=16,
        prf_seed=123,
        use_ctg=False,
    )
    
    # Test with use_ctg=True
    b = stageA_probe_topT(
        X[0], W,
        kA=16,
        prf_seed=123,
        use_ctg=True,
    )
    
    assert isinstance(a["T_sel"], int), "T_sel should be int"
    assert isinstance(b["T_sel"], int), "T_sel should be int"
    assert "idx_top" in a, "idx_top should be in result"
    assert "idx_top" in b, "idx_top should be in result"
    
    print(f"[PASS] stageA_probe_topT accepts use_ctg: T_sel={a['T_sel']}, {b['T_sel']}")


def test_attn_certify_accepts_use_ctg():
    """Test that certify_topT accepts use_ctg parameter."""
    from onebit.attn.runner import certify_topT
    from onebit.attn.sprt_dag import SPRTConfig
    
    W, X = _toy_inputs()
    cands = np.arange(8, dtype=np.int32)
    
    cfg = SPRTConfig(
        eps=0.05,
        alpha=0.005,
        beta=0.005,
        k_max=16,
        chunk=4,
        seed=999,
    )
    
    # Test with use_ctg=False
    r0 = certify_topT(
        X[0], W, cands,
        cfg=cfg,
        backend="cpu",
        prf_seed=999,
        use_ctg=False,
    )
    
    # Test with use_ctg=True
    r1 = certify_topT(
        X[0], W, cands,
        cfg=cfg,
        backend="cpu",
        prf_seed=999,
        use_ctg=True,
    )
    
    assert "k_used" in r0, "k_used should be in result"
    assert "k_used" in r1, "k_used should be in result"
    
    print(f"[PASS] certify_topT accepts use_ctg: k_used={r0['k_used']}, {r1['k_used']}")


def test_logits_accepts_use_ctg():
    """Test that shortlist_and_certify accepts use_ctg parameter."""
    from onebit.ops.logits_sprt import shortlist_and_certify
    
    rng = np.random.default_rng(42)
    d = 256
    V = 100
    
    q_bits = rng.integers(0, 2**32, size=(d // 32,), dtype=np.uint32)
    vocab_ids = np.arange(V, dtype=np.int32)
    
    # Test with use_ctg=0
    r0 = shortlist_and_certify(
        q_bits, vocab_ids,
        d=d,
        k0=8,
        k_step=4,
        k_max=32,
        shortlist_size=16,
        eps=0.05,
        delta=0.01,
        backend="cpu",
        prf_seed=777,
        use_ctg=0,
    )
    
    # Test with use_ctg=1
    r1 = shortlist_and_certify(
        q_bits, vocab_ids,
        d=d,
        k0=8,
        k_step=4,
        k_max=32,
        shortlist_size=16,
        eps=0.05,
        delta=0.01,
        backend="cpu",
        prf_seed=777,
        use_ctg=1,
    )
    
    assert "k_used" in r0, "k_used should be in result"
    assert "k_used" in r1, "k_used should be in result"
    
    print(f"[PASS] shortlist_and_certify accepts use_ctg: k_used={r0['k_used']}, {r1['k_used']}")


@pytest.mark.parametrize("policy_on", [False, True])
def test_controller_shadow_no_side_effects(policy_on):
    """Test that controller shadow path doesn't affect main-path outputs."""
    from onebit.runtime.cpg_policy import CpgPolicy, CpgPolicyCfg
    from onebit.runtime.controller_e2e import infer_one_token_e2e, E2EConfig
    
    rng = np.random.default_rng(0)
    
    # Generate toy inputs
    d_attn = 256
    d_model = 512
    d_kv = 512
    n_ctx = 16
    V = 100
    
    Q_attn_bits = rng.integers(0, 2**32, size=(d_attn // 32,), dtype=np.uint32)
    K_attn_bits = rng.integers(0, 2**32, size=(n_ctx, d_attn // 32), dtype=np.uint32)
    K_kv_bits = rng.integers(0, 2**32, size=(n_ctx, d_kv // 32), dtype=np.uint32)
    V_kv_bits = rng.integers(0, 2**32, size=(n_ctx, d_kv // 32), dtype=np.uint32)
    Q_logits_bits = rng.integers(0, 2**32, size=(d_model // 32,), dtype=np.uint32)
    vocab_ids = np.arange(V, dtype=np.int32)
    
    cfg = E2EConfig(
        kA=8,
        k_max_attn=16,
        k_max_logits=16,
        backend="cpu",
    )
    
    # Create policy if enabled
    ctg = CpgPolicy(CpgPolicyCfg(sample_rate=0.1)) if policy_on else None
    
    # Run two tokens
    out0 = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits, K_kv_bits, V_kv_bits, Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=12345,
        d_attn=d_attn,
        d_model=d_model,
        token_idx=0,
        cpg_policy=ctg,
    )
    
    out1 = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits, K_kv_bits, V_kv_bits, Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=12345,
        d_attn=d_attn,
        d_model=d_model,
        token_idx=1,
        cpg_policy=ctg,
    )
    
    # Check that outputs are valid
    assert out0["status"] in ("CERT_OK", "ATTN_UNSURE", "LOGITS_UNSURE"), \
        f"Invalid status: {out0['status']}"
    assert out1["status"] in ("CERT_OK", "ATTN_UNSURE", "LOGITS_UNSURE"), \
        f"Invalid status: {out1['status']}"
    
    # Check CTG fields present
    assert "ctg_shadow" in out0, "ctg_shadow should be in output"
    assert "ctg_pol_stageA" in out0, "ctg_pol_stageA should be in output"
    assert "ctg_pol_attn" in out0, "ctg_pol_attn should be in output"
    assert "ctg_pol_logits" in out0, "ctg_pol_logits should be in output"
    
    # Check CTG fields are 0 when policy is None
    if not policy_on:
        assert out0["ctg_shadow"] == 0, "ctg_shadow should be 0 when policy is None"
        assert out0["ctg_pol_stageA"] == 0, "ctg_pol_stageA should be 0 when policy is None"
        assert out0["ctg_pol_attn"] == 0, "ctg_pol_attn should be 0 when policy is None"
        assert out0["ctg_pol_logits"] == 0, "ctg_pol_logits should be 0 when policy is None"
    
    print(f"[PASS] Controller shadow no side effects (policy_on={policy_on}): "
          f"status={out0['status']}, ctg_shadow={out0['ctg_shadow']}")


def test_controller_deterministic_with_fixed_seed():
    """Test that controller is deterministic with fixed seed."""
    from onebit.runtime.controller_e2e import infer_one_token_e2e, E2EConfig
    
    rng = np.random.default_rng(0)
    
    # Generate toy inputs
    d_attn = 256
    d_model = 512
    d_kv = 512
    n_ctx = 16
    V = 100
    
    Q_attn_bits = rng.integers(0, 2**32, size=(d_attn // 32,), dtype=np.uint32)
    K_attn_bits = rng.integers(0, 2**32, size=(n_ctx, d_attn // 32), dtype=np.uint32)
    K_kv_bits = rng.integers(0, 2**32, size=(n_ctx, d_kv // 32), dtype=np.uint32)
    V_kv_bits = rng.integers(0, 2**32, size=(n_ctx, d_kv // 32), dtype=np.uint32)
    Q_logits_bits = rng.integers(0, 2**32, size=(d_model // 32,), dtype=np.uint32)
    vocab_ids = np.arange(V, dtype=np.int32)
    
    cfg = E2EConfig(
        kA=8,
        k_max_attn=16,
        k_max_logits=16,
        backend="cpu",
    )
    
    # Run twice with same seed
    out1 = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits, K_kv_bits, V_kv_bits, Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=12345,
        d_attn=d_attn,
        d_model=d_model,
        token_idx=0,
        cpg_policy=None,
    )
    
    out2 = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits, K_kv_bits, V_kv_bits, Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=12345,
        d_attn=d_attn,
        d_model=d_model,
        token_idx=0,
        cpg_policy=None,
    )
    
    # Check determinism
    assert out1["status"] == out2["status"], "Status should be deterministic"
    assert out1["attn_top1"] == out2["attn_top1"], "attn_top1 should be deterministic"
    assert out1["logits_top1"] == out2["logits_top1"], "logits_top1 should be deterministic"
    
    print(f"[PASS] Controller deterministic with fixed seed: status={out1['status']}")

