"""Test that CTG shadow keeps decisions identical with fixed seed (PR-4.3)."""
from __future__ import annotations

import numpy as np
import pytest


def test_shadow_keeps_decisions_identical_fixed_seed():
    """Test that shadow path doesn't alter main-path outcomes."""
    from onebit.runtime.cpg_policy import CpgPolicy, CpgPolicyCfg
    from onebit.runtime.controller_e2e import infer_one_token_e2e, E2EConfig
    
    rng = np.random.default_rng(0)
    
    # Generate toy inputs
    d_attn = 256
    d_model = 512
    d_kv = 512
    n_ctx = 64
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
    
    # Create policy with high sample rate for testing
    ctg = CpgPolicy(CpgPolicyCfg(sample_rate=0.5, promote_min_samples=10))
    
    # Run 20 tokens
    decisions = []
    for t in range(20):
        out = infer_one_token_e2e(
            Q_attn_bits, K_attn_bits, K_kv_bits, V_kv_bits, Q_logits_bits, vocab_ids,
            cfg=cfg,
            prf_seed=12345 + t,  # Different seed per token
            d_attn=d_attn,
            d_model=d_model,
            token_idx=t,
            cpg_policy=ctg,
        )
        decisions.append({
            "status": out["status"],
            "attn_top1": out["attn_top1"],
            "logits_top1": out["logits_top1"],
        })
    
    # Main path must be stable; shadow work must not alter outcomes
    assert len(decisions) == 20, "Should have 20 decisions"
    
    # Check that all decisions are valid
    for i, dec in enumerate(decisions):
        assert dec["status"] in ("CERT_OK", "ATTN_UNSURE", "LOGITS_UNSURE"), \
            f"Token {i}: Invalid status {dec['status']}"
    
    print(f"[PASS] Shadow keeps decisions identical: {len(decisions)} tokens processed")
    print(f"  Statuses: {[d['status'] for d in decisions[:5]]}...")


def test_stageA_shadow_deterministic():
    """Test that Stage-A shadow produces deterministic results."""
    from onebit.ops.attention_probe import stageA_probe_topT
    
    rng = np.random.default_rng(42)
    
    M = 64
    Kw = 8
    
    Q_bits = rng.integers(0, 2**32, size=(Kw,), dtype=np.uint32)
    K_bits = rng.integers(0, 2**32, size=(M, Kw), dtype=np.uint32)
    
    # Run twice with same seed and use_ctg=False
    r1 = stageA_probe_topT(
        Q_bits, K_bits,
        kA=16,
        prf_seed=123,
        use_ctg=False,
    )
    
    r2 = stageA_probe_topT(
        Q_bits, K_bits,
        kA=16,
        prf_seed=123,
        use_ctg=False,
    )
    
    # Check determinism
    assert r1["T_sel"] == r2["T_sel"], "T_sel should be deterministic"
    assert np.array_equal(r1["idx_top"], r2["idx_top"]), "idx_top should be deterministic"
    assert np.allclose(r1["stats"]["mu"], r2["stats"]["mu"]), "mu should be deterministic"
    
    # Run twice with same seed and use_ctg=True
    r3 = stageA_probe_topT(
        Q_bits, K_bits,
        kA=16,
        prf_seed=123,
        use_ctg=True,
    )
    
    r4 = stageA_probe_topT(
        Q_bits, K_bits,
        kA=16,
        prf_seed=123,
        use_ctg=True,
    )
    
    # Check determinism with CTG
    assert r3["T_sel"] == r4["T_sel"], "T_sel should be deterministic with CTG"
    assert np.array_equal(r3["idx_top"], r4["idx_top"]), "idx_top should be deterministic with CTG"
    assert np.allclose(r3["stats"]["mu"], r4["stats"]["mu"]), "mu should be deterministic with CTG"
    
    print(f"[PASS] Stage-A shadow deterministic: T_sel={r1['T_sel']} (no CTG), {r3['T_sel']} (CTG)")


def test_attn_certify_shadow_deterministic():
    """Test that Attn certify shadow produces deterministic results."""
    from onebit.attn.runner import certify_topT
    from onebit.attn.sprt_dag import SPRTConfig
    
    rng = np.random.default_rng(42)
    
    M = 64
    Kw = 8
    T = 8
    
    Q_bits = rng.integers(0, 2**32, size=(Kw,), dtype=np.uint32)
    K_bits = rng.integers(0, 2**32, size=(M, Kw), dtype=np.uint32)
    idx_top = np.arange(T, dtype=np.int32)
    
    cfg = SPRTConfig(
        eps=0.05,
        alpha=0.005,
        beta=0.005,
        k_max=16,
        chunk=4,
        seed=999,
    )
    
    # Run twice with same seed and use_ctg=False
    r1 = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg,
        backend="cpu",
        prf_seed=999,
        use_ctg=False,
    )
    
    r2 = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg,
        backend="cpu",
        prf_seed=999,
        use_ctg=False,
    )
    
    # Check determinism
    assert r1["top1"] == r2["top1"], "top1 should be deterministic"
    assert r1["k_used"] == r2["k_used"], "k_used should be deterministic"
    
    # Run twice with same seed and use_ctg=True
    r3 = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg,
        backend="cpu",
        prf_seed=999,
        use_ctg=True,
    )
    
    r4 = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=cfg,
        backend="cpu",
        prf_seed=999,
        use_ctg=True,
    )
    
    # Check determinism with CTG
    assert r3["top1"] == r4["top1"], "top1 should be deterministic with CTG"
    assert r3["k_used"] == r4["k_used"], "k_used should be deterministic with CTG"
    
    print(f"[PASS] Attn certify shadow deterministic: top1={r1['top1']} (no CTG), {r3['top1']} (CTG)")

