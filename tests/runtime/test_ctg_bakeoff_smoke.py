"""Smoke test for CTG bake-off infrastructure."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.runtime.ctg_grammar import CTG, CTGRule, CTGState, make_default_programs
from onebit.ops.logits_sprt import shortlist_and_certify
from onebit.core.packbits import pack_input_signs


def test_bakeoff_smoke_ctg_fixed_vs_prog():
    """Smoke test comparing CTG-FIXED vs CTG-PROG on same input."""
    rng = np.random.default_rng(42)
    
    d = 256
    vocab_size = 128
    d_words = d // 32
    
    # Generate test data
    q_bits = rng.integers(0, 2**32 - 1, size=(d_words,), dtype=np.uint32)
    v_ids = np.arange(vocab_size, dtype=np.int32)
    
    kwargs = dict(
        d=d,
        k0=8,
        k_step=4,
        k_max=32,
        shortlist_size=16,
        eps=0.05,
        delta=0.01,
        backend="cpu",
        prf_seed=12345,
        use_ctg=0,
    )
    
    # Baseline (no CTG)
    baseline = shortlist_and_certify(q_bits, v_ids, **kwargs)
    
    # CTG-FIXED
    rules_fixed = [CTGRule(op="PASS", ids=None)]
    ctg_fixed = CTG(rules=rules_fixed, vocab_size=vocab_size)
    state_fixed = CTGState()
    
    result_fixed = shortlist_and_certify(
        q_bits, v_ids,
        ctg=ctg_fixed,
        ctg_state=state_fixed,
        ctg_program_id=0,
        **kwargs
    )
    
    # CTG-PROG with program 0 (PASS-biased)
    programs = make_default_programs(vocab_size, K=4)
    ctg_prog = CTG(programs=programs, vocab_size=vocab_size)
    state_prog = CTGState()
    
    result_prog = shortlist_and_certify(
        q_bits, v_ids,
        ctg=ctg_prog,
        ctg_state=state_prog,
        ctg_program_id=0,
        **kwargs
    )
    
    # Verify all runs completed
    assert "k_used" in baseline
    assert "k_used" in result_fixed
    assert "k_used" in result_prog
    
    # Verify CTG telemetry is present
    assert "ctg_prog_id" in result_fixed
    assert "ctg_prog_id" in result_prog
    assert result_fixed["ctg_prog_id"] == 0
    assert result_prog["ctg_prog_id"] == 0
    
    # Verify work is reasonable (CTG should not increase work dramatically)
    baseline_pairs = baseline.get("pairs_evaluated", 0)
    fixed_pairs = result_fixed.get("pairs_evaluated", 0)
    prog_pairs = result_prog.get("pairs_evaluated", 0)
    
    # Allow some variance but ensure CTG doesn't explode work
    assert fixed_pairs <= baseline_pairs * 1.5, f"CTG-FIXED increased work too much: {fixed_pairs} vs {baseline_pairs}"
    assert prog_pairs <= baseline_pairs * 1.5, f"CTG-PROG increased work too much: {prog_pairs} vs {baseline_pairs}"


def test_bakeoff_different_programs():
    """Test that different program IDs produce different behavior."""
    rng = np.random.default_rng(123)
    
    d = 256
    vocab_size = 128
    d_words = d // 32
    
    q_bits = rng.integers(0, 2**32 - 1, size=(d_words,), dtype=np.uint32)
    v_ids = np.arange(vocab_size, dtype=np.int32)
    
    kwargs = dict(
        d=d,
        k0=8,
        k_step=4,
        k_max=32,
        shortlist_size=16,
        eps=0.05,
        delta=0.01,
        backend="cpu",
        prf_seed=12345,
        use_ctg=0,
    )
    
    programs = make_default_programs(vocab_size, K=4)
    ctg = CTG(programs=programs, vocab_size=vocab_size)
    
    results = []
    for prog_id in range(4):
        state = CTGState()
        result = shortlist_and_certify(
            q_bits, v_ids,
            ctg=ctg,
            ctg_state=state,
            ctg_program_id=prog_id,
            **kwargs
        )
        results.append(result)
    
    # Verify all programs ran
    for i, r in enumerate(results):
        assert r["ctg_prog_id"] == i
    
    # Verify at least some variation in outcomes
    # (This is probabilistic but should hold with high probability)
    pairs_list = [r.get("pairs_evaluated", 0) for r in results]
    assert len(set(pairs_list)) > 1 or all(p == 0 for p in pairs_list), \
        "All programs produced identical results (unlikely unless all PASS)"


def test_bakeoff_metrics_structure():
    """Test that bake-off metrics have expected structure."""
    from onebit.cli.bench_bakeoff_ctg import BakeoffMetrics
    
    # Create a sample metrics object
    metrics = BakeoffMetrics(
        config_name="test",
        ppl=10.5,
        ppl_delta_pct=5.0,
        k_mean=20.0,
        k_p95=30.0,
        pairs_mean=100.0,
        pairs_delta_pct=10.0,
        latency_mean_ms=5.0,
        latency_p95_ms=8.0,
        latency_delta_pct=2.0,
        variance_k=5.0,
        variance_ratio=1.2,
        tokens_processed=100,
        unsure_rate=0.05,
    )
    
    # Verify all fields are accessible
    assert metrics.config_name == "test"
    assert metrics.ppl == 10.5
    assert metrics.k_mean == 20.0
    assert metrics.tokens_processed == 100

