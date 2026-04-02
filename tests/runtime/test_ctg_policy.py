"""Tests for CTG policy (PR-4.3).

Goal: Verify deterministic shadow schedule, promotion/demotion logic.

Tests:
- test_shadow_schedule_deterministic(): Same seed → identical mask
- test_promote_after_clean_agreement(): 300 clean samples → enabled
- test_demote_on_disagreement(): After enabled, disagreement → disabled
- test_demote_on_overhead(): After enabled, high overhead → disabled
- test_demote_on_k_used_increase(): After enabled, k_used increase → disabled
"""
from __future__ import annotations

import numpy as np
import pytest

from onebit.runtime.cpg_policy import CpgPolicy, CpgPolicyCfg, CpgPolicyState, _Stat


def test_shadow_schedule_deterministic():
    """Test that shadow schedule is deterministic for fixed seed."""
    cfg1 = CpgPolicyCfg(sample_rate=0.01, seed=12345)
    policy1 = CpgPolicy(cfg1)
    
    cfg2 = CpgPolicyCfg(sample_rate=0.01, seed=12345)
    policy2 = CpgPolicy(cfg2)
    
    # Check 50k tokens
    n_tokens = 50_000
    mask1 = [policy1.should_shadow(t) for t in range(n_tokens)]
    mask2 = [policy2.should_shadow(t) for t in range(n_tokens)]
    
    # Check determinism
    assert mask1 == mask2, "Shadow schedule not deterministic"
    
    # Check sample rate (should be ~1%)
    sample_count = sum(mask1)
    expected = int(n_tokens * 0.01)
    tolerance = int(n_tokens * 0.002)  # ±0.2%
    
    assert abs(sample_count - expected) < tolerance, \
        f"Sample rate {sample_count / n_tokens:.4f} not close to 0.01"
    
    print(f"[PASS] Shadow schedule deterministic: {sample_count}/{n_tokens} = {sample_count / n_tokens:.4f}")


def test_shadow_schedule_different_seeds():
    """Test that different seeds produce different schedules."""
    cfg1 = CpgPolicyCfg(sample_rate=0.01, seed=12345)
    policy1 = CpgPolicy(cfg1)
    
    cfg2 = CpgPolicyCfg(sample_rate=0.01, seed=54321)
    policy2 = CpgPolicy(cfg2)
    
    # Check 1000 tokens
    n_tokens = 1000
    mask1 = [policy1.should_shadow(t) for t in range(n_tokens)]
    mask2 = [policy2.should_shadow(t) for t in range(n_tokens)]
    
    # Check that schedules differ
    assert mask1 != mask2, "Different seeds should produce different schedules"
    
    print(f"[PASS] Different seeds produce different schedules")


def test_promote_after_clean_agreement():
    """Test that module is promoted after clean agreement."""
    cfg = CpgPolicyCfg(
        promote_min_samples=200,
        agree_target=0.999,
        ymean_tol=1e-6,
        k_used_tol=0,
        overhead_tol=0.15,
    )
    policy = CpgPolicy(cfg)
    
    # Initially disabled
    assert not policy.decide("stageA"), "stageA should start disabled"
    
    # Feed 300 clean samples
    for i in range(300):
        policy.update(
            "stageA",
            agree=True,
            ymean_diff=1e-8,  # Well below tolerance
            k_used_delta=0,
            overhead_ratio=0.02,  # 2% overhead
        )
        policy.maybe_promote("stageA")
    
    # Should be promoted
    assert policy.decide("stageA"), "stageA should be promoted after 300 clean samples"
    
    stats = policy.stats()
    assert stats["stageA"]["samples"] == 300
    assert stats["stageA"]["agree"] == 1.0
    assert stats["stageA"]["enabled"] == True
    
    print(f"[PASS] Promoted after 300 clean samples: {stats['stageA']}")


def test_no_promote_before_min_samples():
    """Test that module is not promoted before min samples."""
    cfg = CpgPolicyCfg(
        promote_min_samples=200,
        agree_target=0.999,
        ymean_tol=1e-6,
        k_used_tol=0,
        overhead_tol=0.15,
    )
    policy = CpgPolicy(cfg)
    
    # Feed 100 clean samples (below min)
    for i in range(100):
        policy.update(
            "stageA",
            agree=True,
            ymean_diff=1e-8,
            k_used_delta=0,
            overhead_ratio=0.02,
        )
        policy.maybe_promote("stageA")
    
    # Should NOT be promoted
    assert not policy.decide("stageA"), "stageA should not be promoted before min samples"
    
    print(f"[PASS] Not promoted before min samples (100 < 200)")


def test_no_promote_on_low_agreement():
    """Test that module is not promoted on low agreement."""
    cfg = CpgPolicyCfg(
        promote_min_samples=200,
        agree_target=0.999,
        ymean_tol=1e-6,
        k_used_tol=0,
        overhead_tol=0.15,
    )
    policy = CpgPolicy(cfg)
    
    # Feed 250 samples with 95% agreement (below 99.9%)
    for i in range(250):
        agree = (i % 20) != 0  # 95% agreement
        policy.update(
            "stageA",
            agree=agree,
            ymean_diff=1e-8,
            k_used_delta=0,
            overhead_ratio=0.02,
        )
        policy.maybe_promote("stageA")
    
    # Should NOT be promoted
    assert not policy.decide("stageA"), "stageA should not be promoted with low agreement"
    
    stats = policy.stats()
    assert stats["stageA"]["agree"] < 0.999, f"Agreement {stats['stageA']['agree']} should be < 0.999"
    
    print(f"[PASS] Not promoted on low agreement: {stats['stageA']['agree']:.4f}")


def test_demote_on_disagreement():
    """Test that module is demoted on disagreement."""
    cfg = CpgPolicyCfg(
        promote_min_samples=200,
        demote_window=200,
        agree_target=0.999,
        ymean_tol=1e-6,
        k_used_tol=0,
        overhead_tol=0.15,
    )
    policy = CpgPolicy(cfg)
    
    # Promote first
    for i in range(300):
        policy.update(
            "stageA",
            agree=True,
            ymean_diff=1e-8,
            k_used_delta=0,
            overhead_ratio=0.02,
        )
        policy.maybe_promote("stageA")
    
    assert policy.decide("stageA"), "stageA should be promoted"
    
    # Feed 20 disagreements
    for i in range(20):
        policy.update(
            "stageA",
            agree=False,
            ymean_diff=1e-8,
            k_used_delta=0,
            overhead_ratio=0.02,
        )
        policy.maybe_demote("stageA")
    
    # Should be demoted
    assert not policy.decide("stageA"), "stageA should be demoted after disagreements"
    
    print(f"[PASS] Demoted after 20 disagreements")


def test_demote_on_overhead():
    """Test that module is demoted on high overhead."""
    cfg = CpgPolicyCfg(
        promote_min_samples=200,
        demote_window=200,
        agree_target=0.999,
        ymean_tol=1e-6,
        k_used_tol=0,
        overhead_tol=0.15,
    )
    policy = CpgPolicy(cfg)
    
    # Promote first
    for i in range(300):
        policy.update(
            "attn",
            agree=True,
            ymean_diff=1e-8,
            k_used_delta=0,
            overhead_ratio=0.02,
        )
        policy.maybe_promote("attn")
    
    assert policy.decide("attn"), "attn should be promoted"
    
    # Feed high overhead samples
    for i in range(50):
        policy.update(
            "attn",
            agree=True,
            ymean_diff=1e-8,
            k_used_delta=0,
            overhead_ratio=0.25,  # 25% overhead (> 18% threshold)
        )
        policy.maybe_demote("attn")
    
    # Should be demoted
    assert not policy.decide("attn"), "attn should be demoted on high overhead"
    
    stats = policy.stats()
    print(f"[PASS] Demoted on high overhead: {stats['attn']['overhead_ema']:.4f}")


def test_demote_on_k_used_increase():
    """Test that module is demoted on k_used increase."""
    cfg = CpgPolicyCfg(
        promote_min_samples=200,
        demote_window=200,
        agree_target=0.999,
        ymean_tol=1e-6,
        k_used_tol=0,
        overhead_tol=0.15,
    )
    policy = CpgPolicy(cfg)
    
    # Promote first
    for i in range(300):
        policy.update(
            "logits",
            agree=True,
            ymean_diff=1e-8,
            k_used_delta=0,
            overhead_ratio=0.02,
        )
        policy.maybe_promote("logits")
    
    assert policy.decide("logits"), "logits should be promoted"

    # Feed k_used increases (need >100 samples to shift median in window of 200)
    for i in range(150):
        policy.update(
            "logits",
            agree=True,
            ymean_diff=1e-8,
            k_used_delta=2,  # Positive delta (increase)
            overhead_ratio=0.02,
        )
        policy.maybe_demote("logits")

    # Should be demoted
    assert not policy.decide("logits"), "logits should be demoted on k_used increase"
    
    stats = policy.stats()
    print(f"[PASS] Demoted on k_used increase: median_k_delta={stats['logits']['median_k_delta']}")


def test_stat_ema_updates():
    """Test that _Stat EMA updates work correctly."""
    stat = _Stat(window_size=10, ema_alpha=0.5)
    
    # First update
    stat.update(agree=True, ymean_diff=1.0, k_used_delta=0, overhead_ratio=0.1)
    assert stat.ymean_ema == 1.0, "First EMA should equal first value"
    assert stat.overhead_ema == 0.1, "First EMA should equal first value"
    
    # Second update
    stat.update(agree=True, ymean_diff=0.0, k_used_delta=0, overhead_ratio=0.2)
    expected_ymean = 0.5 * 1.0 + 0.5 * 0.0
    expected_overhead = 0.5 * 0.1 + 0.5 * 0.2
    assert abs(stat.ymean_ema - expected_ymean) < 1e-9, f"EMA {stat.ymean_ema} != {expected_ymean}"
    assert abs(stat.overhead_ema - expected_overhead) < 1e-9, f"EMA {stat.overhead_ema} != {expected_overhead}"
    
    print(f"[PASS] _Stat EMA updates: ymean={stat.ymean_ema:.4f}, overhead={stat.overhead_ema:.4f}")


def test_stat_windowed_history():
    """Test that _Stat windowed history works correctly."""
    stat = _Stat(window_size=5, ema_alpha=0.1)
    
    # Add 10 samples
    for i in range(10):
        stat.update(agree=(i % 2 == 0), ymean_diff=0.0, k_used_delta=i, overhead_ratio=0.0)
    
    # Window should contain last 5 samples
    assert len(stat.k_used_deltas) == 5, f"Window size {len(stat.k_used_deltas)} != 5"
    assert stat.k_used_deltas == [5, 6, 7, 8, 9], f"Window {stat.k_used_deltas} != [5, 6, 7, 8, 9]"
    
    # Recent agree rate should be 40% (2/5: indices 6, 8)
    assert stat.recent_agree_rate() == 0.4, f"Recent agree rate {stat.recent_agree_rate()} != 0.4"
    
    # Overall agree rate should be 50% (5/10)
    assert stat.agree_rate() == 0.5, f"Overall agree rate {stat.agree_rate()} != 0.5"
    
    print(f"[PASS] _Stat windowed history: window={stat.k_used_deltas}, recent_agree={stat.recent_agree_rate():.2f}")


def test_policy_stats_export():
    """Test that policy stats export works correctly."""
    cfg = CpgPolicyCfg()
    policy = CpgPolicy(cfg)
    
    # Add some samples
    for i in range(10):
        policy.update("stageA", agree=True, ymean_diff=1e-8, k_used_delta=0, overhead_ratio=0.02)
    
    stats = policy.stats()
    
    assert "stageA" in stats
    assert "attn" in stats
    assert "logits" in stats
    
    assert stats["stageA"]["samples"] == 10
    assert stats["stageA"]["agree"] == 1.0
    assert stats["stageA"]["enabled"] == False
    
    print(f"[PASS] Policy stats export: {stats['stageA']}")


def test_multiple_modules_independent():
    """Test that multiple modules can be promoted/demoted independently."""
    cfg = CpgPolicyCfg(promote_min_samples=100)
    policy = CpgPolicy(cfg)
    
    # Promote stageA
    for i in range(150):
        policy.update("stageA", agree=True, ymean_diff=1e-8, k_used_delta=0, overhead_ratio=0.02)
        policy.maybe_promote("stageA")
    
    # Promote attn
    for i in range(150):
        policy.update("attn", agree=True, ymean_diff=1e-8, k_used_delta=0, overhead_ratio=0.02)
        policy.maybe_promote("attn")
    
    # Don't promote logits (low agreement)
    for i in range(150):
        agree = (i % 10) != 0  # 90% agreement
        policy.update("logits", agree=agree, ymean_diff=1e-8, k_used_delta=0, overhead_ratio=0.02)
        policy.maybe_promote("logits")
    
    assert policy.decide("stageA") == True, "stageA should be promoted"
    assert policy.decide("attn") == True, "attn should be promoted"
    assert policy.decide("logits") == False, "logits should not be promoted"
    
    print(f"[PASS] Multiple modules independent: stageA=True, attn=True, logits=False")

