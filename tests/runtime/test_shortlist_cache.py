"""Tests for shortlist cache (PR-4.2).

Goal: Verify deterministic ordering, TTL eviction, EMA updates, and cap enforcement.

Tests:
- test_put_and_carry_order_deterministic(): Deterministic priority ordering
- test_ttl_eviction(): IDs disappear after ttl tokens of no sighting
- test_ema_updates_and_priority(): EMA updates and priority changes
- test_cap_enforced_and_stable_evict(): Cap enforced with stable eviction
- test_topk_heap_deterministic(): TopKHeap deterministic ordering
"""
from __future__ import annotations

import numpy as np
import pytest

from onebit.runtime.shortlist import ShortlistCache, TopKHeap, CarryCfg


def test_put_and_carry_order_deterministic():
    """Test that put_many and carry produce deterministic ordering."""
    cap, ttl, ema, seed = 100, 8, 0.3, 12345
    
    # Run 1
    cache1 = ShortlistCache(cap, ttl, ema, seed)
    ids1 = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    scores1 = np.array([1.0, 2.0, 3.0, 2.5, 1.5], dtype=np.float32)
    cache1.put_many(ids1, scores1, now=0)
    
    carry1 = cache1.carry(k=3, now=0)
    
    # Run 2 (same seed, same data)
    cache2 = ShortlistCache(cap, ttl, ema, seed)
    ids2 = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    scores2 = np.array([1.0, 2.0, 3.0, 2.5, 1.5], dtype=np.float32)
    cache2.put_many(ids2, scores2, now=0)
    
    carry2 = cache2.carry(k=3, now=0)
    
    # Check determinism
    assert np.array_equal(carry1, carry2), \
        f"Carry order not deterministic: {carry1} != {carry2}"
    
    # Check priority (highest scores first)
    # Expected order: 30 (3.0), 40 (2.5), 20 (2.0)
    assert carry1[0] == 30, f"Top ID should be 30, got {carry1[0]}"
    assert carry1[1] == 40, f"Second ID should be 40, got {carry1[1]}"
    assert carry1[2] == 20, f"Third ID should be 20, got {carry1[2]}"
    
    print(f"[PASS] Deterministic carry order: {carry1}")


def test_ttl_eviction():
    """Test that IDs disappear after ttl tokens of no sighting."""
    cap, ttl, ema, seed = 100, 3, 0.3, 12345
    
    cache = ShortlistCache(cap, ttl, ema, seed)
    
    # Add IDs at token 0
    ids = np.array([10, 20, 30], dtype=np.int32)
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.put_many(ids, scores, now=0)
    
    # Check at token 0
    carry0 = cache.carry(k=10, now=0)
    assert len(carry0) == 3, f"Should have 3 IDs at token 0, got {len(carry0)}"
    
    # Update ID 20 at token 2 (within TTL)
    cache.update_seen(np.array([20], dtype=np.int32), np.array([2.5], dtype=np.float32), now=2)
    
    # Evict at token 4 (ttl=3, so IDs not seen since token 0 should expire)
    cache.evict_expired(now=4)
    
    carry4 = cache.carry(k=10, now=4)
    
    # ID 20 should survive (last_seen=2, now=4, diff=2 < ttl=3)
    # IDs 10, 30 should be evicted (last_seen=0, now=4, diff=4 > ttl=3)
    assert 20 in carry4, f"ID 20 should survive TTL, got {carry4}"
    assert 10 not in carry4, f"ID 10 should be evicted, got {carry4}"
    assert 30 not in carry4, f"ID 30 should be evicted, got {carry4}"
    
    print(f"[PASS] TTL eviction: {len(carry0)} → {len(carry4)} IDs (20 survived)")


def test_ema_updates_and_priority():
    """Test that EMA updates change priority ordering."""
    cap, ttl, ema, seed = 100, 8, 0.5, 12345
    
    cache = ShortlistCache(cap, ttl, ema, seed)
    
    # Initial: ID 10 has score 1.0, ID 20 has score 2.0
    cache.put_many(
        np.array([10, 20], dtype=np.int32),
        np.array([1.0, 2.0], dtype=np.float32),
        now=0
    )
    
    carry0 = cache.carry(k=2, now=0)
    assert carry0[0] == 20, f"ID 20 should be first (higher score), got {carry0}"
    
    # Update ID 10 with high score (3.0)
    # New EMA for ID 10: 0.5 * 1.0 + 0.5 * 3.0 = 2.0
    cache.update_seen(
        np.array([10], dtype=np.int32),
        np.array([3.0], dtype=np.float32),
        now=1
    )
    
    # Update ID 20 with same score (2.0)
    # New EMA for ID 20: 0.5 * 2.0 + 0.5 * 2.0 = 2.0
    cache.update_seen(
        np.array([20], dtype=np.int32),
        np.array([2.0], dtype=np.float32),
        now=1
    )
    
    carry1 = cache.carry(k=2, now=1)
    
    # Both have same EMA (2.0), but ID 10 has hit_count=2, ID 20 has hit_count=2
    # Tie-break by PRF(id, seed)
    print(f"[INFO] After EMA update: carry={carry1}")
    print(f"[INFO] Stats: {cache.stats()}")
    
    # Update ID 10 again with very high score (5.0)
    # New EMA for ID 10: 0.5 * 2.0 + 0.5 * 5.0 = 3.5
    cache.update_seen(
        np.array([10], dtype=np.int32),
        np.array([5.0], dtype=np.float32),
        now=2
    )
    
    carry2 = cache.carry(k=2, now=2)
    assert carry2[0] == 10, f"ID 10 should be first (higher EMA), got {carry2}"
    
    print(f"[PASS] EMA updates change priority: {carry0} → {carry1} → {carry2}")


def test_cap_enforced_and_stable_evict():
    """Test that cap is enforced with stable eviction."""
    cap, ttl, ema, seed = 10, 8, 0.3, 12345
    
    cache = ShortlistCache(cap, ttl, ema, seed)
    
    # Add 15 IDs (exceeds cap=10)
    ids = np.arange(100, 115, dtype=np.int32)
    scores = np.random.RandomState(42).uniform(1.0, 5.0, size=15).astype(np.float32)
    
    cache.put_many(ids, scores, now=0)
    
    # Check that only cap=10 IDs remain
    stats = cache.stats()
    assert stats["size"] == cap, \
        f"Cache size {stats['size']} should equal cap {cap}"
    
    # Carry all IDs
    carry = cache.carry(k=100, now=0)
    assert len(carry) == cap, \
        f"Carry should return {cap} IDs, got {len(carry)}"
    
    # Run again with same seed and data
    cache2 = ShortlistCache(cap, ttl, ema, seed)
    cache2.put_many(ids, scores, now=0)
    carry2 = cache2.carry(k=100, now=0)
    
    # Check deterministic eviction
    assert np.array_equal(carry, carry2), \
        f"Eviction not deterministic: {carry} != {carry2}"
    
    print(f"[PASS] Cap enforced: {len(ids)} → {len(carry)} IDs (deterministic)")


def test_topk_heap_deterministic():
    """Test that TopKHeap produces deterministic ordering."""
    k, seed = 5, 12345
    
    # Run 1
    heap1 = TopKHeap(k, seed)
    ids1 = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int32)
    scores1 = np.array([1.0, 3.0, 2.0, 5.0, 1.5, 4.0, 2.5], dtype=np.float32)
    heap1.push_many(scores1, ids1)
    
    sorted1 = heap1.sorted_ids()
    
    # Run 2 (same seed, same data)
    heap2 = TopKHeap(k, seed)
    ids2 = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int32)
    scores2 = np.array([1.0, 3.0, 2.0, 5.0, 1.5, 4.0, 2.5], dtype=np.float32)
    heap2.push_many(scores2, ids2)
    
    sorted2 = heap2.sorted_ids()
    
    # Check determinism
    assert np.array_equal(sorted1, sorted2), \
        f"TopKHeap not deterministic: {sorted1} != {sorted2}"
    
    # Check that we got top-5 by score
    # Expected: 40 (5.0), 60 (4.0), 20 (3.0), 70 (2.5), 30 (2.0)
    assert len(sorted1) == k, f"Should have {k} IDs, got {len(sorted1)}"
    assert sorted1[0] == 40, f"Top ID should be 40 (score 5.0), got {sorted1[0]}"
    assert sorted1[1] == 60, f"Second ID should be 60 (score 4.0), got {sorted1[1]}"
    
    print(f"[PASS] TopKHeap deterministic: {sorted1}")


def test_carry_cfg_defaults():
    """Test CarryCfg default values."""
    cfg = CarryCfg()
    
    assert cfg.enable == True
    assert cfg.frac == 0.50
    assert cfg.cap == 256
    assert cfg.ttl == 8
    assert cfg.ema == 0.30
    assert cfg.explore == 128
    assert cfg.seed == 12345
    
    print("[PASS] CarryCfg defaults")


def test_cache_clear():
    """Test cache clear operation."""
    cache = ShortlistCache(cap=100, ttl=8, ema=0.3, seed=12345)
    
    # Add IDs
    cache.put_many(
        np.array([10, 20, 30], dtype=np.int32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        now=0
    )
    
    assert cache.stats()["size"] == 3
    
    # Clear
    cache.clear()
    
    assert cache.stats()["size"] == 0
    assert len(cache.carry(k=10, now=0)) == 0
    
    print("[PASS] Cache clear")


def test_empty_cache_operations():
    """Test operations on empty cache."""
    cache = ShortlistCache(cap=100, ttl=8, ema=0.3, seed=12345)
    
    # Carry from empty cache
    carry = cache.carry(k=10, now=0)
    assert len(carry) == 0, "Empty cache should return empty carry"
    
    # Stats from empty cache
    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["avg_score"] == 0.0
    assert stats["avg_hits"] == 0.0
    
    # Evict from empty cache (should not crash)
    cache.evict_expired(now=10)
    
    print("[PASS] Empty cache operations")


def test_hit_count_increments():
    """Test that hit_count increments on repeated updates."""
    cache = ShortlistCache(cap=100, ttl=8, ema=0.3, seed=12345)
    
    # Add ID 10 three times
    for i in range(3):
        cache.update_seen(
            np.array([10], dtype=np.int32),
            np.array([2.0], dtype=np.float32),
            now=i
        )
    
    # Add ID 20 once
    cache.update_seen(
        np.array([20], dtype=np.int32),
        np.array([2.0], dtype=np.float32),
        now=0
    )
    
    # ID 10 should have higher priority (same score, higher hit_count)
    carry = cache.carry(k=2, now=3)
    
    # Both have same EMA score, but ID 10 has hit_count=3, ID 20 has hit_count=1
    # Priority: (-score, -hit_count, prf_tie)
    # ID 10: (-2.0, -3, prf(10))
    # ID 20: (-2.0, -1, prf(20))
    # ID 10 should be first (higher hit_count)
    assert carry[0] == 10, f"ID 10 should be first (higher hit_count), got {carry}"
    
    print(f"[PASS] Hit count increments: ID 10 (3 hits) > ID 20 (1 hit)")

