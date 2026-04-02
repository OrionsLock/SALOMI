"""Shortlist cache for logits warm-start (PR-4.2).

Goal: Carry a small, deterministic shortlist from token t-1 to token t,
reducing HCL chunk evaluations and pair tests while maintaining identical
final decisions vs baseline.

Invariants:
- Deterministic ordering for fixed seed
- No correctness risk (still compute full vocab energies)
- Zero storage growth beyond cap * sizeof(entry) per head
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CarryCfg:
    """Configuration for shortlist carry-over (PR-4.2).
    
    Args:
        enable: Enable carry-over (default: True)
        frac: Fraction of shortlist to carry from t-1 (default: 0.50)
        cap: Hard cap on carried IDs per head (default: 256)
        ttl: Tokens to keep if unseen (default: 8)
        ema: EMA decay for energy scores (default: 0.30)
        explore: Min fresh candidates to add each token (default: 128)
        seed: PRF seed for deterministic tie-breaking (default: 12345)
    """
    enable: bool = True
    frac: float = 0.50
    cap: int = 256
    ttl: int = 8
    ema: float = 0.30
    explore: int = 128
    seed: int = 12345


class ShortlistCache:
    """Per-head shortlist cache with deterministic priority ordering.
    
    Stores vocab IDs with simple stats:
    - score_ema: EMA of energy scores (float32)
    - last_seen: Token index when last seen (int32)
    - hit_count: Number of times seen (int32)
    
    Priority key: (-score_ema, -hit_count, prf_tie(id, seed))
    Deterministic eviction: lowest priority then oldest.
    """
    
    def __init__(self, cap: int, ttl: int, ema: float, seed: int):
        """Initialize shortlist cache.
        
        Args:
            cap: Maximum number of IDs to store
            ttl: Tokens to keep if unseen
            ema: EMA decay for energy scores
            seed: PRF seed for deterministic tie-breaking
        """
        self.cap = cap
        self.ttl = ttl
        self.ema = ema
        self.seed = seed
        
        # Storage: dict[id] -> (score_ema, last_seen, hit_count)
        self._entries: dict[int, tuple[float, int, int]] = {}
    
    def _prf_tie(self, id: int) -> int:
        """Deterministic tie-breaker using PRF."""
        return (self.seed ^ id) & 0x7FFFFFFF
    
    def _priority_key(self, id: int, score_ema: float, hit_count: int) -> tuple:
        """Compute priority key for sorting.
        
        Returns:
            (-score_ema, -hit_count, prf_tie(id))
        """
        return (-score_ema, -hit_count, self._prf_tie(id))
    
    def put_many(self, ids: np.ndarray, scores: np.ndarray, now: int):
        """Add or update multiple IDs with scores.
        
        Args:
            ids: Vocab IDs [N]
            scores: Energy scores [N]
            now: Current token index
        """
        for id, score in zip(ids, scores):
            id = int(id)
            score = float(score)
            
            if id in self._entries:
                # Update existing entry
                old_score, _, old_count = self._entries[id]
                new_score = (1.0 - self.ema) * old_score + self.ema * score
                self._entries[id] = (new_score, now, old_count + 1)
            else:
                # New entry
                self._entries[id] = (score, now, 1)
        
        # Enforce cap
        if len(self._entries) > self.cap:
            self._evict_to_cap()
    
    def _evict_to_cap(self):
        """Evict lowest priority entries to enforce cap."""
        if len(self._entries) <= self.cap:
            return
        
        # Sort by priority (lowest first)
        items = [
            (self._priority_key(id, score, count), id)
            for id, (score, last_seen, count) in self._entries.items()
        ]
        items.sort(reverse=True)  # Highest priority first
        
        # Keep top cap entries
        keep_ids = {id for _, id in items[:self.cap]}
        self._entries = {
            id: entry
            for id, entry in self._entries.items()
            if id in keep_ids
        }
    
    def carry(self, k: int, now: int) -> np.ndarray:
        """Return up to k IDs by priority.
        
        Args:
            k: Number of IDs to carry
            now: Current token index
        
        Returns:
            Vocab IDs [≤k] sorted by priority (highest first)
        """
        if not self._entries:
            return np.array([], dtype=np.int32)
        
        # Sort by priority (highest first)
        items = [
            (self._priority_key(id, score, count), id)
            for id, (score, last_seen, count) in self._entries.items()
        ]
        items.sort()  # Lowest priority key first (highest priority)
        
        # Take top k
        carry_ids = [id for _, id in items[:k]]
        
        return np.array(carry_ids, dtype=np.int32)
    
    def update_seen(self, ids: np.ndarray, scores: np.ndarray, now: int):
        """Update entries with new scores (alias for put_many).
        
        Args:
            ids: Vocab IDs [N]
            scores: Energy scores [N]
            now: Current token index
        """
        self.put_many(ids, scores, now)
    
    def evict_expired(self, now: int):
        """Evict entries not seen in ttl tokens.
        
        Args:
            now: Current token index
        """
        expired = [
            id
            for id, (_, last_seen, _) in self._entries.items()
            if now - last_seen > self.ttl
        ]
        
        for id in expired:
            del self._entries[id]
    
    def stats(self) -> dict:
        """Return cache statistics.
        
        Returns:
            {
                "size": int - Number of entries
                "cap": int - Maximum capacity
                "avg_score": float - Average score_ema
                "avg_hits": float - Average hit_count
            }
        """
        if not self._entries:
            return {
                "size": 0,
                "cap": self.cap,
                "avg_score": 0.0,
                "avg_hits": 0.0,
            }
        
        scores = [score for score, _, _ in self._entries.values()]
        hits = [count for _, _, count in self._entries.values()]
        
        return {
            "size": len(self._entries),
            "cap": self.cap,
            "avg_score": float(np.mean(scores)),
            "avg_hits": float(np.mean(hits)),
        }
    
    def clear(self):
        """Clear all entries."""
        self._entries.clear()


class TopKHeap:
    """Min-heap for tracking top-K items by score.
    
    Used for collecting fresh candidates during HCL chunk iteration.
    """
    
    def __init__(self, k: int, seed: int):
        """Initialize top-K heap.
        
        Args:
            k: Number of items to track
            seed: PRF seed for deterministic tie-breaking
        """
        self.k = k
        self.seed = seed
        
        # Storage: list of (score, id)
        self._items: list[tuple[float, int]] = []
    
    def _prf_tie(self, id: int) -> int:
        """Deterministic tie-breaker using PRF."""
        return (self.seed ^ id) & 0x7FFFFFFF
    
    def push(self, score: float, id: int):
        """Add item to heap.
        
        Args:
            score: Energy score
            id: Vocab ID
        """
        # Key: (-score, prf_tie(id)) for max-heap behavior
        key = (-score, self._prf_tie(id))
        
        if len(self._items) < self.k:
            # Heap not full, add item
            self._items.append((key, id))
            self._items.sort()  # Simple sort (could use heapq for efficiency)
        else:
            # Heap full, replace min if new item is better
            if key < self._items[-1][0]:
                self._items[-1] = (key, id)
                self._items.sort()
    
    def push_many(self, scores: np.ndarray, ids: np.ndarray):
        """Add multiple items to heap.
        
        Args:
            scores: Energy scores [N]
            ids: Vocab IDs [N]
        """
        for score, id in zip(scores, ids):
            self.push(float(score), int(id))
    
    def full(self) -> bool:
        """Check if heap is full."""
        return len(self._items) >= self.k
    
    def sorted_ids(self) -> np.ndarray:
        """Return IDs sorted by score (highest first).
        
        Returns:
            Vocab IDs [≤k]
        """
        # Items are sorted by key (lowest first), so reverse
        return np.array([id for _, id in self._items], dtype=np.int32)
    
    def size(self) -> int:
        """Return number of items in heap."""
        return len(self._items)

