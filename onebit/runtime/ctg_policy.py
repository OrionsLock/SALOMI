"""DEPRECATED: Use cpg_policy instead.

This module is DEPRECATED. The shadow policy has been renamed from CTG to CPG
(Contrast/Perf Gating) to avoid confusion with CTG (Constant-Time Grammar).

Please use::

    from onebit.runtime.cpg_policy import CpgPolicy, CpgPolicyCfg, CpgPolicyState

This module will be removed in a future version.

---

Original docstring (for reference):

CTG policy with 1% shadow sampling (PR-4.3).

Goal: Enable CTG only where it's provably safe. Use a 1% deterministic "shadow"
sample to A/B CTG(on) vs CTG(off) without affecting outputs. Promote CTG per-module
once the shadow agrees and overhead stays ≤15%. Demote on regressions.

Invariants:
- Export = 1.00 bpp
- Deterministic for fixed seeds
- Main-path decisions and certificates remain identical during learning
- No extra resident buffers; shadow uses transient compute only
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class CtgPolicyCfg:
    """Configuration for CTG policy (PR-4.3).
    
    Args:
        sample_rate: Shadow sampling rate (default: 0.01 = 1%)
        seed: PRF seed for deterministic shadow schedule
        promote_min_samples: Min samples before promotion (default: 200)
        demote_window: Window for demote checks (default: 200)
        agree_target: Min agreement rate for promotion (default: 0.999 = 99.9%)
        ymean_tol: Max |Δ y_mean| EMA for promotion (default: 1e-6)
        k_used_tol: Max median k_used delta for promotion (default: 0)
        overhead_tol: Max time overhead EMA for promotion (default: 0.15 = 15%)
    """
    sample_rate: float = 0.01
    seed: int = 12345
    promote_min_samples: int = 200
    demote_window: int = 200
    agree_target: float = 0.999
    ymean_tol: float = 1e-6
    k_used_tol: int = 0
    overhead_tol: float = 0.15


@dataclass
class CtgPolicyState:
    """Per-module CTG enable flags."""
    enabled_attn_stageA: bool = False
    enabled_attn_sprt: bool = False
    enabled_logits: bool = False


class _Stat:
    """Rolling statistics for one module.
    
    Tracks:
    - count: Total samples
    - agree_count: Samples where shadow agreed with main
    - ymean_ema: EMA of |Δ y_mean|
    - k_used_deltas: Windowed history of k_used deltas
    - overhead_ema: EMA of overhead ratio
    - recent_agrees: Windowed history of agreement flags
    """
    
    def __init__(self, window_size: int = 200, ema_alpha: float = 0.1):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        
        # Counters
        self.count = 0
        self.agree_count = 0
        
        # EMAs
        self.ymean_ema = 0.0
        self.overhead_ema = 0.0
        
        # Windowed history for demote checks
        self.k_used_deltas: list[int] = []
        self.recent_agrees: list[bool] = []
    
    def update(self, agree: bool, ymean_diff: float, k_used_delta: int, overhead_ratio: float):
        """Update statistics with new sample."""
        self.count += 1
        
        if agree:
            self.agree_count += 1
        
        # Update EMAs
        if self.count == 1:
            self.ymean_ema = ymean_diff
            self.overhead_ema = overhead_ratio
        else:
            self.ymean_ema = (1.0 - self.ema_alpha) * self.ymean_ema + self.ema_alpha * ymean_diff
            self.overhead_ema = (1.0 - self.ema_alpha) * self.overhead_ema + self.ema_alpha * overhead_ratio
        
        # Update windowed history
        self.k_used_deltas.append(k_used_delta)
        if len(self.k_used_deltas) > self.window_size:
            self.k_used_deltas.pop(0)
        
        self.recent_agrees.append(agree)
        if len(self.recent_agrees) > self.window_size:
            self.recent_agrees.pop(0)
    
    def agree_rate(self) -> float:
        """Compute overall agreement rate."""
        if self.count == 0:
            return 0.0
        return self.agree_count / self.count
    
    def recent_agree_rate(self) -> float:
        """Compute agreement rate in recent window."""
        if not self.recent_agrees:
            return 0.0
        return sum(self.recent_agrees) / len(self.recent_agrees)
    
    def median_k_used_delta(self) -> int:
        """Compute median k_used delta."""
        if not self.k_used_deltas:
            return 0
        return int(np.median(self.k_used_deltas))
    
    def to_dict(self) -> dict:
        """Export statistics as dict."""
        return {
            "samples": self.count,
            "agree": self.agree_rate(),
            "overhead_ema": self.overhead_ema,
            "ymean_ema": self.ymean_ema,
            "median_k_delta": self.median_k_used_delta(),
        }


class CtgPolicy:
    """CTG policy with 1% shadow sampling.
    
    Manages per-module CTG enable flags based on shadow A/B testing.
    Promotes CTG when shadow agrees and overhead is acceptable.
    Demotes CTG on regressions.
    """
    
    # Module name mapping
    _MODULE_MAP = {
        "stageA": "enabled_attn_stageA",
        "attn": "enabled_attn_sprt",
        "logits": "enabled_logits",
    }
    
    def __init__(self, cfg: CtgPolicyCfg):
        """Initialize CTG policy.
        
        Args:
            cfg: Policy configuration
        """
        self.cfg = cfg
        self.state = CtgPolicyState()
        
        # Per-module statistics
        self._stats: Dict[str, _Stat] = {
            "stageA": _Stat(window_size=cfg.demote_window),
            "attn": _Stat(window_size=cfg.demote_window),
            "logits": _Stat(window_size=cfg.demote_window),
        }
    
    def should_shadow(self, token_idx: int) -> bool:
        """Determine if token should be shadowed.
        
        Uses deterministic hash to select ~sample_rate fraction of tokens.
        
        Args:
            token_idx: Current token index
        
        Returns:
            True if this token should run shadow A/B test
        """
        # Deterministic hash: hash64(seed ^ token_idx) % 10_000 < sample_rate * 10_000
        h = self._hash64(self.cfg.seed ^ token_idx)
        threshold = int(self.cfg.sample_rate * 10_000)
        return (h % 10_000) < threshold
    
    def _hash64(self, x: int) -> int:
        """Simple 64-bit hash (SplitMix64-like)."""
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB
        x = x ^ (x >> 31)
        return x & 0xFFFFFFFFFFFFFFFF
    
    def decide(self, module: str) -> bool:
        """Get current CTG enable flag for module.
        
        Args:
            module: Module name ("stageA", "attn", "logits")
        
        Returns:
            True if CTG should be enabled for this module
        """
        if module not in self._MODULE_MAP:
            raise ValueError(f"Unknown module: {module}")
        
        attr_name = self._MODULE_MAP[module]
        return getattr(self.state, attr_name)
    
    def update(
        self,
        module: str,
        *,
        agree: bool,
        ymean_diff: float,
        k_used_delta: int,
        overhead_ratio: float,
    ):
        """Update statistics for module with shadow sample result.
        
        Args:
            module: Module name ("stageA", "attn", "logits")
            agree: True if shadow agreed with main (same decisions)
            ymean_diff: |Δ y_mean| between main and shadow
            k_used_delta: k_used difference (shadow - main)
            overhead_ratio: Time overhead ratio (shadow_time / main_time - 1.0)
        """
        if module not in self._stats:
            raise ValueError(f"Unknown module: {module}")
        
        self._stats[module].update(agree, ymean_diff, k_used_delta, overhead_ratio)
    
    def maybe_promote(self, module: str):
        """Check if module should be promoted (CTG enabled).
        
        Promotion criteria:
        - samples >= promote_min_samples
        - agree_rate >= agree_target
        - ymean_ema <= ymean_tol
        - median(k_used_delta) <= k_used_tol
        - overhead_ema <= overhead_tol
        
        Args:
            module: Module name ("stageA", "attn", "logits")
        """
        if module not in self._stats:
            raise ValueError(f"Unknown module: {module}")
        
        # Already enabled, skip
        if self.decide(module):
            return
        
        stat = self._stats[module]
        
        # Check promotion criteria
        if stat.count < self.cfg.promote_min_samples:
            return
        
        if stat.agree_rate() < self.cfg.agree_target:
            return
        
        if stat.ymean_ema > self.cfg.ymean_tol:
            return
        
        if stat.median_k_used_delta() > self.cfg.k_used_tol:
            return
        
        if stat.overhead_ema > self.cfg.overhead_tol:
            return
        
        # All criteria met, promote
        attr_name = self._MODULE_MAP[module]
        setattr(self.state, attr_name, True)
    
    def maybe_demote(self, module: str):
        """Check if module should be demoted (CTG disabled).
        
        Demotion criteria (any of):
        - recent_agree_rate < 0.999
        - overhead_ema > 0.18
        - median(k_used_delta) > 0
        
        Args:
            module: Module name ("stageA", "attn", "logits")
        """
        if module not in self._stats:
            raise ValueError(f"Unknown module: {module}")
        
        # Not enabled, skip
        if not self.decide(module):
            return
        
        stat = self._stats[module]
        
        # Check demotion criteria
        if stat.recent_agree_rate() < 0.999:
            # Demote on disagreement
            attr_name = self._MODULE_MAP[module]
            setattr(self.state, attr_name, False)
            return
        
        if stat.overhead_ema > 0.18:
            # Demote on high overhead
            attr_name = self._MODULE_MAP[module]
            setattr(self.state, attr_name, False)
            return
        
        if stat.median_k_used_delta() > 0:
            # Demote on increased k_used
            attr_name = self._MODULE_MAP[module]
            setattr(self.state, attr_name, False)
            return
    
    def stats(self) -> dict:
        """Export all statistics as dict.
        
        Returns:
            Dict with per-module stats and enable flags
        """
        return {
            "stageA": {
                **self._stats["stageA"].to_dict(),
                "enabled": self.state.enabled_attn_stageA,
            },
            "attn": {
                **self._stats["attn"].to_dict(),
                "enabled": self.state.enabled_attn_sprt,
            },
            "logits": {
                **self._stats["logits"].to_dict(),
                "enabled": self.state.enabled_logits,
            },
        }

