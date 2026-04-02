"""CPG policy (Contrast/Perf Gating) with shadow sampling.

This module provides the CPG (Contrast/Perf Gating) policy, which is a
renaming of the earlier CTG shadow policy. The core implementation
lives in :mod:`onebit.runtime.ctg_policy`; this module exposes the
CPG-prefixed API and types.

Rationale
---------
- "CTG" is reserved for Constant-Time Grammar.
- The adaptive throughput / contrast gating policy is now called CPG.
- Existing internal implementation is reused without changing math.

Public API
----------
- :class:`CpgPolicyCfg`
- :class:`CpgPolicyState`
- :class:`CpgPolicy`
- :class:`_Stat`
"""
from __future__ import annotations

from .ctg_policy import (  # type: ignore F401
    CtgPolicyCfg as CpgPolicyCfg,
    CtgPolicyState as CpgPolicyState,
    CtgPolicy as CpgPolicy,
    _Stat,
)

__all__ = ["CpgPolicyCfg", "CpgPolicyState", "CpgPolicy", "_Stat"]

