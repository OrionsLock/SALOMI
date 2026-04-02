from __future__ import annotations

"""Constant-Time Grammar (CTG-FIXED v1).

This module implements a minimal Constant-Time Grammar (CTG) engine with
PASS / INHIBIT / INVERT / PHASE operations over token IDs.

The design follows the CTG-FIXED v1 spec:
- Stateful per sequence via :class:`CTGState` (O(1) state).
- ID-level rules with optional duty cycles.
- Deterministic, seed-free core (PRF hooks can be added later).

The engine is intentionally simple: it produces

    (new_state, mask, invert_flag)

for a given shortlist of token IDs. Integration with HCL / BSDM-W is
handled by the caller.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CTGState:
    """Per-sequence CTG state.

    Attributes
    ----------
    phase:
        Small integer phase counter, advanced once per token.
    mask_digest:
        64-bit rolling digest for determinism / logging.
    """

    phase: int = 0
    mask_digest: int = 0


@dataclass(frozen=True)
class CTGRule:
    """CTG rule over token IDs.

    Parameters
    ----------
    op:
        One of "PASS", "INHIBIT", "INVERT", "PHASE".
    ids:
        Optional int32 array of token IDs this rule applies to. ``None``
        means "all IDs in the shortlist".
    period:
        Period for PHASE / duty-cycle behaviour.
    prob_num, prob_den:
        Rational p = prob_num / prob_den controlling duty cycle. For
        CTG-FIXED v1 this is interpreted deterministically using the
        phase counter (no extra PRF yet).
    """

    op: Literal["PASS", "INHIBIT", "INVERT", "PHASE"]
    ids: Optional[np.ndarray] = None
    period: int = 8
    prob_num: int = 1
    prob_den: int = 1


class CTG:
    """CTG-PROG v1 engine.

    The engine is backend-agnostic and purely combinatorial. Callers are
    expected to thread the returned state across tokens.

    CTG-PROG extends CTG-FIXED with multiple programs: each program is a
    separate rule set. The caller selects a program_id at runtime (via
    program_id_fn) to choose which rule set to apply.
    """

    def __init__(
        self,
        rules: Optional[Sequence[CTGRule]] = None,
        vocab_size: int = 32000,
        class_map: Optional[Dict[str, Iterable[int]]] = None,
        phase_period: int = 8,
        programs: Optional[Sequence[Sequence[CTGRule]]] = None,
    ) -> None:
        """Initialize CTG engine.

        Parameters
        ----------
        rules:
            Single rule set (CTG-FIXED mode). If provided, programs must be None.
        vocab_size:
            Vocabulary size.
        class_map:
            Optional mapping from class names to token ID sets.
        phase_period:
            Phase counter period.
        programs:
            Multiple rule sets (CTG-PROG mode). If provided, rules must be None.
            Each program is a list of CTGRule objects.
        """
        if rules is not None and programs is not None:
            raise ValueError("Cannot specify both rules and programs")

        if programs is not None:
            # CTG-PROG mode: multiple programs
            self.programs = [list(prog) for prog in programs]
            self.rules = []  # Unused in PROG mode
        elif rules is not None:
            # CTG-FIXED mode: single program
            self.rules = list(rules)
            self.programs = [self.rules]  # Wrap as single program
        else:
            # Default: empty PASS-only program
            self.rules = []
            self.programs = [[]]

        self.vocab_size = int(vocab_size)
        self.phase_period = max(1, int(phase_period))
        self.class_map: Dict[str, np.ndarray] = {}
        if class_map:
            for name, ids in class_map.items():
                self.class_map[name] = np.asarray(list(ids), dtype=np.int32)

    # --- helpers -----------------------------------------------------

    @staticmethod
    def _splitmix64(x: int) -> int:
        """Small 64-bit hash for mask_digest updates."""

        x &= 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
        return x & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _ids_mask(shortlist_ids: np.ndarray, ids: Optional[np.ndarray]) -> np.ndarray:
        if ids is None:
            return np.ones_like(shortlist_ids, dtype=bool)
        # Use vectorised membership via broadcasting; shortlist is small.
        return np.isin(shortlist_ids, ids)

    @staticmethod
    def _duty_active(phase: int, rule: CTGRule) -> bool:
        """Deterministic duty-cycle using phase and (prob_num/prob_den).

        This is a simple, seed-free stand-in for a PRF-based schedule.
        Over many phases, the fraction of active phases approaches
        prob_num/prob_den.
        """

        if rule.prob_den <= 0 or rule.prob_num <= 0:
            return False
        slot = phase % max(1, rule.period)
        threshold = (rule.prob_num * max(1, rule.period)) // rule.prob_den
        return slot < threshold

    # --- main API ----------------------------------------------------

    def apply(
        self,
        state: CTGState,
        shortlist_ids: np.ndarray,
        program_id: int = 0,
    ) -> Tuple[CTGState, np.ndarray, bool]:
        """Apply CTG rules to a shortlist.

        Parameters
        ----------
        state:
            Current CTG state.
        shortlist_ids:
            1D int32 array of token IDs.
        program_id:
            Program index (0..K-1). Selects which rule set to apply.
            Default: 0 (backward compatible with CTG-FIXED).

        Returns
        -------
        new_state, mask, invert_flag
            ``mask`` is uint8 array (0/1) aligned with ``shortlist_ids``;
            ``invert_flag`` is a global sign flip indicator for this tick.
        """

        if shortlist_ids.ndim != 1:
            raise ValueError("shortlist_ids must be 1D")

        # Select program
        if not (0 <= program_id < len(self.programs)):
            raise ValueError(f"program_id {program_id} out of range [0, {len(self.programs)})")

        rules = self.programs[program_id]

        phase = (state.phase + 1) % self.phase_period
        mask = np.ones_like(shortlist_ids, dtype=np.uint8)
        invert_flag = False

        # PHASE / INHIBIT / INVERT / PASS ordering
        for rule in rules:
            if not self._duty_active(phase, rule):
                continue
            local_mask = self._ids_mask(shortlist_ids, rule.ids)

            if rule.op == "PHASE":
                # For v1, PHASE does not mutate state beyond global phase.
                continue
            if rule.op == "INHIBIT":
                mask[local_mask] = 0
            elif rule.op == "INVERT":
                if np.any(local_mask):
                    invert_flag = not invert_flag
            elif rule.op == "PASS":
                # PASS is implicit; no-op.
                continue
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unknown CTG op: {rule.op}")

        # Update digest using shortlist IDs, phase, mask, and program_id.
        digest_input = int(shortlist_ids.size ^ phase ^ state.mask_digest ^ program_id)
        digest_input = self._splitmix64(digest_input)
        digest_input ^= int(mask.sum())
        new_digest = self._splitmix64(digest_input)

        new_state = CTGState(phase=phase, mask_digest=new_digest)
        return new_state, mask, invert_flag


# --- CTG-PROG helpers --------------------------------------------------------


def make_default_programs(vocab_size: int, K: int = 4) -> Sequence[Sequence[CTGRule]]:
    """Create K default CTG-PROG programs with distinct biases.

    Parameters
    ----------
    vocab_size:
        Vocabulary size.
    K:
        Number of programs (default: 4).

    Returns
    -------
    programs:
        List of K rule sets, each a list of CTGRule objects.

    Default programs (K=4):
        p=0: PASS-biased (low INHIBIT).
        p=1: PHASE-heavy (maximize carrier diversity).
        p=2: INHIBIT-spiky (aggressively prune tails).
        p=3: INVERT-accent (flip on near-ties).
    """
    if K == 1:
        # Single program: trivial PASS
        return [[CTGRule(op="PASS", ids=None)]]

    if K == 4:
        # p=0: PASS-biased (minimal intervention)
        prog0 = [CTGRule(op="PASS", ids=None)]

        # p=1: PHASE-heavy (maximize carrier diversity)
        prog1 = [
            CTGRule(op="PHASE", ids=None, period=8, prob_num=1, prob_den=1),
            CTGRule(op="PASS", ids=None),
        ]

        # p=2: INHIBIT-spiky (aggressively prune tails)
        # Inhibit 50% of IDs on duty cycle
        prog2 = [
            CTGRule(op="INHIBIT", ids=None, period=8, prob_num=1, prob_den=2),
        ]

        # p=3: INVERT-accent (flip on near-ties)
        prog3 = [
            CTGRule(op="INVERT", ids=None, period=8, prob_num=1, prob_den=4),
        ]

        return [prog0, prog1, prog2, prog3]

    # Fallback: K copies of PASS
    return [[CTGRule(op="PASS", ids=None)] for _ in range(K)]


def default_program_id_fn(h_state: dict) -> int:
    """Stub program selector: always returns 0.

    Parameters
    ----------
    h_state:
        Dictionary with runtime features (e.g., shortlist_entropy, top2_margin,
        head_agreement). For CTG-PROG v1, this is a minimal stub.

    Returns
    -------
    program_id:
        Integer in [0, K-1].
    """
    # Stub: always select program 0 (PASS-biased)
    return 0

