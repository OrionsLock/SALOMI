from __future__ import annotations

import numpy as np

from onebit.runtime.ctg_grammar import CTG, CTGRule, CTGState


def test_ctg_grammar_deterministic_fixed_inputs() -> None:
    """CTG.apply must be deterministic for fixed inputs/state.

    This is a hard guard: given identical (state, shortlist_ids), we expect
    identical (phase, mask_digest, invert_flag, mask).
    """
    vocab_size = 1024
    rules = [
        CTGRule(op="INHIBIT", ids=np.array([1, 3, 5, 7], dtype=np.int32)),
        CTGRule(op="INVERT", ids=np.array([11, 13], dtype=np.int32)),
        CTGRule(op="PHASE", ids=None, period=8, prob_num=1, prob_den=1),
    ]
    ctg = CTG(rules=rules, vocab_size=vocab_size, phase_period=8)

    shortlist = np.array([0, 1, 2, 3, 11, 13, 42], dtype=np.int32)
    state0 = CTGState(phase=0, mask_digest=0)

    state1, mask1, inv1 = ctg.apply(state0, shortlist)
    state2, mask2, inv2 = ctg.apply(state0, shortlist)

    assert state1.phase == state2.phase
    assert state1.mask_digest == state2.mask_digest
    assert bool(inv1) == bool(inv2)
    assert np.array_equal(mask1, mask2)


def test_ctg_phase_advances_mod_period() -> None:
    """Phase must advance deterministically modulo phase_period."""
    vocab_size = 16
    period = 5
    rules = [CTGRule(op="PASS", ids=None)]
    ctg = CTG(rules=rules, vocab_size=vocab_size, phase_period=period)

    shortlist = np.array([0, 1, 2], dtype=np.int32)
    state = CTGState(phase=0, mask_digest=0)

    phases = []
    for _ in range(2 * period + 3):
        state, _mask, _inv = ctg.apply(state, shortlist)
        phases.append(state.phase)

    # All phases must be in [0, period-1]
    assert all(0 <= p < period for p in phases)

    # We must have seen at least one full cycle
    assert set(phases) == set(range(period))


def test_ctg_zero_bias_over_trivial_pass() -> None:
    """Trivial PASS-only CTG should be exactly unbiased.

    We exercise a few steps just to ensure the plumbing doesn't introduce
    accidental masking or inversion.
    """
    vocab_size = 128
    rules = [CTGRule(op="PASS", ids=None)]
    ctg = CTG(rules=rules, vocab_size=vocab_size, phase_period=8)

    shortlist = np.arange(32, dtype=np.int32)
    state = CTGState(phase=0, mask_digest=0)

    kept_total = 0
    T = 16
    for _ in range(T):
        state, mask, inv = ctg.apply(state, shortlist)
        kept_total += int(mask.sum())
        # PASS rule must never invert
        assert not inv

    mean_kept = kept_total / (T * len(shortlist))
    # Exactly 1.0 for trivial PASS-only CTG
    assert abs(mean_kept - 1.0) < 1e-9

