from __future__ import annotations

import numpy as np

from onebit.ops.logits_sprt import shortlist_and_certify
from onebit.runtime.ctg_grammar import CTG, CTGRule, CTGState


def test_ctg_variance_smoke_no_assert() -> None:
    """Smoke-check that CTG wiring does not crash and reports variance.

    This is intentionally *not* a hard gate on variance reduction; it just
    exercises the plumbing and prints the observed variances for manual
    inspection when needed.
    """
    rng = np.random.default_rng(42)

    d = 256
    vocab_size = 64
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
        prf_seed=123,
        use_ctg=0,
    )

    # Baseline without CTG
    base_runs = []
    for _ in range(4):
        base_runs.append(shortlist_and_certify(q_bits, v_ids, **kwargs))

    # Simple CTG that inhibits a fixed subset of IDs
    inhibit_ids = np.arange(0, vocab_size, 2, dtype=np.int32)
    ctg = CTG(
        rules=[CTGRule(op="INHIBIT", ids=inhibit_ids, period=8, prob_num=1, prob_den=1)],
        vocab_size=vocab_size,
        phase_period=8,
    )
    state = CTGState(phase=0, mask_digest=0)

    ctg_runs = []
    for _ in range(4):
        state = CTGState(phase=0, mask_digest=0)
        ctg_runs.append(shortlist_and_certify(q_bits, v_ids, ctg=ctg, ctg_state=state, **kwargs))

    def _extract_top1_scores(results: list[dict]) -> np.ndarray:
        vals = []
        for r in results:
            top1 = r.get("top1")
            if top1 is not None:
                vals.append(top1)
        return np.asarray(vals, dtype=np.int32)

    base_vals = _extract_top1_scores(base_runs)
    ctg_vals = _extract_top1_scores(ctg_runs)

    # The exact numbers are not important; this is primarily a regression
    # sentinel that we can instrument when needed.
    if base_vals.size and ctg_vals.size:
        base_var = float(np.var(base_vals))
        ctg_var = float(np.var(ctg_vals))
        print(f"[CTG variance smoke] base_var={base_var:.6f}, ctg_var={ctg_var:.6f}")

