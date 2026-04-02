from __future__ import annotations

import numpy as np

from onebit.ops.logits_sprt import shortlist_and_certify
from onebit.runtime.ctg_grammar import CTG, CTGRule, CTGState


def test_ctg_never_increases_pairs_evaluated() -> None:
    """With fixed shortlist, CTG pruning must not increase work.

    We compare pairs_evaluated with and without CTG on the same synthetic
    problem. CTG is configured to inhibit half of the shortlist IDs.
    """
    rng = np.random.default_rng(123)

    d = 256
    vocab_size = 64
    d_words = d // 32

    # Synthetic packed query bits (values are irrelevant beyond determinism)
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
        prf_seed=999,
        use_ctg=0,
    )

    # Baseline without CTG grammar
    base = shortlist_and_certify(q_bits, v_ids, **kwargs)
    base_pairs = int(base.get("pairs_evaluated", 0))

    # Now build a CTG that inhibits every other ID in the baseline shortlist
    baseline_shortlist = base["shortlist"]
    inhibit_ids = baseline_shortlist[::2]

    ctg = CTG(
        rules=[CTGRule(op="INHIBIT", ids=inhibit_ids, period=8, prob_num=1, prob_den=1)],
        vocab_size=vocab_size,
        phase_period=8,
    )
    state = CTGState(phase=0, mask_digest=0)

    with_ctg = shortlist_and_certify(q_bits, v_ids, ctg=ctg, ctg_state=state, **kwargs)
    ctg_pairs = int(with_ctg.get("pairs_evaluated", 0))

    # CTG is only allowed to reduce work (or keep it equal in degenerate cases)
    assert ctg_pairs <= base_pairs

