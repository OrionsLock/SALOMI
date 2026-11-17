#!/usr/bin/env python3
"""SALOMI public baseline reproducer.

Thin wrapper around the shared evaluation harness (:mod:`onebit.eval.harness`).
Runs a small, fixed set of configurations on GPT-2 124M and prints a table built
*only* from the records returned by ``harness.run`` -- there are no hardcoded
numbers in this script.  Each run's JSON record (git commit, config, environment)
is written by the harness to ``results/phase0/`` (nothing is written to the repo
root).

Configurations:
  * fp16                         -- reference
  * rtn   w4 g128                -- standard 4-bit RTN baseline
  * rtn   w3 g128                -- standard 3-bit RTN baseline
  * ternary per-channel          -- BitNet-b1.58-style absmean ternary (PTQ)
  * binary  per-channel          -- sign * mean|w| per output row (PTQ)
  * legacy_mixed                 -- SALOMI's historical "Mixed (L0/11 protected)" config

Protocol, bit accounting and caveats are documented in ``onebit/eval/harness.py``.

Usage:
    python -m onebit.repro.run_public_baseline                 # full WikiText-2 test set
    python -m onebit.repro.run_public_baseline --max-windows 8 # quick smoke run (not reportable)
    python -m onebit.repro.run_public_baseline --only fp16,rtn_w4_g128
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from onebit.eval import harness  # noqa: E402

# (key, label, RunConfig kwargs)
CONFIGS = [
    ("fp16", "FP16", dict(method="fp16")),
    ("rtn_w4_g128", "RTN W4 g128", dict(method="rtn", bits=4, group=128)),
    ("rtn_w3_g128", "RTN W3 g128", dict(method="rtn", bits=3, group=128)),
    ("ternary_perchan", "Ternary absmean (per-channel)", dict(method="ternary", group=0)),
    ("binary_perchan", "Binary absmean (per-channel)", dict(method="binary", group=0)),
    ("legacy_mixed", "SALOMI legacy mixed (L0/11 prot.)", dict(method="legacy_mixed", group=0)),
]


def run_baseline(
    max_windows: Optional[int] = None,
    model: str = "gpt2",
    only: Optional[List[str]] = None,
    out_dir: str = harness.DEFAULT_RESULTS_DIR,
) -> List[Dict]:
    selected = [c for c in CONFIGS if not only or c[0] in only]
    if only:
        unknown = set(only) - {c[0] for c in CONFIGS}
        if unknown:
            raise ValueError(f"Unknown config(s) {sorted(unknown)}; choose from {[c[0] for c in CONFIGS]}")

    t0 = time.time()
    rows = []
    for key, label, kw in selected:
        cfg = harness.RunConfig(model=model, max_windows=max_windows, out_dir=out_dir, **kw)
        record = harness.run(cfg)
        rows.append((label, record))

    print()
    print("=" * 92)
    scope = "FULL WikiText-2 test set" if max_windows is None else f"first {max_windows} windows ONLY (not reportable)"
    print(f"SALOMI public baseline -- {model} -- {scope}")
    print("=" * 92)
    print(f"{'Method':<36} {'PPL':>12} {'bpp_linear':>11} {'bpp_whole':>10} {'tokens':>9}  record")
    print("-" * 92)
    for label, rec in rows:
        r = rec["result"]
        print(
            f"{label:<36} {r['ppl']:>12.3f} {r['bpp_linear']:>11.4f} {r['bpp_whole_model']:>10.4f} "
            f"{rec['protocol']['scored_tokens']:>9d}  {rec['run_name']}.json"
        )
    print("-" * 92)
    print(f"Records written to {os.path.relpath(out_dir, harness.REPO_ROOT)}/  ({time.time() - t0:.0f}s total)")
    return [rec for _, rec in rows]


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-windows", type=int, default=None,
                   help="debug only: cap on 1024-token eval windows (default: full test set)")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--only", default=None,
                   help="comma-separated subset of: " + ",".join(c[0] for c in CONFIGS))
    p.add_argument("--out-dir", default=harness.DEFAULT_RESULTS_DIR)
    a = p.parse_args(argv)
    only = [s.strip() for s in a.only.split(",")] if a.only else None
    run_baseline(max_windows=a.max_windows, model=a.model, only=only, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
