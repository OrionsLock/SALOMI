#!/usr/bin/env python3
"""Run the Phase-0 baseline suite through the shared harness (full WikiText-2 test set).

Resumable: configs whose JSON already exists in results/phase0/ are skipped.

    python scripts/run_phase0_baselines.py              # everything
    python scripts/run_phase0_baselines.py --only gptq  # subset by substring
    python scripts/run_phase0_baselines.py --table      # just print the table from JSON
"""
import argparse
import glob
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onebit.eval.harness import DEFAULT_RESULTS_DIR, RunConfig, run  # noqa: E402

SUITE = [
    RunConfig(method="fp32"),
    RunConfig(method="fp16"),
    RunConfig(method="rtn", bits=8, group=0),
    RunConfig(method="rtn", bits=4, group=0),
    RunConfig(method="rtn", bits=4, group=128),
    RunConfig(method="rtn", bits=3, group=128),
    RunConfig(method="rtn", bits=2, group=128),
    RunConfig(method="gptq", bits=4, group=128),
    RunConfig(method="gptq", bits=3, group=128),
    RunConfig(method="gptq", bits=2, group=128),
    RunConfig(method="ternary", group=0),
    RunConfig(method="ternary_twn", group=0),
    RunConfig(method="binary", group=0),
    RunConfig(method="binary", group=128),
    RunConfig(method="binary_svid", group=0),
    RunConfig(method="legacy_lowrank_r8", group=0),
    RunConfig(method="legacy_mixed", group=0),
]


def print_table(out_dir: str) -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        commit = r["env"]["git_commit"][:8] + ("+dirty" if r["env"].get("git_dirty") else "")
        rows.append((r["result"]["bpp_linear"], r["run_name"], r["result"]["ppl"],
                     r["result"]["bpp_whole_model"], r["protocol"]["full_test_set"], commit))
    rows.sort(key=lambda x: -x[0])
    print(f"| {'run':<38} | {'bpp (linear)':>12} | {'bpp (whole)':>11} | {'WikiText-2 PPL':>14} | full | commit |")
    print(f"|{'-'*40}|{'-'*14}|{'-'*13}|{'-'*16}|------|--------|")
    for bpp, name, ppl, whole, full, commit in rows:
        print(f"| {name:<38} | {bpp:>12.3f} | {whole:>11.3f} | {ppl:>14,.2f} | {'yes' if full else 'no':>4} | {commit} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--out-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.table:
        print_table(a.out_dir)
        return
    for cfg in SUITE:
        cfg.max_windows = a.max_windows
        cfg.out_dir = a.out_dir
        name = cfg.default_name()
        if a.only and a.only not in name:
            continue
        if not a.force and os.path.exists(os.path.join(a.out_dir, f"{name}.json")):
            print(f"[skip] {name} (exists)", flush=True)
            continue
        try:
            run(cfg)
        except Exception:  # keep the suite going, but make the failure loud
            print(f"[FAILED] {name}", flush=True)
            traceback.print_exc()
    print_table(a.out_dir)


if __name__ == "__main__":
    main()
