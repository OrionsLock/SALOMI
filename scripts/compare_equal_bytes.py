#!/usr/bin/env python3
"""Compare Phase 0 & Phase 1 harness JSON records at equal total model bytes.

Reads ONLY measured JSON records written by ``onebit/eval/harness.py`` (never
hardcodes perplexity numbers) and optionally prints an analytical GPT-2 family
byte-budget table showing exact equal-byte pairings between larger ~1-bit/ternary
models and smaller GPTQ INT3/INT4 baselines.

Examples:
    python scripts/compare_equal_bytes.py
    python scripts/compare_equal_bytes.py --sort linear
    python scripts/compare_equal_bytes.py --dirs results/phase0 results/phase1 --budget-analysis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onebit.eval.harness import REPO_ROOT  # noqa: E402

FP_META_BITS = 16.0
TERNARY_BITS = 1.6


# ---------------------------------------------------------------------------
# Measured JSON record loading & Pareto frontier
# ---------------------------------------------------------------------------


@dataclass
class MeasuredRecord:
    run_name: str
    source_path: str
    model: str
    ppl: float
    bpp_linear: float
    bpp_whole_model: float
    linear_params: int
    total_params: int
    embed_params: int
    bpp_embed: Optional[float]
    full_test_set: bool
    commit: str

    @property
    def linear_bytes(self) -> float:
        return self.linear_params * self.bpp_linear / 8.0

    @property
    def whole_bytes(self) -> float:
        return self.total_params * self.bpp_whole_model / 8.0

    @property
    def linear_mb(self) -> float:
        return self.linear_bytes / 1e6

    @property
    def whole_mb(self) -> float:
        return self.whole_bytes / 1e6


def load_harness_records(dirs: List[str]) -> List[MeasuredRecord]:
    records: List[MeasuredRecord] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if "result" not in raw or "protocol" not in raw:
                continue
            res = raw["result"]
            proto = raw["protocol"]
            cfg = raw.get("config", {})
            env = raw.get("env", {})
            commit = env.get("git_commit", "unknown")[:8]
            if env.get("git_dirty"):
                commit += "+dirty"
            records.append(
                MeasuredRecord(
                    run_name=raw["run_name"],
                    source_path=os.path.relpath(path, REPO_ROOT),
                    model=str(cfg.get("model", "gpt2")),
                    ppl=float(res["ppl"]),
                    bpp_linear=float(res["bpp_linear"]),
                    bpp_whole_model=float(res["bpp_whole_model"]),
                    linear_params=int(res["linear_params"]),
                    total_params=int(res["total_params"]),
                    embed_params=int(res.get("embed_params", 0)),
                    bpp_embed=float(res["bpp_embed"]) if "bpp_embed" in res else None,
                    full_test_set=bool(proto.get("full_test_set", False)),
                    commit=commit,
                )
            )
    return records


def pareto_frontier_names(records: List[MeasuredRecord], key: str = "whole") -> set:
    """Return run names on the lower (size, ppl) Pareto frontier among full-test-set runs."""
    full_runs = [r for r in records if r.full_test_set]
    size_fn = (lambda r: r.whole_bytes) if key == "whole" else (lambda r: r.linear_bytes)
    full_runs.sort(key=lambda r: (size_fn(r), r.ppl))
    frontier = set()
    best_ppl = float("inf")
    for r in full_runs:
        if r.ppl < best_ppl:
            frontier.add(r.run_name)
            best_ppl = r.ppl
    return frontier


def format_measured_table(records: List[MeasuredRecord], sort_by: str = "whole") -> str:
    if not records:
        return "_No harness JSON records found._"

    fp_ref: Dict[str, float] = {}
    for r in records:
        if r.full_test_set and r.run_name.startswith(("fp32_", "fp16_")):
            fp_ref.setdefault(r.model, r.ppl)

    frontier = pareto_frontier_names(records, key=sort_by)
    if sort_by == "linear":
        ordered = sorted(records, key=lambda r: (r.linear_bytes, r.ppl))
    else:
        ordered = sorted(records, key=lambda r: (r.whole_bytes, r.ppl))

    lines = [
        f"## Measured Harness Results (sorted by {sort_by}-model bytes; `*` = full-test Pareto frontier)",
        "",
        "| Run | Model | Whole MB | Linear MB | bpp (lin) | bpp (whole) | WikiText-2 PPL | × FP | Full | Commit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in ordered:
        star = " *" if r.run_name in frontier else ""
        ref = fp_ref.get(r.model) if r.full_test_set else None
        ratio = f"{r.ppl / ref:.2f}×" if ref else "—"
        full_s = "yes" if r.full_test_set else "no"
        lines.append(
            f"| `{r.run_name}`{star} | `{r.model}` | {r.whole_mb:7.1f} | {r.linear_mb:7.1f} | "
            f"{r.bpp_linear:6.3f} | {r.bpp_whole_model:6.3f} | {r.ppl:12,.2f} | {ratio:>6} | {full_s} | `{r.commit}` |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analytical GPT-2 family storage-byte calculator (exact architecture geometry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPT2Arch:
    name: str
    n_layer: int
    d_model: int
    vocab_size: int = 50257
    n_ctx: int = 1024

    @property
    def linear_params(self) -> int:
        # Per block: c_attn (d, 3d), c_proj (d, d), c_fc (d, 4d), c_proj (4d, d) -> 12 * d^2
        return self.n_layer * 12 * self.d_model * self.d_model

    @property
    def wte_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def wpe_params(self) -> int:
        return self.n_ctx * self.d_model

    @property
    def embed_params(self) -> int:
        return self.wte_params + self.wpe_params

    @property
    def norm_and_bias_params(self) -> int:
        # Per block: biases (3d + d + 4d + d = 9d) + 2 LayerNorms (4d) = 13d; plus final LN (2d)
        return (13 * self.n_layer + 2) * self.d_model

    @property
    def total_params(self) -> int:
        return self.linear_params + self.embed_params + self.norm_and_bias_params

    def linear_bits(self, scheme: str) -> float:
        d = self.d_model
        L = self.n_layer
        # Output/input shapes per block: (3d, d), (d, d), (4d, d), (d, 4d)
        shapes = [(3 * d, d), (d, d), (4 * d, d), (d, 4 * d)]
        bits = 0.0
        for out_f, in_f in shapes:
            n = out_f * in_f
            if scheme == "fp16":
                b = n * 16.0
            elif scheme == "gptq_int4_g128":
                b = n * 4.0 + (out_f * ((in_f + 127) // 128)) * 2.0 * FP_META_BITS
            elif scheme == "gptq_int3_g128":
                b = n * 3.0 + (out_f * ((in_f + 127) // 128)) * 2.0 * FP_META_BITS
            elif scheme == "ternary_perchan":
                b = n * TERNARY_BITS + out_f * FP_META_BITS
            elif scheme == "binary_onebit":
                b = n * 1.0 + (out_f + in_f) * FP_META_BITS
            else:
                raise ValueError(f"Unknown scheme: {scheme}")
            bits += L * b
        return bits

    def embed_bits(self, embed_bits: int) -> float:
        if embed_bits == 16:
            return self.embed_params * 16.0
        # Per-row symmetric INTb: b bits/weight + 16-bit scale per row (wte + wpe)
        rows = self.vocab_size + self.n_ctx
        return self.embed_params * float(embed_bits) + rows * FP_META_BITS

    def storage_summary(self, scheme: str, embed_bits: int = 16) -> Dict[str, float]:
        l_bits = self.linear_bits(scheme)
        e_bits = self.embed_bits(embed_bits)
        other_bits = self.norm_and_bias_params * 16.0
        tot_bits = l_bits + e_bits + other_bits
        return {
            "bpp_linear": l_bits / self.linear_params,
            "bpp_whole": tot_bits / self.total_params,
            "linear_mb": l_bits / 8.0 / 1e6,
            "whole_mb": tot_bits / 8.0 / 1e6,
            "embed_share_pct": 100.0 * self.embed_params / self.total_params,
        }


GPT2_FAMILY = [
    GPT2Arch("gpt2", n_layer=12, d_model=768),
    GPT2Arch("gpt2-medium", n_layer=24, d_model=1024),
    GPT2Arch("gpt2-large", n_layer=36, d_model=1280),
    GPT2Arch("gpt2-xl", n_layer=48, d_model=1600),
]


def format_budget_analysis() -> str:
    schemes: List[Tuple[str, str, int]] = [
        ("FP16", "fp16", 16),
        ("GPTQ INT4 g128 (FP16 emb)", "gptq_int4_g128", 16),
        ("GPTQ INT3 g128 (FP16 emb)", "gptq_int3_g128", 16),
        ("Ternary QAT (FP16 emb)", "ternary_perchan", 16),
        ("Ternary QAT (INT8 emb)", "ternary_perchan", 8),
        ("Ternary QAT (INT4 emb)", "ternary_perchan", 4),
        ("Binary OneBit QAT (FP16 emb)", "binary_onebit", 16),
        ("Binary OneBit QAT (INT8 emb)", "binary_onebit", 8),
        ("Binary OneBit QAT (INT4 emb)", "binary_onebit", 4),
    ]

    lines = [
        "## Analytical Equal-Byte Budget Table (GPT-2 Family, exact storage bits)",
        "",
        "Sizes are in decimal MB (10^6 bytes), computed from exact GPT-2 parameter geometry.",
        "No PPLs are shown here; run `onebit/eval/harness.py` or `scripts/train_kd_qat.py` to populate measured quality.",
        "",
        "| Model | Params | Emb % | Config | bpp (lin) | bpp (whole) | Linear MB | Whole MB |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for arch in GPT2_FAMILY:
        for label, scheme, eb in schemes:
            s = arch.storage_summary(scheme, embed_bits=eb)
            lines.append(
                f"| `{arch.name}` | {arch.total_params / 1e6:6.1f}M | {s['embed_share_pct']:4.1f}% | "
                f"{label} | {s['bpp_linear']:6.3f} | {s['bpp_whole']:6.3f} | "
                f"{s['linear_mb']:7.1f} | {s['whole_mb']:7.1f} |"
            )

    # Highlight concrete equal-byte pairings: larger ~1-bit/ternary model vs smaller GPTQ INT4/INT3.
    g_small, g_med, g_large = GPT2_FAMILY[0], GPT2_FAMILY[1], GPT2_FAMILY[2]
    small_int4 = g_small.storage_summary("gptq_int4_g128", 16)
    small_int3 = g_small.storage_summary("gptq_int3_g128", 16)
    med_bin_e8 = g_med.storage_summary("binary_onebit", 8)
    med_tern_e8 = g_med.storage_summary("ternary_perchan", 8)
    med_int4 = g_med.storage_summary("gptq_int4_g128", 16)
    large_bin_e8 = g_large.storage_summary("binary_onebit", 8)
    large_tern_e8 = g_large.storage_summary("ternary_perchan", 8)

    def _pct_smaller(a: float, ref: float) -> str:
        return f"{100.0 * (ref - a) / ref:.0f}% smaller"

    lines.extend([
        "",
        "### Key Equal-Byte Matchups (Larger Low-Bit Model vs Smaller GPTQ INT4/INT3)",
        "",
        f"- **`gpt2` (124M) GPTQ INT4 g128**: {small_int4['whole_mb']:.1f} MB whole ({small_int4['linear_mb']:.1f} MB linear) vs",
        f"  - **`gpt2-medium` (355M) Binary OneBit + INT8 emb**: {med_bin_e8['whole_mb']:.1f} MB whole ({med_bin_e8['linear_mb']:.1f} MB linear) — **{_pct_smaller(med_bin_e8['whole_mb'], small_int4['whole_mb'])} than 124M INT4!**",
        f"  - **`gpt2-medium` (355M) Ternary + INT8 emb**: {med_tern_e8['whole_mb']:.1f} MB whole ({med_tern_e8['linear_mb']:.1f} MB linear) — **{_pct_smaller(med_tern_e8['whole_mb'], small_int4['whole_mb'])} than 124M INT4!**",
        f"- **`gpt2` (124M) GPTQ INT3 g128**: {small_int3['whole_mb']:.1f} MB whole ({small_int3['linear_mb']:.1f} MB linear) vs",
        f"  - **`gpt2-medium` (355M) Binary OneBit + INT8 emb**: {med_bin_e8['whole_mb']:.1f} MB whole — **{_pct_smaller(med_bin_e8['whole_mb'], small_int3['whole_mb'])} than 124M INT3!**",
        f"- **`gpt2-medium` (355M) GPTQ INT4 g128**: {med_int4['whole_mb']:.1f} MB whole ({med_int4['linear_mb']:.1f} MB linear) vs",
        f"  - **`gpt2-large` (774M) Binary OneBit + INT8 emb**: {large_bin_e8['whole_mb']:.1f} MB whole ({large_bin_e8['linear_mb']:.1f} MB linear) — **{_pct_smaller(large_bin_e8['whole_mb'], med_int4['whole_mb'])} than 355M INT4!**",
        f"  - **`gpt2-large` (774M) Ternary + INT8 emb**: {large_tern_e8['whole_mb']:.1f} MB whole ({large_tern_e8['linear_mb']:.1f} MB linear) — **{_pct_smaller(large_tern_e8['whole_mb'], med_int4['whole_mb'])} than 355M INT4!**",
    ])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dirs",
        nargs="+",
        default=[
            os.path.join(REPO_ROOT, "results", "phase0"),
            os.path.join(REPO_ROOT, "results", "phase1"),
        ],
        help="directories containing canonical harness JSON files",
    )
    p.add_argument("--sort", choices=["whole", "linear"], default="whole")
    p.add_argument("--no-budget-analysis", action="store_true")
    a = p.parse_args(argv)

    records = load_harness_records(a.dirs)
    print(format_measured_table(records, sort_by=a.sort))
    if not a.no_budget_analysis:
        print()
        print(format_budget_analysis())


if __name__ == "__main__":
    main()
