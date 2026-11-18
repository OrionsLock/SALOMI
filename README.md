# SALOMI

SALOMI is a research repository focused on **extreme low-bit transformer quantization and inference**, especially the question of whether **binary or near-binary** weight representations can approach or exceed **ternary** baselines under realistic evaluation.

---

## Validated Status

> [!IMPORTANT]
> Earlier numbers in this repo (FP32 PPL = 5.92, "~1,200× FP32 at 1.1 bpp", "158,000×", INT4 PPL = 8,812, …) came from a
> repeated-paragraph evaluation text and broken baselines. They are **retracted**; see [`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md).

| | |
|---|---|
| **Task** | Causal language modeling, GPT-2 124M |
| **Dataset / metric** | WikiText-2-raw **test** set, 1024-token non-overlapping windows, token-weighted perplexity (FP32 = 29.95) |
| **Evaluation code** | One shared harness: [`onebit/eval/harness.py`](onebit/eval/harness.py). Every run writes a JSON record (commit, config, PPL, bpp) to `results/` |
| **Bitrate accounting** | Reported two ways: `bpp_linear` (transformer-block weights incl. scales, zero-points, codebooks, factors) and `bpp_whole` (plus embeddings/LayerNorm/biases at 16 bits) |
| **Baselines** | RTN (per-channel / g128), real GPTQ, ternary absmean, binary absmean, OneBit SVID init |
| **What failed** | Every **post-training** method at ~1–1.6 bpp (binary, ternary, HVQ, low-rank residual, mixed precision) gives PPL in the thousands or worse on GPT-2 124M |
| **Current conclusion** | ~1-bit GPT-2 requires **training** (QAT / distillation from the FP model). A preliminary CPU proof-of-concept already reaches PPL ≈ 210 at 1.03 bpp; a proper GPU run is the next phase |

See [`CURRENT_STATE.md`](CURRENT_STATE.md) for the full table and what-works / what-doesn't, and [`docs/VALIDATED_RESULTS.md`](docs/VALIDATED_RESULTS.md) for the canonical numbers.

---

## Reproducing the results

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One configuration (~8 min on CPU for the full test set; --max-windows 20 for a quick check)
python -m onebit.eval.harness --method gptq --bits 3 --group 128

# Full Phase 0 baseline suite (~2–3 h on a 2-core CPU; resumable, skips finished configs)
python scripts/run_phase0_baselines.py
python scripts/run_phase0_baselines.py --table     # markdown table from results/phase0/*.json

# Unit tests for the harness, baselines, GPTQ and bpp accounting
pytest tests/test_phase0_correctness.py tests/test_bpp_guard.py tests/test_bpp_strict.py -q
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for environment notes and reporting rules.

---

## Repository Contents

- `onebit/` — quantization, runtime inference, evaluation, kernels, and tooling
- `tests/` — validation and experimentation test tree
- `docs/` — research writeups (see folder split below)
- `onebit/research/paper/` — historical paper-style materials (earlier, more optimistic draft claims; superseded by later docs)

---

## Document Map

| Document | What it answers |
|---|---|
| [`CURRENT_STATE.md`](CURRENT_STATE.md) | What works, what doesn't, what is exploratory, what is deprecated |
| [`docs/VALIDATED_RESULTS.md`](docs/VALIDATED_RESULTS.md) | Canonical benchmark table — one source of truth for numbers |
| [`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md) | Why earlier numbers were retracted; which code was fixed; rules for new results |
| [`results/`](results/) | Machine-readable JSON for every reported run (`results/phase0/`, `results/experiments/`) |
| [`RESEARCH.md`](RESEARCH.md) | Index of research questions, evidence locations, and main code paths |
| [`docs/HONEST_ASSESSMENT.md`](docs/HONEST_ASSESSMENT.md) | Direct writeup of where binary results fail and why |
| [`docs/ARCHIVE.md`](docs/ARCHIVE.md) | Explanation of historical experiment files and naming |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Environment, rerun guidance, reporting standards |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution and repo hygiene expectations |

### docs/ folder split

| Folder / file | Status |
|---|---|
| `docs/VALIDATED_RESULTS.md` | Validated — canonical numbers (harness JSON only) |
| `docs/PHASE0_AUDIT.md` | Validated — retractions and fixes |
| `docs/HONEST_ASSESSMENT.md` | Validated qualitatively; its old PPL tables are superseded |
| `docs/CORRELATION_FINDINGS.md` | Qualitative point (correlation ≠ PPL) stands; numbers predate the harness |
| `docs/FIXES_IMPLEMENTED.md` | Historical — some "fixes" described there were incomplete (see audit) |
| `docs/IMPLEMENTATION_SUMMARY.md` | Current — implementation reference |
| `docs/REPOSITORY_GUIDE.md` | Current — layout reference |
| `docs/PROJECT_ANALYSIS_SUMMARY.md` | Exploratory — broader analysis, some cautious claims |
| `docs/RIGOROUS_TESTING_PLAN.md` | Exploratory — planning document |
| `docs/BINARY_QUANTIZATION_FINDINGS.md` | Archive — earlier findings, partially superseded |
| `docs/SALOMI-RESEARCH-FINDINGS.md` | Archive — early research notes |
| `docs/SALOMI-PROJECT-PLAN.md` | Archive — original project plan |
| `onebit/research/paper/` | Archive — draft paper materials, earlier optimistic claims |

---

## Note on Development History

This repository was developed primarily offline in a private working directory and later published as a consolidated research snapshot. The low public commit count reflects publication history, not lack of iteration. Where earlier claims were revised after stricter evaluation, the corrections are documented explicitly in `docs/PHASE0_AUDIT.md`, `docs/HONEST_ASSESSMENT.md`, `docs/VALIDATED_RESULTS.md`, and summarized with pointers in `RESEARCH.md`.

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m onebit.eval.harness --method fp32 --max-windows 20   # quick sanity check (~1 min)
```

Notes:
- `pyopencl` is optional (OpenCL backend only).
- GPT-2 and WikiText-2 are downloaded from the Hugging Face Hub on first use.
- For the guided overview, read `CURRENT_STATE.md` before opening historical experiment scripts.

## Recommended Reading Order

1. `README.md` (this file)
2. `CURRENT_STATE.md` — what works and what doesn't, right now
3. `docs/VALIDATED_RESULTS.md` — canonical benchmark numbers
4. `docs/PHASE0_AUDIT.md` — what was retracted and why
5. `RESEARCH.md` — index of questions, docs, and implementation locations
6. `docs/HONEST_ASSESSMENT.md` — documented failure modes

## Naming Note

Some filenames under `onebit/research/` preserve chronology rather than ideal taxonomy. Names like `novel_ideas_v*.py` are part of the research trail. Prioritize summary documents and validated test paths over historical experiment filenames.

## License

This repository is licensed under **Apache-2.0**. See `LICENSE`.
