# SALOMI

SALOMI is a research repository focused on **extreme low-bit transformer quantization and inference**, especially the question of whether **binary or near-binary** weight representations can approach or exceed **ternary** baselines under realistic evaluation.

---

## Validated Status

| | |
|---|---|
| **Task** | Causal language modeling (GPT-2 124M; Pythia 6.9B probed) |
| **Dataset** | WikiText / calibration sentences; GPT-2 evaluation text |
| **Primary metric** | End-to-end perplexity (PPL). Correlation is logged but not treated as sufficient evidence. |
| **Bitrate accounting** | Strict: codebook indices + codebook storage + sign bits + metadata all counted. |
| **Best validated result** | ~1.1–1.2 bpp with mixed-precision Hessian-guided VQ (PPL ratio ~1,200× worse than FP32 on 124M; larger models expected to degrade less). |
| **What failed** | Strict 1.00 bpp post-hoc binary quantization is not viable for GPT-2-class LM quality (PPL 158,000× worse than FP32). |
| **Current conclusion** | On GPT-2-class language modeling, strict 1.00 bpp post-hoc binary does not currently preserve usable perplexity under this repo's evaluation. Practical promise sits at ~1.1–1.35 bpp using Hessian-weighted VQ and mixed-precision layer allocation. Proxy metrics like hidden-state correlation are not treated as sufficient evidence and have been explicitly documented as misleading. |

See [`CURRENT_STATE.md`](CURRENT_STATE.md) for a full what-works / what-doesn't breakdown, and [`docs/VALIDATED_RESULTS.md`](docs/VALIDATED_RESULTS.md) for the canonical benchmark table.

---

## One-Command Reproducibility

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python onebit/repro/run_public_baseline.py
```

Expected output: correlation and PPL table for GPT-2 124M, finishing in ~5 minutes on CPU. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for full environment notes.

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
| [`RESEARCH.md`](RESEARCH.md) | Index of research questions, evidence locations, and main code paths |
| [`docs/HONEST_ASSESSMENT.md`](docs/HONEST_ASSESSMENT.md) | Direct writeup of where binary results fail and why |
| [`docs/ARCHIVE.md`](docs/ARCHIVE.md) | Explanation of historical experiment files and naming |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Environment, rerun guidance, reporting standards |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution and repo hygiene expectations |

### docs/ folder split

| Folder / file | Status |
|---|---|
| `docs/VALIDATED_RESULTS.md` | Validated — canonical numbers |
| `docs/HONEST_ASSESSMENT.md` | Validated — confirmed failure analysis |
| `docs/CORRELATION_FINDINGS.md` | Validated — explains correlation/PPL gap |
| `docs/FIXES_IMPLEMENTED.md` | Validated — documents algorithm corrections |
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

This repository was developed primarily offline in a private working directory and later published as a consolidated research snapshot. The low public commit count reflects publication history, not lack of iteration. Where earlier claims were revised after stricter evaluation, the corrections are documented explicitly in `docs/HONEST_ASSESSMENT.md`, `docs/VALIDATED_RESULTS.md`, and summarized with pointers in `RESEARCH.md`.

---

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python onebit/repro/run_public_baseline.py
```

Notes:
- `pyopencl` is optional (OpenCL backend only).
- Some research scripts require HuggingFace model downloads.
- For the guided overview, read `CURRENT_STATE.md` before opening historical experiment scripts.

## Recommended Reading Order

1. `README.md` (this file)
2. `CURRENT_STATE.md` — what works and what doesn't, right now
3. `docs/VALIDATED_RESULTS.md` — canonical benchmark numbers
4. `RESEARCH.md` — index of questions, docs, and implementation locations
5. `docs/HONEST_ASSESSMENT.md` — documented failure modes

## Naming Note

Some filenames under `onebit/research/` preserve chronology rather than ideal taxonomy. Names like `novel_ideas_v*.py` are part of the research trail. Prioritize summary documents and validated test paths over historical experiment filenames.

## License

This repository is licensed under **Apache-2.0**. See `LICENSE`.
