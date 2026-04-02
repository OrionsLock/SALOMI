# SALOMI

SALOMI is a research repository focused on **extreme low-bit transformer quantization and inference**, especially the question of whether **binary or near-binary** weight representations can approach or exceed **ternary** baselines under realistic evaluation.

This repository contains:

- the `onebit/` package for quantization, runtime inference, evaluation, kernels, and related tooling,
- a large `tests/` tree for validation and experimentation,
- research writeups under `docs/`,
- and historical paper-style materials under `onebit/research/paper/`.

## Quick Start

This repository is best treated as a research workspace rather than a one-command product package.

Typical setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Notes:

- `pyopencl` is optional unless you want to explore the OpenCL backend.
- some research scripts expect Hugging Face model/data downloads and may require extra environment setup or credentials depending on your machine state.
- for a guided overview, read `RESEARCH.md` before running older experiment scripts.

## Status

This is a **research repository**, not a polished production package.

The most important repo-level conclusion is:

- **strict 1.00 bpp post-hoc binary quantization does not hold up as a strong GPT-2–class language modeling solution under rigorous evaluation**
- more credible practical results in this repo cluster around **~1.2-1.35 bpp** using Hessian-guided VQ, mixed precision, or magnitude-recovery methods

## Start Here

- `RESEARCH.md` — comprehensive repo-level research report and maturity assessment
- `docs/HONEST_ASSESSMENT.md` — strongest reality-check document
- `docs/PROJECT_ANALYSIS_SUMMARY.md` — validation and failure-mode summary
- `docs/REPOSITORY_GUIDE.md` — curated technical guide to the repository
- `docs/ARCHIVE.md` — explanation of historical experiment files and naming
- `REPRODUCIBILITY.md` — environment and rerun guidance
- `CONTRIBUTING.md` — contribution and repo hygiene expectations

## Important Note on Claims

Some materials under `onebit/research/paper/` preserve **earlier, more optimistic draft claims**. For the most defensible current interpretation of the repository, prefer:

- `RESEARCH.md`
- `docs/`
- `tests/`

over historical paper-draft numbers when they conflict.

## What Makes This Public-Ready

This repo has been curated to improve GitHub readiness:

- `README.md` gives the top-level framing
- `RESEARCH.md` is the comprehensive research report
- `requirements.txt` documents the dependency surface
- `.gitignore` excludes common local caches and transient files
- `LICENSE` now provides clear reuse terms under Apache-2.0

## License

This repository is licensed under **Apache-2.0**. See `LICENSE`.

## Repository Shape

```text
SALOMI/
├── README.md
├── RESEARCH.md
├── onebit/
├── docs/
├── tests/
└── research/result artifacts and experiment scripts
```

## Public Positioning

The strongest honest framing for this project is:

> A serious research and systems exploration of extreme LLM quantization, including both promising methods and rigorous evidence about where naive sub-1-bit claims break down.

## Naming Note

Some filenames, especially under `onebit/research/`, preserve the chronology of the work rather than an ideal public taxonomy. Names like `novel_ideas_v*.py` are intentionally kept as part of the research trail. Public-facing readers should prioritize the curated documents and validated test paths over historical experiment filenames.

## Recommended Reading Order

1. `README.md`
2. `RESEARCH.md`
3. `docs/HONEST_ASSESSMENT.md`
4. `docs/PROJECT_ANALYSIS_SUMMARY.md`
5. `docs/REPOSITORY_GUIDE.md`

If you want the corrected, defensible story of the repo, read in that order before opening the historical paper drafts.
