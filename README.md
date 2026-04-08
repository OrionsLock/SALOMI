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

## Current validated takeaways

- **Strict 1.00 bpp post-hoc binary does not hold up as a strong GPT-2–class language-modeling solution under the repo's later evaluation docs and tests.**
- **Correlation is not a reliable substitute for perplexity.**
- **More credible practical results in this repo cluster around ~1.2-1.35 bpp**, especially for Hessian-guided VQ and magnitude-recovery methods.
- **Some earlier paper-style claims were revised after stricter bpp accounting and end-to-end checks.**

## Start Here

- `docs/VALIDATED_RESULTS.md` — shortest summary of the repo's narrowest defensible claims
- `RESEARCH.md` — longer repo-level report and technical context
- `docs/HONEST_ASSESSMENT.md` — direct writeup of where binary results fail
- `docs/PROJECT_ANALYSIS_SUMMARY.md` — validation and failure-mode summary
- `docs/REPOSITORY_GUIDE.md` — technical guide to the repository layout
- `docs/ARCHIVE.md` — explanation of historical experiment files and naming
- `REPRODUCIBILITY.md` — environment and rerun guidance
- `CONTRIBUTING.md` — contribution and repo hygiene expectations

## Important Note on Claims

Some materials under `onebit/research/paper/` preserve **earlier, more optimistic draft claims**. For the most defensible current interpretation of the repository, prefer:

- `RESEARCH.md`
- `docs/`
- `tests/`

over historical paper-draft numbers when they conflict.

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

## Naming Note

Some filenames, especially under `onebit/research/`, preserve the chronology of the work rather than an ideal public taxonomy. Names like `novel_ideas_v*.py` are intentionally kept as part of the research trail. Public-facing readers should prioritize the summary documents and validated test paths over historical experiment filenames.

## Recommended Reading Order

1. `README.md`
2. `docs/VALIDATED_RESULTS.md`
3. `RESEARCH.md`
4. `docs/HONEST_ASSESSMENT.md`
5. `docs/PROJECT_ANALYSIS_SUMMARY.md`
6. `docs/REPOSITORY_GUIDE.md`

If you want the corrected, defensible story of the repo, read in that order before opening the historical paper drafts.
