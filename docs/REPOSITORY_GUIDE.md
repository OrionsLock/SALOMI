# SALOMI Repository Guide

This document is the curated technical guide to the repository layout, main methods, and where to look first.

For the full repo-level interpretation of claims and maturity, read `RESEARCH.md` first. For the strongest evaluation caveats, read `docs/HONEST_ASSESSMENT.md`.

---

## Project Summary

SALOMI is a research codebase for **extreme low-bit transformer quantization and inference**. The repository explores whether binary or near-binary parameterizations can approach or exceed ternary baselines under realistic constraints.

The validated high-level takeaways are:

- **strict post-hoc 1.00 bpp binary does not hold up** as a strong GPT-2–class language-modeling solution,
- **correlation is not the same as perplexity**,
- **Hessian-guided VQ and magnitude-aware methods** are the most credible practical directions in this codebase,
- and the most plausible practical range is closer to **~1.2-1.35 bpp** than true 1.00 bpp.

---

## Repository Layout

```text
SALOMI/
├── README.md
├── RESEARCH.md
├── REPRODUCIBILITY.md
├── CONTRIBUTING.md
├── LICENSE
├── onebit/
├── docs/
├── tests/
└── root-level research scripts and result artifacts
```

### `onebit/`

The main package. Key areas include:

- `onebit/core/` — bit packing, bpp accounting, transforms, core math utilities
- `onebit/ops/` — low-bit operators such as BSDM-W and fast VQ paths
- `onebit/quantization/` — quantization algorithms such as `HessianVQ`
- `onebit/model/` — GPT-2 quantization and runtime transformer logic
- `onebit/eval/` — evaluation helpers such as perplexity support
- `onebit/backends/` — optional backend implementations including OpenCL
- `onebit/autotune/` — tuning/performance helpers
- `onebit/attn/` — attention-related certification and runner code
- `onebit/research/` — experiment scripts, analysis, and historical paper-style drafts

### `docs/`

Curated narrative documents. Recommended order:

1. `docs/HONEST_ASSESSMENT.md`
2. `docs/PROJECT_ANALYSIS_SUMMARY.md`
3. `docs/RIGOROUS_TESTING_PLAN.md`
4. `docs/CORRELATION_FINDINGS.md`
5. `docs/ARCHIVE.md` for historical experiment naming/context

### `tests/`

Validation-oriented code. This includes:

- strict bpp tests,
- correlation tests,
- perplexity tests,
- speed benchmarks,
- runtime/backend parity tests,
- attention/certification tests,
- and stress/soak tests.

---

## Main Methods

### HessianVQ

Primary quantization direction based on block/vector quantization with Hessian-motivated weighting.

Relevant files:

- `onebit/quantization/hessian_vq.py`
- `onebit/quantization/functional.py`
- `onebit/ops/vq_optimized.py`

### BSDM-W

Binary stochastic dot-product machinery for low-bit runtime inference.

Relevant files:

- `onebit/ops/bsdm_w.py`
- `onebit/model/runtime_transformer.py`

### Calibration and magnitude recovery

Research directions that restore magnitude information on top of binary signs.

Relevant files:

- `onebit/research/calibration_scaling.py`
- `onebit/research/gelu_aware.py`
- `onebit/research/calibrated_binary_v2.py`

---

## Evaluation Priorities

This repository distinguishes between metrics that are often conflated:

- **per-layer correlation**
- **end-to-end hidden-state behavior**
- **perplexity**
- **strict bits per parameter**

For this project, those metrics are not interchangeable. A method that looks good on correlation may still fail badly on perplexity.

---

## Best Entry Points for Readers

If you want to understand the project quickly:

1. `README.md`
2. `RESEARCH.md`
3. `docs/HONEST_ASSESSMENT.md`
4. `docs/PROJECT_ANALYSIS_SUMMARY.md`
5. `tests/test_bpp_strict.py`
6. `tests/test_perplexity_real.py`
7. `onebit/model/runtime_transformer.py`

If you want to inspect the historical research trail, then move into `onebit/research/` after reading the curated materials above.

---

## Naming Convention Note

Some files under `onebit/research/` use chronological names such as `novel_ideas_v*.py`. These are preserved as part of the research history. They should be read as experiment lineage, not as the polished public API of the repository.
