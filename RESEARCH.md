# SALOMI — research scope and evidence

SALOMI is a codebase for **extreme low-bit post-training quantization** of transformers, with emphasis on honest evaluation (strict bits-per-parameter accounting, end-to-end perplexity, and correlation as a limited proxy).

This file is a **pointer and index**: detailed numbers and arguments live in the linked documents and in code/tests.

## Questions the repo investigates

- How far **1-bit or near–1-bit weight formats** can go on GPT-2–scale models versus **ternary (~1.58 bpp)** and mixed schemes.
- When **layer-wise reconstruction metrics** mislead relative to **full-model perplexity**.
- How **Hessian-aware block VQ**, **low-rank residuals**, and **mixed precision** trade bpp for quality under audited storage.

## Canonical write-ups (read these for claims)

| Document | Role |
|----------|------|
| [`docs/VALIDATED_RESULTS.md`](docs/VALIDATED_RESULTS.md) | Tightest summary of claims the repo treats as validated |
| [`docs/HONEST_ASSESSMENT.md`](docs/HONEST_ASSESSMENT.md) | Failure modes, caveats, where optimistic readings break |
| [`docs/PROJECT_ANALYSIS_SUMMARY.md`](docs/PROJECT_ANALYSIS_SUMMARY.md) | Broader validation vs. claim gap |
| [`docs/RIGOROUS_TESTING_PLAN.md`](docs/RIGOROUS_TESTING_PLAN.md) | Testing intent and methodology framing |
| [`onebit/research/RIGOROUS_TEST_RESULTS.md`](onebit/research/RIGOROUS_TEST_RESULTS.md) | Research-side rigorous test notes |
| [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md) | Layout and navigation |

Historical / draft material under [`onebit/research/paper/`](onebit/research/paper/) may predate stricter evaluation; treat it as **archive**, not current truth.

## Implemented systems (where the work lives)

- **Quantization:** `onebit/quantization/` (e.g. `hessian_vq.py`, `lowrank_residual.py`, `mixed_precision.py`, `redun_score.py`)
- **BPP accounting:** `onebit/core/bpp_guard.py`
- **Runtime / packed ops:** `onebit/ops/`, `onebit/model/runtime_transformer.py`
- **OpenCL backend:** `onebit/backends/opencl/`
- **Eval helpers:** `onebit/eval/perplexity.py`, `onebit/data/wikitext.py`
- **Tests:** `tests/` — bpp, e2e, runtime, and related checks

## Conclusions repeatedly supported in-repo

The following are **not independent results**; they are the reconciled position repeated across `docs/VALIDATED_RESULTS.md`, `docs/HONEST_ASSESSMENT.md`, and related notes:

1. **Strict ~1.00 bpp post-hoc binary** is insufficient for acceptable GPT-2–class LM quality in the settings exercised here.
2. **High per-layer correlation** does not guarantee acceptable **perplexity**; full-model loss is the harsher metric.
3. **Effective bpp** must include indices, codebooks, scales, and routing metadata—not sign bits alone.
4. **Hessian-weighted VQ**, **INT8 low-rank residual**, and **mixed precision** are the more credible operating points in the **~1.1–1.4 bpp** range documented in the validated docs.

For exact figures and experimental conditions, use the documents above and the scripts they reference.

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md). When citing numbers, cite the **git commit** and the **script or test** that produced them.
