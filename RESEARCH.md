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
| [`CURRENT_STATE.md`](CURRENT_STATE.md) | Canonical results table (harness JSON) and what works / what doesn't |
| [`docs/VALIDATED_RESULTS.md`](docs/VALIDATED_RESULTS.md) | Tightest summary of claims the repo treats as validated |
| [`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md) | Retractions (old eval text, broken baselines, bpp errors) and code fixes |
| [`docs/HONEST_ASSESSMENT.md`](docs/HONEST_ASSESSMENT.md) | Failure modes, caveats, where optimistic readings break |
| [`docs/PROJECT_ANALYSIS_SUMMARY.md`](docs/PROJECT_ANALYSIS_SUMMARY.md) | Broader validation vs. claim gap |
| [`docs/RIGOROUS_TESTING_PLAN.md`](docs/RIGOROUS_TESTING_PLAN.md) | Testing intent and methodology framing |
| [`onebit/research/RIGOROUS_TEST_RESULTS.md`](onebit/research/RIGOROUS_TEST_RESULTS.md) | Research-side rigorous test notes |
| [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md) | Layout and navigation |

Historical / draft material under [`onebit/research/paper/`](onebit/research/paper/) may predate stricter evaluation; treat it as **archive**, not current truth.

## Implemented systems (where the work lives)

- **Evaluation (canonical):** `onebit/eval/harness.py` — WikiText-2 test PPL, two-way bpp, JSON records in `results/`
- **Baselines:** `onebit/quantization/baselines.py` (RTN, ternary, binary, SVID), `onebit/quantization/gptq.py` (GPTQ)
- **Quantization research:** `onebit/quantization/` (e.g. `hessian_vq.py`, `lowrank_residual.py`, `mixed_precision.py`, `redun_score.py`)
- **BPP accounting:** `onebit/core/bpp_guard.py`, `BitReport` in the harness
- **KD-QAT proof-of-concept:** `scripts/onebit_kd_poc.py`
- **Runtime / packed ops:** `onebit/ops/`, `onebit/model/runtime_transformer.py`
- **OpenCL backend:** `onebit/backends/opencl/`
- **Tests:** `tests/` — `test_phase0_correctness.py`, bpp, e2e, runtime, and related checks

> [!WARNING]
> `onebit/eval/baselines.py` returns random logits and `onebit/data/wikitext.py` loads WikiText-103; neither is used by the harness. Do not use them for reported numbers.

## Conclusions supported by the harness

1. **Post-training ~1–1.6 bpp** (binary, ternary, HVQ, low-rank residual, mixed precision) is unusable on GPT-2 124M; see `CURRENT_STATE.md`.
2. **High per-layer correlation** does not guarantee acceptable **perplexity**; full-model loss is the harsher metric.
3. **Effective bpp** must include indices, codebooks, scales, zero-points and factors, and should be stated both for linear layers and the whole model.
4. The credible route to ~1 bit is **training** (QAT / distillation from the FP model), as in OneBit, BitDistiller and ParetoQ. A preliminary CPU run in this repo supports that.

For exact figures and experimental conditions, use the JSON in `results/` and the documents above.

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md). Numbers in docs must come from a harness JSON record, which already stores the **git commit**, config and environment.
