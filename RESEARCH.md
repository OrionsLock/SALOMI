# SALOMI Research Report

**Repository:** SALOMI  
**Primary package:** `onebit` v0.0.1  
**Last synthesized:** 2026-04-01  
**Purpose:** Publishable, repo-level research report covering project scope, methods, evidence, contradictions, strengths, risks, and publication readiness.

---

## Executive Summary

SALOMI is an ambitious research codebase focused on **extreme transformer quantization**, especially the question of whether **binary or near-binary weight representations** can approach or exceed the quality of **ternary quantization** while improving storage and runtime efficiency.

The repository is technically substantial. It contains:

- a nontrivial Python package with quantization, runtime, evaluation, OpenCL backend, autotuning, attention certification, and CLI tooling,
- a large experimental history under `onebit/research/`,
- a broad test suite under `tests/`,
- and multiple rounds of documentation that move from **optimistic early claims** to **much more rigorous, self-correcting conclusions**.

The central reconciled conclusion across the most credible materials in the repo is:

1. **Pure post-hoc 1.00 bpp binary quantization is not good enough for usable GPT-2–class language modeling quality.**
2. **Per-layer correlation is not a reliable substitute for held-out perplexity.**
3. **Strict bpp accounting materially changes several early headline claims.**
4. **The most credible practical region is not true 1.00 bpp, but roughly 1.2-1.35 bpp using Hessian-guided VQ, mixed precision, or magnitude-recovery methods.**

In short: **yes, this project is impressive**, especially as a research and systems exploration effort. It is **not yet clean enough to be presented as a polished public library or as a single authoritative paper artifact without curation**.

---

## Table of Contents

1. [Project Identity](#project-identity)
2. [Repository at a Glance](#repository-at-a-glance)
3. [Research Question and Historical Targets](#research-question-and-historical-targets)
4. [Reconciled Scientific Conclusions](#reconciled-scientific-conclusions)
5. [Core Implemented Systems](#core-implemented-systems)
6. [Experimental Program](#experimental-program)
7. [Documentation and Narrative Quality](#documentation-and-narrative-quality)
8. [Engineering Maturity](#engineering-maturity)
9. [What Is Impressive](#what-is-impressive)
10. [What Needs Work Before Public Release](#what-needs-work-before-public-release)
11. [Recommended Public Positioning](#recommended-public-positioning)
12. [Repository Map](#repository-map)
13. [Suggested Citation](#suggested-citation)

---

## Project Identity

SALOMI is a repository for **extreme low-bit transformer inference and quantization research**.

Two expansions of the SALOMI acronym appear in the repo:

- **Stochastic Approximation for Low-Memory Inference**
- **Scalable Adaptive Low-bitwidth Optimized Model Inference**

These should be treated as naming variants for the same project rather than separate initiatives.

The repository centers on:

- binary and ternary quantization,
- Hessian-guided block/vector quantization,
- packed sign representations,
- stochastic low-bit dot-product operators,
- GPT-2 quantization/runtime experiments,
- and rigorous evaluation of whether these ideas survive end-to-end testing.

---

## Repository at a Glance

Based on the current project tree:

- `onebit/` contains roughly **188 Python files**
- `tests/` contains roughly **91 Python test/experiment files**
- `docs/` contains **10 markdown documents**

This is not a toy repo. It is a large research workspace with both:

- **systems work**: runtime, backends, kernels, packers, autotuning, attention logic
- **science work**: many experimental variants, logs, benchmark scripts, and writeups

At the same time, it still has the texture of an active lab notebook:

- many root-level result files,
- many one-off scripts,
- historical experiment variants,
- and multiple overlapping narratives about what the project has proven.

---

## Research Question and Historical Targets

The core historical question is:

> Can a transformer model be quantized to around **1.00 bits per parameter** while maintaining quality competitive with or better than **ternary 1.58 bpp** methods?

Common target themes across the repo:

- match or beat ternary quality using less memory,
- validate on GPT-2–scale models,
- preserve layer outputs under real activations,
- achieve strong end-to-end perplexity,
- maintain or improve throughput via binary operators and packed compute.

Historically, some materials under `onebit/research/paper/` present a strong claim that **sub-1-bit methods beat ternary**. Later materials in `docs/`, `tests/`, and `onebit/research/RIGOROUS_TEST_RESULTS.md` narrow or refute parts of that story under stricter evaluation.

---

## Reconciled Scientific Conclusions

This section reflects the most trustworthy overall position after comparing `docs/`, `tests/`, and research summaries.

### 1. Strict 1.00 bpp post-hoc binary does not hold up for LLM quality

The strongest repeated conclusion is that **simple sign-based or strict binary post-hoc quantization is not enough** for GPT-2–class language modeling. The repo’s later assessments describe **catastrophic perplexity degradation** even when local metrics look better than expected.

### 2. Correlation is not the same as model quality

Several documents make this explicit:

- high hidden-state or per-layer correlation can coexist with bad perplexity,
- residual connections can preserve apparent signal,
- but language modeling loss is dominated by **softmax sensitivity**, **cross-entropy**, and downstream error compounding.

This is one of the most important intellectual contributions in the repo: the project does not stop at favorable reconstruction metrics, but documents why those metrics can be misleading.

### 3. Bpp accounting became more rigorous over time

The repo repeatedly corrects earlier bpp claims by counting:

- sign bits,
- block indices,
- codebooks,
- row/column scales,
- routing metadata,
- and other overhead.

That shift matters. It changes some sub-1-bit headlines into higher effective bpp regimes.

### 4. Hessian-guided VQ remains one of the strongest practical directions

Across the reconciled docs, **HessianVQ** or related magnitude-aware VQ methods remain among the most credible approaches, especially around **~1.25-1.32 bpp**, where per-layer metrics can beat naive ternary baselines.

### 5. Mixed precision and magnitude recovery are more realistic than pure binary

The repo’s later findings consistently favor:

- low-rank magnitude recovery,
- row/column scaling,
- selective higher precision for sensitive layers or weights,
- and protecting MLP/GELU-heavy paths more than attention.

### 6. MLP blocks are the main bottleneck

GELU-heavy MLP layers are repeatedly identified as far more sensitive than attention, which helps explain why full-model quality collapses faster than isolated layer metrics would suggest.

---

## Core Implemented Systems

### Quantization

Representative files:

- `onebit/quantization/hessian_vq.py`
- `onebit/quantization/functional.py`
- `onebit/ops/vq_optimized.py`

This part of the repo is centered on block magnitude quantization plus sign handling, with Hessian-aware framing and fast decode paths.

One important caveat from the code itself: `onebit/quantization/hessian_vq.py` describes Hessian-weighted VQ, but the current loop explicitly notes it is using **unweighted K-means “for now”**. That does not make the work unimpressive, but it does mean some naming and implementation details should be described carefully in any public writeup.

### 1-bit compute and runtime

Representative files:

- `onebit/ops/bsdm_w.py`
- `onebit/model/runtime_transformer.py`
- `onebit/model/quantize_gpt2.py`
- `onebit/model/hcl_logits_head.py`
- `onebit/model/onebit_logits_head.py`

This is one of the strongest parts of the repo. The code does not stop at offline quantization; it attempts an end-to-end runtime story with:

- packed inputs,
- stochastic dot-product operators,
- configurable compute budgets,
- optional logits heads,
- and GPT-2-shaped inference flows.

### Attention, certification, and scheduling

Representative files:

- `onebit/attn/sprt_dag.py`
- `onebit/attn/runner.py`

The presence of SPRT/DAG-style attention certification is a genuinely interesting systems/research direction and gives the repo more depth than a typical quantization-only experiment dump.

### Backends and performance

Representative files:

- `onebit/backends/opencl/host_opencl.py`
- `onebit/autotune/tuner.py`

The OpenCL kernels and autotuning logic make the project more impressive technically. They suggest the work is aiming at runtime realism, not just paper tables.

### Evaluation and controls

Representative files:

- `onebit/core/bpp_guard.py`
- `onebit/eval/perplexity.py`
- `onebit/research/proper_eval.py`
- `onebit/research/cross_validation.py`

This is where the project’s self-correction becomes visible: the repo includes explicit machinery for bpp auditing and more realistic evaluation instead of relying only on reconstruction proxies.

### Deployment layer

Representative file:

- `onebit/deploy/api.py`

This file is worth flagging carefully. It is labeled as production-ready, but it currently mixes real components with placeholders, simulated outputs, and at least one broken reference pattern (`CrossValidator` is used while the import is commented out). Publicly, this should not be presented as production-ready in its current form.

---

## Experimental Program

### Versioned research lineage

The `onebit/research/novel_ideas_v*.py` series, together with many `results_v*.txt` files, functions as a running laboratory notebook of the project’s search process.

This includes experiments with:

- transform-domain methods,
- sign prediction and encoding tricks,
- low-rank magnitude recovery,
- calibrated binary variants,
- distillation,
- residual-aware ideas,
- strict bpp re-checks,
- and true/near-1-bit analysis.

### What the experiment history shows

At a high level, the repo’s own experiment summaries suggest:

- transform-domain tricks looked promising on synthetic structure but mostly failed on real GPT-2 weights,
- encoding tricks did not recover “free” information,
- magnitude recovery is the strongest direction,
- pure binary remains too destructive,
- and end-to-end evaluation is much harsher than per-layer metrics.

### Tests versus research artifacts

The project has both:

- a meaningful test tree under `tests/`
- and a large number of root-level or research-level standalone scripts

This is good for exploration, but it means the repo is still split between:

- **reproducible validation**
- and **ongoing exploratory experimentation**

That distinction should be made explicit in any public presentation.

---

## Documentation and Narrative Quality

### Strengths

The repo contains unusually strong self-critique for a research codebase. In particular:

- `docs/HONEST_ASSESSMENT.md`
- `docs/PROJECT_ANALYSIS_SUMMARY.md`
- `docs/RIGOROUS_TESTING_PLAN.md`
- `onebit/research/RIGOROUS_TEST_RESULTS.md`

all push the project toward a more trustworthy narrative.

This is a real strength. Many repos keep only the optimistic story. This one preserves both the ambition and the corrections.

### Weaknesses

The public-facing story is still fragmented.

The biggest narrative issue is that two incompatible readings coexist:

- **optimistic paper-style claim**: sub-1-bit beats ternary
- **later rigorous claim**: strict bpp and full-model evaluation make that claim much weaker or false in the original form

Without a guide document, an outside reader can easily quote the wrong numbers.

That is exactly why this report exists.

---

## Engineering Maturity

### What looks solid

- `onebit/model/runtime_transformer.py` is substantial and test-backed.
- `onebit/ops/bsdm_w.py` and related bit-packing/runtime paths show serious implementation work.
- OpenCL backend and autotuning support are strong signals of engineering ambition.
- The test tree is much broader than what most exploratory repos have.

### What looks incomplete or fragile

- No root `pyproject.toml`, `setup.py`, or `requirements.txt` was found.
- No root `README.md` existed before this pass.
- Some directories do not look consistently packaged as formal Python subpackages.
- `onebit/deploy/api.py` is not aligned with its “production-ready” label.
- The repo root still contains many logs, debug files, and result artifacts that make the project feel more like a working lab directory than a curated public release.

### Maturity verdict

This is **research-grade and technically serious**, but **not release-grade as a polished library**.

---

## What Is Impressive

This project is impressive for several reasons.

### 1. Breadth

It spans:

- theory,
- quantization algorithms,
- evaluation,
- runtime inference,
- kernels,
- autotuning,
- attention logic,
- and many rounds of experiments.

Most repos only cover one or two of those layers.

### 2. Honest iteration

The repo does not simply report favorable intermediate metrics. It contains documents that admit when a promising story did not survive stricter testing. That intellectual honesty is a positive signal.

### 3. Systems ambition

The OpenCL backend, runtime transformer, and packed compute story make the work feel materially deeper than a notebook-only research effort.

### 4. Test density

A repo with around ninety test files plus experiment runners is materially more serious than a one-off prototype, even if some of the tests are smoke/regression style rather than publication-grade evaluation.

### 5. Originality of exploration

The project does not just clone standard quantization patterns. It explores VQ, low-rank corrections, stochastic operators, residual analyses, certification ideas, and runtime engineering in a single codebase.

---

## What Needs Work Before Public Release

If you want this to look strong on GitHub, the biggest gaps are presentation and curation, not raw effort.

### Required cleanup

1. Add a clean root `README.md` with:
   project overview, status, honest headline claims, and where to start.
2. Add dependency metadata:
   `pyproject.toml` or at least `requirements.txt`.
3. Separate or ignore noisy artifacts:
   debug logs, transient results, caches, and local workspace files.
4. Mark historical paper claims clearly:
   `onebit/research/paper/` should be labeled as draft or historical if the repo’s later docs supersede it.
5. Tighten naming:
   anything called “production-ready” should actually be production-ready or renamed.

### Optional but high-value cleanup

1. Group research artifacts into clearer folders:
   `experiments/`, `results/`, `paper-drafts/`, `validated-results/`.
2. Add a reproducibility section:
   dataset assumptions, environment, entry commands, and expected outputs.
3. Add one benchmark table that explicitly distinguishes:
   per-layer correlation, end-to-end correlation, and perplexity.

---

## Recommended Public Positioning

The best public framing is not:

> “We solved sub-1-bit LLM quantization.”

The stronger and more defensible framing is:

> “This repository documents a serious research program into extreme LLM quantization, including both promising approaches and rigorous evidence about where naive sub-1-bit claims break down.”

That version is still impressive, but it is also credible.

Recommended status label for GitHub:

- **Research repository**
- **Experimental**
- **Validated in parts; not a stable production package**

---

## Repository Map

### Core package

| Area | Representative paths | Notes |
|------|----------------------|-------|
| Core | `onebit/core/packbits.py`, `onebit/core/bpp_guard.py`, `onebit/core/hadamard.py` | packing, accounting, transforms |
| Ops | `onebit/ops/bsdm_w.py`, `onebit/ops/vq_optimized.py` | low-bit operators and decode |
| Quantization | `onebit/quantization/hessian_vq.py`, `onebit/quantization/functional.py` | block/VQ logic |
| Model | `onebit/model/quantize_gpt2.py`, `onebit/model/runtime_transformer.py` | GPT-2 path and runtime |
| Eval | `onebit/eval/perplexity.py` | model-quality measurement |
| Deploy | `onebit/deploy/api.py` | deployment facade, currently mixed with placeholders |
| Attention | `onebit/attn/sprt_dag.py`, `onebit/attn/runner.py` | certification and attention logic |
| Backends | `onebit/backends/opencl/` | kernels and host code |
| Autotune | `onebit/autotune/tuner.py` | runtime tuning |
| Data | `onebit/data/wikitext.py` | dataset helpers |
| Research | `onebit/research/` | experimental lab notebook and summaries |

### Docs

| File | Role |
|------|------|
| `docs/REPOSITORY_GUIDE.md` | curated technical guide to the repository |
| `docs/PROJECT_ANALYSIS_SUMMARY.md` | validation/failure mode summary |
| `docs/HONEST_ASSESSMENT.md` | strongest reality check |
| `docs/RIGOROUS_TESTING_PLAN.md` | claimed vs verified gap framing |
| `docs/BINARY_QUANTIZATION_FINDINGS.md` | binary-specific findings |
| `docs/CORRELATION_FINDINGS.md` | correlation-focused findings with perplexity caveats |

### Tests

Representative categories under `tests/`:

- strict bpp accounting
- real perplexity checks
- end-to-end correlation
- speed benchmarks
- GELU and propagation studies
- runtime, backend, and OpenCL parity checks
- attention and certification tests
- stress and soak tests

---

## Suggested Citation

If this repository is cited publicly, cite a specific commit and identify which evaluation script produced the headline numbers.

```bibtex
@misc{salomi2026repo,
  title        = {SALOMI: Research code for extreme LLM weight quantization},
  howpublished = {\url{<repository-url>}},
  note         = {Commit <hash>. See README.md and RESEARCH.md for validated claims and caveats.},
  year         = {2026}
}
```

---

## Final Assessment

### Is this impressive?

**Yes.** The repository is technically ambitious, unusually broad, and more honest than most research repos about what failed.

### Is it publication-ready as-is?

**No, not as-is.** It is publishable to GitHub, but it still needs curation before it will read as a polished public research release rather than an internal research workspace.

### Best concise verdict

> Impressive research project, credible technical depth, strong experimentation, but still needs packaging, cleanup, and narrative consolidation before public release.
