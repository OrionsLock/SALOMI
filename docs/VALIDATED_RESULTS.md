# Validated Results

This document is the shortest summary of the claims in SALOMI that are most defensible from the current repository state.

It is intentionally narrower than the historical paper drafts and narrower than some earlier experiment summaries.

## Scope

These points reflect the later evaluation-oriented materials in the repo, especially:

- `docs/HONEST_ASSESSMENT.md`
- `docs/PROJECT_ANALYSIS_SUMMARY.md`
- `onebit/research/RIGOROUS_TEST_RESULTS.md`
- `tests/test_bpp_strict.py`
- `tests/test_perplexity_real.py`
- `tests/test_improvements.py` (comprehensive improvement test, April 2026)

## Latest quantitative results (April 2026)

### Per-layer output correlation (MLP c_fc, layers 0/5/11, GPT-2 124M)

| Method | Avg Corr | Min Corr | BPP | vs Old HVQ |
|--------|----------|----------|-----|------------|
| Binary (sign*scale) | 0.816 | 0.804 | 1.00 | -9.2% |
| Ternary (30% sparse) | 0.896 | 0.889 | 1.58 | -0.3% |
| OLD HVQ K=32 (unweighted, 5 iters) | 0.899 | 0.880 | 1.31 | baseline |
| **NEW HVQ K=64 (Hessian-weighted, 15 iters)** | **0.913** | **0.895** | **1.38** | **+1.6%** |
| **NEW HVQ K=64 + GPTQ refinement** | **0.917** | **0.901** | **1.38** | **+2.0%** |
| LowRank r=4 FP32 (old) | 0.885 | 0.875 | 1.21 | -1.5% |
| **LowRank r=8 INT8 (new)** | **0.898** | **0.893** | **1.10** | -0.1% |
| **LowRank r=12 INT8 (new)** | **0.906** | **0.902** | **1.16** | +0.8% |
| **Two-Stage VQ 64+32** | **0.982** | **0.978** | **1.69** | **+9.3%** |

### End-to-end perplexity (all 12 layers, all weights, GPT-2 124M)

| Method | PPL | Ratio vs FP32 | BPP |
|--------|-----|---------------|-----|
| FP32 Baseline | 5.92 | 1.0x | 32.0 |
| Binary (sign*scale) | 935,427 | 158,027x | 1.000 |
| NEW HVQ K=64 (Hessian-weighted) | 25,735 | 4,348x | 1.380 |
| **LowRank r=8 INT8** | **8,629** | **1,458x** | **1.111** |
| **Mixed-precision (L0/11 protected)** | **7,152** | **1,208x** | **1.175** |

## Validated or defensible claims

### 1. Strict post-hoc 1.00 bpp binary is not enough for GPT-2-class LM quality

Sign-only binary post-hoc quantization produces 158,000x worse perplexity. Even with learned scales the best 1.00 bpp result is 400x worse. This is confirmed in the latest test.

### 2. Correlation and perplexity should not be treated as interchangeable

The Two-Stage VQ achieves 0.982 average correlation yet still produces catastrophic perplexity when applied end-to-end. The correlation-to-PPL gap remains the most important finding in this repo.

### 3. Early sub-1-bit claims were too optimistic under strict accounting

Later bpp analysis in the repo argues that some early sub-1-bit figures omitted or undercounted costs such as sign bits, metadata, indices, or codebook overhead.

### 4. Hessian-weighted VQ is measurably better than unweighted

The April 2026 improvements fixed the core `HessianVQ` class to actually use Hessian-weighted K-means (the previous implementation computed weights but never used them). This produced a +1.6% correlation improvement at similar BPP, and +2.0% with GPTQ refinement.

### 5. INT8 low-rank factors dominate FP32 at lower BPP

LowRank r=8 with INT8 factors achieves better correlation (0.898) than FP32 r=4 (0.885) at lower BPP (1.10 vs 1.21). Quantising the correction factors wastes almost no quality while halving the overhead.

### 6. Mixed-precision layer allocation helps

Protecting layers 0 and 11 with higher-rank correction and allocating more bits to MLP paths improved end-to-end PPL by 17% (8,629 to 7,152) at only 0.06 additional BPP.

### 7. MLP/GELU sensitivity is a major bottleneck

MLP blocks are 77-200x more sensitive than attention layers due to GELU curvature near zero. This is confirmed both by isolated tests and by the effectiveness of the mixed-precision scheme.

## Claims that should be treated cautiously

- `sub-1-bit beats ternary` as a general repo-level conclusion
- `0.58 bpp` DualPath-style claims without strict overhead accounting
- correlation-only claims presented as evidence of usable LM behavior
- anything described as "production-ready" without checking the implementation directly
- perplexity improvement percentages that are not tied to a specific logged experiment

## Best current one-sentence summary

SALOMI is a research codebase for extreme low-bit transformer quantization whose strongest current contributions are (1) rigorous evidence about where post-hoc 1-bit breaks down, (2) concrete improvements to Hessian-weighted VQ and INT8 low-rank residual methods at 1.1-1.4 bpp, and (3) a quantified demonstration that mixed-precision layer allocation materially helps end-to-end quality.
