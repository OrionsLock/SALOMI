# SALOMI — Current State

**Last updated:** April 2026  
**Branch:** `proxy-sr-vq-pipeline`

This document answers in 30 seconds: what works, what does not, what is exploratory, what is deprecated, and what result you can stand behind today.

---

## What Works

| Component | Status | Evidence |
|---|---|---|
| `HessianVQ` weighted K-means (K=64, 15 iters) | **Working** | +1.6% correlation vs old unweighted; confirmed in `tests/test_improvements.py` |
| `HessianVQ` + GPTQ sequential refinement | **Working** | +2.0% correlation improvement; same test |
| `LowRankResidual` with INT8 factors (r=8, r=12) | **Working** | Better correlation at lower BPP than FP32 r=4; confirmed |
| Mixed-precision layer allocation | **Working** | 17% PPL improvement (8,629 → 7,152) protecting layers 0 and 11 |
| Strict BPP accounting (`onebit/core/bpp_guard.py`) | **Working** | Codebook + indices + overhead all counted |
| End-to-end PPL evaluation (`onebit/research/proper_eval.py`) | **Working** | Tested on GPT-2 124M all-layer quantization |
| Cross-architecture `ModelAdapter` (GPT-2, OPT, Pythia/NeoX) | **Working** | Validated: Pythia 6.9B loaded and probed successfully |
| Proxy-SR-VQ Redun Score probe | **Working** | 124M and 6.9B both produce stable, non-NaN scores |
| Dynamic quantizer allocation from Redun Score | **Working** | 10 HessianVQ + 1 TernarySparse on 124M; 30 HessianVQ on 6.9B |
| Scaling law fit across model sizes | **Partial** | R² = 0.006–0.017 with 2 models; needs more data points |
| Block-wise calibration (`qat_loop.py` Phase 1) | **Working** | Tested on GPT-2 124M; not yet run on 6.9B |
| Policy export/import (`policy_export.py`) | **Working** | JSON policy file written for 124M |

---

## What Does Not Work

| Claim / Method | Status | Why |
|---|---|---|
| Strict 1.00 bpp post-hoc binary quantization | **Fails** | PPL 158,000× worse than FP32 on GPT-2 124M |
| Binary with residual-connection argument | **Insufficient** | 0.92 correlation ≠ usable PPL; PPL still 590× worse |
| Sub-1-bit (0.58 bpp DualPath-style) claims | **Not validated** | Strict BPP accounting not applied; overhead not counted |
| Correlation as a sufficient quality proxy | **Rejected** | Two-Stage VQ achieves 0.982 correlation, still catastrophic PPL |
| End-to-end PPL under 100× FP32 at 1.1 bpp | **Not yet achieved** | Best result is ~1,200× worse on 124M; larger models expected better |

---

## What Is Exploratory

| Component | Notes |
|---|---|
| Proxy-SR-VQ scaling law extrapolation to 70B | Only 2 data points so far; directionally promising |
| TernarySparse quantizer | Implemented and allocated but not yet ablated against plain ternary |
| QAT loop Phase 2 (STE fine-tuning) | Implemented; not yet benchmarked end-to-end |
| Redun Score coefficient tuning (α, β, γ) | Grid search implemented; not yet run with quality feedback |
| OPT family support | Adapter implemented; not yet tested against a real OPT model |
| 6.9B post-calibration PPL | Calibration not yet run; Redun Scores suggest it should work |

---

## What Is Deprecated / Archived

| Component | Notes |
|---|---|
| `onebit/research/novel_ideas_v*.py` | Research scratchpad scripts; superseded by structured modules |
| `onebit/research/paper/` | Draft paper materials with earlier, more optimistic claims |
| Per-layer correlation as a headline metric | Replaced by end-to-end PPL as governing metric |
| Early "sub-1-bit beats ternary" framing | Revised after strict BPP accounting |
| `onebit/research/EXPERIMENT_SUMMARY.md` | Historical; for context only |

---

## The One Result You Can Stand Behind

> **Mixed-precision Hessian-weighted VQ with INT8 low-rank residual at ~1.1–1.2 bpp delivers the best PPL in this repo at 7,152 PPL on GPT-2 124M (1,208× FP32), with all overhead counted.** Binary 1.00 bpp is not viable. The practical floor for usable quality in this codebase is currently ~1.1 bpp.

---

## Research Chronology

### Phase 1 — Initial hypothesis (early 2026)
- Hypothesis: residual connections make binary quantization viable because errors are additive, not multiplicative.
- Initial correlation numbers looked promising (0.92 hidden-state correlation).
- BPP accounting was loose; sub-1-bit claims not yet stress-tested.

### Phase 2 — Stricter evaluation reveals problems
- End-to-end perplexity tests showed catastrophic degradation at 1.00 bpp (158,000× FP32).
- Correlation-to-PPL gap documented explicitly: 0.982 correlation ≠ usable language model.
- BPP accounting corrected: codebook overhead, sign bits, and indices added back in.
- Several earlier headline claims revised in `docs/HONEST_ASSESSMENT.md`.

### Phase 3 — Algorithm improvements (April 2026)
- Fixed core `HessianVQ` bug: weighted K-means was computing weights but not using them.
- Added GPTQ-style sequential error compensation.
- Switched low-rank residual factors from FP32 to INT8 — better quality at lower BPP.
- Implemented mixed-precision layer allocation with MLP sensitivity analysis.
- Added `BPPCalculator` for strict accounting.
- Best PPL improved from >25,000 to 7,152 at 1.175 bpp.

### Phase 4 — Proxy-SR-VQ pipeline (April 2026)
- Implemented full Proxy Scale-Redundancy VQ pipeline: Redun Score probe, dynamic allocation, block-wise calibration, scaling law fitting, policy export.
- Extended to multi-architecture via `ModelAdapter` (GPT-2, OPT, GPT-NeoX/Pythia).
- Validated: Pythia 6.9B (6.86B params) produces stable Redun Scores (mean 0.905 vs 0.906 for 124M), confirming the proxy-transfer hypothesis.
- Scaling law fit with 2 data points: MLP R² = 0.017, Attn R² = 0.006. More model sizes needed.

### Next steps
- Run block-wise calibration on Pythia 6.9B and measure post-quantization PPL.
- Add intermediate proxy sizes (Pythia 1.4B, 2.8B) to tighten the scaling law.
- Ablate TernarySparse vs plain ternary baseline.
- Run QAT Phase 2 loop and measure improvement over calibration-only baseline.

---

## Canonical Benchmark Table

> Perplexity is the governing metric. Correlation is logged as a diagnostic but is not sufficient evidence of quality.

| Method | True BPP | Dataset | PPL | Corr (avg) | Status |
|---|---|---|---|---|---|
| FP32 baseline | 32.0 | GPT-2 eval | 5.92 | 1.000 | Reference |
| Binary (sign×scale) | 1.000 | GPT-2 eval | 935,427 | ~0.82 | **Fails** |
| OLD HVQ K=32 unweighted | 1.310 | GPT-2 eval | ~25,735 | 0.899 | Deprecated |
| NEW HVQ K=64 Hessian-weighted | 1.380 | GPT-2 eval | 25,735 | 0.913 | Exploratory |
| LowRank r=8 INT8 | 1.111 | GPT-2 eval | 8,629 | 0.898 | **Validated** |
| Mixed-precision (L0/11 protected) | 1.175 | GPT-2 eval | 7,152 | — | **Best validated** |
| Two-Stage VQ 64+32 | 1.690 | GPT-2 eval | — | 0.982 | Exploratory |
| Pythia 6.9B FP16 baseline | 16.0 | Calib sentences | 24.12 | — | Reference |
| Pythia 6.9B quantized (~1.1 bpp) | ~1.1 | — | **not yet run** | — | Pending |
