# Phase 0 Audit: Measurement Fixes and Retractions

**Branch:** `phase0-honest-eval`  
**Audited commit:** `9c3e106`  
**Status:** this document supersedes the quantitative claims in earlier docs wherever they conflict.

This file records *why* the headline numbers changed. Current numbers live in
[`VALIDATED_RESULTS.md`](VALIDATED_RESULTS.md); each one is produced by
[`onebit/eval/harness.py`](../onebit/eval/harness.py) and stored as JSON under `results/phase0/`.

---

## 1. Summary

The repo's earlier conclusions ("binary fails", "~1.1 bpp is the practical floor", "INT4 is 381× worse than FP32") were partly
correct and partly artifacts of the measurement setup:

| Earlier claim | What was actually going on | Status |
|---|---|---|
| FP32 GPT-2 PPL = 5.92 | Eval text was one paragraph repeated 20× (1,024 tokens). GPT-2 copies repeated text, so PPL is artificially low. On the full WikiText-2 test set, FP32 is 29.95 (`fp32_gpt2`). | **Retracted** |
| "1,208× / 1,458× / 158,027× worse than FP32" | The denominators come from the repeated-text eval | **Retracted** |
| INT4 = 8,812 PPL (381× worse); INT2 better than INT4 | Per-tensor absmax scale, wrong level counts, embeddings and tied LM head quantized, ~60-token eval. Correct group-128 INT4 is 33.26 with RTN (+11%) and 31.28 with GPTQ (+4.5%). | **Retracted** |
| "Usable quality needs ~8 bits" table | Derived from the broken baselines above | **Retracted** |
| HessianVQ "GPTQ refinement +2%" | `_gptq_refine` multiplied its result by 0 and kept only a per-row DC offset (uncounted bits). Not GPTQ. | **Retracted**; real GPTQ now in `onebit/quantization/gptq.py` |
| "Hessian-weighted" end-to-end PPLs | PART 2 of `tests/test_improvements.py` passed `H_diag = np.ones`, and the assignment step ignored weights anyway | **Retracted** (the numbers were unweighted K-means) |
| Two-Stage VQ: 0.982 corr @ 1.69 bpp | `effective_bpp` subtracted 1 bpp for "shared signs" that weren't shared; the real figure is about 2.69 bpp | **Retracted** |
| Mixed precision "allocates more bits to MLP" | The config gives MLPs rank-6 low-rank (~1.08 bpp) and attention HVQ (~1.39 bpp), the opposite | **Corrected** |
| `proper_eval.py` "Working" | Could not run: wrong kwarg `input=`, swallowed exceptions, unmasked pads | **Fixed** (delegates to the harness) |
| One-command repro | `run_public_baseline.py` printed a **hardcoded** "validated" table | **Fixed** (computes everything) |
| Proxy-SR-VQ Redun Score "transfers 124M → 6.9B" | The Hessian term is ~1e-3, so the score is basically the magnitude CV of a Gaussian (~0.76 + act var); equal means are expected for any two models | **Retracted** |
| Scaling law to 70B | 2 model sizes, different architectures and tokenizers, R² ≈ 0.01, and the 88 "points" are per-layer repeats | **Retracted** |
| QAT Phase 2 "implemented" | Could not train (STE under `no_grad`, all params frozen, tied LM head binarized). Never run. | **Disabled** with an explicit error |
| Strict bpp "all overhead counted" | Per-matrix accounting was mostly right, but embeddings (39.4M of 124M params) were silently excluded | **Fixed**: the harness reports both linear-only and whole-model bpp |
| "Distillation doesn't rescue binary" | The earlier attempts trained on ~8 hand-written sentences | **Never actually tested**, see §4 |

What *does* survive:
- Strict ~1-bit **post-training** quantization of GPT-2 124M is not usable, under any harness.
- Correlation is a poor proxy for perplexity.
- The 0.58-bpp DualPath claim was wrong.

---

## 2. Reproducing the old numbers (to confirm the diagnosis)

Running the *old* code paths at `9c3e106` side by side with the standard protocol
(GPT-2 124M; WikiText-2 test, 1024-token non-overlapping windows, first 20 windows):

| Config | bpp (linear) | Old harness (repeated paragraph) | WikiText-2 | Old doc |
|---|---|---|---|---|
| FP32 | 32 | 5.92 | 31.02 | 5.92 |
| Binary sign × mean (per-tensor) | 1.00 | 935,434 | 3,307,061 | 935,427 |
| LowRank r=8 INT8 | 1.111 | 8,629 | 7,839 | 8,629 |
| Mixed (L0/11 protected) | 1.176 | 8,007 (K-means was unseeded) | 5,626 | 7,152 |
| Old "INT4" (per-tensor, incl. embeddings) | 4 | 16,881 | 15,102 | 8,812 |
| INT4 RTN, group 128 (correct) | 4.25 | 6.21 | 34.31 | n/a |

The old harness reproduces the old numbers almost exactly, so the code was run as described. The *measurement* is what was wrong.

---

## 3. Code-level findings and fixes

| Area | File | Finding | Fix on this branch |
|---|---|---|---|
| Eval | `tests/test_improvements.py` | Repeated-paragraph PPL; `H_diag = ones` in PART 2; correlation measured on the calibration tokens | Harness WikiText-2 eval; real calibration Hessian; held-out tokens |
| Eval | `onebit/research/proper_eval.py` | Unrunnable | Delegates to the harness |
| Eval | `onebit/repro/run_public_baseline.py` | Hardcoded table; wrong HVQ bpp (4-weight vs 16-weight blocks) | Thin wrapper over the harness |
| Baselines | `tests/comprehensive_real_test.py` | Per-tensor absmax, wrong levels, embeddings quantized | Uses `onebit/quantization/baselines.py` on block linears only |
| VQ | `onebit/quantization/hessian_vq.py` | Weighted assignment was a no-op; fake GPTQ; unseeded; uncounted mean shift | Per-element weighted distance; seeded; `gptq_refine=True` raises and points to real GPTQ |
| VQ | `onebit/quantization/lowrank_residual.py` | Two-Stage bpp −1.0; one global INT8 scale for factors | Honest bpp; per-rank scales (counted) |
| BPP | `onebit/core/bpp_guard.py` | Params double-counted; 8 bits/index regardless of K | Count once; ⌈log₂K⌉ |
| QAT | `onebit/proxy/qat_loop.py` | Phase 2 cannot train; hardcoded bpp map; swallowed exceptions | Phase 2 raises; real bpp; errors propagate |
| New | `onebit/eval/harness.py` | n/a | Standard protocol, JSON records, two-way bpp |
| New | `onebit/quantization/baselines.py`, `gptq.py` | n/a | Correct RTN / ternary / binary / SVID and real GPTQ |

Not fixed (out of Phase 0 scope, treat as archive): `onebit/research/novel_ideas_v*.py`, `redunforge_proxy_policy.py`
(dense Adam reconstruction discards the claimed low-bit structure; random-token calibration), `onebit/eval/baselines.py`
(returns random logits), and `onebit/data/wikitext.py` (tokenized WikiText-103, not the raw protocol).

---

## 4. Preliminary evidence for the next phase (not a validated result)

A CPU proof-of-concept of OneBit-style KD-QAT (`W ≈ diag(a)·sign(W)·diag(b)`, 1.03 bpp linear, SVID init, logit-KL from FP GPT-2):

| Tokens trained | 0 | 51k | 102k | 154k | 205k | 256k | 298k |
|---|---|---|---|---|---|---|---|
| WikiText-2 PPL (first 20 windows) | 81,759 | 359 | 295 | 242 | 228 | 212 | 209.5 |

Caveats:
- Single seed and a 20-window eval.
- Trained on WikiText-2 *train*, the same domain as the test set.
- Still about 7× FP.

Even so, with 0.0003B tokens it is far ahead of every post-training result at ≤1.6 bpp (best: legacy low-rank r=8, 15,444 on the
full test set; the eval sets differ, so the ratio is only indicative). It strongly suggests that the
research direction should move to training-based quantization, consistent with OneBit (arXiv:2402.11295) and ParetoQ (arXiv:2502.02631).
Script: [`scripts/onebit_kd_poc.py`](../scripts/onebit_kd_poc.py).

---

## 5. Rules going forward

1. A number goes into a doc only if it comes from a JSON file under `results/` produced by the harness, quoted with its run name and commit.
2. Always report FP16 under the same protocol, plus `bpp_linear` **and** `bpp_whole_model`.
3. Compare against the strongest baseline at equal bits (GPTQ for PTQ; equal-byte INT4/INT3 for near-1-bit claims), not against a weak one.
4. Correlation and MSE are diagnostics only.
5. Calibration data comes from the train split; evaluation from the test split. Never the same tokens.
