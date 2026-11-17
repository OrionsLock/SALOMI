# SALOMI: Current State

**Last updated:** November 2026 (Phase 0 measurement fixes, branch `phase0-honest-eval`)

This page gives the 30-second view of what works, what doesn't, and what you can stand behind today.
For why the numbers changed from earlier versions, see [`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md).

> [!IMPORTANT]
> Every number below comes from [`onebit/eval/harness.py`](onebit/eval/harness.py): GPT-2 124M, full WikiText-2-raw test set,
> 1024-token non-overlapping windows. The JSON records are in [`results/phase0/`](results/phase0/).
> Earlier figures (FP32 = 5.92, "1,208× FP32", INT4 = 8,812, …) came from a repeated-paragraph eval and broken baselines, and are retracted.

---

## Canonical results (GPT-2 124M, WikiText-2 test)

<!-- PHASE0_TABLE -->

`bpp (linear)` covers the transformer-block weights, including all scales, zero-points, codebooks and factors.
`bpp (whole)` additionally stores embeddings, LayerNorm and biases at 16 bits. Ternary is counted at 1.6 bits (5 trits/byte).

---

## What works

| Component | Status | Evidence |
|---|---|---|
| Shared evaluation harness (`onebit/eval/harness.py`) | **Working** | FP32 = 29.95, matching the standard GPT-2 reference (~29.9) |
| Correct baselines (`onebit/quantization/baselines.py`) | **Working** | RTN per-channel / g128, ternary absmean, binary absmean, OneBit SVID init |
| Real GPTQ (`onebit/quantization/gptq.py`) | **Working** | Beats RTN at 3 and 2 bits, consistent with the GPTQ paper |
| Two-way bpp accounting (`BitReport`) | **Working** | Linear-only and whole-model, recorded with every run |
| HessianVQ, per-element weighted K-means (fixed) | **Working** | Unit-tested; seeded; no fake GPTQ |
| Unit tests for the above | **Working** | `tests/test_phase0_correctness.py` (21 tests) |

## What does not work

| Claim / method | Status | Why |
|---|---|---|
| ~1-bit **post-training** quantization of GPT-2 124M (binary, ternary, HVQ, low-rank, mixed) | **Fails** | All produce PPL in the thousands or worse (table above). Consistent with the literature: BiLLM is ~4.8× FP even at 1.3B |
| 2-bit RTN / GPTQ on GPT-2 124M | **Poor** | See table; small models are hard to quantize post-training |
| Correlation / MSE as a quality proxy | **Rejected** | High correlation does not imply usable PPL |
| 0.58-bpp DualPath, sub-1-bit claims | **Retracted** | Accounting errors (`onebit/research/RIGOROUS_TEST_RESULTS.md`) |
| Proxy-SR-VQ Redun Score transfer, 70B scaling law | **Retracted** | Score is dominated by Gaussian magnitude CV; 2 model sizes, R² ≈ 0.01 |
| QAT Phase 2 in `onebit/proxy/qat_loop.py` | **Disabled** | Could not train; raises `NotImplementedError` |

## What is promising (preliminary, not validated)

| Direction | Evidence so far |
|---|---|
| **KD-QAT at ~1 bit** (OneBit-style `diag(a)·sign(W)·diag(b)`, 1.03 bpp linear) | CPU proof-of-concept: PPL 81,759 → **209.5** after 298k training tokens (first 20 test windows; trained on WikiText-2 train; single seed). That is ~27× better than any post-training result here, at lower bpp. Script: `scripts/onebit_kd_poc.py`; record: `results/experiments/onebit_kd_poc_cpu40min_preliminary.json` |

## Deprecated / archived

| Component | Notes |
|---|---|
| `onebit/research/novel_ideas_v*.py`, root-level `results_v*.txt` | Research scratchpad; correlation-only or unreproducible |
| `onebit/research/paper/`, `docs/SALOMI-*`, `docs/BINARY_QUANTIZATION_FINDINGS.md` | Early optimistic drafts |
| `redunforge_proxy_policy.py` | Reported bpp does not describe the stored weights (see audit) |
| `onebit/eval/baselines.py` | Returns random logits; do not use |

---

## The result you can stand behind

> On GPT-2 124M, post-training quantization below ~3 bits per weight does not preserve usable language-modelling quality with
> any method in this repo, including a correct GPTQ. ~1-bit operation requires **training** (quantization-aware training /
> distillation from the FP model). A first CPU proof-of-concept of that route already improves on every post-training
> result here by more than an order of magnitude.

## Next steps (Phase 1)

1. KD-QAT from pretrained GPT-2 on a general corpus (FineWeb-Edu / OpenWebText), ternary first, then binary. Learnable
   per-channel scales; embeddings and LM head at 8-bit and counted. Budget: 3–5B tokens for a first result (single rented GPU).
2. Evaluate with the harness on the full test set; compare against **equal-byte** INT4/INT3 GPTQ models, not just FP16.
3. Side track: fix the VQ geometry (signed 8-dim vectors, K=256) and fine-tune codebooks end to end with KD.

Success criterion for a near-1-bit claim: ≤1.1 bpp (linear; whole-model also stated), WikiText-2 PPL within ~2× of FP,
produced by a committed script with harness JSON in `results/`.
