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

| Method | Config | bpp (linear) | bpp (whole) | WikiText-2 PPL | × FP32 | Record |
|---|---|---:|---:|---:|---:|---|
| FP32 | — | 32.000 | 26.921 | 29.95 | 1.00 | `fp32_gpt2` |
| FP16 | — | 16.000 | 16.000 | 29.95 | 1.00 | `fp16_gpt2` |
| RTN INT8 | per-channel | 8.031 | 10.561 | 29.99 | 1.00 | `rtn_w8_perchan_gpt2` |
| **GPTQ INT4** | group 128 | 4.250 | 7.980 | 31.28 | 1.04 | `gptq_w4_g128_gpt2` |
| RTN INT4 | group 128 | 4.250 | 7.980 | 33.26 | 1.11 | `rtn_w4_g128_gpt2` |
| RTN INT4 | per-channel | 4.031 | 7.831 | 38.35 | 1.28 | `rtn_w4_perchan_gpt2` |
| **GPTQ INT3** | group 128 | 3.250 | 7.298 | 41.43 | 1.38 | `gptq_w3_g128_gpt2` |
| RTN INT3 | group 128 | 3.250 | 7.298 | 66.19 | 2.21 | `rtn_w3_g128_gpt2` |
| **GPTQ INT2** | group 128 | 2.250 | 6.615 | 689.11 | 23 | `gptq_w2_g128_gpt2` |
| RTN INT2 | group 128 | 2.250 | 6.615 | 3,017 | 101 | `rtn_w2_g128_gpt2` |
| Ternary TWN | per-channel, thr 0.7·mean | 1.616 | 6.182 | 34,466 | 1,151 | `ternary_twn_perchan_gpt2` |
| Ternary absmean (BitNet rule) | per-channel | 1.616 | 6.182 | 1,696,137 | 56,631 | `ternary_perchan_gpt2` |
| Legacy mixed (HVQ + low-rank) | L0/L11 protected | 1.164 | 5.874 | 17,266 | 576 | `legacy_mixed_perchan_gpt2` |
| Binary absmean | group 128 | 1.125 | 5.847 | 293,959 | 9,815 | `binary_g128_gpt2` |
| Legacy low-rank residual r=8 | INT8 factors | 1.111 | 5.838 | 15,444 | 516 | `legacy_lowrank_r8_perchan_gpt2` |
| Binary SVID (OneBit init, no training) | row + col vectors | 1.028 | 5.781 | 92,925 | 3,103 | `binary_svid_perchan_gpt2` |
| Binary absmean | per-channel | 1.016 | 5.773 | 190,302 | 6,354 | `binary_perchan_gpt2` |

`bpp (linear)` covers the transformer-block weights, including all scales, zero-points, codebooks and factors.
`bpp (whole)` additionally stores embeddings, LayerNorm and biases at 16 bits. Ternary is counted at 1.6 bits (5 trits/byte).

How to read it:
- Every method at ≤1.6 bpp has PPL ≥ 15,000, within ~3× of (or far worse than) guessing uniformly over the 50,257-token
  vocabulary. These are all broken models; their relative ordering is not a useful signal for which method to pursue.
- The embeddings (39.4M of 124M parameters) set a **whole-model floor of ~5.7 bpp** when kept at 16 bits. A whole-model
  ~1-bit claim needs low-bit embeddings too; report both numbers.
- The two ternary rows show how much the rounding rule matters: the BitNet absmean rule is a *training* recipe and
  is a strawman as a post-training baseline. TWN is the fair post-training ternary.

---

## What works

| Component | Status | Evidence |
|---|---|---|
| Shared evaluation harness (`onebit/eval/harness.py`) | **Working** | FP32 = 29.95, FP16 = 29.95: in the expected range (~30) for GPT-2 small under this protocol |
| Correct baselines (`onebit/quantization/baselines.py`) | **Working** | RTN per-channel / g128, ternary TWN and absmean, binary absmean, OneBit SVID init |
| Real GPTQ (`onebit/quantization/gptq.py`) | **Working** | Beats RTN at every width: INT4 1.04× vs 1.11× FP, INT3 1.38× vs 2.21×, INT2 23× vs 101× |
| Two-way bpp accounting (`BitReport`) | **Working** | Linear-only and whole-model, recorded with every run |
| HessianVQ, per-element weighted K-means (fixed) | **Working** | Unit-tested; seeded; no fake GPTQ |
| Unit tests for the above | **Working** | `tests/test_phase0_correctness.py` (19 tests); 32 together with the bpp tests |

## What does not work

| Claim / method | Status | Why |
|---|---|---|
| ~1-bit **post-training** quantization of GPT-2 124M (binary, ternary, HVQ, low-rank, mixed) | **Fails** | All give PPL ≥ 15,000 (≥ 515× FP; table above). Consistent with the literature: BiLLM is ~4.8× FP even at 1.3B |
| 2-bit RTN / GPTQ on GPT-2 124M | **Poor** | GPTQ INT2 g128 = 689 (23× FP); RTN = 3,017 |
| "Mixed precision helps" (legacy) | **Not supported** | Under the harness the legacy mixed config (17,266) is *worse* than plain low-rank r=8 (15,444) |
| Correlation / MSE as a quality proxy | **Rejected** | High correlation does not imply usable PPL. Example: absmean ternary has *lower* weight MSE than binary (spot-checked on two matrices) yet ~9× higher PPL |
| 0.58-bpp DualPath, sub-1-bit claims | **Retracted** | Accounting errors (`onebit/research/RIGOROUS_TEST_RESULTS.md`) |
| Proxy-SR-VQ Redun Score transfer, 70B scaling law | **Retracted** | Score is dominated by Gaussian magnitude CV; 2 model sizes, R² ≈ 0.01 |
| QAT Phase 2 in `onebit/proxy/qat_loop.py` | **Disabled** | Could not train; raises `NotImplementedError` |

## What is promising (preliminary, not validated)

| Direction | Evidence so far |
|---|---|
| **KD-QAT at ~1 bit** (OneBit-style `diag(a)·sign(W)·diag(b)`, 1.03 bpp linear) | CPU proof-of-concept: PPL 81,759 → **209.5** after 298k training tokens (first 20 test windows; trained on WikiText-2 train; single seed). Every post-training result at ≤1.6 bpp in the table is ≥ 15,000 on the full test set; the eval sets differ, so the ~70× gap is indicative only. Script: `scripts/onebit_kd_poc.py`; record: `results/experiments/onebit_kd_poc_cpu40min_preliminary.json` |

## Deprecated / archived

| Component | Notes |
|---|---|
| `onebit/research/novel_ideas_v*.py`, root-level `results_v*.txt` | Research scratchpad; correlation-only or unreproducible |
| `onebit/research/paper/`, `docs/SALOMI-*`, `docs/BINARY_QUANTIZATION_FINDINGS.md` | Early optimistic drafts |
| `redunforge_proxy_policy.py` | Reported bpp does not describe the stored weights (see audit) |
| `onebit/eval/baselines.py` | Returns random logits; do not use |
| `onebit/data/wikitext.py` | Loads tokenized WikiText-103, not the raw WikiText-2 protocol; not used by the harness |

---

## The result you can stand behind

> On GPT-2 124M, post-training quantization holds up to 3 bits with GPTQ (1.38× FP at 3.25 bpp), collapses at 2 bits
> (23×), and is unusable at ≤1.6 bpp with every method in this repo (≥ 515×). ~1-bit operation requires **training**
> (quantization-aware training / distillation from the FP model). A first CPU proof-of-concept of that route reaches
> PPL ≈ 210 at 1.03 bpp, far ahead of every post-training ~1-bit result here.

## Next steps (Phase 1)

1. KD-QAT from pretrained GPT-2 on a general corpus (FineWeb-Edu / OpenWebText), ternary first, then binary. Learnable
   per-channel scales; embeddings and LM head at 8-bit and counted. Budget: 3–5B tokens for a first result (single rented GPU).
2. Evaluate with the harness on the full test set; compare against **equal-byte** INT4/INT3 GPTQ models, not just FP16.
3. Side track: fix the VQ geometry (signed 8-dim vectors, K=256) and fine-tune codebooks end to end with KD.

Success criterion for a near-1-bit claim: ≤1.1 bpp (linear; whole-model also stated), WikiText-2 PPL within ~2× of FP,
produced by a committed script with harness JSON in `results/`.
