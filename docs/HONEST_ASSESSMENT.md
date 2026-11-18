# HONEST ASSESSMENT: Binary Quantization Reality

> [!IMPORTANT]
> **November 2026 update (Phase 0).** The qualitative conclusion of this document still holds: post-training binary
> quantization does not work for GPT-2 and ~1-bit needs training. But the perplexity numbers in the sections below were produced
> by older scripts with a non-standard eval text and broken INT baselines (e.g. "INT4 = 8,812" came from per-tensor absmax
> quantization that also quantized the embeddings). Canonical numbers are now in [`CURRENT_STATE.md`](../CURRENT_STATE.md),
> produced by `onebit/eval/harness.py`; see [`PHASE0_AUDIT.md`](PHASE0_AUDIT.md) for details. The tables marked *superseded*
> are kept only for history.

## Executive Summary

**Binary (1.00 bpp) quantization DOES NOT WORK for LLMs without extensive calibration/training.**

## Novel Residual Connection Insight

The hypothesis was: Residual connections (`x = x + layer(x)`) preserve the original signal while quantization only affects the delta, making errors ADDITIVE rather than MULTIPLICATIVE.

### Test Results (MLP-only binary with residual awareness)

| Metric | Value |
|--------|-------|
| Weight correlation | 0.78 |
| **Final hidden state correlation** | **0.92** |
| Perplexity | 23,389 (590x worse) |

### Analysis

The residual connection hypothesis is **PARTIALLY VALIDATED**:
- Final hidden state achieves 0.92 correlation despite binary MLP weights
- This is much better than without residuals (would be ~0.34 correlation)
- The "information highway" does preserve the original signal

BUT **0.92 correlation is NOT ENOUGH** for language modeling because:
1. **Softmax exponential sensitivity**: `softmax(x)` amplifies small differences
2. **Cross-entropy is harsh**: Even small probability errors → large loss
3. **Perplexity = exp(loss)**: Further exponential amplification
4. **Vocabulary is huge**: 50,257 classes means tiny probability errors matter

### Layer-by-Layer Correlation (12-layer GPT-2)

```
Layer 0  (input):  1.0000  (unchanged embeddings)
Layer 1:          0.6246
Layer 2:          0.4941  (lowest)
Layer 3-8:        0.58-0.82 (partial recovery)
Layer 9-11:       0.34-0.52 (degrading)
Layer 12 (final): 0.9160  (residual recovery!)
```

The residual connections successfully recover correlation at the final layer, but the intermediate degradation still causes problems.

### Conclusion on Residual Insight

**The insight is correct but insufficient:**
- Residual connections DO preserve signal (0.92 vs ~0.34 without)
- But 0.92 correlation → 590x worse PPL, not usable
- Would need 0.999+ correlation for acceptable PPL
- Binary quantization cannot achieve 0.999+ correlation

## Real Test Results on GPT-2 (124M parameters) — *superseded*

> [!WARNING]
> Superseded. This table came from `tests/comprehensive_real_test.py` before the fix (per-tensor absmax INT quantizers that also
> quantized the embeddings, short eval text). In particular, "INT4 = 8,812" and "INT2 better than INT4" are artifacts.
> Harness numbers: INT4 g128 = 33.26 (RTN) / 31.28 (GPTQ); INT2 g128 = 3,017 (RTN) / 689 (GPTQ); FP32 = 29.95. See `CURRENT_STATE.md`.

| Method | Bits Per Param | Perplexity | vs FP32 |
|--------|----------------|------------|---------|
| FP32 (baseline) | 32.0 | 23.12 | 1.00x |
| Binary (sign) | 1.0 | 294,457 | **12,736x** |
| Binary (per-row scale) | 1.0 | 98,246 | **4,249x** |
| Ternary | 1.58 | 15,367 | **665x** |
| INT2 | 2.0 | 7,913 | **342x** |
| INT4 | 4.0 | 8,812 | **381x** |
| INT8 | 8.0 | 28.57 | **1.24x** |

## Key Findings

### 1. Binary Quantization is Catastrophic
- Simple sign-based binary quantization increases perplexity by **12,736x**
- The model generates garbage text (repeated tokens)
- Per-row scaling helps slightly but still produces unusable output

### 2. The "Correlation" Metric is Misleading
- Previous experiments showed "0.98 correlation" on synthetic data
- This does NOT translate to usable perplexity on real models
- Correlation measures local similarity, not functional correctness

### 3. Residual Connections Don't Save Binary
- While residual connections help preserve signal in synthetic tests
- They cannot compensate for the ~12,000x perplexity degradation
- The MLP layers' GELU nonlinearity amplifies quantization error

### 4. Only INT8 Maintains Quality — *superseded*
> Under the harness, INT8 is 29.99 (1.00×) and INT4 g128 + GPTQ is 31.28 (1.04×); INT4 is fine when quantized correctly.

- INT8 (8 bpp) achieves just 1.24x perplexity increase
- This is a 31.75x compression from FP32
- This aligns with industry practice (GPTQ, AWQ, etc.)

## Why Binary Fails

### Information Theory Perspective
- GPT-2 has ~124M parameters trained with massive gradients
- Each parameter encodes specific information learned from training
- Reducing 32 bits → 1 bit loses 31 bits of information per parameter
- Total information loss: 124M × 31 = 3.84 billion bits

### Error Propagation
- Layer 1 error: ~50% (binary can only represent sign)
- After 12 layers: (0.5)^12 correlation ≈ 0.00024
- This explains the ~4000-12000x perplexity increase

### GELU Sensitivity
- MLP layers use GELU: `x * Φ(x)`
- GELU has high curvature near 0
- Binary weights create large errors that GELU amplifies nonlinearly

## What WOULD Be Needed for 1.00 BPP

To achieve usable quality at 1.00 bpp, you would need:

1. **Calibration-Aware Training**: Train the model knowing it will be binary
2. **Knowledge Distillation**: Use FP32 teacher to guide binary student
3. **Learned Scaling**: Learn optimal per-row or per-group scaling factors
4. **Error Correction**: Additional bits for residual errors
5. **Architecture Changes**: Replace GELU with more quantization-friendly activations

### Realistic BPP Estimates for Usable Quality

Updated with harness numbers (GPT-2 124M, WikiText-2 test, post-training, bpp of the transformer-block linears):

| Target quality | Post-training bpp needed | Evidence |
|---|---|---|
| < 1.1× FP PPL | ~4.25 (INT4 g128 + GPTQ) | `gptq_w4_g128_gpt2`: 31.28 |
| < 1.5× FP PPL | ~3.25 (INT3 g128 + GPTQ) | `gptq_w3_g128_gpt2`: 41.43 |
| < 100× FP PPL | ~2.25 (INT2 g128 + GPTQ) | `gptq_w2_g128_gpt2`: 689 (23×) |
| Usable at ~1–1.6 bpp | Not achievable post-training | best ≤1.6 bpp result is 15,444 (516×) |

The old version of this table ("< 2× needs ~8 bits") was derived from the broken INT baselines.

## Conclusion

**The SALOMI project's claims of achieving "good quality at 1.00 bpp" are not supported by rigorous testing.**

The honest reality:
- Simple binary quantization produces unusable models
- Correctly implemented INT4 (group-wise, GPTQ) is near-lossless; the old INT8-only conclusion came from broken INT4 code
- True binary quantization requires model retraining, not just post-training quantization

## Recommendations

1. **For production**: Use INT4 with GPTQ/AWQ (near-lossless on GPT-2 under the harness); INT8 if you need zero risk
2. **For research**: Focus on training-aware binary quantization
3. **For SALOMI**: The stochastic BSDM-W approach may help but needs proper validation

## Files Created

- `tests/test_real_gpt2_validation.py` - Initial real GPT-2 test
- `tests/comprehensive_real_test.py` - Multi-method comparison
- This document: Honest assessment of findings

---

*Generated from rigorous testing on real GPT-2 weights, not synthetic data.*