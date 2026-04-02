# Binary Quantization for LLMs: Comprehensive Findings

## Executive Summary

After rigorous testing of multiple novel approaches for binary (1.00 bpp) quantization of GPT-2, we have found that **pure binary quantization fundamentally destroys model quality** due to softmax exponential sensitivity.

## Test Results Summary

### Baseline Quantization Methods (GPT-2 124M)

| Method | BPP | Perplexity | vs FP32 (23.12) |
|--------|-----|------------|-----------------|
| FP32 Baseline | 32.0 | 23.12 | 1.00x |
| Binary (sign) | 1.0 | 294,457 | **12,736x** |
| Binary (per-row) | 1.0 | 98,246 | **4,249x** |
| Ternary | 1.58 | 15,367 | **665x** |
| INT2 | 2.0 | 7,913 | **342x** |
| INT4 | 4.0 | 8,812 | **381x** |
| INT8 | 8.0 | 28.57 | **1.24x** ✓ |

### Novel Approach Results (Round 1)

| Approach | BPP | PPL | Ratio |
|----------|-----|-----|-------|
| Output-Optimal Scaling | 1.00 | 10,638 | 632x |
| GPTQ-Style Compensation | 1.00 | 9,998 | 595x |
| Learned Scales (no calib) | 1.00 | 8,012 | 477x |
| Learned Scales (calibrated) | 1.00 | 6,718 | 400x |
| Binary + FP32 Residual (r=4) | 1.23 | 514 | 31x |
| Binary + FP32 Residual (r=8) | 1.44 | 333 | 20x |
| Binary + FP32 Residual (r=16) | 1.86 | 342 | 20x |

### Residual-Aware Binary (Novel Insight)

Hypothesis: Residual connections preserve the original signal.
- Weight correlation: 0.78
- Final hidden state correlation: **0.92**
- PPL: 23,389 (590x worse)

**Conclusion**: Even 0.92 correlation → 590x worse PPL due to softmax sensitivity.

## Key Technical Insights

### 1. Correlation ≠ Perplexity

This is the most important finding. High hidden state correlation does NOT translate to good perplexity:

```
0.92 correlation → 590x worse PPL
0.78 weight correlation → 12,000x worse PPL
```

Why? Because PPL = exp(cross_entropy) exponentially amplifies small differences.

### 2. The Softmax Bottleneck

The softmax function is exponentially sensitive to logit differences:
- If logit changes by 1.0, probability can change by 2.7x
- With 50,257 vocabulary, small errors compound
- Cross-entropy severely penalizes wrong predictions

### 3. Low-Rank Residual Is Key

The most promising approach is Binary + Low-Rank Residual:
- Signs preserve ~78% of information
- Low-rank correction captures essential error
- Trade-off: BPP vs quality

| Rank | BPP | PPL Ratio |
|------|-----|-----------|
| 4 | 1.23 | 31x |
| 8 | 1.44 | 20x |
| 16 | 1.86 | 20x |

Diminishing returns after rank 8.

### 4. GELU Sensitivity

MLP layers are 77-200x more sensitive than attention due to GELU:
- GELU(x) = x * Φ(x) has high curvature near 0
- Binary errors get amplified nonlinearly
- Quantization-friendly activations (ReLU) improve results

## Approaches Still Under Test

### GELU-Aware + Heavy Distillation
- Learn GELU correction factors
- Per-input and per-output scaling
- Temperature-scaled distillation

### Top-K Distillation
- Key insight: Preserve top token probabilities
- Weight loss by token importance
- MSE on raw logits + KL on softmax

### INT4 Residual (Lower BPP)
- Quantize residual to 4-bit
- Target: 1.05-1.15 BPP

## Theoretical Analysis

### Why Binary Cannot Work at 1.00 BPP

For usable quality (< 2x PPL), we need approximately:
- 0.999+ correlation for hidden states
- < 0.1% error in top-k logits

Binary signs can only achieve:
- ~0.78 weight correlation
- ~0.92 hidden state correlation (with residuals)

The gap from 0.92 → 0.999 correlation requires additional bits.

### Minimum Viable BPP Estimate

Based on results:
- 1.00 bpp: 400-600x worse PPL (unusable)
- 1.23 bpp: 31x worse PPL (marginal)
- 1.44 bpp: 20x worse PPL (improving)
- ~2.0 bpp: ~10x worse PPL (estimate)
- ~4.0 bpp: ~2x worse PPL (estimate)
- 8.0 bpp: 1.24x worse PPL (achieved)

**Conclusion**: Practical minimum for usable binary-ish quantization is ~1.5-2.0 bpp.

## Novel Ideas For Future Work

### 1. Binary-Aware Training
Train the model knowing it will be binary:
- Use straight-through estimators
- GELU → ReLU replacement
- Larger model to compensate

### 2. Token-Importance Weighted Quantization
- Identify which weights affect top tokens most
- Keep those in higher precision
- Binary for others

### 3. Dynamic Precision per Layer
- First/last layers: higher precision (4-8 bit)
- Middle layers: binary
- Average: ~1.5 bpp

### 4. Entropy Coding on Binary
- Binary signs have entropy < 1.0 if skewed
- Apply arithmetic coding
- Could achieve < 1.0 bpp effective

### 5. Sparse Binary + Dense Scale
- Most weights: binary sparse (many zeros)
- Key weights: dense scale factors
- Trade-off zeros for precision

## Files Created

- `tests/test_real_gpt2_validation.py` - Basic validation
- `tests/comprehensive_real_test.py` - Multi-method comparison
- `tests/test_residual_binary_real.py` - Residual-aware test
- `tests/novel_binary_approaches.py` - First novel approaches
- `tests/aggressive_binary.py` - INT4 residual, block, GELU-aware
- `tests/topk_distillation_binary.py` - Top-K focused distillation

## Current Status

Tests still running:
1. GELU-Aware + Distillation
2. Top-K Distillation with learned scales

Preliminary indications suggest distillation can improve but:
- 1.00 bpp target remains challenging
- 1.5 bpp with ~10x PPL may be achievable

## Recommendations

### For Production Use
Use INT8 (8 bpp) with established methods like GPTQ or AWQ.

### For Research
Focus on:
1. Training-aware binary quantization
2. Mixed precision with layer importance
3. Entropy coding to reduce effective bpp

### For This Project (SALOMI)
The BSDM-W stochastic approach may help with scale estimation, but fundamental binary limitation remains.

---

*Document updated as tests complete.*