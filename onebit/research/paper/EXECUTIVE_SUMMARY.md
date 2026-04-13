# SALOMI Research: Executive Summary

> Status note: This summary reflects an earlier, more optimistic draft narrative. Several headline claims were later re-evaluated under stricter bpp accounting and end-to-end testing. For the current interpretation, use `README.md`, `docs/VALIDATED_RESULTS.md`, `docs/HONEST_ASSESSMENT.md`, and the index in `RESEARCH.md`.

## Sub-1-Bit Neural Network Quantization That Beats Ternary

---

## The Bottom Line

We achieved **sub-1-bit quantization (0.58-0.94 bpp)** that **outperforms ternary quantization (1.58 bpp)** in reconstruction quality.

| Metric | Ternary (BitNet) | Our Method | Winner |
|--------|------------------|------------|--------|
| **Bits per weight** | 1.58 | **0.94** | ✅ Ours (41% smaller) |
| **Correlation** | 0.7348 | **0.8961** | ✅ Ours (22% better) |
| **Consistency** | σ = 0.157 | **σ = 0.052** | ✅ Ours (3× more stable) |

---

## What We Built

### HessianVQ: Hessian-Weighted Block Vector Quantization

Instead of quantizing each weight independently to {-1, 0, +1}, we:

1. **Group weights into 4×4 blocks** (16 weights each)
2. **Use the Hessian** to identify which weights matter most
3. **Apply Vector Quantization** to find similar blocks
4. **Store just the codebook index** (7 bits for 128 entries)

Result: **0.94 bpp with 22% better quality than ternary's 1.58 bpp**

### DualPathVQ: Adaptive Bit Allocation

Not all blocks are equally important. We route:
- **Important blocks** (60%) → High-quality path (K=32, 5 bits)
- **Less important blocks** (40%) → Efficient path (K=8, 3 bits)

Result: **0.58 bpp with 26% better quality than ternary**

---

## Key Results (48 GPT-2 Weight Matrices)

```
Method              Mean Corr    Std Dev     BPP      vs Ternary
────────────────────────────────────────────────────────────────
Ternary             0.7348       0.1567      1.58     baseline
HessianVQ-32        0.8113       0.0847      0.81     +10.4%
HessianVQ-128       0.8961       0.0518      0.94     +21.96%  ★
DualPathVQ          0.9237       0.0421      0.58     +25.7%   ★
```

---

## Why It Works

### 1. Neural Network Weights Have Structure

Nearby weights in a matrix are often similar. Block VQ captures this:
- 16 weights per block share a single codebook entry
- The codebook adapts to the weight distribution

### 2. Not All Weights Are Equal

The Hessian tells us which weights matter:
- High Hessian = output is sensitive to this weight → preserve carefully
- Low Hessian = output barely changes → approximate roughly

### 3. Ternary Wastes Bits on Unimportant Weights

Ternary gives equal precision everywhere. We allocate bits where they matter.

---

## Visual Comparison

```
TERNARY QUANTIZATION (1.58 bpp):
    Original:  [0.02, -0.15, 0.08, 0.01, 0.25, -0.18, ...]
    Quantized: [0,    -0.12, 0.12, 0,    0.12, -0.12, ...]  ← Only 3 values!
    
HESSIANVQ (0.94 bpp):  
    Original:  [0.02, -0.15, 0.08, 0.01, 0.25, -0.18, ...]
    Quantized: [0.03, -0.14, 0.09, 0.02, 0.24, -0.17, ...]  ← Much closer!
```

---

## Module-by-Module Breakdown

| Module Type | Ternary | HessianVQ | Improvement |
|-------------|---------|-----------|-------------|
| attn.c_attn | 0.8027 | 0.9297 | **+15.8%** |
| attn.c_proj | 0.6961 | 0.8646 | **+24.2%** |
| mlp.c_fc | 0.8824 | 0.9140 | **+3.6%** |
| mlp.c_proj | 0.5579 | 0.8762 | **+57.1%** |

Ternary struggles most on projection layers. HessianVQ excels everywhere.

---

## Implications

### For Model Deployment
- **41% memory reduction** vs ternary at equal or better quality
- **63% memory reduction** with DualPathVQ at better quality
- More models fit in consumer GPU memory

### For Research
- Sub-1-bit quantization is viable and can beat higher-precision methods
- Hessian-guided methods are key to extreme compression
- Block-based VQ is underexplored for neural networks

### For Hardware
- VQ-based inference needs codebook lookups (different from matmul)
- Potential for specialized accelerators
- Sign + index format is amenable to bit-packing

---

## Next Steps

1. **Full-Model Perplexity**: Validate end-to-end quality (current results are per-layer)
2. **Larger Models**: Test on 7B+ parameter models
3. **Inference Kernels**: Build optimized CUDA/Triton kernels
4. **Quantization-Aware Training**: Train models to be VQ-friendly

---

## Citation

```bibtex
@article{salomi2024,
  title={Sub-1-Bit Neural Network Quantization via Hessian-Weighted Vector Quantization},
  author={SALOMI Research Team},
  journal={arXiv preprint},
  year={2024}
}
```

---

*SALOMI: Stochastic Approximation for Low-Memory Inference*
*December 2024*

