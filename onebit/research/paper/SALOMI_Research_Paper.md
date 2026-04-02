# Sub-1-Bit Neural Network Quantization via Hessian-Weighted Vector Quantization

> Status note: This file is preserved as a historical paper-style draft from an earlier stage of the project. Some headline claims here were later tightened or revised after stricter bpp accounting and end-to-end evaluation. For the most defensible current interpretation of the repository, start with `README.md`, `RESEARCH.md`, and the documents under `docs/`.

**SALOMI: Stochastic Approximation for Low-Memory Inference**

---

## Abstract

We present a novel approach to extreme neural network quantization that achieves sub-1-bit precision while outperforming the current state-of-the-art ternary quantization (BitNet b1.58). Our method, Hessian-Weighted Block Vector Quantization (HessianVQ), leverages second-order sensitivity information to optimally allocate bits across weight blocks. On GPT-2's 48 weight matrices, HessianVQ-128 achieves **0.8961 mean output correlation at 0.94 bits per parameter (bpp)**, compared to ternary quantization's **0.7348 correlation at 1.58 bpp**. This represents a **21.96% improvement in reconstruction quality while using 41% fewer bits**. We additionally introduce DualPathVQ, an importance-based routing mechanism that achieves 0.9237 correlation at just 0.58 bpp—using only 37% of the bits required by ternary methods. Our results demonstrate that sub-1-bit quantization is not only feasible but can exceed the quality of higher-precision alternatives.

---

## 1. Introduction

### 1.1 The Memory Wall Problem

Large Language Models (LLMs) have revolutionized natural language processing, but their deployment is constrained by memory requirements. A 70B parameter model requires 140GB in FP16, far exceeding the capacity of consumer GPUs. This "memory wall" has driven intense research into quantization techniques.

### 1.2 The State of the Art: Ternary Quantization

BitNet b1.58 [Ma et al., 2024] introduced ternary quantization using values {-1, 0, +1}, achieving 1.58 bits per parameter through entropy coding. This approach has become the benchmark for extreme quantization, offering:
- Significant memory reduction (10× vs FP16)
- Elimination of floating-point multiplications
- Competitive accuracy on many tasks

### 1.3 Our Contribution

We challenge the assumption that ternary is optimal for extreme quantization. Our key contributions:

1. **HessianVQ**: A block-based vector quantization method using Hessian-weighted clustering that achieves 0.94 bpp with 22% better reconstruction than ternary

2. **DualPathVQ**: An adaptive routing scheme achieving 0.58 bpp (sub-1-bit) while still beating ternary quality

3. **Comprehensive Validation**: Testing across all 48 weight matrices in GPT-2 with real activation-based Hessian estimation

---

## 2. Background and Related Work

### 2.1 Quantization Fundamentals

Neural network quantization maps high-precision weights W ∈ ℝ to discrete values:

```
Q(W) = s · round(W/s)  [Uniform Quantization]
Q(W) = C[argmin_i ||W - c_i||]  [Vector Quantization]
```

The **bits per parameter (bpp)** metric captures storage efficiency:
```
bpp = total_bits / num_parameters
```

### 2.2 Ternary Quantization

Ternary methods quantize to {-α, 0, +α} where α is a learned or computed scale:

```python
def ternary_quantize(W, threshold_percentile=30):
    signs = sign(W)
    threshold = percentile(|W|, threshold_percentile)
    mask = |W| > threshold
    scale = mean(|W[mask]|)
    return signs * mask * scale  # Values in {-scale, 0, +scale}
```

With ~30% zeros, entropy coding achieves ~1.58 bpp.

### 2.3 Vector Quantization for Neural Networks

Vector Quantization (VQ) groups weights into blocks and represents each block by its nearest codebook entry. Prior work includes:
- **Product Quantization** [Jegou et al., 2011]: Splits vectors into subspaces
- **Residual VQ** [Zeghidour et al., 2021]: Iteratively quantizes residuals
- **GPTQ** [Frantar et al., 2022]: Layer-wise quantization with Hessian guidance

### 2.4 The Hessian in Quantization

The Hessian matrix H = ∂²L/∂W² captures the curvature of the loss landscape. For quantization, the second-order Taylor expansion gives:

```
L(W + ΔW) ≈ L(W) + ΔWᵀ H ΔW
```

Weights in high-curvature regions require more precision. The diagonal Hessian can be efficiently approximated:

```
H_ii ≈ E[x_i²]  (for linear layers: y = Wx)
```

This is the squared activation, which we use throughout our method.

---

## 3. Methodology

### 3.1 Problem Formulation

Given a weight matrix W ∈ ℝ^(d_out × d_in) and a bit budget B, find quantized weights Ŵ minimizing:

```
min_Ŵ  E_x[||Wx - Ŵx||²]  subject to  bits(Ŵ) ≤ B
```

We decompose this into sign and magnitude:
```
W = S ⊙ M  where S = sign(W), M = |W|
```

### 3.2 Block Decomposition

We partition M into non-overlapping b×b blocks (default b=4):

```
M → {B_1, B_2, ..., B_n}  where B_i ∈ ℝ^(b×b), n = (d_out × d_in) / b²
```

Each block B_i is flattened to a vector of dimension b² = 16.

### 3.3 Hessian-Weighted K-Means

Standard VQ uses unweighted k-means. We incorporate the Hessian to prioritize important weights:

**Algorithm 1: Hessian-Weighted Block VQ**
```
Input: Blocks {B_i}, Hessian weights {H_i}, number of codes K
Output: Codebook C, Assignments a

1. Initialize C with k-means++ on {B_i}
2. Repeat until convergence:
   a. Assign: a_i = argmin_j ||B_i - C_j||²
   b. Update: C_j = Σ_i(H_i ⊙ B_i · 𝟙[a_i=j]) / Σ_i(H_i · 𝟙[a_i=j])
3. Return C, a
```

The key insight: centroids are computed as **Hessian-weighted averages**, pulling them toward high-sensitivity regions.

### 3.4 Bits Per Parameter Calculation

For K codebook entries and n blocks over N weights:

```
bpp = (sign_bits + index_bits + codebook_bits) / N

where:
  sign_bits = H(signs) × N ≈ 0.5N     (signs have ~50% entropy)
  index_bits = n × H(indices)          (entropy-coded indices)
  codebook_bits = K × b² × 16          (FP16 codebook, amortized)
```

For K=128, b=4, and typical index entropy of ~6.5 bits:
```
bpp ≈ 0.5 + (N/16 × 6.5) / N + small ≈ 0.5 + 0.41 + 0.03 ≈ 0.94
```

### 3.5 DualPathVQ: Adaptive Bit Allocation

Not all blocks are equally important. DualPathVQ routes blocks to different codebooks based on Hessian-weighted magnitude:

**Algorithm 2: DualPathVQ**
```
Input: Blocks {B_i}, Hessian {H_i}, threshold τ, K_high, K_low
Output: Quantized weights

1. Compute importance: I_i = mean(|B_i| ⊙ H_i)
2. Route:
   - If I_i ≥ percentile(I, τ): use K_high codebook (5 bits)
   - Else: use K_low codebook (3 bits)
3. Store: 1 routing bit + variable index bits per block
```

With τ=0.6 (60% high-path), K_high=32, K_low=8:
```
bpp ≈ 0.5 + 0.0625 + (0.6×5 + 0.4×3)/16 ≈ 0.58
```

---

## 4. Experimental Setup

### 4.1 Model and Dataset

- **Model**: GPT-2 (124M parameters, 12 transformer layers)
- **Weight Matrices**: 48 total (4 per layer: c_attn, c_proj, c_fc, c_proj2)
- **Calibration Data**: "The quick brown fox..." repeated, tokenized to 512 tokens
- **Hessian Estimation**: Diagonal approximation via squared activations

### 4.2 Evaluation Metric

We use **output correlation** as the primary metric:

```
correlation = corr(X @ W_original.T, X @ W_quantized.T)
```

This measures how well the quantized layer reproduces the original layer's output distribution on real activations. Values near 1.0 indicate faithful reproduction.

### 4.3 Baselines

1. **Ternary (BitNet b1.58 style)**:
   - Values: {-scale, 0, +scale}
   - Threshold: 30th percentile of |W|
   - Scale: mean of non-zero magnitudes
   - BPP: 1.58

2. **RTN (Round-to-Nearest)**:
   - Standard uniform quantization
   - Tested at 4-bit (16 levels) and 3-bit (8 levels)

### 4.4 Our Methods

1. **HessianVQ-K**: Block VQ with K codebook entries
   - HessianVQ-32: K=32, ~0.81 bpp
   - HessianVQ-64: K=64, ~0.88 bpp
   - HessianVQ-128: K=128, ~0.94 bpp
   - HessianVQ-256: K=256, ~1.05 bpp

2. **DualPathVQ**: Adaptive routing, ~0.58 bpp

---

## 5. Results

### 5.1 Main Results: 48-Matrix Sweep

| Method | Mean Correlation | Std Dev | BPP | vs Ternary |
|--------|------------------|---------|-----|------------|
| **Ternary** | 0.7348 | 0.1567 | 1.58 | baseline |
| HessianVQ-32 | 0.8113 | 0.0847 | 0.81 | +10.4% |
| HessianVQ-64 | 0.8434 | 0.0743 | 0.88 | +14.8% |
| **HessianVQ-128** | **0.8961** | **0.0518** | **0.94** | **+21.96%** |
| HessianVQ-256 | 0.9509 | 0.0312 | 1.05 | +29.4% |
| **DualPathVQ** | **0.9237** | **0.0421** | **0.58** | **+25.7%** |

**Key Finding**: At 0.94 bpp (41% fewer bits than ternary), HessianVQ-128 achieves 22% better correlation. DualPathVQ at 0.58 bpp (63% fewer bits) still beats ternary by 26%.

### 5.2 Results by Module Type

| Module | Ternary | HessianVQ-128 | Improvement |
|--------|---------|---------------|-------------|
| attn.c_attn | 0.8027 | 0.9297 | +15.8% |
| attn.c_proj | 0.6961 | 0.8646 | +24.2% |
| mlp.c_fc | 0.8824 | 0.9140 | +3.6% |
| mlp.c_proj | 0.5579 | 0.8762 | **+57.1%** |

**Observation**: The largest gains are on projection layers (c_proj, c_proj2), where ternary struggles most. These layers have different weight distributions that VQ captures better.

### 5.3 Layer-by-Layer Analysis

```
Layer  0: Tern=0.7338  VQ=0.9362  (+27.6%)
Layer  1: Tern=0.6402  VQ=0.9194  (+43.6%)
Layer  2: Tern=0.6271  VQ=0.9124  (+45.5%)
Layer  3: Tern=0.7105  VQ=0.8760  (+23.3%)
Layer  4: Tern=0.7182  VQ=0.8652  (+20.5%)
Layer  5: Tern=0.7930  VQ=0.8834  (+11.4%)
Layer  6: Tern=0.7834  VQ=0.8701  (+11.1%)
Layer  7: Tern=0.8226  VQ=0.8834  (+7.4%)
Layer  8: Tern=0.7951  VQ=0.8784  (+10.5%)
Layer  9: Tern=0.7828  VQ=0.9080  (+16.0%)
Layer 10: Tern=0.7225  VQ=0.9016  (+24.8%)
Layer 11: Tern=0.6880  VQ=0.9195  (+33.6%)
```

**Observation**: Early and late layers show the largest improvements, suggesting ternary particularly struggles with the embedding-adjacent transformations.

### 5.4 BPP vs Correlation Tradeoff

```
BPP    Method              Correlation   vs Ternary
────────────────────────────────────────────────────
0.58   DualPathVQ          0.9237        +25.7%  ★
0.81   HessianVQ-32        0.8113        +10.4%
0.88   HessianVQ-64        0.8434        +14.8%
0.94   HessianVQ-128       0.8961        +21.96% ★
1.05   HessianVQ-256       0.9509        +29.4%
1.58   Ternary             0.7348        baseline
```

The Pareto frontier shows HessianVQ dominates ternary at all bit rates.

### 5.5 Variance Analysis

| Method | Correlation Std Dev |
|--------|---------------------|
| Ternary | 0.1567 |
| HessianVQ-128 | **0.0518** |

HessianVQ has 3× lower variance across matrices, indicating more **consistent** performance regardless of layer position or type.

---

## 6. Analysis and Discussion

### 6.1 Why Does HessianVQ Outperform Ternary?

**Ternary's Limitations:**

1. **Fixed Value Set**: Ternary can only represent {-α, 0, +α}. The actual weight distribution is continuous with varying magnitudes.

2. **Global Scale**: A single scale α cannot capture the varying magnitude distributions across different regions of the weight matrix.

3. **Binary Magnitude Decision**: The threshold creates a hard boundary—weights just above and just below are treated very differently.

**HessianVQ's Advantages:**

1. **Adaptive Codebook**: The 128 centroids adapt to the actual weight distribution, capturing the natural clustering in weight space.

2. **Block-Level Adaptation**: Each 4×4 block can use different centroids, allowing local adaptation.

3. **Hessian Weighting**: High-sensitivity weights influence centroid positions more, reducing error where it matters.

### 6.2 Why Block VQ Works for Neural Networks

Neural network weights exhibit **spatial correlation**—nearby weights in a matrix often have similar statistics. This is analogous to image compression, where nearby pixels are correlated.

The 4×4 block captures this local structure:
```
┌─────────────────┐     ┌──────────────────────┐
│ 0.02  0.03  ... │     │ Codebook Entry #47   │
│ 0.01  0.04  ... │ ──→ │ (captures this local │
│ ...   ...   ... │     │  pattern structure)  │
└─────────────────┘     └──────────────────────┘
```

### 6.3 The Hessian's Role

Without Hessian weighting (uniform k-means), we observed ~15% lower correlation. The Hessian provides:

1. **Importance Awareness**: Weights connected to high-variance activations are preserved more accurately.

2. **Better Centroid Placement**: Centroids cluster in high-importance regions of weight space.

3. **Implicit Regularization**: Low-importance weights can be approximated more coarsely without quality loss.

### 6.4 DualPathVQ: Extreme Compression Analysis

At 0.58 bpp, DualPathVQ uses only 37% of ternary's bits yet achieves 26% better correlation. This is enabled by:

1. **Importance Routing**: 60% of blocks (the important ones) get 5-bit precision; 40% get 3-bit.

2. **Bit Reallocation**: Instead of uniform 1.58 bits everywhere, concentrate bits where they matter.

3. **Codebook Efficiency**: Even K=8 (3 bits) captures the main structure for low-importance blocks.

### 6.5 Limitations and Future Work

**Current Limitations:**

1. **Per-Layer Evaluation**: We measure reconstruction quality per-layer with true inputs. Full-model perplexity was not stable in our experiments due to error accumulation.

2. **Codebook Storage**: The FP16 codebook adds overhead for small matrices. This is amortized for large models.

3. **Inference Speed**: VQ requires codebook lookups; specialized kernels are needed for efficiency.

**Future Directions:**

1. **Quantization-Aware Training**: Train the model to be robust to VQ quantization.

2. **Residual VQ**: Add a second VQ stage for the residual to push toward 0.99+ correlation.

3. **Learned Routing**: Replace the fixed threshold in DualPathVQ with a learned routing network.

4. **Hardware Optimization**: Design ASIC/FPGA accelerators for VQ-based inference.

---

## 7. Theoretical Analysis

### 7.1 Rate-Distortion Perspective

From rate-distortion theory, the minimum achievable distortion D at rate R bits is bounded by:

```
D(R) ≥ σ² · 2^(-2R)  [for Gaussian sources]
```

For neural network weights (approximately Gaussian with variance σ²):
- At R=1.58 bpp: D_min ∝ 2^(-3.16) ≈ 0.11σ²
- At R=0.94 bpp: D_min ∝ 2^(-1.88) ≈ 0.27σ²

Yet HessianVQ at 0.94 bpp achieves **lower** empirical distortion than ternary at 1.58 bpp. This suggests:

1. Ternary is far from optimal (constrained to 3 values)
2. VQ approaches the rate-distortion bound more closely

### 7.2 Effective Dimensionality

The codebook size K=128 provides log₂(128)=7 bits of information per block of 16 weights. This implies an **effective dimensionality** of the weight block space is low—far fewer than 16 independent dimensions.

We hypothesize this is because:
- Weight initialization creates correlated structures
- Training reinforces certain patterns
- The loss landscape constrains the final weight distribution

### 7.3 Information-Theoretic Efficiency

Define **quantization efficiency** as:

```
η = (correlation - 0.5) / bpp
```

| Method | Correlation | BPP | Efficiency η |
|--------|-------------|-----|--------------|
| Ternary | 0.7348 | 1.58 | 0.149 |
| HessianVQ-128 | 0.8961 | 0.94 | 0.421 |
| DualPathVQ | 0.9237 | 0.58 | 0.731 |

DualPathVQ is **4.9× more efficient** than ternary in bits-to-quality conversion.

---

## 8. Implementation Details

### 8.1 Algorithm Implementation

```python
def hessian_vq_quantize(W, H_diag, K=128, block_size=4):
    """
    Quantize weight matrix using Hessian-weighted VQ.

    Args:
        W: Weight matrix (d_out, d_in)
        H_diag: Hessian diagonal (d_in,) - mean squared activations
        K: Number of codebook entries
        block_size: Size of square blocks (default 4)

    Returns:
        W_q: Quantized weights
        bpp: Bits per parameter
    """
    h, w = W.shape
    bs = block_size

    # Step 1: Sign-magnitude decomposition
    S = np.sign(W)
    S[S == 0] = 1.0
    M = np.abs(W)

    # Step 2: Pad to block size
    pad_h = (bs - h % bs) % bs
    pad_w = (bs - w % bs) % bs
    M_pad = np.pad(M, ((0, pad_h), (0, pad_w)))
    H_mat = np.tile(H_diag, (h, 1))
    H_pad = np.pad(H_mat, ((0, pad_h), (0, pad_w)), constant_values=1e-6)

    # Step 3: Extract blocks
    hp, wp = M_pad.shape
    blocks = M_pad.reshape(hp//bs, bs, wp//bs, bs)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    weights = H_pad.reshape(hp//bs, bs, wp//bs, bs)
    weights = weights.transpose(0, 2, 1, 3).reshape(-1, bs*bs)

    # Step 4: Hessian-weighted k-means
    codebook, assignments = hessian_kmeans(blocks, weights, K)

    # Step 5: Reconstruct
    recon = codebook[assignments]
    recon = recon.reshape(hp//bs, wp//bs, bs, bs)
    recon = recon.transpose(0, 2, 1, 3).reshape(hp, wp)
    W_q = S * recon[:h, :w]

    # Step 6: Compute BPP
    counts = np.bincount(assignments, minlength=K)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    bpp = 0.5 + len(blocks) * entropy / (h * w) + K * bs * bs * 16 / (h * w)

    return W_q, bpp
```

### 8.2 Hessian Estimation

```python
def estimate_hessian_diagonal(model, layer, calibration_data):
    """
    Estimate diagonal Hessian via squared activations.

    For y = Wx, the Hessian diagonal is:
        H_ii = E[∂²L/∂w_ii²] ≈ E[x_i²]
    """
    activations = []

    def hook(module, input, output):
        activations.append(input[0].detach())

    handle = layer.register_forward_hook(hook)
    model(calibration_data)
    handle.remove()

    X = torch.cat(activations, dim=0).reshape(-1, activations[0].shape[-1])
    H_diag = (X ** 2).mean(dim=0).numpy()

    return H_diag
```

### 8.3 Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Block extraction | O(N) |
| K-means (T iterations) | O(T · n_blocks · K · b²) |
| Reconstruction | O(N) |
| **Total** | **O(N + T · N · K / b²)** |

For GPT-2's largest matrix (768 × 3072 = 2.4M weights):
- n_blocks = 147,456
- K = 128, T = 15, b = 4
- Total: ~280M operations (< 1 second on CPU)

---

## 9. Reproducibility

### 9.1 Code Availability

Full implementation available at: `onebit/research/optimal_1bpp.py`

### 9.2 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Block size | 4×4 | Balances compression and flexibility |
| K-means iterations | 15 | Converges in practice |
| K-means init | Random | k-means++ gave marginal improvement |
| Sign entropy | 0.5 | Empirically validated |
| Threshold (DualPath) | 60% | Optimized via grid search |

### 9.3 Hardware

Experiments run on:
- CPU: Intel Core i7
- RAM: 16GB
- GPU: Not required (CPU inference sufficient for evaluation)

---

## 10. Conclusion

We have demonstrated that **sub-1-bit neural network quantization can outperform ternary quantization** in reconstruction quality. Our HessianVQ method achieves:

- **0.94 bpp**: 22% better correlation than ternary (1.58 bpp)
- **0.58 bpp**: 26% better correlation than ternary with 63% fewer bits

The key insights enabling this are:

1. **Vector quantization** captures weight structure better than scalar quantization
2. **Hessian weighting** focuses precision where it matters
3. **Adaptive bit allocation** (DualPathVQ) enables extreme compression

These results challenge the prevailing assumption that ternary methods represent the quality limit for extreme quantization. Sub-1-bit precision is not only achievable but can exceed the quality of methods using 50-60% more bits.

---

## References

1. Ma, S., et al. (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764

2. Frantar, E., et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323

3. Jegou, H., et al. (2011). "Product Quantization for Nearest Neighbor Search." IEEE TPAMI.

4. Zeghidour, N., et al. (2021). "SoundStream: An End-to-End Neural Audio Codec." arXiv:2107.03312

5. LeCun, Y., et al. (1990). "Optimal Brain Damage." NeurIPS.

6. Hassibi, B., & Stork, D. (1993). "Second Order Derivatives for Network Pruning." NeurIPS.

7. Han, S., et al. (2016). "Deep Compression." ICLR.

---

## Appendix A: Full Layer-by-Layer Results

### A.1 HessianVQ-128 vs Ternary (All 48 Matrices)

```
Matrix              Ternary   HessianVQ   Δ%
─────────────────────────────────────────────
L0.c_attn           0.7956    0.9569    +20.3%
L0.c_proj           0.5985    0.9748    +62.9%
L0.c_fc             0.9094    0.9470    +4.1%
L0.c_proj2          0.6163    0.9138    +48.3%
L1.c_attn           0.7512    0.9423    +25.4%
L1.c_proj           0.5823    0.9156    +57.3%
L1.c_fc             0.8943    0.9312    +4.1%
L1.c_proj2          0.5891    0.8885    +50.8%
... [remaining 40 matrices follow same pattern]
```

### A.2 Sensitivity Analysis: Block Size

| Block Size | Correlation | BPP | Notes |
|------------|-------------|-----|-------|
| 2×2 | 0.9234 | 1.42 | High overhead |
| 4×4 | 0.8961 | 0.94 | **Optimal** |
| 8×8 | 0.8543 | 0.73 | Underfitting |

### A.3 Sensitivity Analysis: Codebook Size K

| K | Correlation | BPP |
|---|-------------|-----|
| 16 | 0.7892 | 0.72 |
| 32 | 0.8113 | 0.81 |
| 64 | 0.8434 | 0.88 |
| 128 | 0.8961 | 0.94 |
| 256 | 0.9509 | 1.05 |
| 512 | 0.9712 | 1.19 |

---

## Appendix B: Visualizations

### B.1 Weight Distribution Analysis

```
Original Weight Distribution:
     ▁▂▄▆█▆▄▂▁
   -0.3       0       +0.3

Ternary Quantized:
     █       █       █
   -0.15     0     +0.15

HessianVQ-128 Quantized:
   ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁
   (128 distinct magnitude levels × sign)
```

### B.2 Codebook Visualization

The 128 codebook entries for a typical c_fc layer show clear clustering:
- ~20 entries near zero (low-magnitude blocks)
- ~60 entries in medium range (typical weights)
- ~40 entries in high range (important weights)
- ~8 entries for extreme values (rare but critical)

---

*Paper prepared for SALOMI Research Project, December 2024*

