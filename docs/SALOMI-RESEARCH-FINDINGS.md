# SALOMI Research Findings: Breaking the Binary-Ternary Barrier

**Stochastic Approximation for Low-Memory Inference**

---

## Executive Summary

This document presents comprehensive research findings from the SALOMI project, focused on achieving **1.00 bits-per-parameter (bpp)** quantization that matches or exceeds the quality of **1.58-bit ternary (BitNet b1.58)** quantization.

### Key Findings

After testing **30+ quantization methods** including ideas from ChatGPT and Claude, we found:

1. **Single-layer correlation improvements DO NOT translate to end-to-end quality**
   - Methods showing -4% gap on single layers → +2,000,000% perplexity degradation

2. **MLP layers are 200x more sensitive than attention** due to GELU amplification
   - Binary Attention: +11,161% perplexity
   - Binary MLP: +2,167,425% perplexity

3. **GELU amplifies quantization errors exponentially**
   - 96% of activations in GELU-sensitive region
   - Sign flip near 0 → 185% error per activation
   - Errors compound through 12 layers

4. **Post-hoc quantization fundamentally breaks pre-trained models**
   - The only working path is training-time quantization (BitNet approach)
   - Model must learn to avoid GELU-sensitive regions during training

5. Previous findings on single-layer metrics:
   - LowRank r=2 at 1.10 bpp: -4.2% vs ternary (single-layer only)
   - Row+Col at 1.05 bpp: -4.7% vs ternary (single-layer only)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background: Why Ternary Beats Binary](#2-background-why-ternary-beats-binary)
3. [Initial Approaches Tested](#3-initial-approaches-tested)
4. [The Key Insight: Compression Changes Everything](#4-the-key-insight-compression-changes-everything)
5. [Entropy-Shaped Binary Training](#5-entropy-shaped-binary-training)
6. [Block-Structured Signs](#6-block-structured-signs)
7. [Hybrid Approach: The Breakthrough](#7-hybrid-approach-the-breakthrough)
8. [CTG Integration](#8-ctg-integration)
9. [BSDM-W Calibration](#9-bsdm-w-calibration)
10. [Conclusions and Future Work](#10-conclusions-and-future-work)
11. [Advanced Approaches: ChatGPT's Novel Ideas](#11-advanced-approaches-chatgpts-novel-ideas)
12. [Implementation Files](#12-implementation-files)
13. [Transform-Domain Experiments](#13-transform-domain-experiments-novel-ideas-v1--v2)
14. [Magnitude Recovery: The Breakthrough](#14-magnitude-recovery-the-breakthrough)
15. [The Core Problem Statement](#15-the-core-problem-statement)
16. [Solution Space for True 1.0 bpp](#16-solution-space-for-true-10-bpp)
17. [Files Created in This Research](#17-files-created-in-this-research)
18. [Final Summary Table](#18-final-summary-table)
19. [Claude's First-Principles Ideas](#19-claudes-first-principles-ideas-november-2025)
20. [Corrected Summary: What Actually Works](#20-corrected-summary-what-actually-works)
21. [End-to-End Perplexity Analysis (Critical Discovery)](#21-end-to-end-perplexity-analysis-critical-discovery)

---

## 1. Problem Statement

### The Challenge

Modern neural networks require billions of parameters stored as 32-bit floats. Quantization reduces memory and compute requirements, with two main approaches:

| Quantization | Levels | BPP | Example |
|--------------|--------|-----|---------|
| Binary | 2: {-1, +1} | 1.00 | Sign-only |
| Ternary | 3: {-1, 0, +1} | 1.58 | BitNet b1.58 |

**Goal**: Achieve ternary-level quality at binary bit-rate (1.00 bpp).

### Why This Matters

- **Memory**: 1.0 bpp uses 37% less memory than 1.58 bpp
- **Compute**: Binary operations are faster (XNOR + popcount)
- **Deployment**: Enables larger models on edge devices

---

## 2. Background: Why Ternary Beats Binary

### The Information-Theoretic View

Ternary quantization stores two pieces of information:
1. **Sign**: Which direction ({-1, +1}) - 1 bit
2. **Importance**: Whether to include (0 = skip) - ~0.58 bits

Binary only stores the sign, treating all weights as equally important.

### The Zero-Mask Advantage

For Gaussian weights, ~30% have small magnitudes that add noise rather than signal:

```
Weight Distribution:
|.....****####****.....|
      ↑    ↑    ↑
    small  OK  small
    (noise)    (noise)

Ternary: zeros out small weights → cleaner signal
Binary:  keeps all weights → noise accumulates
```

### Experimental Baseline

On random regression tasks with Gaussian weights:

| Method | Correlation | BPP |
|--------|-------------|-----|
| FP32 | 1.000 | 32.00 |
| Binary | 0.72 | 1.00 |
| **Ternary** | **0.89** | **1.58** |

Gap: Binary is ~17-20% worse than ternary.

---

## 3. Initial Approaches Tested

We implemented and tested 15+ quantization approaches in `onebit/research/unified_1bit.py`:

### 3.1 Post-Training Quantization Methods

| Method | Description | Result |
|--------|-------------|--------|
| Sign-only | Basic {-1, +1} | 0.72 correlation |
| Learned Basis | α*B + β with learned scales | Minimal improvement |
| Importance-Weighted | Weight by magnitude | Requires magnitude storage |
| Hadamard Domain | Quantize in Walsh-Hadamard basis | No significant gain |
| Residual Binary | Two-level residual | ~0.80 correlation (2 bpp) |

### 3.2 Training-Aware Methods

| Method | Description | Result |
|--------|-------------|--------|
| STE Binary | Straight-Through Estimator | 73% of FP32 quality |
| Magnitude Regularization | Encourage uniform |W| | Made things WORSE |
| CTG-Trained | Train with CTG patterns | Matches STE Binary |

### 3.3 Key Finding from Initial Tests

**All post-training and training-aware methods showed the same pattern:**
- Binary: ~73% of FP32 quality
- Ternary: ~86% of FP32 quality
- Gap: ~13% in favor of ternary

This suggested an **information-theoretic barrier** that couldn't be overcome with clever quantization alone.

---

## 4. The Key Insight: Compression Changes Everything

### The Flawed Assumption

Our initial analysis assumed:
- Each weight gets exactly 1 independent bit
- No compression or structure
- Worst-case random Gaussian weights

### ChatGPT's Crucial Reframing

> "You're not constrained to 1 raw bit per weight. The constraint is:
> **total bits in file / number of weights = 1.0 bpp**"

This means:
1. **Entropy coding** can compress structured signs
2. **Freed bits** can encode magnitude information
3. **Real networks** have exploitable structure (not random Gaussian)

### The New Strategy

```
Traditional: 1 bit/weight × N weights = N bits (fixed)

Entropy-based:
  - Compress signs: 0.7 bits/weight × N = 0.7N bits
  - Magnitude info: 0.3 bits/weight × N = 0.3N bits
  - Total: 1.0N bits (same budget, more information!)
```

---

## 5. Entropy-Shaped Binary Training

### Concept

Train binary networks with a regularization term that encourages **low-entropy sign patterns**:

```python
class EntropyShapedBinary:
    def train_entropy_shaped(self, X, Y_target, entropy_reg=0.1):
        # Task loss + entropy regularization
        task_loss = MSE(Y_pred, Y_target)
        entropy_loss = sign_correlation_loss(W_latent)
        total_loss = task_loss + entropy_reg * entropy_loss
```

### Sign Correlation Loss

Encourages neighboring signs to be similar (compressible):

```python
def sign_correlation_loss(W):
    signs = np.sign(W)
    h_corr = mean(signs[:, :-1] * signs[:, 1:])  # Horizontal
    v_corr = mean(signs[:-1, :] * signs[1:, :])  # Vertical
    return 2.0 - (h_corr + v_corr)  # Low when neighbors match
```

### Results

| Config | Sign Entropy | Freed Bits | Gap Reduction |
|--------|--------------|------------|---------------|
| entropy_reg=0.0 | 0.99 bpp | 0.01 | Baseline |
| entropy_reg=0.1 | 0.95 bpp | 0.05 | ~3% |
| entropy_reg=0.5 | 0.95 bpp | 0.05 | ~3% |

**Finding**: Entropy regularization alone freed only ~5% of bits, insufficient for major improvement.

---

## 6. Block-Structured Signs

### Concept

Force entire blocks to have the same sign. This compresses dramatically:

| Block Size | Signs BPP | Freed BPP |
|------------|-----------|-----------|
| 1×1 | 1.000 | 0.000 |
| 2×2 | 0.250 | 0.750 |
| 4×4 | 0.062 | 0.938 |
| 8×8 | 0.016 | 0.984 |

### Implementation

```python
def train_block_structured(self, X, Y_target, block_size=4):
    # Regularization: encourage uniform signs within blocks
    for each block:
        block_loss += variance(W_block)

    total_loss = task_loss + structure_reg * block_loss
```

### Results

| Method | Correlation | BPP |
|--------|-------------|-----|
| Binary (1×1) | 0.72 | 1.000 |
| Block 2×2 | 0.71 | 0.250 |
| Block 4×4 | 0.71 | 0.062 |
| Block 8×8 | 0.71 | 0.016 |
| Ternary | 0.89 | 1.580 |

**Key Finding**: Block structure preserves ~99% of binary quality at **16-60× lower BPP**!

This means we have **massive bit budget remaining** for magnitude information.

---

## 7. Hybrid Approach: Results on Synthetic vs Real Weights

### Concept

Combine block-structured signs with per-weight magnitude levels:

```
Budget: 1.0 bpp total
├── Block signs (4×4): 0.062 bpp
├── Magnitude levels:  0.938 bpp → 2^0.938 ≈ 2 levels
└── Total: ~1.0 bpp
```

### Implementation

```python
def train_hybrid_block_plus_magnitude(self, X, Y_target, block_size=2):
    # 1. Train block-structured signs
    W_block = train_block_structured(X, Y_target, block_size)

    # 2. Quantize magnitudes to n_levels
    n_levels = 2^(1.0 - sign_bpp)  # Use freed bits
    mag_quantized = quantize_to_levels(|W_fp32|, n_levels)

    # 3. Combine: sign × magnitude_level × scale
    W_hybrid = sign(W_block) * level_scales[mag_quantized]
```

### Results on Synthetic Trained Weights (128×128)

| Method | Correlation | BPP | vs Ternary |
|--------|-------------|-----|------------|
| Binary | 0.72 | 1.00 | -19% |
| Hybrid Block-2 | 0.89 | 1.25 | +0.6% |
| **Hybrid Block-4** | **0.89** | **1.06** | **+0.5%** |
| Ternary | 0.88 | 1.58 | baseline |

**On synthetic trained weights, Hybrid BEATS ternary!**

### ⚠️ CRITICAL: Results on Real GPT-2 Weights

| Method | Correlation | BPP | vs Ternary |
|--------|-------------|-----|------------|
| Binary | 0.765 | 1.00 | -8.0% |
| **Hybrid Block-4** | **0.264** | 1.06 | **-68.3%** ❌ |
| **Hybrid Block-2** | **0.511** | 1.25 | **-38.5%** ❌ |
| LowRank r=2 | 0.797 | 1.10 | -4.2% ✓ |
| Ternary | 0.832 | 1.58 | baseline |

**Hybrid Block FAILS CATASTROPHICALLY on real GPT-2 weights!**

### Why It Fails on Real Weights

Hybrid Block assumes **spatial coherence** (nearby weights have similar signs):
- **Synthetic trained weights**: Training creates this structure ✓
- **Real GPT-2 weights**: No spatial structure; neighbors are random ✗

When you use "majority sign per block" on random signs:
- ~50% of weights get the WRONG sign
- Magnitude levels can't compensate for wrong signs
- Result: -68% correlation (catastrophic)

---

## 8. CTG Integration

### Constant-Time Grammar (CTG)

CTG provides structured patterns for 1-bit inference with four operations:

| Operation | Effect | Use Case |
|-----------|--------|----------|
| PASS | w → w | Default |
| INHIBIT | w → 0 | Procedural zeros |
| INVERT | w → -w | Phase correction |
| PHASE | Complex rotation | Frequency domain |

### CTG-FIXED vs CTG-PROG

- **CTG-FIXED**: Static pattern (e.g., zero every 5th position)
- **CTG-PROG**: Adaptive programs selected per-layer/per-head

### Integration with Hybrid Approach

CTG can provide the **block structure** procedurally:

```python
# CTG provides structured sparsity pattern
ctg_pattern = generate_ctg_pattern(layer_id, head_id)

# Pattern defines which blocks are active
W_hybrid = apply_ctg_blocks(W_signs, W_magnitudes, ctg_pattern)
```

### Benefits

1. **No storage cost** for block assignments (procedural)
2. **Deterministic** (reproducible across devices)
3. **Hardware-friendly** (constant-time operations)

---

## 9. BSDM-W Calibration

### Binary Stochastic Dot-product Matching (BSDM-W)

BSDM-W estimates dot products from binary representations using Walsh-Hadamard carriers and Sigma-Delta modulators.

### Calibration Formula

For 1-bit quantization with known activations, use:

```
ŷ_t = a·||h_t||₂·√d·ẑ_t + b
```

Where:
- `h_t`: Activation vector at time t
- `d`: Dimension
- `ẑ_t`: BSDM-W raw estimate
- `a, b`: Calibration parameters (fit via linear regression)

### Calibration Procedure

1. **Collect calibration set**: (h_t, y_t) on 5k-10k tokens
2. **Fit parameters**: Linear regression on actual dot-products
3. **High T for calibration**: Use T=64-128 ticks during calibration
4. **Lower T for production**: Can reduce T after calibration

### Memory from Previous Sessions

> "For 1-bit quantization calibration: collect calibration set (h_t, y_t) on 5k-10k tokens, fit global scaling parameters a,b using formula ŷ_t = a·||h_t||₂·√d·ẑ_t + b via linear regression to anchor BSDM-W estimates to actual dot-product scale, use high T (64-128) for calibration only, then evaluate with lower T for production."

---

## 10. Conclusions and Future Work

### Key Findings

1. **The constraint is file size, not raw bits per weight**
   - Compression enables trading sign entropy for magnitude information

2. **Block-structured signs are highly efficient**
   - 4×4 blocks: 0.062 bpp for signs (94% savings)
   - Quality preserved: ~99% of unstructured binary

3. **Hybrid approach matches ternary at lower BPP**
   - Block signs + magnitude levels ≈ ternary quality
   - 33% fewer bits than ternary (1.06 vs 1.58 bpp)

4. **CTG provides procedural structure**
   - No storage overhead for block patterns
   - Hardware-friendly, deterministic

### What We Proved

| Claim | Status | Evidence |
|-------|--------|----------|
| 1.0 bpp can match 1.58 bpp quality | ✓ VALIDATED | Hybrid: 0.89 @ 1.06 bpp |
| Block structure enables compression | ✓ VALIDATED | 4×4: 0.71 @ 0.06 bpp |
| ChatGPT's reframing was correct | ✓ VALIDATED | Effective bpp ≠ raw bpp |

### Future Work

1. **Real Transformer Training**
   - Apply hybrid approach to GPT-2/LLaMA training
   - Measure perplexity, not just correlation

2. **Optimal Block Size Selection**
   - Per-layer adaptive block sizes
   - CTG-PROG for block pattern selection

3. **Hardware Implementation**
   - XNOR + popcount for block-structured matmul
   - OpenCL/CUDA kernels for hybrid inference

4. **Entropy Coding Integration**
   - Arithmetic coding for sign compression
   - Side-channel encoding for magnitude levels

5. **Scaling Laws**
   - Does hybrid advantage grow with model size?
   - Optimal bit allocation at different scales

---

## Appendix A: Code Structure

```
onebit/
├── research/
│   └── unified_1bit.py      # Main research framework
│       ├── EntropyShapedBinary
│       ├── TrainingAwareCTG
│       ├── PerfectBinaryOracle
│       └── TrainingAwareSimulation
├── runtime/
│   ├── ctg_grammar.py       # CTG engine
│   ├── ctg_selector.py      # Adaptive program selection
│   └── controller_e2e.py    # End-to-end controller
├── ops/
│   ├── bsdm_w.py            # BSDM-W implementation
│   └── bsdm_w_torch.py      # PyTorch wrapper
└── core/
    ├── hadamard.py          # Walsh-Hadamard transform
    └── packbits.py          # Bit packing utilities
```

## Appendix B: Key Classes

### EntropyShapedBinary

```python
class EntropyShapedBinary:
    """Train binary weights to be compressible."""

    def estimate_entropy(signs, block_size=8) -> float
    def sign_correlation_loss(W) -> float
    def train_entropy_shaped(X, Y_target, entropy_reg) -> Tuple
    def train_block_structured(X, Y_target, block_size) -> Tuple
    def train_hybrid_block_plus_magnitude(X, Y_target, block_size) -> Tuple
    def compare_hybrid(n_samples) -> Dict
```

### TrainingAwareCTG

```python
class TrainingAwareCTG:
    """Simulate training-aware CTG quantization."""

    def redistribute_importance(W_latent, inhibit_mask) -> W_redistributed
    def quantize_with_ctg(W_redistributed) -> W_binary
    def simulate_trained_ctg(W_random) -> W_quantized
```

## Appendix C: Experimental Results Summary

### Post-Training Quantization (Random Gaussian Weights)

| Method | BPP | Correlation | % of FP32 |
|--------|-----|-------------|-----------|
| FP32 | 32.00 | 1.000 | 100% |
| Ternary | 1.58 | 0.86 | 86% |
| Binary (sign) | 1.00 | 0.72 | 72% |

### Training-Aware (Learnable Task)

| Method | BPP | Correlation | % of FP32 |
|--------|-----|-------------|-----------|
| FP32 | 32.00 | 1.000 | 100% |
| Ternary | 1.58 | 0.86 | 86% |
| Binary STE | 1.00 | 0.73 | 73% |
| Binary + CTG | 0.80 | 0.65 | 65% |

### Hybrid Approach (128×128, Final Results)

| Method | BPP | Correlation | vs Ternary |
|--------|-----|-------------|------------|
| Binary | 1.00 | 0.717 | -19.1% |
| Hybrid Block-2 | 1.25 | 0.888 | +0.0% |
| **Hybrid Block-4** | **1.06** | **0.891** | **+0.3%** |
| Ternary | 1.58 | 0.889 | baseline |

---

## 11. Advanced Approaches: ChatGPT's Novel Ideas

After initial experiments, we received a comprehensive set of novel ideas from ChatGPT for beating ternary with 1-bit quantization. This section documents our systematic evaluation of these approaches.

### 11.1 The Key Reframing

ChatGPT identified a crucial insight:

> **At fixed memory M bits:**
> - Ternary: M ÷ 1.58 = N_t parameters
> - Binary: M ÷ 1.00 = N_b parameters = **1.58× more parameters!**

This means comparing binary vs ternary at the *same parameter count* is unfair to binary. The fair comparison is at the *same memory budget*.

### 11.2 Ideas Evaluated

| # | Idea | Concept | Result |
|---|------|---------|--------|
| 0 | **1.58× more params** | Use extra binary params for width | ❌ More params = more noise |
| 1 | **Binary MoE** | Hash routing to binary experts | ❌ FP projection dominates memory |
| 2 | **Duty-cycle magnitude** | Use time/iterations as magnitude | ❌ Same as binary (-9.7%) |
| 4 | **Hashed weight sharing** | Share weights, gain width | ❌ Collisions hurt (-16.7%) |
| 6 | **Binary kernel + FP head** | Binary features + tiny FP | ✅ +13% but 41× memory |
| 8 | **Binary basis selection** | Dictionary of binary bases | ❌ FP overhead too high |
| 9 | **Iterative refinement** | Error-feedback loops | ✅ Works at 1.27× memory |

### 11.3 Detailed Results

#### Fixed Memory Comparison (d=256×256)

```
Method              Corr      Memory    vs Ternary
─────────────────────────────────────────────────
ternary            0.8852     1.00×      baseline
binary_std         0.7987     0.63×       -9.8%
binary_wider       0.7987     1.00×       -9.8%
binary_hashed      0.7374     1.00×      -16.7%
binary_2layer      0.4703     1.00×      -46.9%
```

**Finding**: Simply having more binary parameters doesn't help. The extra width/depth doesn't compensate for lack of zeros.

#### Iterative Binary Refinement

```
Method        Corr      Memory    vs Ternary
────────────────────────────────────────────
iter_2       0.9350     1.27×      +5.6% ✓
iter_3       0.9724     1.90×      +9.9% ✓
ternary      0.8852     1.00×      baseline
binary       0.7987     0.63×      -9.8%
```

**Finding**: Iterative binary BEATS ternary, but requires more memory (1.27× to 1.9×).

#### Structured Sparsity Approaches

```
Method              Corr      BPP     vs Ternary
───────────────────────────────────────────────
zero40_none        0.8980    1.571     +1.4% ✓
ternary            0.8852    1.580     baseline
col_sparse_70      0.6840    0.703    -22.7%
block_sparse       0.7587    0.845    -14.3%
```

**Finding**: More zeros (40% vs 30%) can beat ternary, but requires ~1.57 bpp (essentially still ternary).

### 11.4 Why the 10% Gap Exists

The persistent ~10% gap between binary and ternary is **NOT about bit-rate**. It's about **quantization error**:

```
Binary: All weights have magnitude = scale
        Error on small weight w: |scale - |w||

Ternary: Small weights become zero
         Error on small weight w: |0 - w| = |w| (smaller!)
```

The zeros in ternary perform **implicit pruning** of noisy small weights. Binary has no mechanism to "ignore" unimportant weights.

### 11.5 What Would Actually Beat Ternary at 1.0 bpp

Based on our experiments, these approaches show promise:

1. **Training-time quantization** (like BitNet)
   - Model learns to use binary weights effectively
   - Representations adapt to binary constraints
   - Not possible with post-hoc quantization

2. **Iterative binary at ~1.3× memory**
   - Trade extra memory for iterations
   - Each iteration corrects previous errors
   - Beats ternary at 1.27× memory (+5.6%)

3. **Compressible binary patterns**
   - Train for low-entropy sign patterns
   - Freed bits encode magnitude side-channel
   - Block-structured signs achieve 0.06 bpp
   - Combined with 2-level magnitude: 1.06 bpp total

4. **CTG-based procedural patterns**
   - Generate weights from small programs
   - Nearly 0 bpp for base patterns
   - Sparse corrections for task-specific adjustments

### 11.6 The Fundamental Trade-off

```
Quality = f(Information Content)

┌────────────────────────────────────────────────────┐
│  Approach          │ BPP  │ Quality │ Trade-off   │
├────────────────────┼──────┼─────────┼─────────────┤
│  Binary (raw)      │ 1.00 │  ~80%   │ No zeros    │
│  Binary (iter×2)   │ 1.27 │  ~94%   │ +27% memory │
│  Binary (iter×3)   │ 1.90 │  ~97%   │ +90% memory │
│  Ternary           │ 1.58 │  ~89%   │ Baseline    │
│  Hybrid block-4    │ 1.06 │  ~89%   │ Compression │
└────────────────────┴──────┴─────────┴─────────────┘
```

### 11.7 Conclusions from ChatGPT Ideas

1. **The 1.58× more params insight is correct** but insufficient alone
2. **Extra parameters need structured use** (iterations, not just width)
3. **The ~10% gap is fundamental** for post-hoc single-pass binary
4. **Iterative approaches work** but trade memory for quality
5. **Compression-based approaches** (hybrid block) remain most promising for true 1.0 bpp

---

## 12. Implementation Files

### Research Modules Created

| File | Purpose |
|------|---------|
| `onebit/research/unified_1bit.py` | Original quantization methods |
| `onebit/research/structured_1bit.py` | Column/group structured sparsity |
| `onebit/research/error_optimal_1bit.py` | Error-minimizing approaches |
| `onebit/research/training_time_binary.py` | Training-time optimization |
| `onebit/research/lowrank_binary.py` | Low-rank binary factorization |
| `onebit/research/fair_memory_comparison.py` | Fixed-memory comparison |
| `onebit/research/iterative_binary.py` | Iterative refinement (ChatGPT #9) |

### Key Experimental Commands

```bash
# Run iterative binary experiments
python -m onebit.research.iterative_binary

# Run fair memory comparison
python -m onebit.research.fair_memory_comparison

# Run structured approaches
python -m onebit.research.structured_1bit
```

---

## References

1. BitNet b1.58: 1.58-bit Large Language Models (Microsoft Research)
2. BinaryConnect: Training Deep Neural Networks with binary weights (Courbariaux et al.)
3. XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks (Rastegari et al.)
4. Trained Ternary Quantization (Zhu et al.)

---

## 13. Transform-Domain Experiments (Novel Ideas V1 & V2)

Following ChatGPT's suggestion of "crazy novel ideas," we implemented and tested 30+ approaches.

### 13.1 Ideas Tested on Synthetic Data

| Category | Methods | Best at 1.0 bpp |
|----------|---------|-----------------|
| Transform Domain | DCT, Hadamard, Walsh, Block DCT, Adaptive | +0.8% vs ternary |
| Encoding Tricks | Codon (DNA-style), Sign Prediction, Huffman | -8% to -65% |
| Pseudo-Ternary | CTG Inhibit, Self-Referential, Periodic | -5% to -22% |
| Iterative | DCT+Residual, Multi-pass | +17% (at 2.0 bpp) |

**Synthetic Result**: DCT/Hadamard binary appeared to beat ternary at 1.0 bpp on structured synthetic data.

### 13.2 The Critical GPT-2 Test

We then tested the winning methods on **real GPT-2 weights**. Results were dramatically different:

| Method | BPP | Synthetic | Real GPT-2 | Δ |
|--------|-----|-----------|------------|---|
| DCT Binary | 1.00 | +0.4% | **-35%** | Catastrophic failure |
| Hadamard | 1.00 | +0.2% | **-14%** | Worse than binary |
| Walsh Carrier | 1.00 | +0.2% | -14% | Same as Hadamard |
| Block DCT 32 | 1.02 | +0.4% | **-29%** | Still failing |
| Binary baseline | 1.00 | -16% | -11% | Consistent |

**Key Finding**: Transform-domain binarization only works when weights have compressible structure. Real LLM weights don't have this structure.

### 13.3 Why Transforms Failed on Real Weights

Our synthetic data was accidentally structured:
- Low-rank (rank=16)
- Sparse (30% zeros)
- DCT/Hadamard captured this well

Real GPT-2 weights:
- Full rank
- Not sparse
- No frequency-domain structure

**Conclusion**: Transforms can't create information that isn't there.

---

## 14. Magnitude Recovery: The Breakthrough

### 14.1 The Real Problem Identified

```
Ternary {-1, 0, +1}:
  +1 = "positive AND important"
   0 = "NOT important"        ← This is FREE magnitude info!
  -1 = "negative AND important"

Binary {-1, +1}:
  +1 = "positive" (importance unknown)
  -1 = "negative" (importance unknown)
```

**The zero in ternary is a GATE that says "ignore this weight."**

### 14.2 Magnitude Has Low-Rank Structure

Key insight: Even when W is full-rank, |W| (the magnitude matrix) has low-rank structure.

```python
W_quant = sign(W) * (U_k @ S_k @ V_k^T)
```

Where U_k, S_k, V_k are from SVD of |W|.

### 14.3 Real GPT-2 Results

| Method | BPP | vs Ternary | Notes |
|--------|-----|------------|-------|
| Binary | 1.00 | -11.0% | Baseline |
| Row Scale | 1.04 | -8.0% | Per-row magnitude |
| Col Scale | 1.01 | -6.3% | Per-column magnitude |
| Row+Col Scale | 1.06 | **-3.2%** | Rank-1 magnitude |
| LowRank r=1 | 1.06 | -3.2% | Same as Row+Col |
| LowRank r=2 | 1.11 | **-1.3%** | Getting close |
| LowRank r=4 | 1.22 | **-0.4%** | Nearly matches! |
| **LowRank r=8** | **1.44** | **+0.6%** | **BEATS TERNARY!** |
| Ternary | 1.58 | baseline | |

### 14.4 Layer-by-Layer Analysis

Performance varies by matrix shape:

| Matrix Type | Best Method | vs Ternary |
|-------------|-------------|------------|
| Square (768×768) | LowRank r=8 | **+44%** |
| Rectangular (3072×768) | LowRank r=8 | -4% to -8% |
| Attention (2304×768) | LowRank r=8 | +0.3% to +5% |

**Finding**: Square matrices benefit most from low-rank magnitude.

---

## 15. The Core Problem Statement

### 15.1 Information Theory View

```
Ternary: log₂(3) = 1.58 bits per weight
├── Sign: 1.0 bits
└── Importance: 0.58 bits

Binary: log₂(2) = 1.00 bits per weight
├── Sign: 1.0 bits
└── Importance: 0 bits  ← THE GAP
```

### 15.2 What Binary Must Solve

To match ternary at 1.0 bpp, binary must answer:

**"Which weights should be ignored?"**

This requires ~0.58 bits of "importance" information PER WEIGHT.

### 15.3 Sources of Free Importance Information (Tested)

| Source | Works? | Why? |
|--------|--------|------|
| Sign patterns | ❌ | Signs don't predict magnitude |
| Transform domain | ❌ | Real weights lack structure |
| Position in matrix | ❌ | No correlation |
| Neighboring weights | ❌ | No local correlation |
| **Low-rank structure of |W|** | ✅ | Magnitude IS low-rank! |
| Input activations | ❓ | Untested at scale |
| Layer statistics | ❓ | Untested |
| Training-time adaptation | ❓ | Requires retraining |

### 15.4 The Fundamental Question

**Can we find 0.58 bits of magnitude information for FREE?**

At 1.0 bpp budget:
- LowRank r=1 (1.06 bpp): -3.2% vs ternary
- LowRank r=2 (1.11 bpp): -1.3% vs ternary
- LowRank r=4 (1.22 bpp): -0.4% vs ternary

**Gap to close**: 0.22 bpp worth of magnitude info at zero cost.

---

## 16. Solution Space for True 1.0 bpp

### Option A: Free Magnitude Signal
Find something that predicts weight importance without storing it:
- Input activation patterns?
- Cross-layer correlations?
- Learnable importance predictors (amortized cost)?

### Option B: Better Magnitude Compression
Store magnitude but compress below 0.58 bpp:
- Low-rank works but costs 0.44 bpp for r=8
- Can we achieve rank-4 quality at rank-1 cost?

### Option C: Training-Time Adaptation
Train model to work with binary constraints:
- Model learns to encode importance in signs
- Requires full retraining

### Option D: Different "Ternary Expression"
Express 3 states using only sign bits:
- Temporal encoding (this timestep vs last)
- Spatial encoding (this weight vs neighbor)
- Activation-conditional (sign depends on input)

---

## 17. Files Created in This Research

| File | Purpose |
|------|---------|
| `onebit/research/novel_ideas_test.py` | 15 crazy ideas (synthetic) |
| `onebit/research/novel_ideas_v2.py` | 30+ methods with variations |
| `onebit/research/dct_binary.py` | PyTorch DCT/Hadamard + GPT-2 testing |
| `onebit/research/EXPERIMENT_SUMMARY.md` | Detailed experiment log |

### Key Commands

```bash
# Test all novel ideas on synthetic data
python -m onebit.research.novel_ideas_v2

# Test on real GPT-2 weights
python -m onebit.research.dct_binary
```

---

## 18. Final Summary Table

### All Methods Tested (Comprehensive)

| Method | BPP | Synthetic | Real GPT-2 | Verdict |
|--------|-----|-----------|------------|---------|
| **Transforms** |
| DCT Binary | 1.00 | +0.4% | -35% | ❌ |
| Hadamard Binary | 1.00 | +0.2% | -14% | ❌ |
| Block DCT | 1.02 | +0.4% | -29% | ❌ |
| Adaptive Transform | 1.00 | +0.8% | -15% | ❌ |
| **Encoding Tricks** |
| Codon (DNA) | 0.77 | -8% | - | ❌ |
| Sign Prediction | 1.00 | -31% | - | ❌ |
| **Pseudo-Ternary** |
| CTG Inhibit | 1.3 | -5% | - | ❌ |
| Self-Referential | 1.00 | -22% | - | ❌ |
| **Magnitude Recovery** |
| Row+Col Scale | 1.06 | - | -3.2% | ⚠️ Close |
| LowRank r=2 | 1.11 | - | -1.3% | ⚠️ Close |
| LowRank r=4 | 1.22 | - | -0.4% | ✅ Matches |
| **LowRank r=8** | **1.44** | - | **+0.6%** | ✅ **BEATS** |
| **Baselines** |
| Binary | 1.00 | -16% | -11% | ❌ Gap exists |
| Ternary | 1.58 | baseline | baseline | Reference |

### The Bottom Line

1. **At exactly 1.0 bpp**: Binary is ~11% behind ternary (fundamental gap)
2. **At 1.22 bpp**: LowRank r=4 nearly matches ternary (-0.4%)
3. **At 1.44 bpp**: LowRank r=8 BEATS ternary (+0.6%)
4. **To beat ternary at 1.0 bpp**: Need free source of magnitude info

---

## 19. Claude's First-Principles Ideas (November 2025)

After reviewing our results, Claude (Claude Opus 4.5) proposed several novel approaches derived from first principles. We systematically tested each one.

### 19.1 Claude's Analysis

Claude correctly identified that:
> "Ternary's zeros silence noisy weights. The ~10% gap exists because binary forces every weight to contribute, including the ~30% that should shut up."

And proposed:
> "To match ternary at 1.0 bpp, we need to either:
> 1. DERIVE importance from stored signs (0 extra bits)
> 2. ENCODE importance in a computable pattern (0 extra bits)
> 3. Use EXISTING computation to infer importance (0 extra bits)"

### 19.2 Ideas Tested

| # | Idea | Concept | Result | Why Failed |
|---|------|---------|--------|------------|
| 1 | **Procedural Zero Mask** | Position-based zeros | -20% to -27% | Random position ≠ importance |
| 2 | **Activation-Gated Binary** | Small inputs × ±1 ≈ 0 | -10% (same as binary) | Gates wrong signal |
| 3 | **Procedural Zeros + Training** | Train with fixed zero pattern | -26% to -32% | Random capacity loss |
| 4 | **Sign-Texture Importance** | Neighbor agreement → importance | -28% | No spatial structure in real weights |
| 5 | **Output-Weighted Binary** | Output magnitude = importance | -9% (same as binary) | Emergent importance is wrong signal |
| 6 | **Implicit Neural Weights** | Tiny network generates W | ~0% | Can't compress arbitrary patterns |

### 19.3 Detailed Results

#### Activation-Gated Binary

Claude's insight: "Small activations × ±1 ≈ zero contribution naturally"

```python
def gate_activations(x, threshold=0.05):
    gate = sigmoid((abs(x) - threshold) * sharpness)
    return x * gate  # Small inputs → ~0 contribution
```

**Result on GPT-2**: -10.0% vs ternary (identical to plain binary)

**Why it failed**: The importance is in the WEIGHT magnitude, not input magnitude. A small input × important weight should still contribute.

#### Procedural Zeros WITH Training

Tested whether training with known zero positions would help:

| Method | Correlation | BPP | vs Ternary |
|--------|-------------|-----|------------|
| Ternary (trained) | 0.9681 | 1.58 | baseline |
| Binary (trained) | 0.7213 | 1.00 | -25.5% |
| ProcZero 20% (trained) | 0.7086 | 1.00 | -26.8% |
| ProcZero 30% (trained) | 0.6596 | 1.00 | -31.9% |

**Finding**: Even when training knows which positions will be zero, it can't compensate. Random zeros remove capacity; training can't route around random holes.

#### Sign-Texture Importance

Idea: Local sign agreement indicates importance (edges = important, uniform = less important)

```python
def compute_sign_agreement(W_binary, kernel_size=3):
    neighbor_avg = conv2d(W_binary, averaging_kernel)
    agreement = abs(W_binary * neighbor_avg)  # High = agrees with neighbors
    return agreement  # Use as importance
```

**Result on GPT-2**: -28% vs ternary (much worse than binary!)

**Why it failed**: GPT-2 weights have NO spatial structure. Sign agreement is random, not correlated with magnitude.

### 19.4 The Critical Hybrid Block Finding

Our documented "1.06 bpp matches ternary" was based on **synthetic trained weights**:

| Dimension | Method | Correlation | BPP | vs Ternary |
|-----------|--------|-------------|-----|------------|
| 128×128 (synthetic) | Hybrid Block-4 | 0.8878 | 1.062 | **+0.5%** ✓ |
| 128×128 (synthetic) | Ternary | 0.8835 | 1.580 | baseline |

**BUT on real GPT-2 weights:**

| Method | Correlation | BPP | vs Ternary |
|--------|-------------|-----|------------|
| Binary | 0.7652 | 1.00 | -8.0% |
| **Hybrid Block-4** | **0.2639** | 1.06 | **-68.3%** ❌ |
| **Hybrid Block-2** | **0.5113** | 1.25 | **-38.5%** ❌ |
| LowRank r=2 | 0.7966 | 1.10 | **-4.2%** ✓ |
| Ternary | 0.8315 | 1.58 | baseline |

**Why Hybrid Block fails on real weights**: It assumes spatial coherence (nearby weights have similar signs). This is true for trained synthetic weights but NOT for GPT-2. Real LLM weights have no spatial structure—neighboring weights are essentially random.

### 19.5 Claude's Ideas: Summary

| Idea | Status | Reason |
|------|--------|--------|
| Procedural zeros | ❌ Failed | Position ≠ importance |
| Activation gating | ❌ Failed | Gates wrong signal (input, not weight) |
| Training with zeros | ❌ Failed | Random holes can't be routed around |
| Sign-texture importance | ❌ Failed | No spatial structure in real weights |
| Output-weighted | ❌ Failed | Emergent signal is wrong |
| Implicit neural weights | ❌ Failed | Can't compress arbitrary patterns |

**Conclusion**: All approaches that try to derive importance from signs, position, or activation patterns fail because importance is fundamentally tied to weight MAGNITUDE, which is independent of these signals.

---

## 20. Corrected Summary: What Actually Works

### 20.1 Methods That Work on Real GPT-2 Weights

| Method | BPP | vs Ternary | Notes |
|--------|-----|------------|-------|
| Binary | 1.00 | **-8.4%** | Baseline |
| Row+Col Scale (4-bit) | 1.007 | **-4.5%** | Best near-1.0 bpp |
| Row+Col Scale (32-bit) | 1.05 | -4.7% | Minimal overhead |
| LowRank r=2 | 1.10 | **-4.2%** | Good balance |
| LowRank r=4 | 1.21 | **-2.4%** | Very close |
| LowRank r=8 | 1.42 | **-1.7%** | Nearly matches |
| Ternary | 1.58 | baseline | |

### 20.2 Methods That ONLY Work on Synthetic Trained Weights

| Method | Synthetic | Real GPT-2 | Notes |
|--------|-----------|------------|-------|
| Hybrid Block-4 | +0.5% | **-68%** | Catastrophic failure |
| Hybrid Block-2 | +0.6% | **-38%** | Also fails |
| DCT Binary | +0.4% | **-35%** | No frequency structure |
| Hadamard Binary | +0.2% | **-14%** | Same issue |

### 20.3 The Pareto Frontier (Real Weights Only)

```
Quality (vs Ternary)
  |
  |                                      Ternary (1.58 bpp, 0%)
  |                             LowRank r=8 (1.42 bpp, -1.7%)
  |                      LowRank r=4 (1.21 bpp, -2.4%)
  |                LowRank r=2 (1.10 bpp, -4.2%)
  |          Row+Col 4-bit (1.01 bpp, -4.5%)
  |    Binary (1.00 bpp, -8.4%)
  +---------------------------------------------------------> BPP
```

### 20.4 Practical Recommendations

| Goal | Method | BPP | Quality Loss | Storage Savings |
|------|--------|-----|--------------|-----------------|
| **Minimum storage** | Binary | 1.00 | -8.4% | 37% vs ternary |
| **Best balance** | Row+Col 4-bit | 1.01 | -4.5% | 36% vs ternary |
| **Near-ternary quality** | LowRank r=4 | 1.21 | -2.4% | 23% vs ternary |
| **Match ternary** | LowRank r=8 | 1.42 | -1.7% | 10% vs ternary |

### 20.5 The Fundamental Gap Confirmed

After testing 30+ methods across both ChatGPT and Claude suggestions:

**True 1.0 bpp binary is fundamentally ~8-10% behind ternary on real pre-trained weights.**

The only paths that close this gap require:
1. **Extra bits for magnitude** (LowRank, Row+Col scaling)
2. **Training-time adaptation** (model learns binary constraints from scratch)

No post-hoc method that operates at exactly 1.0 bpp can match ternary because:
- Ternary's zeros encode "importance" (which weights to ignore)
- This information is tied to magnitude
- Magnitude is independent of sign, position, and activation patterns
- The only way to know magnitude is to store it (~0.06-0.44 extra bpp)

---

## 21. End-to-End Perplexity Analysis (Critical Discovery)

### 21.1 The Correlation vs Perplexity Disconnect

Previous experiments measured **single-layer correlation** (how well binary MatMul matches FP32). However, testing **end-to-end perplexity** reveals a critical disconnect:

| Metric | Single-Layer | End-to-End |
|--------|-------------|------------|
| Binary vs Ternary gap | ~8-10% | **2,000,000%+** |
| Row+Col improvement | -4.7% | **Makes it WORSE** |
| LowRank improvement | -3.2% | **Makes it WORSE** |

**Key finding**: Single-layer correlation improvements DO NOT translate to end-to-end quality.

### 21.2 Layer Sensitivity Analysis

Testing binarization on individual GPT-2 layers:

| Layer Binarized | Perplexity | vs FP32 |
|-----------------|------------|---------|
| Layer 0 only | 35,082 | **+41,448%** |
| Layer 1 only | 96 | +14% |
| Layer 2-11 (each) | 96-138 | +14-64% |
| **ALL layers** | **1,748,427** | **+2,000,000%** |

**Critical finding**: Layer 0 is catastrophically sensitive. Errors compound exponentially through 12 layers.

### 21.3 Component Sensitivity: MLP vs Attention

Testing binarization by component type:

| Component | Parameters | PPL when Binary | vs FP32 |
|-----------|-----------|-----------------|---------|
| All Attention | 28M (33%) | 7,544 | +11,161% |
| **All MLP** | **57M (67%)** | **1,452,001** | **+2,167,425%** |
| Everything | 85M | 1,238,130 | +1,848,162% |

**MLP layers are ~200x more sensitive than attention layers!**

### 21.4 Root Cause: GELU Amplification

**Why MLP is so sensitive:**

```
Pre-GELU activations in GPT-2:
- 96% are in GELU-sensitive region (|x| < 1)
- GELU is asymmetric: GELU(-x) ≠ -GELU(x)

When binary quantization flips a sign near 0:
  FP32:   x = +0.10 → GELU(x) = +0.054
  Binary: x = -0.10 → GELU(x) = -0.046
                      ↓
              185% ERROR from tiny sign flip!
```

The GELU activation AMPLIFIES sign errors:
1. Binary can flip signs of small weights
2. Small weights → pre-GELU values near 0
3. GELU is asymmetric at 0
4. Sign flip → 100-200% error per activation
5. Errors compound through layers → catastrophic perplexity

### 21.5 Why Attention is More Robust

Attention uses **softmax** which is:
- Symmetric and smooth
- No asymmetric nonlinearity like GELU
- Errors don't get amplified the same way

### 21.6 Why Ternary's Zeros Help

| Method | Single-Layer Error | Why |
|--------|-------------------|-----|
| Binary | 63.5% | Small weights get full scale → wrong contributions |
| Ternary | 60.4% | Small weights → 0 → no contribution |

**For GELU-sensitive regions, "no contribution" is better than "wrong contribution".**

The ~3% single-layer improvement from ternary becomes massive when compounded through 12 layers with GELU amplification.

### 21.7 Implications for 1.0 BPP

This analysis reveals why post-hoc binary fails so catastrophically:

1. **GELU amplifies errors** - small quantization errors become huge
2. **Errors compound** - 12 layers × 6 weights each = exponential error growth
3. **MLP dominates** - 67% of parameters are in the most sensitive component

**Possible mitigations:**
1. Keep MLP at higher precision (ternary/4-bit), only binarize attention
2. Replace GELU with more robust activation (ReLU) during training
3. Train from scratch with binary constraints (model learns to avoid GELU-sensitive regions)

### 21.8 Hybrid Quantization Results

Testing binary attention + higher-precision MLP:

| Config | PPL | vs FP32 | Effective BPP |
|--------|-----|---------|---------------|
| FP32 + FP32 | 62 | 0% | 32.0 |
| Binary Attn + FP32 MLP | 6,216 | +9,981% | 21.7 |
| Binary Attn + Ternary MLP | 2,520,339 | +4,087,100% | 1.39 |
| Ternary + Ternary | 567,045 | +919,469% | 1.58 |
| Binary + Binary | 918,172 | +1,488,887% | 1.00 |

Even keeping MLP at FP32 still gives ~10,000% worse perplexity due to attention errors compounding.

### 21.9 Conclusion: The True Barrier

**Post-hoc quantization at any precision fundamentally breaks GPT-2** because:

1. The model was trained with FP32 precision
2. Weights have evolved to work together with tiny precision
3. GELU amplifies any quantization error
4. Errors compound exponentially through layers

**The only path to working low-bit models is training-time quantization:**
- BitNet trains with ternary constraints from scratch
- Model learns to avoid GELU-sensitive regions
- Weights co-adapt to quantization noise

This explains why BitNet b1.58 works while post-hoc quantization fails.

---

## 22. Calibrated Binary: The New Regime

### 22.1 Changing the Rules

After discovering that post-hoc quantization fundamentally breaks pre-trained models, we explored a new regime:

> **"Allow a sliver of metadata"** - Instead of pure 1.0 bpp, use binary signs + learnable calibration parameters.

### 22.2 The Calibrated Binary Approach

```python
class CalibratedBinaryConv1D(nn.Module):
    """Binary signs (frozen) + learnable low-rank magnitude calibration."""
    def __init__(self, orig_weight, rank=8):
        # Freeze binary signs
        sign = torch.sign(orig_weight)
        self.register_buffer('sign', sign)

        # Learnable calibration: magnitude = a + U @ V^T
        self.a = nn.Parameter(orig_weight.abs().mean())
        self.U = nn.Parameter(torch.randn(n_in, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n_out, rank) * 0.01)

    def forward(self, x):
        mag = F.relu(self.a + self.U @ self.V.T) + 1e-6
        return x @ (self.sign * mag)
```

### 22.3 Key Results

#### Final Comparison (Same Evaluation Data)

| Method | BPP | PPL | vs FP32 |
|--------|-----|-----|---------|
| FP32 | 32.00 | 137.48 | baseline |
| **Post-hoc Ternary** | 1.58 | **1,560,063** | **+1,134,691%** |
| **Calibrated Binary** | **1.11** | **139.55** | **+1.5%** |

**Calibrated Binary is 10,000x better than post-hoc ternary!**

#### Realistic Evaluation (WikiText-2 train→val)

| Method | BPP | PPL | vs FP32 |
|--------|-----|-----|---------|
| FP32 | 32.00 | 44.73 | baseline |
| Post-hoc Binary | 1.00 | 3,773,701 | +8,437,000% |
| Calibrated Binary | 1.11 | 2,926 | +6,440% |

On held-out data, calibrated binary is still ~2000x better than post-hoc binary, but not close to FP32.

### 22.4 Why Calibration Works

1. **Binary signs preserve direction** - The sign of each weight is correct
2. **Low-rank calibration recovers magnitude** - U @ V^T approximates |W|
3. **Knowledge distillation** - Calibration parameters learn to match FP32 outputs
4. **Frozen signs = stable training** - No gradient through sign function

### 22.5 The Generalization Gap

| Evaluation | Calibrated Binary vs FP32 |
|------------|---------------------------|
| Same data as training | +1.5% |
| Held-out validation | +6,440% |

**Key insight**: Calibration can perfectly match FP32 on training data, but doesn't generalize well to unseen data. This suggests:
- The calibration is overfitting to the training distribution
- More calibration data is needed
- Or: the binary signs fundamentally lose information that can't be recovered

### 22.6 Progressive Calibration

Layer-by-layer calibration with end-to-end distillation:

| Layer | Val PPL | Notes |
|-------|---------|-------|
| 0 | 109 | First layer calibrated |
| 1 | 148 | Errors start compounding |
| ... | ... | ... |
| 11 | 921 | Final: +1192% vs FP32 |

Progressive calibration is more stable than end-to-end, but errors still compound.

### 22.7 Block-Structured Calibration (CTG-Aware)

Instead of per-element calibration, use block-structured scales:

| Block Size | BPP | PPL | vs FP32 |
|------------|-----|-----|---------|
| 16 | 1.031 | 84.17 | +0.0%* |
| 32 | 1.008 | 84.17 | +0.0%* |
| 64 | 1.002 | 84.17 | +0.0%* |
| 128 | 1.000 | 84.17 | +0.0%* |

*On training data only - overfitting

### 22.8 Conclusions from Calibration Experiments

1. **Calibration dramatically improves over post-hoc quantization**
   - 10,000x better than post-hoc ternary on same data
   - 2,000x better than post-hoc binary on held-out data

2. **The generalization gap is significant**
   - Perfect on training data (+1.5%)
   - Poor on held-out data (+6,440%)

3. **More calibration data helps but doesn't solve the problem**
   - WikiText-2 train (200 samples) → 2,926 PPL
   - Need much larger calibration sets

4. **Block-structured calibration is efficient**
   - Block size 128 achieves ~1.0 bpp
   - But still overfits to training data

### 22.9 The Path Forward

Based on these experiments, the viable paths are:

1. **Large-scale calibration** - Use thousands of samples for calibration
2. **Training-time quantization** - Train with binary constraints from scratch
3. **Hybrid precision** - Keep critical layers at higher precision
4. **Accept the gap** - Use calibrated binary at 1.11 bpp with ~6000% gap

---

*Document updated: 2025-11-30*
*SALOMI Project - Stochastic Approximation for Low-Memory Inference*

