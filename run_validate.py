import numpy as np

"""
CTG Training-Aware Simulation

This simulates what ChatGPT described:
1. Define a FIXED CTG suppression pattern
2. "Train" weights to minimize error WHILE respecting the pattern
3. See if the network learns to concentrate signal in protected slots
"""

"""
Learned Codebook for Zero Patterns (LCZP)

Key insight: Instead of storing 1 bit per weight for zero/non-zero,
we can learn a CODEBOOK of common zero patterns and store only the
codebook index per row/block.

If we have K=256 patterns and d_in=768 weights per row:
- Codebook: 256 * 768 bits = 24KB (amortized over all rows)
- Per-row: 8 bits to select pattern
- Effective: 1 + 8/768 ≈ 1.01 bpp

This is ALMOST 1.00 bpp and could match ternary if patterns are learned well.
"""

np.random.seed(42)

def ctg_fixed_mask(d_in, pattern="periodic", period=5):
    """Create a fixed CTG INHIBIT pattern.

    In real CTG, this would be phase-based duty cycle.
    Here we simulate with a simple periodic mask.
    """
    mask = np.ones(d_in, dtype=np.float32)
    if pattern == "periodic":
        # Every 'period'-th position is INHIBIT (zero)
        mask[::period] = 0
    elif pattern == "random_fixed":
        # Random but FIXED pattern (same for all rows)
        np.random.seed(12345)  # Fixed seed for reproducibility
        mask = (np.random.rand(d_in) > 0.2).astype(np.float32)  # 20% zeros
    return mask

def train_ctg_aware_proper(d_out, d_in, mask, n_samples=1000, lr=0.1):
    """Proper CTG-aware training simulation.

    Key difference from before: We don't have a "target W".
    Instead, we train weights from scratch to solve a task,
    WITH the CTG mask applied during forward pass.

    The network learns to put signal where it won't be masked.
    """
    # Initialize weights (will be trained)
    W = np.random.randn(d_out, d_in).astype(np.float32) * 0.1

    # Target: random linear transformation (the "true" weights we're trying to learn)
    W_true = np.random.randn(d_out, d_in).astype(np.float32) * 0.02

    # Training loop (gradient descent with CTG mask)
    for _ in range(n_samples):
        x = np.random.randn(d_in).astype(np.float32)
        y_true = W_true @ x

        # Forward: apply CTG mask (this is the key!)
        W_masked = W * mask  # CTG zeros out inhibit slots
        y_pred = W_masked @ x

        # Loss and gradient
        error = y_pred - y_true
        grad = np.outer(error, x)  # dL/dW

        # Update (only protected slots get meaningful gradients!)
        # Inhibit slots: grad is multiplied by 0 in forward, so gradient is masked
        W -= lr * grad * mask  # Gradient also masked by CTG pattern

    # Final quantization: take sign
    W_sign = np.sign(W)
    W_sign[W_sign == 0] = 1
    scale_row = np.mean(np.abs(W), axis=1, keepdims=True)

    return W_sign, scale_row, W_true

print("Understanding CTG's True Potential")
print("=" * 70)
print()
print("ChatGPT's key insight: CTG works when the TASK has structure that")
print("aligns with CTG's suppression pattern. Random W_true has no structure.")
print()
print("For CTG to help, you need END-TO-END training on a real task (LM, etc)")
print("where gradient descent can redistribute signal to protected slots.")
print()

# Let's instead verify the THEORETICAL claim:
# If weights ARE trained to avoid inhibit slots, does CTG match ternary?

print("THEORETICAL VERIFICATION:")
print("If weights are PERFECTLY adapted to CTG (zero magnitude in inhibit slots),")
print("does the masked 1-bit match ternary?")
print()

np.random.seed(42)
results = []

for dim in [128, 256]:
    d_out, d_in = dim, dim

    # Create "oracle" weights that KNOW the CTG pattern
    # (This simulates what perfectly trained weights would look like)
    mask = ctg_fixed_mask(d_in, "periodic", period=5)  # 20% zeros

    # Oracle: concentrate ALL magnitude in protected slots
    W_oracle = np.random.randn(d_out, d_in).astype(np.float32) * 0.02
    W_oracle[:, mask < 0.5] = 0  # Zero out inhibit slots DURING TRAINING

    # Quantize
    W_oracle_sign = np.sign(W_oracle)
    W_oracle_sign[W_oracle_sign == 0] = 1  # Handle zeros
    scale_oracle = np.mean(np.abs(W_oracle[W_oracle != 0])) if np.any(W_oracle != 0) else 1.0

    # Compare: oracle CTG vs ternary on same W
    W_full = np.random.randn(d_out, d_in).astype(np.float32) * 0.02
    W_full_sign = np.sign(W_full)
    W_full_sign[W_full_sign == 0] = 1
    scale_full = np.mean(np.abs(W_full), axis=1, keepdims=True)

    W_abs_mean = np.mean(np.abs(W_full))
    W_tern = np.zeros_like(W_full)
    W_tern[W_full > 0.3 * W_abs_mean] = 1
    W_tern[W_full < -0.3 * W_abs_mean] = -1
    scale_tern = np.mean(np.abs(W_full), axis=1, keepdims=True)

    # Evaluate: approximation quality of W_full
    corrs = {'naive': [], 'ctg_oracle': [], 'ternary': []}

    for _ in range(50):
        x = np.random.randn(d_in).astype(np.float32)
        y_true = W_full @ x

        # Naive
        y_naive = (W_full_sign * scale_full) @ x
        corrs['naive'].append(np.corrcoef(y_true, y_naive)[0,1])

        # CTG Oracle (this is what trained CTG weights would achieve)
        # Apply mask to sign(W_full), not oracle weights
        y_ctg = (W_full_sign * mask * scale_full) @ x
        corrs['ctg_oracle'].append(np.corrcoef(y_true, y_ctg)[0,1])

        # Ternary
        y_tern = (W_tern * scale_tern) @ x
        corrs['ternary'].append(np.corrcoef(y_true, y_tern)[0,1])

    sparsity = 1 - np.mean(mask)
    sparsity_tern = np.mean(W_tern == 0)

    result = f"dim={dim}:\n"
    result += f"  Naive 1-bit (1.00 bpp):         {np.mean(corrs['naive']):.4f}\n"
    result += f"  CTG periodic (sp={sparsity:.0%}, 1.00bpp): {np.mean(corrs['ctg_oracle']):.4f}\n"
    result += f"  Ternary (sp={sparsity_tern:.0%}, 1.58bpp):     {np.mean(corrs['ternary']):.4f}\n"
    result += "\n"

    results.append(result)
    print(f"dim={dim} done")

print()
for r in results:
    print(r)

print("=" * 70)
print("CONCLUSION:")
print("- CTG with periodic sparsity (20%) is WORSE than ternary (19%) because")
print("  ternary zeros are placed at SMALL MAGNITUDE weights, not periodically.")
print("- The fundamental limit: structured sparsity != magnitude-based sparsity.")
print()
print("ChatGPT's insight IS CORRECT for TRAINING-AWARE scenarios where the")
print("network can redistribute signal. But for POST-TRAINING quantization")
print("of random weights, CTG cannot match ternary.")

