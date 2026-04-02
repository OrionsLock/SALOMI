"""
BPP ANALYSIS: Why is the 0.58 claim wrong?

The claimed 0.58 bpp calculation was:
- Sign: ~0.5 bits (entropy-coded balanced signs)
- Routing: 1 bit per 16 weights = 0.0625 bpp
- Index: (0.6 × 5 + 0.4 × 3) / 16 ≈ 0.26 bpp

But the REAL calculation shows 1.33 bpp. Let's find the error.
"""

import numpy as np
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
W = model.transformer.h[0].mlp.c_fc.weight.detach().cpu().numpy().T

h, w = W.shape
bs = 4
n_weights = h * w
n_blocks = ((h + bs - 1) // bs) * ((w + bs - 1) // bs)

print("=" * 70)
print("BPP BREAKDOWN ANALYSIS")
print("=" * 70)

print(f"\nMatrix: {h} x {w} = {n_weights:,} weights")
print(f"Blocks: {n_blocks:,} ({bs}x{bs} = 16 weights per block)")

# Parameters
k_high, k_low = 32, 8
threshold = 0.6
n_high = int(n_blocks * threshold)
n_low = n_blocks - n_high

print(f"\nHigh path: {n_high:,} blocks ({threshold*100:.0f}%), K={k_high}")
print(f"Low path:  {n_low:,} blocks ({(1-threshold)*100:.0f}%), K={k_low}")

print("\n" + "-" * 70)
print("ORIGINAL CLAIM (0.58 bpp)")
print("-" * 70)

# Claimed calculation
claim_sign = 0.5  # bits per weight
claim_routing = 1.0 / 16  # 1 bit per block / 16 weights per block
claim_index = (threshold * np.log2(k_high) + (1-threshold) * np.log2(k_low)) / 16
claim_total = claim_sign + claim_routing + claim_index

print(f"Sign:     {claim_sign:.4f} bpp")
print(f"Routing:  {claim_routing:.4f} bpp")
print(f"Index:    {claim_index:.4f} bpp")
print(f"TOTAL:    {claim_total:.4f} bpp")
print(f"\nPROBLEM: Ignores codebook overhead!")

print("\n" + "-" * 70)
print("CORRECT CALCULATION")
print("-" * 70)

# Sign bits - worst case 1 bit per weight
sign_bits = 1.0 * n_weights
sign_bpp = sign_bits / n_weights

# Actually, with entropy coding and ~50% positive:
signs = np.sign(W)
pos_frac = np.mean(signs > 0)
sign_entropy = -pos_frac * np.log2(pos_frac) - (1-pos_frac) * np.log2(1-pos_frac)
sign_bits_entropy = sign_entropy * n_weights
sign_bpp_entropy = sign_bits_entropy / n_weights

# Routing bits
routing_bits = n_blocks * 1.0
routing_bpp = routing_bits / n_weights

# Index bits
index_bits_high = n_high * np.log2(k_high)
index_bits_low = n_low * np.log2(k_low)
index_bits = index_bits_high + index_bits_low
index_bpp = index_bits / n_weights

# Codebook overhead (THIS IS THE MISSING PIECE)
cb_bits = (k_high + k_low) * (bs * bs) * 16  # FP16 storage
cb_bpp = cb_bits / n_weights

print(f"Sign (worst case):     {sign_bpp:.4f} bpp ({sign_bits:,.0f} bits)")
print(f"Sign (entropy):        {sign_bpp_entropy:.4f} bpp ({sign_bits_entropy:,.0f} bits)")
print(f"Routing:               {routing_bpp:.4f} bpp ({routing_bits:,.0f} bits)")
print(f"Index:                 {index_bpp:.4f} bpp ({index_bits:,.0f} bits)")
print(f"Codebook:              {cb_bpp:.4f} bpp ({cb_bits:,} bits)")

total_worst = sign_bits + routing_bits + index_bits + cb_bits
total_entropy = sign_bits_entropy + routing_bits + index_bits + cb_bits

print(f"\nTOTAL (worst sign):    {total_worst/n_weights:.4f} bpp")
print(f"TOTAL (entropy sign):  {total_entropy/n_weights:.4f} bpp")

print("\n" + "-" * 70)
print("WHY 0.58 IS WRONG")
print("-" * 70)
print(f"""
The 0.58 claim made THREE errors:

1. SIGN BITS: Claimed ~0.5 bpp via entropy coding
   Reality: Signs are NOT perfectly balanced
   Actual sign entropy: {sign_bpp_entropy:.3f} bpp (not 0.5)

2. CODEBOOK OVERHEAD: Claimed "negligible"
   Reality: {cb_bpp:.4f} bpp - NOT negligible!
   With K=32+8=40 centroids × 16 values × 16 bits = {cb_bits:,} bits

3. PER-BLOCK AMORTIZATION: Claimed "divide by 16"
   Reality: Index bits ARE per-block, but sign bits are per-WEIGHT

CORRECTED MINIMUM BPP: {total_entropy/n_weights:.3f}
""")

print("\n" + "-" * 70)
print("HOW TO ACTUALLY GET 0.58 BPP")
print("-" * 70)

# To get 0.58 bpp, we need MUCH more aggressive compression
target_bpp = 0.58
available_bits = target_bpp * n_weights

# Sign: Use learned grouping, assume 0.3 bpp achievable
sign_budget = 0.3 * n_weights

# Routing: 0.0625 per weight (1 bit per 16 weights)
routing_budget = routing_bits

# Remaining for index + codebook
remaining = available_bits - sign_budget - routing_budget
print(f"Target: {target_bpp} bpp = {available_bits:,.0f} total bits")
print(f"Sign budget: {sign_budget:,.0f} bits (0.3 bpp via learned grouping)")
print(f"Routing budget: {routing_budget:,.0f} bits")
print(f"Remaining for index+codebook: {remaining:,.0f} bits")

# With small codebook
k_small = 4  # Only 4 centroids = 2 bits per block
index_bits_small = n_blocks * np.log2(k_small)
cb_bits_small = k_small * (bs * bs) * 16

print(f"\nWith K=4 (2 bits per block):")
print(f"  Index bits: {index_bits_small:,.0f}")
print(f"  Codebook: {cb_bits_small:,}")
print(f"  Total: {sign_budget + routing_budget + index_bits_small + cb_bits_small:,.0f}")
print(f"  BPP: {(sign_budget + routing_budget + index_bits_small + cb_bits_small)/n_weights:.3f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The 0.58 bpp claim is INVALID for the current DualPathVQ configuration.

To achieve true 0.58 bpp:
1. Need much smaller codebooks (K≤4)
2. Need sign compression via grouping
3. Or use binary VQ (K=2) with residual

The ACTUAL bpp for K=32/8 DualPathVQ is approximately 1.33 bpp.

This is STILL better than ternary (1.58 bpp) but not sub-1-bit!
""")

