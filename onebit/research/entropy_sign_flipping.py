"""
Research Experiment: Entropy-Constrained Sign Flipping

Goal: Achieve True 1.00 bpp by trading sign accuracy on small weights for
budget to store low-rank magnitude information.

Hypothesis:
    Error(flipped_sign_on_small_weight) < Error(missing_magnitude_info)

Methodology:
1.  Low-Rank Magnitude: Store |W| as U*S*V^T (Rank r).
2.  Sign Compression: Flip signs in 4x4 blocks to make them uniform if cost is low.
3.  Budget: Target 1.00 bpp total.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List

from onebit.model.quantize_gpt2 import GPT2Config, create_mock_gpt2_weights

@dataclass
class CompressionStats:
    bpp_signs: float
    bpp_magnitude: float
    bpp_total: float
    correlation: float
    mse: float

def get_low_rank_magnitude(W: np.ndarray, rank: int = 8) -> Tuple[np.ndarray, float]:
    """
    Approximates |W| using Rank-r SVD.
    Returns:
        Mag_approx: The approximated magnitude matrix.
        bpp_cost: Estimated BPP cost for storing U, S, V in Int8.
    """
    d_out, d_in = W.shape
    W_abs = np.abs(W)
    
    # SVD
    U, S, Vt = np.linalg.svd(W_abs, full_matrices=False)
    
    # Truncate
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    
    # Reconstruct
    Mag_approx = U_r @ np.diag(S_r) @ Vt_r
    
    # Calculate BPP cost (assuming Int8 quantization for factors)
    # U_r: d_out * r bytes
    # S_r: r * 4 bytes (keep float32 for singular values, negligible)
    # Vt_r: r * d_in bytes
    total_bits = (d_out * rank * 8) + (rank * 32) + (rank * d_in * 8)
    bpp_cost = total_bits / W.size
    
    return Mag_approx, bpp_cost

def greedy_block_flipping(W: np.ndarray, target_sign_bpp: float, block_size: int = 4) -> Tuple[np.ndarray, float]:
    """
    Greedily flips signs to make blocks uniform to reduce entropy.
    
    Args:
        W: Original weights (for cost calculation).
        target_sign_bpp: Target BPP for signs.
        block_size: Size of blocks (block_size x block_size).
        
    Returns:
        S_flipped: Sign matrix with some blocks forced uniform.
        actual_bpp: Estimated BPP after compression.
    """
    d_out, d_in = W.shape
    S = np.sign(W)
    W_abs = np.abs(W)
    
    # Pad if necessary
    pad_h = (block_size - d_out % block_size) % block_size
    pad_w = (block_size - d_in % block_size) % block_size
    
    if pad_h > 0 or pad_w > 0:
        # Simple padding for now (just ignore edge cases in cost calc or pad with zeros)
        # For prototype, let's just truncate to multiple of block_size
        h_new = d_out - (d_out % block_size)
        w_new = d_in - (d_in % block_size)
        S = S[:h_new, :w_new]
        W_abs = W_abs[:h_new, :w_new]
        d_out, d_in = h_new, w_new
        
    n_blocks_h = d_out // block_size
    n_blocks_w = d_in // block_size
    total_blocks = n_blocks_h * n_blocks_w
    
    # Calculate cost for each block to be forced to +1 or -1
    # Cost = sum(|W|) of elements that need to flip
    
    block_costs = []
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            r_start, r_end = i*block_size, (i+1)*block_size
            c_start, c_end = j*block_size, (j+1)*block_size
            
            w_block = W[r_start:r_end, c_start:c_end]
            s_block = S[r_start:r_end, c_start:c_end]
            
            # Cost to force all +1: sum(|w| where s=-1)
            cost_pos = np.sum(np.abs(w_block[s_block < 0]))
            
            # Cost to force all -1: sum(|w| where s=+1)
            cost_neg = np.sum(np.abs(w_block[s_block > 0]))
            
            # Choose cheaper direction
            if cost_pos < cost_neg:
                cost = cost_pos
                target_val = 1.0
            else:
                cost = cost_neg
                target_val = -1.0
                
            # Bits saved:
            # Uncompressed: block_size^2 bits
            # Compressed: 1 bit (sign) + overhead (say 1 bit to flag "is_compressed")
            # Let's assume a simple scheme:
            # 1 bit per block to say "Uniform" or "Raw"
            # If Uniform: +1 bit for sign -> Total 2 bits
            # If Raw: +16 bits -> Total 17 bits
            # Saving = 17 - 2 = 15 bits
            bits_saved = (block_size * block_size + 1) - 2
            
            block_costs.append({
                'r': i, 'c': j,
                'cost': cost,
                'target': target_val,
                'bits_saved': bits_saved,
                'cost_per_bit': cost / bits_saved if bits_saved > 0 else float('inf')
            })
            
    # Sort by cost per bit (cheapest first)
    block_costs.sort(key=lambda x: x['cost_per_bit'])
    
    # Current BPP (assuming all raw + 1 bit overhead)
    current_bits = total_blocks * (block_size * block_size + 1)
    target_bits = target_sign_bpp * (d_out * d_in)
    
    S_flipped = S.copy()
    
    # Greedily flip
    for item in block_costs:
        if current_bits <= target_bits:
            break
            
        # Apply flip
        r, c = item['r'], item['c']
        r_start, r_end = r*block_size, (r+1)*block_size
        c_start, c_end = c*block_size, (c+1)*block_size
        
        S_flipped[r_start:r_end, c_start:c_end] = item['target']
        current_bits -= item['bits_saved']
        
    final_bpp = current_bits / (d_out * d_in)
    return S_flipped, final_bpp

def run_experiment():
    print("="*60)
    print("Experiment: Entropy-Constrained Sign Flipping")
    print("="*60)
    
    # 1. Create Data
    print("Creating mock GPT-2 weights...")
    cfg = GPT2Config(n_layers=1, d_model=768, d_ff=3072)
    weights = create_mock_gpt2_weights(cfg)
    
    # Pick a nice big matrix: FFN Up projection [768, 3072]
    # Note: create_mock_gpt2_weights returns [d_model, d_ff]
    # We transpose it to [d_ff, d_model] as per quantization logic
    W_orig = weights["h.0.mlp.c_fc.w"].T 
    print(f"Target Matrix: {W_orig.shape}")
    
    # 2. Baseline: Binary (Sign Only)
    S_orig = np.sign(W_orig)
    scale_scalar = np.mean(np.abs(W_orig))
    W_binary = S_orig * scale_scalar
    corr_binary = np.corrcoef(W_orig.flatten(), W_binary.flatten())[0, 1]
    print(f"Baseline Binary (1.00 bpp): Corr = {corr_binary:.4f}")
    
    # 3. Baseline: Ternary (1.58 bpp)
    # Simple thresholding for ternary
    threshold = 0.05 * np.max(np.abs(W_orig)) # Arbitrary threshold for mock
    W_ternary = W_orig.copy()
    W_ternary[np.abs(W_ternary) < threshold] = 0
    W_ternary = np.sign(W_ternary) * scale_scalar # Simplified
    corr_ternary = np.corrcoef(W_orig.flatten(), W_ternary.flatten())[0, 1]
    print(f"Baseline Ternary (~1.58 bpp): Corr = {corr_ternary:.4f}")
    print("-" * 60)
    
    # 4. Hybrid Approach (Rank 8)
    print("Running Hybrid Approach (Rank 8)...")
    rank = 8
    Mag_approx, bpp_mag = get_low_rank_magnitude(W_orig, rank=rank)
    print(f"  Low-Rank Magnitude (r={rank}): Cost = {bpp_mag:.4f} bpp")
    
    target_total_bpp = 1.00
    target_sign_bpp = target_total_bpp - bpp_mag
    print(f"  Target Sign BPP: {target_sign_bpp:.4f}")
    
    S_flipped, bpp_signs = greedy_block_flipping(W_orig, target_sign_bpp, block_size=4)
    print(f"  Compressed Signs: Cost = {bpp_signs:.4f} bpp")
    
    W_hybrid = S_flipped * Mag_approx
    corr_hybrid = np.corrcoef(W_orig.flatten(), W_hybrid.flatten())[0, 1]
    total_bpp = bpp_mag + bpp_signs
    
    print(f"  Hybrid (r=8) Correlation: {corr_hybrid:.4f}")
    print(f"  vs Binary: {corr_hybrid - corr_binary:+.4f}")

    print("-" * 60)

    # 5. Hybrid Approach (Rank 1)
    print("Running Hybrid Approach (Rank 1)...")
    rank = 1
    Mag_approx, bpp_mag = get_low_rank_magnitude(W_orig, rank=rank)
    print(f"  Low-Rank Magnitude (r={rank}): Cost = {bpp_mag:.4f} bpp")
    
    target_total_bpp = 1.00
    target_sign_bpp = target_total_bpp - bpp_mag
    print(f"  Target Sign BPP: {target_sign_bpp:.4f}")
    
    S_flipped, bpp_signs = greedy_block_flipping(W_orig, target_sign_bpp, block_size=4)
    print(f"  Compressed Signs: Cost = {bpp_signs:.4f} bpp")
    
    W_hybrid = S_flipped * Mag_approx
    corr_hybrid_r1 = np.corrcoef(W_orig.flatten(), W_hybrid.flatten())[0, 1]
    total_bpp = bpp_mag + bpp_signs
    
    print(f"  Hybrid (r=1) Correlation: {corr_hybrid_r1:.4f}")
    print(f"  vs Binary: {corr_hybrid_r1 - corr_binary:+.4f}")
    
    if corr_hybrid_r1 > corr_binary:
        print("  SUCCESS: Beat standard binary with Rank-1!")
    else:
        print("  FAILURE: Did not beat standard binary with Rank-1.")

if __name__ == "__main__":
    import sys
    # Redirect stdout to file
    with open("experiment_results_utf8.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        run_experiment()
        sys.stdout = sys.__stdout__
    
    # Also print to console for sanity
    with open("experiment_results_utf8.txt", "r", encoding="utf-8") as f:
        print(f.read())
