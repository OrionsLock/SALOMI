"""
Novel Ideas V3: Pushing the boundaries of 1-bit quantization.

Experiments:
1. Iterative Rotated Binary (IRB):
   - Learn an orthogonal rotation R to align weights with the binary hypercube.
   - Algorithm: Alternating optimization of B = sign(W R^T) and R = Procrustes(W, B).
   - Hypothesis: A rotated basis captures more information in the signs.

2. VQ-Magnitude Binary:
   - Instead of Low-Rank approximation for magnitudes, use Vector Quantization.
   - Encode 4x4 magnitude blocks using a learned codebook.
   - Hypothesis: Local magnitude patterns are repetitive and can be compressed efficiently.

3. Sparse Correction Binary:
   - Standard Binary + Sparse "Fixes".
   - Store indices and values of the top-k errors.
   - Hypothesis: Fixing a few large errors is more efficient than global improvements.
"""

import numpy as np
import torch
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
import time

def simple_kmeans(X: np.ndarray, n_clusters: int, n_iter: int = 10):
    """Simple K-Means implementation using NumPy."""
    n_samples, n_features = X.shape
    
    # Random initialization
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[indices]
    
    labels = np.zeros(n_samples, dtype=int)
    
    for _ in range(n_iter):
        # Assign labels
        # Compute distances (n_samples, n_clusters)
        # ||x - c||^2 = x^2 + c^2 - 2xc
        dists = np.sum(X**2, axis=1, keepdims=True) + \
                np.sum(centroids**2, axis=1) - \
                2 * X @ centroids.T
        labels = np.argmin(dists, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(n_clusters)
        
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = np.mean(X[mask], axis=0)
                counts[k] = np.sum(mask)
            else:
                # Re-init empty cluster
                new_centroids[k] = X[np.random.randint(n_samples)]
                
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
        
    return centroids, labels


# =============================================================================
# 1. ITERATIVE ROTATED BINARY (IRB)
# =============================================================================

class IterativeRotatedBinary:
    """
    Learns an orthogonal rotation R such that W ≈ B @ R.
    B is binary {-1, +1}. R is orthogonal.
    
    Optimization:
    1. Fix R, B = sign(W @ R.T)
    2. Fix B, R = argmin ||W - B @ R||_F  (Orthogonal Procrustes)
       M = B.T @ W
       U, _, Vt = SVD(M)
       R = U @ Vt
    """
    def __init__(self, d_in: int, d_out: int, n_iter: int = 20):
        self.d_in = d_in
        self.d_out = d_out
        self.n_iter = n_iter
        
        self.R = None # (d_in, d_in) orthogonal matrix
        self.B = None # (d_out, d_in) binary matrix
        self.scale = 1.0
        
    def train(self, W_target: np.ndarray):
        # Initialize R as identity
        self.R = np.eye(self.d_in, dtype=np.float32)
        
        for i in range(self.n_iter):
            # 1. Update B (Project W onto rotated basis)
            W_rotated = W_target @ self.R.T
            self.B = np.sign(W_rotated).astype(np.float32)
            self.B[self.B == 0] = 1.0
            
            # 2. Update R (Orthogonal Procrustes)
            # We want R that minimizes ||W - B @ R||
            # Solution involves SVD of B.T @ W
            M = self.B.T @ W_target
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            self.R = U @ Vt
            
            # Calculate error
            W_recon = self.B @ self.R
            # Optimal scale
            self.scale = np.sum(W_recon * W_target) / np.sum(W_recon**2)
            error = np.mean((W_recon * self.scale - W_target)**2)
            # print(f"Iter {i}: MSE = {error:.6f}")
            
    def get_weights(self) -> np.ndarray:
        return (self.B @ self.R) * self.scale
    
    def effective_bpp(self) -> float:
        # Storage:
        # B: d_out * d_in bits
        # R: d_in * d_in floats (Can we quantize R? Maybe, but let's assume FP16 or similar overhead)
        # Actually, R is shared across rows? No, R is d_in x d_in.
        # If d_out >> d_in, R is negligible?
        # For GPT-2: d_in = d_out = 768.
        # R is 768*768 floats. That's huge. 32 bits * 768 = 24576 bits per row equivalent.
        # This is only viable if R is shared or structured.
        # Let's assume R is stored in FP16 (16 bits).
        
        n_weights = self.d_out * self.d_in
        b_bits = n_weights
        r_bits = self.d_in * self.d_in * 16
        return (b_bits + r_bits) / n_weights

# =============================================================================
# 2. VQ-MAGNITUDE BINARY
# =============================================================================

class VQMagnitudeBinary:
    """
    W = Sign * Magnitude
    Sign is 1-bit.
    Magnitude is compressed using Vector Quantization on 4x4 blocks.
    """
    def __init__(self, d_in: int, d_out: int, block_size: int = 4, n_codes: int = 256):
        self.d_in = d_in
        self.d_out = d_out
        self.block_size = block_size
        self.n_codes = n_codes
        
        self.S = None # Signs
        self.codebook = None # (n_codes, block_size*block_size)
        self.indices = None # (n_blocks_h, n_blocks_w)
        
    def train(self, W_target: np.ndarray):
        # 1. Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # 2. Magnitudes
        M = np.abs(W_target)
        
        # Extract blocks
        h, w = M.shape
        bh, bw = self.block_size, self.block_size
        
        # Pad if needed
        pad_h = (bh - h % bh) % bh
        pad_w = (bw - w % bw) % bw
        M_padded = np.pad(M, ((0, pad_h), (0, pad_w)), mode='constant')
        
        blocks = []
        n_h = M_padded.shape[0] // bh
        n_w = M_padded.shape[1] // bw
        
        for i in range(n_h):
            for j in range(n_w):
                block = M_padded[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                blocks.append(block.flatten())
        
        blocks = np.array(blocks)
        
        # K-Means Clustering
        self.codebook, labels = simple_kmeans(blocks, n_clusters=self.n_codes)
        self.indices = labels.reshape(n_h, n_w)
        
    def get_weights(self) -> np.ndarray:
        # Reconstruct Magnitude
        bh, bw = self.block_size, self.block_size
        n_h, n_w = self.indices.shape
        
        M_rec = np.zeros((n_h * bh, n_w * bw), dtype=np.float32)
        
        for i in range(n_h):
            for j in range(n_w):
                idx = self.indices[i, j]
                block = self.codebook[idx].reshape(bh, bw)
                M_rec[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = block
                
        # Crop padding
        M_rec = M_rec[:self.d_out, :self.d_in]
        
        return self.S * M_rec
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        
        # Signs: 1 bit
        sign_bits = n_weights
        
        # Indices: log2(n_codes) per block
        n_blocks = self.indices.size
        index_bits = n_blocks * np.log2(self.n_codes)
        
        # Codebook: n_codes * block_area * 32 (or 16)
        codebook_bits = self.n_codes * (self.block_size**2) * 32
        
        return (sign_bits + index_bits + codebook_bits) / n_weights

# =============================================================================
# 3. SPARSE CORRECTION BINARY
# =============================================================================

class SparseCorrectionBinary:
    """
    W = Scale * Sign + SparseCorrection
    
    SparseCorrection stores (index, value) for the worst errors.
    """
    def __init__(self, d_in: int, d_out: int, sparsity: float = 0.05):
        self.d_in = d_in
        self.d_out = d_out
        self.sparsity = sparsity
        
        self.S = None
        self.scale = 1.0
        self.corr_indices = None
        self.corr_values = None
        
    def train(self, W_target: np.ndarray):
        # Base Binary
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Optimal Scale
        self.scale = np.mean(np.abs(W_target))
        
        # Residual
        W_approx = self.S * self.scale
        Residual = W_target - W_approx
        
        # Find top-k errors
        k = int(self.d_in * self.d_out * self.sparsity)
        flat_res = np.abs(Residual.flatten())
        top_k_idx = np.argsort(flat_res)[-k:]
        
        self.corr_indices = top_k_idx
        self.corr_values = Residual.flatten()[top_k_idx]
        
    def get_weights(self) -> np.ndarray:
        W = self.S * self.scale
        
        flat_W = W.flatten()
        flat_W[self.corr_indices] += self.corr_values
        
        return flat_W.reshape(self.d_out, self.d_in)
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        
        # Base: 1 bit + scale
        base_bits = n_weights + 32
        
        # Correction: k * (log2(N) + 16)  (assuming FP16 for values)
        k = len(self.corr_indices)
        idx_bits = np.ceil(np.log2(n_weights))
        val_bits = 16 
        corr_bits = k * (idx_bits + val_bits)
        
        return (base_bits + corr_bits) / n_weights

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V3: Rotated, VQ, and Sparse Correction")
    print("="*80)
    
    # Setup Data
    d = 256
    np.random.seed(42)
    # Synthetic weights with some structure (SVD decay)
    U = np.random.randn(d, d)
    U, _ = np.linalg.qr(U)
    Vt = np.random.randn(d, d)
    Vt, _ = np.linalg.qr(Vt)
    S = np.exp(-np.linspace(0, 5, d)) # Exponential decay singular values
    W_true = U @ np.diag(S) @ Vt
    W_true = W_true.astype(np.float32)
    
    # Generate Data
    X_test = np.random.randn(1000, d).astype(np.float32)
    Y_test = X_test @ W_true.T
    
    results = {}
    
    # 0. Baselines
    # Binary
    S_bin = np.sign(W_true)
    S_bin[S_bin==0] = 1
    scale_bin = np.mean(np.abs(W_true))
    W_bin = S_bin * scale_bin
    corr_bin = np.corrcoef((X_test @ W_bin.T).flatten(), Y_test.flatten())[0,1]
    results['Binary Baseline'] = {'corr': corr_bin, 'bpp': 1.0}
    
    # Ternary
    thresh = np.percentile(np.abs(W_true), 30)
    W_tern = S_bin * (np.abs(W_true) > thresh)
    scale_tern = np.mean(np.abs(W_true[np.abs(W_true) > thresh]))
    W_tern = W_tern * scale_tern
    corr_tern = np.corrcoef((X_test @ W_tern.T).flatten(), Y_test.flatten())[0,1]
    results['Ternary Baseline'] = {'corr': corr_tern, 'bpp': 1.58}
    
    print(f"Binary Baseline: {corr_bin:.4f} @ 1.00 bpp")
    print(f"Ternary Baseline: {corr_tern:.4f} @ 1.58 bpp")
    print("-" * 40)

    # 1. Iterative Rotated Binary
    print("\nRunning Iterative Rotated Binary...")
    irb = IterativeRotatedBinary(d, d)
    irb.train(W_true)
    W_irb = irb.get_weights()
    corr_irb = np.corrcoef((X_test @ W_irb.T).flatten(), Y_test.flatten())[0,1]
    bpp_irb = irb.effective_bpp()
    results['Rotated Binary'] = {'corr': corr_irb, 'bpp': bpp_irb}
    print(f"Result: {corr_irb:.4f} @ {bpp_irb:.2f} bpp")
    
    # 2. VQ Magnitude
    print("\nRunning VQ Magnitude Binary...")
    # Try different codebook sizes
    for n_codes in [16, 64, 256]:
        vqm = VQMagnitudeBinary(d, d, block_size=4, n_codes=n_codes)
        vqm.train(W_true)
        W_vqm = vqm.get_weights()
        corr_vqm = np.corrcoef((X_test @ W_vqm.T).flatten(), Y_test.flatten())[0,1]
        bpp_vqm = vqm.effective_bpp()
        results[f'VQ-Mag (K={n_codes})'] = {'corr': corr_vqm, 'bpp': bpp_vqm}
        print(f"K={n_codes}: {corr_vqm:.4f} @ {bpp_vqm:.2f} bpp")

    # 3. Sparse Correction
    print("\nRunning Sparse Correction Binary...")
    for sp in [0.01, 0.05, 0.10]:
        scb = SparseCorrectionBinary(d, d, sparsity=sp)
        scb.train(W_true)
        W_scb = scb.get_weights()
        corr_scb = np.corrcoef((X_test @ W_scb.T).flatten(), Y_test.flatten())[0,1]
        bpp_scb = scb.effective_bpp()
        results[f'Sparse Corr ({sp*100}%)'] = {'corr': corr_scb, 'bpp': bpp_scb}
        print(f"Sparsity={sp}: {corr_scb:.4f} @ {bpp_scb:.2f} bpp")

    # Summary
    with open("results_v3_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<25} {'Corr':>8} {'BPP':>8} {'vs Tern':>10}\n")
        f.write("-" * 60 + "\n")
        
        for name, res in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (res['corr'] - corr_tern) / corr_tern * 100
            line = f"{name:<25} {res['corr']:>8.4f} {res['bpp']:>8.2f} {vs_tern:>+9.1f}%\n"
            print(line.strip())
            f.write(line)

if __name__ == "__main__":
    run_experiments()
