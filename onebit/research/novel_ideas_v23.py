"""
Novel Ideas V23: High-Dimensional VQ
Hypothesis: VQ efficiency improves with dimensionality.
Goal: Test if larger blocks (D=8, D=16) achieve better correlation at fixed 1.00 bpp.

Configs:
1. D=4,  K=16    (1.00 bpp) - Baseline (V22)
2. D=8,  K=256   (1.00 bpp) - 8 bits / 8 weights
3. D=16, K=65536 (1.00 bpp) - 16 bits / 16 weights
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel
from tqdm import tqdm
import os

def get_gpt2_weights():
    print("Loading GPT-2 weights...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    weights = []
    for name, param in model.named_parameters():
        if "c_attn" in name or "c_proj" in name or "c_fc" in name:
            w = param.detach().numpy()
            if w.ndim == 2:
                if "c_fc" in name or "c_attn" in name or "c_proj" in name:
                    w = w.T
                weights.append(w)
    print(f"Loaded {len(weights)} matrices.")
    return weights

class HighDimVQ:
    def __init__(self, n_codes, block_size):
        self.n_codes = n_codes
        self.block_size = block_size
        
    def run(self, W):
        d_out, d_in = W.shape
        bs = self.block_size
        
        # Pad
        pad = (bs - (W.size % bs)) % bs
        W_flat = W.flatten()
        W_pad = np.pad(W_flat, (0, pad))
        
        vectors = W_pad.reshape(-1, bs)
        
        # Train K-Means
        # For K=65536, standard K-means is slow. Use MiniBatch or just random sampling if too slow.
        # Let's use a fast approximation: Train on subset, assign all.
        
        n_vectors = len(vectors)
        # Ensure we have enough training data for K codes
        target_train_size = max(50000, self.n_codes)
        if n_vectors < target_train_size:
             # If dataset is small, use all of it, and handle K > N later
             train_size = n_vectors
        else:
             train_size = target_train_size
             
        train_indices = np.random.choice(n_vectors, train_size, replace=False)
        train_vectors = vectors[train_indices]
        
        # Init centroids
        if self.n_codes > train_size:
            # Edge case: more codes than training data
            # Initialize with random noise or sample with replacement
            indices = np.random.choice(len(train_vectors), self.n_codes, replace=True)
            centroids = train_vectors[indices].copy()
        else:
            indices = np.random.choice(len(train_vectors), self.n_codes, replace=False)
            centroids = train_vectors[indices].copy()
        
        # Fast K-Means (3 iters)
        for _ in range(3):
            # Chunked distance calculation to save memory
            # dists matrix: (Batch, K) -> can be huge for K=65536
            # We need to be careful.
            # Use FAISS-like logic: ||x-c||^2 = ||x||^2 + ||c||^2 - 2xc
            
            # Precompute centroid norms
            c_norms = (centroids**2).sum(axis=1) # (K,)
            
            # Update centroids accumulator
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(self.n_codes)
            
            # Process in batches
            batch_size = 1024
            for i in range(0, len(train_vectors), batch_size):
                batch = train_vectors[i:i+batch_size]
                
                # Distance
                # (B, D) @ (D, K) -> (B, K)
                dots = np.dot(batch, centroids.T)
                dists = -2 * dots + c_norms[None, :] # Ignore ||x||^2 as it's constant for argmin
                
                assigns = np.argmin(dists, axis=1)
                
                # Accumulate
                for j, a in enumerate(assigns):
                    new_centroids[a] += batch[j]
                    counts[a] += 1
            
            # Average
            mask = counts > 0
            centroids[mask] = new_centroids[mask] / counts[mask][:, None]
            # Empty clusters stay same (or could re-init)
            
        # Quantize All
        # Process in batches
        recon_vectors = np.zeros_like(vectors)
        c_norms = (centroids**2).sum(axis=1)
        
        batch_size = 1024
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            dots = np.dot(batch, centroids.T)
            dists = -2 * dots + c_norms[None, :]
            assigns = np.argmin(dists, axis=1)
            recon_vectors[i:i+batch_size] = centroids[assigns]
            
        W_recon = recon_vectors.flatten()[:W.size].reshape(d_out, d_in)
        return W_recon

def run_experiments():
    weights = get_gpt2_weights()
    weights = weights[:3] # Test on first 3 matrices
    
    print("\n" + "="*60)
    print("EXPERIMENT 19: NOVEL IDEAS V23 (High-Dim VQ)")
    print("="*60)
    
    results = []
    
    # 1. D=4, K=16 (1.00 bpp)
    print("\nRunning D=4, K=16 (1.00 bpp)...")
    corrs = []
    for W in tqdm(weights):
        vq = HighDimVQ(n_codes=16, block_size=4)
        W_rec = vq.run(W)
        corr = np.corrcoef(W.flatten(), W_rec.flatten())[0, 1]
        corrs.append(corr)
    print(f"D=4, K=16: {np.mean(corrs):.4f}")
    results.append(f"D=4, K=16 | 1.00 bpp | Corr: {np.mean(corrs):.4f}")
    
    # 2. D=8, K=256 (1.00 bpp)
    print("\nRunning D=8, K=256 (1.00 bpp)...")
    corrs = []
    for W in tqdm(weights):
        vq = HighDimVQ(n_codes=256, block_size=8)
        W_rec = vq.run(W)
        corr = np.corrcoef(W.flatten(), W_rec.flatten())[0, 1]
        corrs.append(corr)
    print(f"D=8, K=256: {np.mean(corrs):.4f}")
    results.append(f"D=8, K=256 | 1.00 bpp | Corr: {np.mean(corrs):.4f}")
    
    # 3. D=16, K=65536 (1.00 bpp)
    print("\nRunning D=16, K=65536 (1.00 bpp)...")
    corrs = []
    for W in tqdm(weights):
        # Note: K=65536 might be slow, but we optimized the kernel
        vq = HighDimVQ(n_codes=65536, block_size=16)
        W_rec = vq.run(W)
        corr = np.corrcoef(W.flatten(), W_rec.flatten())[0, 1]
        corrs.append(corr)
    print(f"D=16, K=65536: {np.mean(corrs):.4f}")
    results.append(f"D=16, K=65536 | 1.00 bpp | Corr: {np.mean(corrs):.4f}")

    with open("results_v23.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    run_experiments()
