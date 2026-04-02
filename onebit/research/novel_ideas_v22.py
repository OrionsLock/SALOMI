"""
Novel Ideas V22: The Quest for True 1.00 BPP
Goal: Achieve >0.95 correlation at strictly 1.00 bpp.

Methods:
1. SignedBlockVQ: Joint sign-magnitude VQ (K=16, Block=4 -> 1.00 bpp)
2. RotatedBlockVQ: Rotate blocks to align energy, then VQ
3. PredictiveSignVQ: Predict signs to lower entropy
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel
from tqdm import tqdm
import os

def get_gpt2_weights():
    """Load real GPT-2 weights for testing"""
    print("Loading GPT-2 weights...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    weights = []
    for name, param in model.named_parameters():
        if "c_attn" in name or "c_proj" in name or "c_fc" in name:
            w = param.detach().numpy()
            if w.ndim == 2:
                # Transpose Conv1D weights to (Out, In)
                if "c_fc" in name or "c_attn" in name or "c_proj" in name:
                    w = w.T
                weights.append(w)
    print(f"Loaded {len(weights)} matrices.")
    return weights

class SignedBlockVQ:
    """
    Jointly quantize Sign and Magnitude using VQ.
    Block Size = 4
    Codebook Size = 16 (4 bits)
    BPP = 4/4 = 1.00 bits (plus negligible codebook overhead)
    """
    def __init__(self, n_codes=16, block_size=4):
        self.n_codes = n_codes
        self.block_size = block_size
        
    def run(self, W):
        d_out, d_in = W.shape
        bs = self.block_size
        
        # Pad
        pad_h = (bs - d_out % bs) % bs
        pad_w = (bs - d_in % bs) % bs
        W_pad = np.pad(W, ((0, pad_h), (0, pad_w)))
        
        h_p, w_p = W_pad.shape
        n_blocks = (h_p * w_p) // (bs * bs)
        
        # Reshape to (N_blocks, BlockSize^2)
        # Note: We use BlockSize^2 vectors (e.g. 16-dim vectors for 4x4 blocks? No, usually 1x4 or 4x1)
        # Let's try 1x4 blocks (vectors of length 4) to keep codebook small?
        # If BlockSize=4 means 4x4=16 elements, then K=16 is tiny (1 bit per 4 weights -> 0.25 bpp).
        # Wait, standard VQ usually does blocks of size B.
        # To get 1.00 bpp:
        # If vector dim is D, we need D bits per vector.
        # So K = 2^D.
        # If D=4, K=16.
        # So we treat every 4 weights as a vector.
        
        # Reshape to (N_vectors, 4)
        vectors = W_pad.reshape(-1, 4)
        
        # Train K-Means
        # Init
        indices = np.random.choice(len(vectors), self.n_codes, replace=False)
        centroids = vectors[indices].copy()
        
        # Iterations
        for _ in range(10):
            dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_codes):
                mask = (assignments == k)
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(vectors[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]
            centroids = new_centroids
            
        # Reconstruct
        recon_vectors = centroids[assignments]
        W_recon = recon_vectors.reshape(h_p, w_p)[:d_out, :d_in]
        
        # Calc BPP
        # Indices: log2(K) bits per vector
        # Vectors: D weights
        # BPP = log2(K) / D
        bpp = np.log2(self.n_codes) / 4.0
        
        return W_recon, bpp

class RotatedBlockVQ:
    """
    Rotate vectors before VQ to align energy.
    Uses PCA to find best rotation.
    """
    def __init__(self, n_codes=16, block_size=4):
        self.n_codes = n_codes
        self.block_size = block_size
        
    def run(self, W):
        d_out, d_in = W.shape
        
        # Pad
        pad = (4 - (W.size % 4)) % 4
        W_flat = W.flatten()
        W_pad = np.pad(W_flat, (0, pad))
        
        vectors = W_pad.reshape(-1, 4)
        
        # 1. PCA Rotation
        # Center
        mean = np.mean(vectors, axis=0)
        vectors_c = vectors - mean
        
        # Covariance
        cov = np.dot(vectors_c.T, vectors_c) / len(vectors_c)
        evals, evecs = np.linalg.eigh(cov)
        
        # Sort descending
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        
        # Rotate: Y = X @ R
        rotated_vectors = np.dot(vectors_c, evecs)
        
        # 2. VQ on Rotated Vectors
        indices = np.random.choice(len(rotated_vectors), self.n_codes, replace=False)
        centroids = rotated_vectors[indices].copy()
        
        for _ in range(10):
            dists = np.linalg.norm(rotated_vectors[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_codes):
                mask = (assignments == k)
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(rotated_vectors[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]
            centroids = new_centroids
            
        # 3. Inverse Rotate
        recon_rotated = centroids[assignments]
        recon_vectors = np.dot(recon_rotated, evecs.T) + mean
        
        W_recon = recon_vectors.flatten()[:W.size].reshape(d_out, d_in)
        
        bpp = np.log2(self.n_codes) / 4.0
        return W_recon, bpp

def run_experiments():
    weights = get_gpt2_weights()
    # Use a subset for speed
    weights = weights[:5] 
    
    print("\n" + "="*60)
    print("EXPERIMENT 18: NOVEL IDEAS V22 (True 1.00 BPP)")
    print("="*60)
    
    results = []
    
    # 1. SignedBlockVQ (K=16, D=4 -> 1.00 bpp)
    print("\nRunning SignedBlockVQ (K=16)...")
    corrs = []
    for W in tqdm(weights):
        vq = SignedBlockVQ(n_codes=16, block_size=4)
        W_rec, bpp = vq.run(W)
        corr = np.corrcoef(W.flatten(), W_rec.flatten())[0, 1]
        corrs.append(corr)
    
    mean_corr = np.mean(corrs)
    print(f"SignedBlockVQ (1.00 bpp): {mean_corr:.4f}")
    results.append(f"SignedBlockVQ (K=16) | 1.00 bpp | Corr: {mean_corr:.4f}")
    
    # 2. SignedBlockVQ (K=32, D=4 -> 1.25 bpp)
    print("\nRunning SignedBlockVQ (K=32)...")
    corrs = []
    for W in tqdm(weights):
        vq = SignedBlockVQ(n_codes=32, block_size=4)
        W_rec, bpp = vq.run(W)
        corr = np.corrcoef(W.flatten(), W_rec.flatten())[0, 1]
        corrs.append(corr)
        
    mean_corr = np.mean(corrs)
    print(f"SignedBlockVQ (1.25 bpp): {mean_corr:.4f}")
    results.append(f"SignedBlockVQ (K=32) | 1.25 bpp | Corr: {mean_corr:.4f}")
    
    # 3. RotatedBlockVQ (K=16, D=4 -> 1.00 bpp)
    print("\nRunning RotatedBlockVQ (K=16)...")
    corrs = []
    for W in tqdm(weights):
        vq = RotatedBlockVQ(n_codes=16, block_size=4)
        W_rec, bpp = vq.run(W)
        corr = np.corrcoef(W.flatten(), W_rec.flatten())[0, 1]
        corrs.append(corr)
        
    mean_corr = np.mean(corrs)
    print(f"RotatedBlockVQ (1.00 bpp): {mean_corr:.4f}")
    results.append(f"RotatedBlockVQ (K=16) | 1.00 bpp | Corr: {mean_corr:.4f}")
    
    # 4. RotatedBlockVQ (K=256, D=4 -> 2.00 bpp) - Upper bound check
    print("\nRunning RotatedBlockVQ (K=256)...")
    corrs = []
    for W in tqdm(weights):
        vq = RotatedBlockVQ(n_codes=256, block_size=4)
        W_rec, bpp = vq.run(W)
        corr = np.corrcoef(W.flatten(), W_rec.flatten())[0, 1]
        corrs.append(corr)
        
    mean_corr = np.mean(corrs)
    print(f"RotatedBlockVQ (2.00 bpp): {mean_corr:.4f}")
    results.append(f"RotatedBlockVQ (K=256) | 2.00 bpp | Corr: {mean_corr:.4f}")

    # Write results
    with open("results_v22.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    run_experiments()
