"""
Novel Ideas V24: Multi-Metric Push to 1.000 Correlation
Goal: Achieve 1.000 correlation at 1.00 bpp using advanced VQ techniques.

Methods:
1. Product Quantization (8 sub-vectors of 2D, K=256 each)
2. Learned Rotation VQ (PCA + VQ)
3. Multi-metric evaluation (Corr, MSE, NMSE, SNR, Cosine)
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
                if "c_fc" in name or "c_attn" in name or "c_proj" in name:
                    w = w.T
                weights.append(w)
    print(f"Loaded {len(weights)} matrices.")
    return weights

def compute_metrics(W_orig, W_recon):
    """Compute comprehensive metrics"""
    W_orig_flat = W_orig.flatten()
    W_recon_flat = W_recon.flatten()
    
    # 1. Pearson Correlation
    corr = np.corrcoef(W_orig_flat, W_recon_flat)[0, 1]
    
    # 2. MSE
    mse = np.mean((W_orig_flat - W_recon_flat)**2)
    
    # 3. NMSE (Normalized MSE)
    variance = np.var(W_orig_flat)
    nmse = mse / (variance + 1e-10)
    
    # 4. SNR (Signal-to-Noise Ratio in dB)
    signal_power = np.mean(W_orig_flat**2)
    noise_power = mse
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # 5. Cosine Similarity
    dot = np.dot(W_orig_flat, W_recon_flat)
    norm_orig = np.linalg.norm(W_orig_flat)
    norm_recon = np.linalg.norm(W_recon_flat)
    cosine_sim = dot / (norm_orig * norm_recon + 1e-10)
    
    # 6. Max Absolute Error
    max_abs_error = np.max(np.abs(W_orig_flat - W_recon_flat))
    
    return {
        'correlation': corr,
        'mse': mse,
        'nmse': nmse,
        'snr_db': snr_db,
        'cosine_sim': cosine_sim,
        'max_abs_error': max_abs_error
    }

class ProductQuantization:
    """
    Split 16D blocks into 8 sub-vectors of 2D.
    Each sub-vector: K=256 (8 bits)
    Total: 8 bits / 16 weights × 16 = 1.00 bpp
    Codebook: 8 × (256 × 2 × FP16) = 8 KB
    """
    def __init__(self, n_subvectors=8, k_per_subvec=256):
        self.n_subvec = n_subvectors
        self.k_per_subvec = k_per_subvec
        self.block_size = 16  # Total dimension
        self.subvec_dim = self.block_size // self.n_subvec  # 2D per sub-vector
        
    def run(self, W):
        d_out, d_in = W.shape
        bs = self.block_size
        
        # Pad
        pad = (bs - (W.size % bs)) % bs
        W_flat = W.flatten()
        W_pad = np.pad(W_flat, (0, pad))
        
        # Reshape to (N_blocks, 16)
        blocks = W_pad.reshape(-1, bs)
        
        # Split into sub-vectors: (N_blocks, 8, 2)
        subvectors = blocks.reshape(-1, self.n_subvec, self.subvec_dim)
        
        # Quantize each sub-vector independently
        codebooks = []
        assignments_all = []
        
        for s in range(self.n_subvec):
            # Extract all blocks for this sub-vector position
            subvec_data = subvectors[:, s, :]  # (N_blocks, 2)
            
            # K-Means
            indices = np.random.choice(len(subvec_data), min(self.k_per_subvec, len(subvec_data)), replace=False)
            centroids = subvec_data[indices].copy()
            
            # Fast K-Means (3 iters)
            for _ in range(3):
                dists = np.linalg.norm(subvec_data[:, None, :] - centroids[None, :, :], axis=2)
                assigns = np.argmin(dists, axis=1)
                
                new_centroids = np.zeros_like(centroids)
                for k in range(self.k_per_subvec):
                    mask = (assigns == k)
                    if np.sum(mask) > 0:
                        new_centroids[k] = np.mean(subvec_data[mask], axis=0)
                    else:
                        new_centroids[k] = centroids[k]
                centroids = new_centroids
            
            # Final assignment
            dists = np.linalg.norm(subvec_data[:, None, :] - centroids[None, :, :], axis=2)
            assigns = np.argmin(dists, axis=1)
            
            codebooks.append(centroids)
            assignments_all.append(assigns)
        
        # Reconstruct
        recon_subvectors = np.zeros_like(subvectors)
        for s in range(self.n_subvec):
            recon_subvectors[:, s, :] = codebooks[s][assignments_all[s]]
        
        # Reshape back
        recon_blocks = recon_subvectors.reshape(-1, bs)
        W_recon = recon_blocks.flatten()[:W.size].reshape(d_out, d_in)
        
        return W_recon

class LearnedRotationVQ:
    """
    Use PCA to find optimal rotation, then VQ.
    """
    def __init__(self, n_codes=65536, block_size=16):
        self.n_codes = n_codes
        self.block_size = block_size
        
    def run(self, W):
        d_out, d_in = W.shape
        bs = self.block_size
        
        # Pad
        pad = (bs - (W.size % bs)) % bs
        W_flat = W.flatten()
        W_pad = np.pad(W_flat, (0, pad))
        
        blocks = W_pad.reshape(-1, bs)
        
        # 1. PCA for rotation
        mean = np.mean(blocks, axis=0)
        blocks_centered = blocks - mean
        
        cov = np.dot(blocks_centered.T, blocks_centered) / len(blocks_centered)
        evals, evecs = np.linalg.eigh(cov)
        
        # Sort by descending eigenvalue
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        
        # Rotate
        rotated_blocks = np.dot(blocks_centered, evecs)
        
        # 2. VQ on rotated blocks
        # Sample for training
        n_train = min(len(rotated_blocks), 50000)
        train_indices = np.random.choice(len(rotated_blocks), n_train, replace=False)
        train_blocks = rotated_blocks[train_indices]
        
        # Init
        if self.n_codes > n_train:
            indices = np.random.choice(n_train, self.n_codes, replace=True)
        else:
            indices = np.random.choice(n_train, self.n_codes, replace=False)
        centroids = train_blocks[indices].copy()
        
        # Fast K-Means (3 iters on training data)
        for _ in range(3):
            c_norms = (centroids**2).sum(axis=1)
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(self.n_codes)
            
            batch_size = 4096
            for i in range(0, len(train_blocks), batch_size):
                batch = train_blocks[i:i+batch_size]
                dots = np.dot(batch, centroids.T)
                dists = -2 * dots + c_norms[None, :]
                assigns = np.argmin(dists, axis=1)
                
                for j, a in enumerate(assigns):
                    new_centroids[a] += batch[j]
                    counts[a] += 1
            
            mask = counts > 0
            centroids[mask] = new_centroids[mask] / counts[mask][:, None]
        
        # Quantize all blocks
        recon_rotated = np.zeros_like(rotated_blocks)
        c_norms = (centroids**2).sum(axis=1)
        
        batch_size = 4096
        for i in range(0, len(rotated_blocks), batch_size):
            batch = rotated_blocks[i:i+batch_size]
            dots = np.dot(batch, centroids.T)
            dists = -2 * dots + c_norms[None, :]
            assigns = np.argmin(dists, axis=1)
            recon_rotated[i:i+batch_size] = centroids[assigns]
        
        # 3. Inverse rotate
        recon_centered = np.dot(recon_rotated, evecs.T)
        recon_blocks = recon_centered + mean
        
        W_recon = recon_blocks.flatten()[:W.size].reshape(d_out, d_in)
        return W_recon

def run_experiments():
    weights = get_gpt2_weights()
    weights = weights[:5]  # Test on first 5 matrices
    
    print("\n" + "="*80)
    print("EXPERIMENT 21: NOVEL IDEAS V24 (Multi-Metric Push to 1.000)")
    print("="*80)
    
    results = []
    
    # 1. Product Quantization
    print("\n1. Product Quantization (8x2D, K=256 each)...")
    metrics_list = []
    for W in tqdm(weights):
        pq = ProductQuantization(n_subvectors=8, k_per_subvec=256)
        W_rec = pq.run(W)
        metrics = compute_metrics(W, W_rec)
        metrics_list.append(metrics)
    
    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    print(f"  Correlation: {avg_metrics['correlation']:.4f}")
    print(f"  NMSE: {avg_metrics['nmse']:.4f}")
    print(f"  SNR: {avg_metrics['snr_db']:.2f} dB")
    print(f"  Cosine Sim: {avg_metrics['cosine_sim']:.4f}")
    results.append(f"ProductQuantization | {avg_metrics['correlation']:.4f} corr | {avg_metrics['nmse']:.4f} NMSE | {avg_metrics['snr_db']:.2f} dB SNR")
    
    # 2. Learned Rotation VQ
    print("\n2. Learned Rotation VQ (PCA + K=65536)...")
    metrics_list = []
    for W in tqdm(weights):
        lrvq = LearnedRotationVQ(n_codes=65536, block_size=16)
        W_rec = lrvq.run(W)
        metrics = compute_metrics(W, W_rec)
        metrics_list.append(metrics)
    
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    print(f"  Correlation: {avg_metrics['correlation']:.4f}")
    print(f"  NMSE: {avg_metrics['nmse']:.4f}")
    print(f"  SNR: {avg_metrics['snr_db']:.2f} dB")
    print(f"  Cosine Sim: {avg_metrics['cosine_sim']:.4f}")
    results.append(f"LearnedRotationVQ | {avg_metrics['correlation']:.4f} corr | {avg_metrics['nmse']:.4f} NMSE | {avg_metrics['snr_db']:.2f} dB SNR")
    
    # Write results
    with open("results_v24.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    run_experiments()
