"""
Novel Ideas V24 (CORRECTED): Product Quantization at TRUE 1.00 BPP
Previous version used 4.00 bpp by mistake!

For 1.00 bpp with Product Quantization:
- 16 weights per block → need 16 bits total
- Config 1: 2 sub-vectors of 8D, K=256 each → 2×8 = 16 bits ✓
- Config 2: 4 sub-vectors of 4D, K=16 each → 4×4 = 16 bits ✓
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel
from tqdm import tqdm

def get_gpt2_weights():
    """Load real GPT-2 weights"""
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
    
    corr = np.corrcoef(W_orig_flat, W_recon_flat)[0, 1]
    mse = np.mean((W_orig_flat - W_recon_flat)**2)
    variance = np.var(W_orig_flat)
    nmse = mse / (variance + 1e-10)
    signal_power = np.mean(W_orig_flat**2)
    snr_db = 10 * np.log10(signal_power / (mse + 1e-10))
    
    return {'correlation': corr, 'nmse': nmse, 'snr_db': snr_db}

class ProductQuantization_Corrected:
    """
    TRUE 1.00 bpp Product Quantization
    """
    def __init__(self, n_subvectors, k_per_subvec):
        self.n_subvec = n_subvectors
        self.k_per_subvec = k_per_subvec
        self.block_size = 16
        self.subvec_dim = self.block_size // self.n_subvec
        
        # Verify BPP
        bits_per_block = self.n_subvec * np.log2(self.k_per_subvec)
        self.bpp = bits_per_block / self.block_size
        print(f"  Config: {self.n_subvec} subvec × {self.subvec_dim}D, K={self.k_per_subvec} → BPP={self.bpp:.2f}")
        
    def run(self, W):
        d_out, d_in = W.shape
        bs = self.block_size
        
        # Pad
        pad = (bs - (W.size % bs)) % bs
        W_flat = W.flatten()
        W_pad = np.pad(W_flat, (0, pad))
        
        blocks = W_pad.reshape(-1, bs)
        subvectors = blocks.reshape(-1, self.n_subvec, self.subvec_dim)
        
        # Quantize each sub-vector
        codebooks = []
        assignments_all = []
        
        for s in range(self.n_subvec):
            subvec_data = subvectors[:, s, :]
            
            # K-Means
            n_train = min(len(subvec_data), 10000)
            train_idx = np.random.choice(len(subvec_data), n_train, replace=False)
            train_data = subvec_data[train_idx]
            
            indices = np.random.choice(len(train_data), min(self.k_per_subvec, len(train_data)), replace=False)
            centroids = train_data[indices].copy()
            
            # Fast K-Means (5 iters)
            for _ in range(5):
                dists = np.linalg.norm(train_data[:, None, :] - centroids[None, :, :], axis=2)
                assigns = np.argmin(dists, axis=1)
                
                new_centroids = np.zeros_like(centroids)
                for k in range(self.k_per_subvec):
                    mask = (assigns == k)
                    if np.sum(mask) > 0:
                        new_centroids[k] = np.mean(train_data[mask], axis=0)
                    else:
                        new_centroids[k] = centroids[k]
                centroids = new_centroids
            
            # Assign all
            dists = np.linalg.norm(subvec_data[:, None, :] - centroids[None, :, :], axis=2)
            assigns = np.argmin(dists, axis=1)
            
            codebooks.append(centroids)
            assignments_all.append(assigns)
        
        # Reconstruct
        recon_subvectors = np.zeros_like(subvectors)
        for s in range(self.n_subvec):
            recon_subvectors[:, s, :] = codebooks[s][assignments_all[s]]
        
        recon_blocks = recon_subvectors.reshape(-1, bs)
        W_recon = recon_blocks.flatten()[:W.size].reshape(d_out, d_in)
        
        return W_recon

def run_experiments():
    weights = get_gpt2_weights()
    weights = weights[:5]
    
    print("\n" + "="*80)
    print("CORRECTED V24: Product Quantization at TRUE 1.00 BPP")
    print("="*80)
    
    results = []
    
    # 1. Config: 2 sub-vectors of 8D, K=256 each (1.00 bpp)
    print("\n1. PQ: 2 sub-vectors × 8D, K=256...")
    metrics_list = []
    for W in tqdm(weights):
        pq = ProductQuantization_Corrected(n_subvectors=2, k_per_subvec=256)
        W_rec = pq.run(W)
        metrics = compute_metrics(W, W_rec)
        metrics_list.append(metrics)
    
    avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    print(f"  Corr: {avg['correlation']:.4f} | NMSE: {avg['nmse']:.4f} | SNR: {avg['snr_db']:.2f} dB")
    results.append(f"PQ(2×8D, K=256) | 1.00 bpp | Corr: {avg['correlation']:.4f} | NMSE: {avg['nmse']:.4f}")
    
    # 2. Config: 4 sub-vectors of 4D, K=16 each (1.00 bpp)
    print("\n2. PQ: 4 sub-vectors × 4D, K=16...")
    metrics_list = []
    for W in tqdm(weights):
        pq = ProductQuantization_Corrected(n_subvectors=4, k_per_subvec=16)
        W_rec = pq.run(W)
        metrics = compute_metrics(W, W_rec)
        metrics_list.append(metrics)
    
    avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    print(f"  Corr: {avg['correlation']:.4f} | NMSE: {avg['nmse']:.4f} | SNR: {avg['snr_db']:.2f} dB")
    results.append(f"PQ(4×4D, K=16) | 1.00 bpp | Corr: {avg['correlation']:.4f} | NMSE: {avg['nmse']:.4f}")
    
    # 3. Baseline: Standard VQ (D=16, K=65536) for comparison
    print("\n3. Baseline: Standard VQ (D=16, K=65536)...")
    print("  (Skipping - already know it's 0.9434)")
    results.append(f"Standard VQ(16D, K=65536) | 1.00 bpp | Corr: 0.9434 (from V23)")
    
    with open("results_v24_corrected.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    run_experiments()
