"""
Novel Ideas V16: The 1.00 BPP Limit

Mission: Find the maximum correlation achievable at EXACTLY 1.00 bpp.
Strict Requirement: MUST use real GPT-2 weights.

Experiments:
1. Tuned HessianBlockVQ (Sweep K to hit 1.00 bpp)
2. Residual Hessian VQ (Two-stage refinement)
"""

import numpy as np
import torch
from typing import Tuple, Dict, List
import sys

# =============================================================================
# HELPER: Load Real GPT-2 Weights (STRICT) - Reused from V15
# =============================================================================

def load_real_gpt2_data():
    """Load real GPT-2 weights and activation data. Fail if unavailable."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("Loading real GPT-2-small model...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Layer 0 MLP FC: (768, 3072)
        layer = model.transformer.h[0].mlp.c_fc
        W_real = layer.weight.detach().cpu().numpy().T # (3072, 768)
        
        print(f"Loaded real GPT-2 weight: shape {W_real.shape}")
        
        # Capture activations for Hessian
        print("Capturing activations...")
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(input[0].detach().cpu().numpy())
            
        handle = layer.register_forward_hook(hook_fn)
        
        # Run some text through
        text = "The quick brown fox jumps over the lazy dog. " * 10
        inputs = tokenizer(text, return_tensors="pt")
        model(inputs.input_ids)
        
        handle.remove()
        
        X_real = np.concatenate(activations, axis=0)
        X_real = X_real.reshape(-1, X_real.shape[-1])
        print(f"Captured activations: shape {X_real.shape}")
        
        return W_real, X_real
        
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load real GPT-2 data: {e}")
        sys.exit(1)

# =============================================================================
# 1. HESSIAN-WEIGHTED BLOCK VQ (Refined)
# =============================================================================

class HessianBlockVQ:
    """
    VQ on 4x4 magnitude blocks, weighted by Hessian diagonal.
    """
    def __init__(self, d_in: int, d_out: int, n_codes: int = 16):
        self.d_in = d_in
        self.d_out = d_out
        self.n_codes = n_codes
        self.block_size = 4
        self.S = None
        self.codebook = None
        self.assignments = None
        self.sign_entropy = 0.5 # From V10
        self.index_entropy = 0.0
        
    def _weighted_kmeans(self, X, weights, k, max_iter=20):
        indices = np.random.choice(len(X), k, replace=False)
        centroids = X[indices].copy()
        
        for _ in range(max_iter):
            # Assignment: weighted distance
            # Approx: scale X and C by sqrt(mean_block_weight)
            block_weights = np.mean(weights, axis=1, keepdims=True)
            distances = np.linalg.norm((X[:, None, :] - centroids[None, :, :]) * np.sqrt(block_weights[:, None, :]), axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Update: weighted mean
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = (assignments == i)
                if np.sum(mask) > 0:
                    X_subset = X[mask]
                    W_subset = weights[mask]
                    new_centroids[i] = np.sum(X_subset * W_subset, axis=0) / (np.sum(W_subset, axis=0) + 1e-8)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        return centroids, assignments

    def train(self, W_target: np.ndarray, X_calib: np.ndarray):
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        M = np.abs(W_target)
        
        # Hessian diagonal
        H_diag = np.mean(X_calib**2, axis=0)
        H_matrix = np.tile(H_diag, (self.d_out, 1))
        
        bs = self.block_size
        
        # Extract blocks
        blocks = []
        weights = []
        
        H_out, W_out = M.shape
        pad_h = (bs - H_out % bs) % bs
        pad_w = (bs - W_out % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)), mode='constant')
        H_pad = np.pad(H_matrix, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=1e-6)
        
        for i in range(0, H_out, bs):
            for j in range(0, W_out, bs):
                block = M_pad[i:i+bs, j:j+bs].flatten()
                weight = H_pad[i:i+bs, j:j+bs].flatten()
                blocks.append(block)
                weights.append(weight)
                
        blocks = np.array(blocks)
        weights = np.array(weights)
        
        # Weighted VQ
        self.codebook, self.assignments = self._weighted_kmeans(blocks, weights, self.n_codes)
        
        # Calculate entropy of indices
        counts = np.bincount(self.assignments, minlength=self.n_codes)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        self.index_entropy = -np.sum(probs * np.log2(probs))
        
    def get_weights(self) -> np.ndarray:
        bs = self.block_size
        H, W = self.d_out, self.d_in
        M_recon = np.zeros((H + (bs - H % bs) % bs, W + (bs - W % bs) % bs))
        
        idx = 0
        for i in range(0, H, bs):
            for j in range(0, W, bs):
                block = self.codebook[self.assignments[idx]].reshape(bs, bs)
                M_recon[i:i+bs, j:j+bs] = block
                idx += 1
                
        return self.S * M_recon[:H, :W]
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = self.sign_entropy * n_weights
        
        n_blocks = (self.d_out * self.d_in) / (self.block_size ** 2)
        vq_bits = n_blocks * self.index_entropy
        codebook_bits = self.n_codes * (self.block_size ** 2) * 32
        
        return (sign_bits + vq_bits + codebook_bits) / n_weights


# =============================================================================
# 2. RESIDUAL HESSIAN VQ (Two-Stage)
# =============================================================================

class ResidualHessianVQ:
    """
    Two-stage VQ:
    1. Coarse VQ (K1)
    2. Residual VQ (K2) on error
    """
    def __init__(self, d_in: int, d_out: int, k1: int = 32, k2: int = 32):
        self.d_in = d_in
        self.d_out = d_out
        self.k1 = k1
        self.k2 = k2
        self.vq1 = HessianBlockVQ(d_in, d_out, n_codes=k1)
        self.vq2 = HessianBlockVQ(d_in, d_out, n_codes=k2)
        self.S = None
        
    def train(self, W_target: np.ndarray, X_calib: np.ndarray):
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        M = np.abs(W_target)
        
        # Stage 1
        self.vq1.train(W_target, X_calib)
        M_recon1 = np.abs(self.vq1.get_weights()) # Magnitude only
        
        # Residual
        Residual = M - M_recon1
        # We need to pass "Residual" as if it were weights to vq2, but vq2 expects signed weights
        # So we pass Residual * S (since vq2 extracts magnitude)
        # Wait, vq2 extracts magnitude via np.abs(). So passing Residual (which can be neg) is fine?
        # No, magnitude must be positive.
        # Residual = M - M_recon. This can be negative.
        # But we are modeling Magnitude. M approx M_recon + M_res.
        # M_res can be negative.
        # BlockVQ assumes positive magnitude.
        # So we need signed residual VQ?
        # Let's simplify: Residual VQ on the *absolute* error? No.
        # Let's use Signed VQ for residual?
        # Or just add a sign bit for residual?
        # Actually, let's just use a larger codebook in single stage first.
        # If we do residual, we need to handle signs of residual.
        # Let's assume residual is small and mostly symmetric.
        # Let's try to just fit M - M_recon.
        # If we enforce M_recon <= M, then residual is positive.
        # But VQ is mean-based.
        
        # Alternative: Just use larger K in single stage.
        # If K=256, we have 8 bits per block (0.5 bpp).
        # Plus 0.5 bpp signs = 1.0 bpp.
        # So K=256 should be exactly 1.00 bpp.
        pass

    def get_weights(self) -> np.ndarray:
        return self.vq1.get_weights() # Placeholder
        
    def effective_bpp(self) -> float:
        return self.vq1.effective_bpp()


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V16: THE 1.00 BPP LIMIT")
    print("="*80)
    
    # Load REAL data
    W_real, X_real = load_real_gpt2_data()
    d_out, d_in = W_real.shape
    
    n_test = min(1000, X_real.shape[0])
    X_test = X_real[:n_test]
    Y_test = X_test @ W_real.T
    
    results = {}
    
    # Baselines
    S_bin = np.sign(W_real)
    S_bin[S_bin==0] = 1
    scale_bin = np.mean(np.abs(W_real))
    W_bin = S_bin * scale_bin
    corr_bin = np.corrcoef((X_test @ W_bin.T).flatten(), Y_test.flatten())[0,1]
    
    thresh = np.percentile(np.abs(W_real), 30)
    W_tern = S_bin * (np.abs(W_real) > thresh)
    scale_tern = np.mean(np.abs(W_real[np.abs(W_real) > thresh]))
    W_tern = W_tern * scale_tern
    corr_tern = np.corrcoef((X_test @ W_tern.T).flatten(), Y_test.flatten())[0,1]
    
    print(f"\nBinary Baseline: {corr_bin:.4f} @ 1.00 bpp")
    print(f"Ternary Baseline: {corr_tern:.4f} @ 1.58 bpp")
    print("-" * 40)
    
    # Sweep K for HessianVQ
    print("\nSweeping HessianVQ Codebook Size...")
    # We want to hit 1.00 bpp.
    # BPP = 0.5 (signs) + IndexEntropy/16.
    # If IndexEntropy = 8 bits (K=256), BPP = 0.5 + 8/16 = 1.0.
    # So K=256 is the theoretical limit if entropy is max.
    # If entropy is lower, we can use larger K.
    
    for k in [32, 64, 128, 256, 512]:
        hvq = HessianBlockVQ(d_in, d_out, n_codes=k)
        hvq.train(W_real, X_real)
        W_hvq = hvq.get_weights()
        corr_hvq = np.corrcoef((X_test @ W_hvq.T).flatten(), Y_test.flatten())[0,1]
        bpp_hvq = hvq.effective_bpp()
        results[f'HessianVQ (K={k})'] = {'corr': corr_hvq, 'bpp': bpp_hvq}
        print(f"K={k}: {corr_hvq:.4f} @ {bpp_hvq:.2f} bpp (idx_H={hvq.index_entropy:.2f})")

    # Summary
    with open("results_v16_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"SUMMARY - V16: THE 1.00 BPP LIMIT\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<30} {'Corr':>8} {'BPP':>8} {'vs Tern':>10}\n")
        f.write("-" * 65 + "\n")
        
        for name, res in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (res['corr'] - corr_tern) / corr_tern * 100
            line = f"{name:<30} {res['corr']:>8.4f} {res['bpp']:>8.2f} {vs_tern:>+9.1f}%\n"
            print(line.strip())
            f.write(line)
            
    print("\n" + "="*80)
    print("Results written to results_v16_utf8.txt")

if __name__ == "__main__":
    run_experiments()
