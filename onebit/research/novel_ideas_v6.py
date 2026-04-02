"""
Novel Ideas V6: Global Structure & Dynamic Inference

Experiments:
1. Global Magnitude Dictionary:
   - Learn a single codebook of magnitude vectors shared across ALL layers.
   - Hypothesis: Magnitude patterns are universal (e.g., attention heads look similar).

2. Stochastic Resonance Binary:
   - Inject noise during inference to "smooth" the binary quantization.
   - Y = Activation(X @ W_bin.T + noise)
   - Hypothesis: Noise can help represent values between -1 and 1.

3. Hypernetwork Scaling:
   - Predict layer-wise scale from input statistics.
   - scale = MLP(mean(X), var(X))
   - Hypothesis: Optimal scale depends on input distribution.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List

# =============================================================================
# 1. GLOBAL MAGNITUDE DICTIONARY
# =============================================================================

class GlobalMagnitudeDictionary:
    """
    Simulates sharing a magnitude codebook across multiple layers.
    """
    def __init__(self, d_in: int, d_out: int, n_layers: int = 12, n_codes: int = 256):
        self.d_in = d_in
        self.d_out = d_out
        self.n_layers = n_layers
        self.n_codes = n_codes
        
        # Global codebook: (n_codes, d_in) - assuming row-wise magnitude vectors
        self.codebook = None 
        
        # Per-layer indices: (n_layers, d_out)
        self.layer_indices = []
        self.layer_signs = []
        
    def train(self, layers_W: List[np.ndarray]):
        """Train global codebook on multiple layers."""
        # 1. Collect all magnitude vectors (rows)
        all_mags = []
        for W in layers_W:
            # S = sign(W)
            S = np.sign(W).astype(np.float32)
            S[S == 0] = 1.0
            self.layer_signs.append(S)
            
            # M = abs(W)
            # We approximate each row magnitude by a codebook vector
            # But wait, M is (d_out, d_in). A row is (d_in,).
            # If we share row patterns, we need d_in to be same.
            M = np.abs(W)
            all_mags.append(M)
            
        # Stack all rows: (n_layers * d_out, d_in)
        X_train = np.vstack(all_mags)
        
        # 2. Learn Codebook (K-Means)
        # Simple K-Means implementation
        n_samples = X_train.shape[0]
        indices = np.random.choice(n_samples, self.n_codes, replace=False)
        self.codebook = X_train[indices]
        
        for _ in range(10): # 10 iters
            # Assign
            # Chunked distance calculation to avoid OOM
            labels = np.zeros(n_samples, dtype=int)
            chunk_size = 1000
            for i in range(0, n_samples, chunk_size):
                chunk = X_train[i:i+chunk_size]
                dists = np.sum(chunk**2, axis=1, keepdims=True) + \
                        np.sum(self.codebook**2, axis=1) - \
                        2 * chunk @ self.codebook.T
                labels[i:i+chunk_size] = np.argmin(dists, axis=1)
            
            # Update
            new_codebook = np.zeros_like(self.codebook)
            counts = np.zeros(self.n_codes)
            for k in range(self.n_codes):
                mask = labels == k
                if np.any(mask):
                    new_codebook[k] = np.mean(X_train[mask], axis=0)
                else:
                    new_codebook[k] = X_train[np.random.randint(n_samples)]
            
            if np.allclose(self.codebook, new_codebook):
                break
            self.codebook = new_codebook
            
        # 3. Assign indices for each layer
        offset = 0
        for i in range(self.n_layers):
            n_rows = layers_W[i].shape[0]
            # We need to re-assign because we only did it for the whole batch
            # But we can just re-compute for this layer
            M = np.abs(layers_W[i])
            dists = np.sum(M**2, axis=1, keepdims=True) + \
                    np.sum(self.codebook**2, axis=1) - \
                    2 * M @ self.codebook.T
            idxs = np.argmin(dists, axis=1)
            self.layer_indices.append(idxs)
            
    def get_layer_weight(self, layer_idx: int) -> np.ndarray:
        idxs = self.layer_indices[layer_idx]
        M = self.codebook[idxs]
        return self.layer_signs[layer_idx] * M
        
    def effective_bpp(self) -> float:
        # Global cost amortized over n_layers
        codebook_bits = self.n_codes * self.d_in * 32
        
        # Per layer cost
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        index_bits = self.d_out * np.log2(self.n_codes)
        
        total_bits_per_layer = sign_bits + index_bits + (codebook_bits / self.n_layers)
        return total_bits_per_layer / n_weights


# =============================================================================
# 2. STOCHASTIC RESONANCE BINARY
# =============================================================================

class StochasticResonanceBinary:
    """
    Inject noise into pre-activation to improve binary representation.
    """
    def __init__(self, d_in: int, d_out: int, noise_std: float = 0.1):
        self.d_in = d_in
        self.d_out = d_out
        self.noise_std = noise_std
        
        self.W_bin = None
        self.scale = 1.0
        
    def train(self, W_target: np.ndarray):
        # Standard binary
        self.W_bin = np.sign(W_target).astype(np.float32)
        self.W_bin[self.W_bin == 0] = 1.0
        self.scale = np.mean(np.abs(W_target))
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        # Y = (X @ W.T + noise) * scale
        # Noise is added to the DOT PRODUCT result
        pre_act = X @ self.W_bin.T
        noise = np.random.normal(0, self.noise_std, pre_act.shape).astype(np.float32)
        return (pre_act + noise) * self.scale
        
    def effective_bpp(self) -> float:
        return 1.0 + 32/(self.d_in*self.d_out) # Just binary + scale


# =============================================================================
# 3. HYPERNETWORK SCALING
# =============================================================================

class HypernetworkScaling:
    """
    Scale depends on input statistics.
    scale = MLP(mean(X), std(X))
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        
        self.W_bin = None
        # Simple MLP: 2 inputs (mean, std) -> 1 output (scale)
        # We'll implement it in numpy for simplicity
        self.mlp_w1 = np.random.randn(2, 16).astype(np.float32) * 0.1
        self.mlp_b1 = np.zeros(16, dtype=np.float32)
        self.mlp_w2 = np.random.randn(16, 1).astype(np.float32) * 0.1
        self.mlp_b2 = np.array([1.0], dtype=np.float32) # Init bias to 1.0
        
    def train(self, W_target: np.ndarray, X_train: np.ndarray):
        # 1. Binary Weights
        self.W_bin = np.sign(W_target).astype(np.float32)
        self.W_bin[self.W_bin == 0] = 1.0
        
        # 2. Train MLP to predict optimal scale per batch/sample
        # Target scale for a sample x: s* = (x @ W_true.T) / (x @ W_bin.T)
        # This is hard to optimize directly.
        # Let's just optimize MLP to minimize MSE(Y_pred, Y_true)
        
        Y_true = X_train @ W_target.T
        Y_bin = X_train @ self.W_bin.T
        
        # Input features: mean and std of X (per sample)
        means = np.mean(X_train, axis=1, keepdims=True)
        stds = np.std(X_train, axis=1, keepdims=True)
        Feats = np.hstack([means, stds]) # (N, 2)
        
        # Simple Gradient Descent
        lr = 0.001 # Reduced LR
        for _ in range(100):
            # Forward MLP
            h = np.maximum(0, Feats @ self.mlp_w1 + self.mlp_b1)
            scales = h @ self.mlp_w2 + self.mlp_b2 # (N, 1)
            
            # Pred
            Y_pred = Y_bin * scales
            
            # Loss
            diff = Y_pred - Y_true
            loss = np.mean(diff**2)
            
            # Backward
            # dL/dScale = 2 * diff * Y_bin
            # dL/dScale_i = sum_j (2 * diff_ij * Y_bin_ij) / d_out
            dScale = np.mean(2 * diff * Y_bin, axis=1, keepdims=True)
            
            # Clip gradients
            dScale = np.clip(dScale, -1.0, 1.0)
            
            # MLP Backprop
            # scale = h @ w2 + b2
            d_w2 = h.T @ dScale
            d_b2 = np.sum(dScale, axis=0)
            d_h = dScale @ self.mlp_w2.T
            
            # h = ReLU(z)
            d_z = d_h * (h > 0)
            d_w1 = Feats.T @ d_z
            d_b1 = np.sum(d_z, axis=0)
            
            # Update
            self.mlp_w1 -= lr * d_w1 / len(X_train)
            self.mlp_b1 -= lr * d_b1 / len(X_train)
            self.mlp_w2 -= lr * d_w2 / len(X_train)
            self.mlp_b2 -= lr * d_b2 / len(X_train)
            
    def forward(self, X: np.ndarray) -> np.ndarray:
        means = np.mean(X, axis=1, keepdims=True)
        stds = np.std(X, axis=1, keepdims=True)
        Feats = np.hstack([means, stds])
        
        h = np.maximum(0, Feats @ self.mlp_w1 + self.mlp_b1)
        scales = h @ self.mlp_w2 + self.mlp_b2
        
        return (X @ self.W_bin.T) * scales
        
    def effective_bpp(self) -> float:
        # MLP params are negligible
        return 1.0


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V6: Global & Dynamic")
    print("="*80)
    
    # Setup Data
    d = 256
    n_layers = 10
    np.random.seed(42)
    
    # Generate multiple "layers" with similar structure
    layers_W = []
    for _ in range(n_layers):
        U = np.random.randn(d, d)
        U, _ = np.linalg.qr(U)
        Vt = np.random.randn(d, d)
        Vt, _ = np.linalg.qr(Vt)
        S = np.exp(-np.linspace(0, 5, d))
        W = U @ np.diag(S) @ Vt
        layers_W.append(W.astype(np.float32))
        
    # Test Data (using first layer for single-layer tests)
    W_true = layers_W[0]
    X_train = np.random.randn(5000, d).astype(np.float32)
    Y_train = X_train @ W_true.T
    X_test = np.random.randn(1000, d).astype(np.float32)
    Y_test = X_test @ W_true.T
    
    results = {}
    
    # Baselines
    S_bin = np.sign(W_true)
    S_bin[S_bin==0] = 1
    scale_bin = np.mean(np.abs(W_true))
    W_bin = S_bin * scale_bin
    corr_bin = np.corrcoef((X_test @ W_bin.T).flatten(), Y_test.flatten())[0,1]
    results['Binary Baseline'] = {'corr': corr_bin, 'bpp': 1.0}
    
    thresh = np.percentile(np.abs(W_true), 30)
    W_tern = S_bin * (np.abs(W_true) > thresh)
    scale_tern = np.mean(np.abs(W_true[np.abs(W_true) > thresh]))
    W_tern = W_tern * scale_tern
    corr_tern = np.corrcoef((X_test @ W_tern.T).flatten(), Y_test.flatten())[0,1]
    results['Ternary Baseline'] = {'corr': corr_tern, 'bpp': 1.58}
    
    print(f"Binary Baseline: {corr_bin:.4f} @ 1.00 bpp")
    print(f"Ternary Baseline: {corr_tern:.4f} @ 1.58 bpp")
    print("-" * 40)
    
    # 1. Global Magnitude Dictionary
    print("\nRunning Global Magnitude Dictionary...")
    for n_codes in [16, 64, 256]:
        gmd = GlobalMagnitudeDictionary(d, d, n_layers=n_layers, n_codes=n_codes)
        gmd.train(layers_W)
        
        # Evaluate on first layer
        W_gmd = gmd.get_layer_weight(0)
        corr_gmd = np.corrcoef((X_test @ W_gmd.T).flatten(), Y_test.flatten())[0,1]
        bpp_gmd = gmd.effective_bpp()
        results[f'Global-Mag (K={n_codes})'] = {'corr': corr_gmd, 'bpp': bpp_gmd}
        print(f"K={n_codes}: {corr_gmd:.4f} @ {bpp_gmd:.2f} bpp")
        
    # 2. Stochastic Resonance
    print("\nRunning Stochastic Resonance...")
    for noise in [0.01, 0.05, 0.1, 0.2]:
        srb = StochasticResonanceBinary(d, d, noise_std=noise)
        srb.train(W_true)
        Y_pred = srb.forward(X_test)
        corr_srb = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0,1]
        bpp_srb = srb.effective_bpp()
        results[f'Stoch-Res (std={noise})'] = {'corr': corr_srb, 'bpp': bpp_srb}
        print(f"std={noise}: {corr_srb:.4f} @ {bpp_srb:.2f} bpp")
        
    # 3. Hypernetwork Scaling
    print("\nRunning Hypernetwork Scaling...")
    hns = HypernetworkScaling(d, d)
    hns.train(W_true, X_train)
    Y_pred = hns.forward(X_test)
    corr_hns = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0,1]
    bpp_hns = hns.effective_bpp()
    results['Hypernet-Scale'] = {'corr': corr_hns, 'bpp': bpp_hns}
    print(f"Result: {corr_hns:.4f} @ {bpp_hns:.2f} bpp")
    
    # Summary
    with open("results_v6_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY - NOVEL IDEAS V6\n")
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
