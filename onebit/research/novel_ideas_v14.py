"""
Novel Ideas V14: Extreme 1.00 bpp Optimization

Mission: Push the boundary at EXACTLY 1.00 bpp.
Current best at 1.00 bpp: Entropy+LowRank (r=2) @ 0.8045 (-8.8% vs ternary).

Experiments:
1. Sparse Ternary (Entropy-Constrained)
2. Neural Context Entropy Coding
3. Block-wise Magnitude VQ
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List

# =============================================================================
# HELPER: Load Real GPT-2 Weights (Same as V13)
# =============================================================================

def try_load_gpt2_weights():
    """Try to load GPT-2 weights, fallback to realistic synthetic if unavailable."""
    try:
        from transformers import GPT2LMHeadModel
        print("Loading real GPT-2-small weights...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        # Extract a representative weight matrix (e.g., first MLP layer)
        W_real = model.transformer.h[0].mlp.c_fc.weight.detach().cpu().numpy().T
        print(f"Loaded real GPT-2 weight: shape {W_real.shape}")
        return W_real, True
    except Exception as e:
        print(f"Could not load GPT-2 weights: {e}")
        print("Using realistic synthetic weights instead...")
        d_out, d_in = 768, 3072
        np.random.seed(42)
        W_synthetic = np.random.standard_t(df=3, size=(d_out, d_in)).astype(np.float32) * 0.02
        U = np.random.randn(d_out, 20)
        Vt = np.random.randn(20, d_in)
        W_synthetic += 0.1 * (U @ Vt)
        W_synthetic *= 0.02 / np.std(W_synthetic)
        print(f"Created realistic synthetic weight: shape {W_synthetic.shape}")
        return W_synthetic.astype(np.float32), False

# =============================================================================
# 1. SPARSE TERNARY (Entropy-Constrained)
# =============================================================================

class SparseTernary:
    """
    Ternary quantization with sparsity tuned for exactly 1.00 bpp entropy.
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.W_quant = None
        self.entropy = 0.0
        
    def _compute_ternary_entropy(self, W):
        """Compute entropy of ternary stream (-1, 0, +1)."""
        n_total = W.size
        n_zero = np.sum(W == 0)
        n_pos = np.sum(W > 0)
        n_neg = np.sum(W < 0)
        
        probs = np.array([n_zero, n_pos, n_neg]) / n_total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def train(self, W_target: np.ndarray):
        """Find threshold that yields ~1.00 bpp entropy."""
        # Binary search for threshold
        M = np.abs(W_target)
        low = 0.0
        high = np.max(M)
        best_thresh = 0.0
        best_entropy = 0.0
        
        for _ in range(20):
            mid = (low + high) / 2
            # Quantize
            W_temp = np.zeros_like(W_target)
            mask = M > mid
            W_temp[mask] = np.sign(W_target)[mask]
            
            h = self._compute_ternary_entropy(W_temp)
            
            if h > 1.00:
                low = mid # Need more sparsity (higher threshold)
            else:
                high = mid # Can afford less sparsity
                best_thresh = mid
                best_entropy = h
        
        # Final quantization
        self.W_quant = np.zeros_like(W_target)
        mask = M > best_thresh
        
        # Use mean of kept values as scale
        if np.sum(mask) > 0:
            scale = np.mean(M[mask])
            self.W_quant[mask] = np.sign(W_target)[mask] * scale
            
        self.entropy = best_entropy
        
    def get_weights(self) -> np.ndarray:
        return self.W_quant
        
    def effective_bpp(self) -> float:
        # Entropy coding + negligible scale overhead
        return self.entropy


# =============================================================================
# 2. NEURAL CONTEXT ENTROPY CODING
# =============================================================================

class NeuralContextEntropy:
    """
    Use MLP to predict signs from context -> lower entropy.
    Target: 0.40 bpp signs + 0.60 bpp magnitude = 1.00 bpp.
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 2):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank # For magnitude
        self.S = None
        self.U = None
        self.Vt = None
        self.sign_entropy = 1.0
        
    def _train_context_model(self, S):
        """Train small MLP to predict sign from 3x3 context."""
        # Prepare data: (N, 8) context -> (N, 1) target
        # Pad S with zeros
        S_pad = np.pad(S, ((1,1), (1,1)), mode='constant')
        H, W = S.shape
        
        contexts = []
        targets = []
        
        # Sample subset for speed
        n_samples = 50000
        rows = np.random.randint(0, H, n_samples)
        cols = np.random.randint(0, W, n_samples)
        
        for r, c in zip(rows, cols):
            # 3x3 window around r+1, c+1
            ctx = S_pad[r:r+3, c:c+3].flatten()
            # Remove center (target)
            ctx = np.delete(ctx, 4)
            contexts.append(ctx)
            targets.append(1 if S[r, c] > 0 else 0)
            
        X = torch.FloatTensor(np.array(contexts))
        y = torch.FloatTensor(np.array(targets)).unsqueeze(1)
        
        # Simple MLP
        model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Train
        for _ in range(500):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
        # Estimate entropy
        with torch.no_grad():
            probs = model(X).numpy()
            # Clip for stability
            probs = np.clip(probs, 1e-6, 1-1e-6)
            # Cross entropy is the bit rate
            bce = -(y.numpy() * np.log2(probs) + (1-y.numpy()) * np.log2(1-probs))
            return np.mean(bce)

    def train(self, W_target: np.ndarray):
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Estimate neural entropy
        self.sign_entropy = self._train_context_model(self.S)
        
        # Magnitude (Low Rank)
        M = np.abs(W_target)
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        self.U = u[:, :self.rank] * s[:self.rank]
        self.Vt = vt[:self.rank, :]
        
    def get_weights(self) -> np.ndarray:
        M_recon = self.U @ self.Vt
        return self.S * M_recon
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = self.sign_entropy * n_weights
        mag_bits = (self.d_out * self.rank + self.rank * self.d_in) * 32
        return (sign_bits + mag_bits) / n_weights


# =============================================================================
# 3. BLOCK MAGNITUDE VQ
# =============================================================================

class BlockMagnitudeVQ:
    """
    VQ on 4x4 magnitude blocks.
    """
    def __init__(self, d_in: int, d_out: int, n_codes: int = 16):
        self.d_in = d_in
        self.d_out = d_out
        self.n_codes = n_codes
        self.block_size = 4
        self.S = None
        self.codebook = None
        self.assignments = None
        self.sign_entropy = 0.5 # Assume V10's simple context entropy
        
    def _simple_kmeans(self, X, k, max_iter=20):
        indices = np.random.choice(len(X), k, replace=False)
        centroids = X[indices].copy()
        
        for _ in range(max_iter):
            distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            assignments = np.argmin(distances, axis=1)
            new_centroids = np.array([X[assignments == i].mean(axis=0) if np.sum(assignments == i) > 0 
                                      else centroids[i] for i in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return centroids, assignments

    def train(self, W_target: np.ndarray):
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        M = np.abs(W_target)
        bs = self.block_size
        
        # Extract blocks
        blocks = []
        H, W = M.shape
        # Pad if needed
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)), mode='constant')
        
        for i in range(0, H, bs):
            for j in range(0, W, bs):
                block = M_pad[i:i+bs, j:j+bs].flatten()
                blocks.append(block)
                
        blocks = np.array(blocks)
        
        # VQ
        self.codebook, self.assignments = self._simple_kmeans(blocks, self.n_codes)
        
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
        vq_bits = n_blocks * np.log2(self.n_codes)
        codebook_bits = self.n_codes * (self.block_size ** 2) * 32
        
        return (sign_bits + vq_bits + codebook_bits) / n_weights


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V14: EXTREME 1.00 BPP OPTIMIZATION")
    print("="*80)
    
    # Load weights
    W_true, is_real = try_load_gpt2_weights()
    d_out, d_in = W_true.shape
    
    X_test = np.random.randn(1000, d_in).astype(np.float32) * 0.1
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
    
    print(f"\nBinary Baseline: {corr_bin:.4f} @ 1.00 bpp")
    print(f"Ternary Baseline: {corr_tern:.4f} @ 1.58 bpp")
    print("-" * 40)
    
    # 1. Sparse Ternary
    print("\nRunning Sparse Ternary (Entropy-Constrained)...")
    st = SparseTernary(d_in, d_out)
    st.train(W_true)
    W_st = st.get_weights()
    corr_st = np.corrcoef((X_test @ W_st.T).flatten(), Y_test.flatten())[0,1]
    bpp_st = st.effective_bpp()
    results['Sparse Ternary'] = {'corr': corr_st, 'bpp': bpp_st}
    print(f"Result: {corr_st:.4f} @ {bpp_st:.2f} bpp")
    
    # 2. Neural Context Entropy
    print("\nRunning Neural Context Entropy...")
    for rank in [1, 2, 4]:
        nce = NeuralContextEntropy(d_in, d_out, rank=rank)
        nce.train(W_true)
        W_nce = nce.get_weights()
        corr_nce = np.corrcoef((X_test @ W_nce.T).flatten(), Y_test.flatten())[0,1]
        bpp_nce = nce.effective_bpp()
        results[f'NeuralContext+LR (r={rank})'] = {'corr': corr_nce, 'bpp': bpp_nce}
        print(f"Rank={rank}: {corr_nce:.4f} @ {bpp_nce:.2f} bpp (sign_H={nce.sign_entropy:.3f})")
    
    # 3. Block Magnitude VQ
    print("\nRunning Block Magnitude VQ...")
    for k in [16, 32, 64]:
        bmvq = BlockMagnitudeVQ(d_in, d_out, n_codes=k)
        bmvq.train(W_true)
        W_bmvq = bmvq.get_weights()
        corr_bmvq = np.corrcoef((X_test @ W_bmvq.T).flatten(), Y_test.flatten())[0,1]
        bpp_bmvq = bmvq.effective_bpp()
        results[f'BlockVQ (K={k})'] = {'corr': corr_bmvq, 'bpp': bpp_bmvq}
        print(f"K={k}: {corr_bmvq:.4f} @ {bpp_bmvq:.2f} bpp")

    # Summary
    with open("results_v14_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"SUMMARY - V14: EXTREME 1.00 BPP OPTIMIZATION\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<30} {'Corr':>8} {'BPP':>8} {'vs Tern':>10}\n")
        f.write("-" * 65 + "\n")
        
        for name, res in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (res['corr'] - corr_tern) / corr_tern * 100
            line = f"{name:<30} {res['corr']:>8.4f} {res['bpp']:>8.2f} {vs_tern:>+9.1f}%\n"
            print(line.strip())
            f.write(line)
            
    print("\n" + "="*80)
    print("Results written to results_v14_utf8.txt")

if __name__ == "__main__":
    run_experiments()
