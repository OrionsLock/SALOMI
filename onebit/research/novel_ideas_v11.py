"""
Novel Ideas V11: Entropy-Coded Signs + Magnitude Recovery

Exploit V10's breakthrough: Signs compress to 0.50 bpp via entropy coding.
Now combine with magnitude recovery to beat ternary at sub-1.58 bpp!

Experiments:
1. Entropy + Low-Rank: 0.50 + 0.6 = 1.1 bpp
2. Entropy + VQ Magnitude: 0.50 + 0.5 = 1.0 bpp  
3. Entropy + Scale Codebook: 0.50 + 0.15 = 0.65 bpp
"""

import numpy as np
from typing import Tuple, Dict, List

# =============================================================================
# HELPER: Entropy-Coded Signs (from V10)
# =============================================================================

def compute_sign_entropy(S: np.ndarray) -> float:
    """Compute conditional entropy H(S_ij | S_i,j-1)."""
    transitions = {}
    for i in range(S.shape[0]):
        for j in range(1, S.shape[1]):
            prev = int(S[i, j-1])
            curr = int(S[i, j])
            key = (prev, curr)
            transitions[key] = transitions.get(key, 0) + 1
    
    total_prev = {-1: 0, 1: 0}
    for (prev, curr), count in transitions.items():
        total_prev[prev] += count
    
    entropy = 0.0
    for (prev, curr), count in transitions.items():
        if total_prev[prev] > 0:
            p = count / total_prev[prev]
            if p > 0:
                entropy -= p * np.log2(p) * count
    
    total_transitions = sum(transitions.values())
    if total_transitions > 0:
        entropy /= total_transitions
    
    return entropy


# =============================================================================
# 1. ENTROPY + LOW-RANK MAGNITUDE
# =============================================================================

class EntropyLowRank:
    """
    Entropy-coded signs + low-rank magnitude recovery.
    Target: 0.50 (signs) + 0.6 (rank-4) = 1.1 bpp
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 4):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        
        self.S = None
        self.U = None
        self.Vt = None
        self.sign_entropy = 0.0
        
    def train(self, W_target: np.ndarray):
        """Train on target weights."""
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Compute entropy
        self.sign_entropy = compute_sign_entropy(self.S)
        
        # Magnitude via low-rank SVD
        M = np.abs(W_target)
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        self.U = u[:, :self.rank] * s[:self.rank]
        self.Vt = vt[:self.rank, :]
        
    def get_weights(self) -> np.ndarray:
        M_recon = self.U @ self.Vt
        return self.S * M_recon
        
    def effective_bpp(self) -> float:
        """Entropy-coded signs + low-rank magnitude."""
        n_weights = self.d_out * self.d_in
        sign_bits = self.sign_entropy * n_weights
        magnitude_bits = (self.d_out * self.rank + self.rank * self.d_in) * 32
        return (sign_bits + magnitude_bits) / n_weights


# =============================================================================
# 2. ENTROPY + VECTOR QUANTIZED MAGNITUDE
# =============================================================================

class EntropyVQMagnitude:
    """
    Entropy-coded signs + VQ for magnitude.
    Target: 0.50 (signs) + 0.5 (VQ) = 1.0 bpp EXACTLY!
    """
    def __init__(self, d_in: int, d_out: int, n_codes: int = 32):
        self.d_in = d_in
        self.d_out = d_out
        self.n_codes = n_codes
        
        self.S = None
        self.codebook = None
        self.assignments = None
        self.sign_entropy = 0.0
        
    def _simple_kmeans(self, X, k, max_iter=20):
        """Simple K-means clustering."""
        # Initialize centroids randomly
        indices = np.random.choice(len(X), k, replace=False)
        centroids = X[indices].copy()
        
        for _ in range(max_iter):
            # Assign to nearest centroid
            distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([X[assignments == i].mean(axis=0) if np.sum(assignments == i) > 0 
                                      else centroids[i] for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return centroids, assignments
        
    def train(self, W_target: np.ndarray):
        """Train on target weights."""
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Compute entropy
        self.sign_entropy = compute_sign_entropy(self.S)
        
        # Magnitude: VQ on row-magnitude vectors
        M = np.abs(W_target)
        
        # Simple approach: VQ on per-row scale
        row_scales = np.mean(M, axis=1, keepdims=True)
        
        # Cluster scales
        self.codebook, self.assignments = self._simple_kmeans(
            row_scales, self.n_codes
        )
        
    def get_weights(self) -> np.ndarray:
        # Reconstruct magnitude from VQ codes
        M_recon = self.codebook[self.assignments]
        return self.S * M_recon
        
    def effective_bpp(self) -> float:
        """Entropy signs + VQ magnitude."""
        n_weights = self.d_out * self.d_in
        sign_bits = self.sign_entropy * n_weights
        
        # VQ: log2(n_codes) bits per row + codebook storage
        vq_bits = self.d_out * np.log2(self.n_codes)
        codebook_bits = self.n_codes * 32  # Amortized
        
        return (sign_bits + vq_bits + codebook_bits) / n_weights


# =============================================================================
# 3. ENTROPY + LEARNED SCALE CODEBOOK
# =============================================================================

class EntropyScaleCodebook:
    """
    Entropy-coded signs + learned per-row scale codebook.
    Target: 0.50 (signs) + 0.15 (scales) = 0.65 bpp ULTRA-COMPRESSED!
    """
    def __init__(self, d_in: int, d_out: int, n_scale_codes: int = 256):
        self.d_in = d_in
        self.d_out = d_out
        self.n_scale_codes = n_scale_codes
        
        self.S = None
        self.scale_codebook = None
        self.scale_assignments = None
        self.sign_entropy = 0.0
        
    def _simple_kmeans_1d(self, X, k, max_iter=20):
        """K-means for 1D data."""
        # Sort and divide into k bins
        X_sorted = np.sort(X.flatten())
        bin_size = len(X_sorted) // k
        centroids = np.array([X_sorted[i*bin_size:(i+1)*bin_size].mean() 
                              for i in range(k)])
        
        for _ in range(max_iter):
            # Assign to nearest
            distances = np.abs(X[:, None] - centroids[None, :])
            assignments = np.argmin(distances, axis=1)
            
            # Update
            new_centroids = np.array([X[assignments == i].mean() if np.sum(assignments == i) > 0 
                                      else centroids[i] for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return centroids, assignments
        
    def train(self, W_target: np.ndarray):
        """Train on target weights."""
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Compute entropy
        self.sign_entropy = compute_sign_entropy(self.S)
        
        # Per-row scales
        row_scales = np.mean(np.abs(W_target), axis=1)
        
        # Quantize scales to codebook
        self.scale_codebook, self.scale_assignments = self._simple_kmeans_1d(
            row_scales, self.n_scale_codes
        )
        
    def get_weights(self) -> np.ndarray:
        # Reconstruct with quantized scales
        scales = self.scale_codebook[self.scale_assignments]
        return self.S * scales[:, None]
        
    def effective_bpp(self) -> float:
        """Entropy signs + scale codebook."""
        n_weights = self.d_out * self.d_in
        sign_bits = self.sign_entropy * n_weights
        
        # Scale codes: log2(n_codes) per row + codebook
        scale_bits = self.d_out * np.log2(self.n_scale_codes)
        codebook_bits = self.n_scale_codes * 32  # Amortized
        
        return (sign_bits + scale_bits + codebook_bits) / n_weights


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V11: ENTROPY + MAGNITUDE (BEATING TERNARY!)")
    print("="*80)
    
    # Setup
    d = 256
    np.random.seed(42)
    U = np.random.randn(d, d)
    U, _ = np.linalg.qr(U)
    Vt = np.random.randn(d, d)
    Vt, _ = np.linalg.qr(Vt)
    S = np.exp(-np.linspace(0, 5, d))
    W_true = U @ np.diag(S) @ Vt
    W_true = W_true.astype(np.float32)
    
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
    
    # 1. Entropy + Low-Rank
    print("\nRunning Entropy + Low-Rank...")
    for rank in [2, 4, 8]:
        elr = EntropyLowRank(d, d, rank=rank)
        elr.train(W_true)
        W_elr = elr.get_weights()
        corr_elr = np.corrcoef((X_test @ W_elr.T).flatten(), Y_test.flatten())[0,1]
        bpp_elr = elr.effective_bpp()
        results[f'Entropy+LR (r={rank})'] = {'corr': corr_elr, 'bpp': bpp_elr}
        print(f"Rank={rank}: {corr_elr:.4f} @ {bpp_elr:.2f} bpp (entropy={elr.sign_entropy:.3f})")
    
    # 2. Entropy + VQ
    print("\nRunning Entropy + VQ Magnitude...")
    for n_codes in [16, 32, 64]:
        evq = EntropyVQMagnitude(d, d, n_codes=n_codes)
        evq.train(W_true)
        W_evq = evq.get_weights()
        corr_evq = np.corrcoef((X_test @ W_evq.T).flatten(), Y_test.flatten())[0,1]
        bpp_evq = evq.effective_bpp()
        results[f'Entropy+VQ (K={n_codes})'] = {'corr': corr_evq, 'bpp': bpp_evq}
        print(f"Codes={n_codes}: {corr_evq:.4f} @ {bpp_evq:.2f} bpp")
    
    # 3. Entropy + Scale Codebook
    print("\nRunning Entropy + Scale Codebook...")
    for n_scales in [128, 256, 512]:
        esc = EntropyScaleCodebook(d, d, n_scale_codes=n_scales)
        esc.train(W_true)
        W_esc = esc.get_weights()
        corr_esc = np.corrcoef((X_test @ W_esc.T).flatten(), Y_test.flatten())[0,1]
        bpp_esc = esc.effective_bpp()
        results[f'Entropy+Scale (K={n_scales})'] = {'corr': corr_esc, 'bpp': bpp_esc}
        print(f"Scales={n_scales}: {corr_esc:.4f} @ {bpp_esc:.2f} bpp")
    
    # Summary
    with open("results_v11_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY - NOVEL IDEAS V11 (ENTROPY + MAGNITUDE)\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<30} {'Corr':>8} {'BPP':>8} {'vs Tern':>10}\n")
        f.write("-" * 65 + "\n")
        
        for name, res in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (res['corr'] - corr_tern) / corr_tern * 100
            line = f"{name:<30} {res['corr']:>8.4f} {res['bpp']:>8.2f} {vs_tern:>+9.1f}%\n"
            print(line.strip())
            f.write(line)
    
    print("\n" + "="*80)
    print("🎯 GOAL: Beat ternary (0.8819 @ 1.58 bpp) at LOWER BPP")
    print("="*80)

if __name__ == "__main__":
    run_experiments()
