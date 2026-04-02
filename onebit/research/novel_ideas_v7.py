"""
Novel Ideas V7: Structure & Composition

Experiments:
1. Permutation-Optimized Binary:
   - Reorder (permute) weights to make the magnitude matrix smoother or lower-rank.
   - Hypothesis: Natural ordering is not optimal for compression.
   - Method: Sort by magnitude (heuristic for smoothness).

2. Frequency-Sparse Magnitude:
   - Compress magnitude using Discrete Cosine Transform (DCT).
   - Keep top-K DCT coefficients.
   - Hypothesis: Magnitude varies smoothly, so it's sparse in frequency domain.

3. Dual-Binary Decomposition:
   - W approx alpha * B1 + beta * B2
   - Sum of two binary bases.
   - Hypothesis: 2 bits (dual binary) is better than 1.58 bits (ternary).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List

# =============================================================================
# 1. PERMUTATION-OPTIMIZED BINARY
# =============================================================================

class PermutationOptimizedBinary:
    """
    Permute weights to make magnitude low-rank approximation better.
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 4):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        
        self.S = None
        self.U = None
        self.Vt = None
        self.perm_indices = None # Permutation of columns
        
    def train(self, W_target: np.ndarray):
        # 1. Find permutation that sorts columns by L2 norm
        # This clusters high-magnitude columns together
        col_norms = np.linalg.norm(W_target, axis=0)
        self.perm_indices = np.argsort(col_norms)
        
        # Permute W
        W_perm = W_target[:, self.perm_indices]
        
        # 2. Standard Low-Rank Magnitude on Permuted W
        self.S = np.sign(W_perm).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        M = np.abs(W_perm)
        
        # SVD on Magnitude
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        self.U = u[:, :self.rank] * s[:self.rank]
        self.Vt = vt[:self.rank, :]
        
    def get_weights(self) -> np.ndarray:
        # Reconstruct permuted magnitude
        M_perm = self.U @ self.Vt
        W_perm = self.S * M_perm
        
        # Inverse permutation to get original order
        inv_perm = np.argsort(self.perm_indices)
        W_recon = W_perm[:, inv_perm]
        
        return W_recon
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        # Low rank params
        lr_bits = (self.d_out * self.rank + self.rank * self.d_in) * 32
        # Permutation indices: d_in * log2(d_in)
        perm_bits = self.d_in * np.log2(self.d_in)
        
        return (sign_bits + lr_bits + perm_bits) / n_weights


# =============================================================================
# 2. FREQUENCY-SPARSE MAGNITUDE (DCT)
# =============================================================================

class FrequencySparseMagnitude:
    """
    Compress magnitude using DCT and keeping top-K coefficients.
    """
    def __init__(self, d_in: int, d_out: int, n_coeffs: int = 256):
        self.d_in = d_in
        self.d_out = d_out
        self.n_coeffs = n_coeffs
        
        self.S = None
        self.dct_coeffs = None
        self.dct_indices = None
        
        # Precompute DCT matrix
        self.dct_mat_in = self._make_dct_matrix(d_in)
        self.dct_mat_out = self._make_dct_matrix(d_out)
        
    def _make_dct_matrix(self, n):
        # DCT-II matrix
        # C_kn = cos(pi/N * (n + 0.5) * k)
        k = np.arange(n).reshape((n, 1))
        n_idx = np.arange(n).reshape((1, n))
        C = np.cos(np.pi / n * (n_idx + 0.5) * k)
        # Scale rows to make it orthogonal
        C[0, :] *= 1 / np.sqrt(n)
        C[1:, :] *= np.sqrt(2 / n)
        return C
        
    def train(self, W_target: np.ndarray):
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        M = np.abs(W_target)
        
        # 2D DCT: Y = C_out * M * C_in.T
        M_dct = self.dct_mat_out @ M @ self.dct_mat_in.T
        
        # Flatten and keep top-K
        flat_dct = M_dct.flatten()
        top_k_idx = np.argsort(np.abs(flat_dct))[-self.n_coeffs:]
        
        self.dct_indices = top_k_idx
        self.dct_coeffs = flat_dct[top_k_idx]
        
    def get_weights(self) -> np.ndarray:
        # Reconstruct DCT matrix
        M_dct = np.zeros((self.d_out, self.d_in), dtype=np.float32)
        flat_dct = M_dct.flatten()
        flat_dct[self.dct_indices] = self.dct_coeffs
        M_dct = flat_dct.reshape(self.d_out, self.d_in)
        
        # Inverse 2D DCT: M = C_out.T * Y * C_in
        M_recon = self.dct_mat_out.T @ M_dct @ self.dct_mat_in
        
        return self.S * M_recon
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        # Coeffs + Indices
        dct_bits = self.n_coeffs * (32 + np.log2(n_weights))
        
        return (sign_bits + dct_bits) / n_weights


# =============================================================================
# 3. DUAL-BINARY DECOMPOSITION
# =============================================================================

class DualBinaryDecomposition:
    """
    W approx alpha * B1 + beta * B2
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        
        self.B1 = None
        self.B2 = None
        self.alpha = 1.0
        self.beta = 0.5
        
    def train(self, W_target: np.ndarray, n_iter: int = 10):
        # Init B1 as sign(W)
        self.B1 = np.sign(W_target).astype(np.float32)
        self.B1[self.B1 == 0] = 1.0
        
        # Init alpha
        self.alpha = np.mean(np.abs(W_target))
        
        # Residual
        R = W_target - self.alpha * self.B1
        
        # Init B2 as sign(Residual)
        self.B2 = np.sign(R).astype(np.float32)
        self.B2[self.B2 == 0] = 1.0
        
        # Init beta
        self.beta = np.mean(np.abs(R))
        
        # Alternating optimization
        for _ in range(n_iter):
            # Optimize scales given B1, B2
            # Least squares for [alpha, beta]
            # y_i = alpha * b1_i + beta * b2_i
            # A = [B1_flat, B2_flat]^T
            # A^T A [alpha, beta]^T = A^T W_flat
            
            b1_flat = self.B1.flatten()
            b2_flat = self.B2.flatten()
            w_flat = W_target.flatten()
            
            A = np.vstack([b1_flat, b2_flat]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, w_flat, rcond=None)
            self.alpha, self.beta = coeffs
            
            # Optimize B1 given B2, alpha, beta
            # min || W - alpha*B1 - beta*B2 ||^2
            # min || (W - beta*B2) - alpha*B1 ||^2
            # B1 = sign(W - beta*B2) * sign(alpha)
            Target1 = W_target - self.beta * self.B2
            self.B1 = np.sign(Target1 * np.sign(self.alpha)).astype(np.float32)
            self.B1[self.B1 == 0] = 1.0
            
            # Optimize B2 given B1, alpha, beta
            Target2 = W_target - self.alpha * self.B1
            self.B2 = np.sign(Target2 * np.sign(self.beta)).astype(np.float32)
            self.B2[self.B2 == 0] = 1.0
            
    def get_weights(self) -> np.ndarray:
        return self.alpha * self.B1 + self.beta * self.B2
        
    def effective_bpp(self) -> float:
        return 2.0 # 2 bits per weight + negligible scalars


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V7: Structure & Composition")
    print("="*80)
    
    # Setup Data
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
    
    # 1. Permutation Optimized
    print("\nRunning Permutation Optimized Binary...")
    for r in [2, 4, 8]:
        pob = PermutationOptimizedBinary(d, d, rank=r)
        pob.train(W_true)
        W_pob = pob.get_weights()
        corr_pob = np.corrcoef((X_test @ W_pob.T).flatten(), Y_test.flatten())[0,1]
        bpp_pob = pob.effective_bpp()
        results[f'Permutation (r={r})'] = {'corr': corr_pob, 'bpp': bpp_pob}
        print(f"Rank={r}: {corr_pob:.4f} @ {bpp_pob:.2f} bpp")
        
    # 2. Frequency Sparse
    print("\nRunning Frequency Sparse Magnitude...")
    for k in [64, 128, 256]:
        fsm = FrequencySparseMagnitude(d, d, n_coeffs=k)
        fsm.train(W_true)
        W_fsm = fsm.get_weights()
        corr_fsm = np.corrcoef((X_test @ W_fsm.T).flatten(), Y_test.flatten())[0,1]
        bpp_fsm = fsm.effective_bpp()
        results[f'Freq-Sparse (K={k})'] = {'corr': corr_fsm, 'bpp': bpp_fsm}
        print(f"K={k}: {corr_fsm:.4f} @ {bpp_fsm:.2f} bpp")
        
    # 3. Dual Binary
    print("\nRunning Dual Binary Decomposition...")
    dbd = DualBinaryDecomposition(d, d)
    dbd.train(W_true)
    W_dbd = dbd.get_weights()
    corr_dbd = np.corrcoef((X_test @ W_dbd.T).flatten(), Y_test.flatten())[0,1]
    bpp_dbd = dbd.effective_bpp()
    results['Dual-Binary'] = {'corr': corr_dbd, 'bpp': bpp_dbd}
    print(f"Result: {corr_dbd:.4f} @ {bpp_dbd:.2f} bpp")
    
    # Summary
    with open("results_v7_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY - NOVEL IDEAS V7\n")
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
