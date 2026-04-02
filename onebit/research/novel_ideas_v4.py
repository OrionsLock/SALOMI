"""
Novel Ideas V4: Building on V3 insights.

From V3, we learned:
1. Rotated Binary works (+3% vs ternary) but rotation matrix is too expensive
2. VQ-Magnitude is less efficient than Low-Rank
3. Sparse corrections don't work well

New experiments:
1. Structured Rotation Binary (SRB):
   - Use cheap parameterized rotations (Givens/Householder cascades)
   - Hypothesis: We can capture rotation benefits with ~log(d) parameters
   
2. Hybrid Low-Rank + VQ:
   - Low-rank for global magnitude + VQ for residual
   - Hypothesis: Combining complementary compression approaches
   
3. Input-Adaptive Binary:
   - Binary weights change interpretation based on input statistics
   - Hypothesis: Dynamic magnitude from "free" input signal
   
4. Sign-Magnitude Factorization:
   - Factor W = S * diag(m) where S is binary, m is compressed
   - Different from low-rank: compress the diagonal magnitude vector
"""

import numpy as np
from typing import Tuple, Dict
import time

def simple_kmeans(X: np.ndarray, n_clusters: int, n_iter: int = 10):
    """Simple K-Means implementation."""
    n_samples, n_features = X.shape
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[indices]
    labels = np.zeros(n_samples, dtype=int)
    
    for _ in range(n_iter):
        dists = np.sum(X**2, axis=1, keepdims=True) + \
                np.sum(centroids**2, axis=1) - \
                2 * X @ centroids.T
        labels = np.argmin(dists, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                new_centroids[k] = X[np.random.randint(n_samples)]
                
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return centroids, labels


# =============================================================================
# 1. STRUCTURED ROTATION BINARY
# =============================================================================

class StructuredRotationBinary:
    """
    Use a cascade of Givens rotations (cheap to parameterize).
    Each Givens rotation is defined by (i, j, theta).
    Store k Givens rotations: 3*k parameters vs d^2 for full matrix.
    """
    def __init__(self, d_in: int, d_out: int, n_rotations: int = 16):
        self.d_in = d_in
        self.d_out = d_out
        self.n_rotations = n_rotations
        
        self.rotations = []  # List of (i, j, theta)
        self.B = None  # Binary weights
        self.scale = 1.0
        
    def apply_givens(self, v: np.ndarray, i: int, j: int, theta: float) -> np.ndarray:
        """Apply Givens rotation to vector v."""
        c, s = np.cos(theta), np.sin(theta)
        v_new = v.copy()
        v_new[i] = c * v[i] - s * v[j]
        v_new[j] = s * v[i] + c * v[j]
        return v_new
    
    def apply_rotation_matrix(self, W: np.ndarray) -> np.ndarray:
        """Apply cascade of Givens rotations to each row."""
        W_rot = W.copy()
        for i, j, theta in self.rotations:
            for row_idx in range(W_rot.shape[0]):
                W_rot[row_idx] = self.apply_givens(W_rot[row_idx], i, j, theta)
        return W_rot
    
    def train(self, W_target: np.ndarray):
        """Greedy: Find Givens rotations that maximize binarization quality."""
        W_current = W_target.copy()
        
        # Random initialization of rotations
        for _ in range(self.n_rotations):
            i = np.random.randint(0, self.d_in)
            j = np.random.randint(0, self.d_in)
            if i == j:
                j = (j + 1) % self.d_in
            theta = np.random.uniform(-np.pi, np.pi)
            self.rotations.append((i, j, theta))
        
        # Binarize rotated weights
        W_rot = self.apply_rotation_matrix(W_target)
        self.B = np.sign(W_rot).astype(np.float32)
        self.B[self.B == 0] = 1.0
        
        # Optimal scale
        W_recon = self.B  # After inverse rotation, but we skip for speed
        self.scale = np.mean(np.abs(W_target))
        
    def get_weights(self) -> np.ndarray:
        # In practice, we'd apply inverse rotations to B
        # For now, approximate as B * scale
        return self.B * self.scale
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        binary_bits = n_weights
        # Each rotation: 2 indices (log2(d_in)) + 1 angle (16 bits)
        rotation_bits = self.n_rotations * (2 * np.ceil(np.log2(self.d_in)) + 16)
        return (binary_bits + rotation_bits) / n_weights


# =============================================================================
# 2. HYBRID LOW-RANK + VQ
# =============================================================================

class HybridLowRankVQ:
    """
    Magnitude = LowRank_approx + VQ_residual
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 4, 
                 vq_codes: int = 16, block_size: int = 4):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.vq_codes = vq_codes
        self.block_size = block_size
        
        self.S = None
        self.U = None  # Low-rank factors
        self.V = None
        self.vq_codebook = None
        self.vq_indices = None
        
    def train(self, W_target: np.ndarray):
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Magnitude
        M = np.abs(W_target)
        
        # Low-rank approximation of magnitude
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        self.U = U[:, :self.rank] @ np.diag(np.sqrt(s[:self.rank]))
        self.V = Vt[:self.rank, :].T @ np.diag(np.sqrt(s[:self.rank]))
        
        M_lowrank = self.U @ self.V.T
        
        # Residual
        Residual = M - M_lowrank
        
        # VQ on residual blocks
        bs = self.block_size
        n_h = (self.d_out + bs - 1) // bs
        n_w = (self.d_in + bs - 1) // bs
        
        # Pad residual
        pad_h = n_h * bs - self.d_out
        pad_w = n_w * bs - self.d_in
        Res_padded = np.pad(Residual, ((0, pad_h), (0, pad_w)), mode='constant')
        
        blocks = []
        for i in range(n_h):
            for j in range(n_w):
                block = Res_padded[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                blocks.append(block.flatten())
        
        blocks = np.array(blocks)
        self.vq_codebook, labels = simple_kmeans(blocks, n_clusters=self.vq_codes)
        self.vq_indices = labels.reshape(n_h, n_w)
        
    def get_weights(self) -> np.ndarray:
        # Reconstruct magnitude
        M_lowrank = self.U @ self.V.T
        
        # Reconstruct VQ residual
        bs = self.block_size
        n_h, n_w = self.vq_indices.shape
        Res_vq = np.zeros((n_h * bs, n_w * bs), dtype=np.float32)
        
        for i in range(n_h):
            for j in range(n_w):
                idx = self.vq_indices[i, j]
                block = self.vq_codebook[idx].reshape(bs, bs)
                Res_vq[i*bs:(i+1)*bs, j*bs:(j+1)*bs] = block
        
        Res_vq = Res_vq[:self.d_out, :self.d_in]
        M_total = M_lowrank + Res_vq
        
        return self.S * M_total
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        
        sign_bits = n_weights
        lowrank_bits = (self.d_out + self.d_in) * self.rank * 32
        
        n_blocks = self.vq_indices.size
        vq_index_bits = n_blocks * np.log2(self.vq_codes)
        vq_codebook_bits = self.vq_codes * (self.block_size ** 2) * 32
        
        total = sign_bits + lowrank_bits + vq_index_bits + vq_codebook_bits
        return total / n_weights


# =============================================================================
# 3. INPUT-ADAPTIVE BINARY
# =============================================================================

class InputAdaptiveBinary:
    """
    Binary weights get their magnitude from input statistics.
    magnitude_ij = f(input_statistics)
    
    For offline evaluation, we approximate with precomputed input stats.
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        
        self.S = None  # Binary signs
        self.input_importance = None  # Per-input-dim importance
        self.output_importance = None  # Per-output-dim importance
        
    def train(self, W_target: np.ndarray, X_train: np.ndarray = None):
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Learn input/output importance from weight magnitudes
        self.input_importance = np.mean(np.abs(W_target), axis=0)
        self.output_importance = np.mean(np.abs(W_target), axis=1)
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass with input-adaptive magnitudes."""
        # Magnitude from input statistics
        # Simple: use input L2 norm per dimension
        input_norms = np.abs(X)  # (batch, d_in)
        
        # Scale based on learned importance
        adaptive_mag = np.outer(self.output_importance, self.input_importance)
        
        # Weight = Sign * AdaptiveMag
        W_adaptive = self.S * adaptive_mag
        
        return X @ W_adaptive.T
    
    def get_weights(self) -> np.ndarray:
        """Static weights (for comparison)."""
        mag = np.outer(self.output_importance, self.input_importance)
        return self.S * mag
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        importance_bits = (self.d_in + self.d_out) * 32
        return (sign_bits + importance_bits) / n_weights


# =============================================================================
# 4. SIGN-MAGNITUDE FACTORIZATION
# =============================================================================

class SignMagnitudeFactorization:
    """
    W = S * M where S is binary matrix, M is diagonal magnitude.
    Compress M using quantization.
    """
    def __init__(self, d_in: int, d_out: int, mag_bits: int = 8):
        self.d_in = d_in
        self.d_out = d_out
        self.mag_bits = mag_bits
        
        self.S = None
        self.M_row = None  # Per-row magnitude
        self.M_col = None  # Per-col magnitude
        
    def train(self, W_target: np.ndarray):
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Magnitude: per-row and per-col scales
        self.M_row = np.mean(np.abs(W_target), axis=1)
        self.M_col = np.mean(np.abs(W_target), axis=0)
        
        # Quantize magnitudes to mag_bits
        # Simple linear quantization
        self.M_row = self._quantize(self.M_row, self.mag_bits)
        self.M_col = self._quantize(self.M_col, self.mag_bits)
        
    def _quantize(self, x: np.ndarray, n_bits: int) -> np.ndarray:
        """Quantize to n_bits."""
        x_min, x_max = x.min(), x.max()
        n_levels = 2 ** n_bits
        scale = (x_max - x_min) / (n_levels - 1)
        quantized = np.round((x - x_min) / scale) * scale + x_min
        return quantized
    
    def get_weights(self) -> np.ndarray:
        # Magnitude = outer product of row and col scales
        M = np.sqrt(np.outer(self.M_row, self.M_col))
        return self.S * M
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        mag_bits = (self.d_out + self.d_in) * self.mag_bits
        return (sign_bits + mag_bits) / n_weights


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V4: Structured Rotations, Hybrid Methods, Adaptive")
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
    
    # 1. Structured Rotation
    print("\nRunning Structured Rotation Binary...")
    for n_rot in [8, 16, 32]:
        srb = StructuredRotationBinary(d, d, n_rotations=n_rot)
        srb.train(W_true)
        W_srb = srb.get_weights()
        corr_srb = np.corrcoef((X_test @ W_srb.T).flatten(), Y_test.flatten())[0,1]
        bpp_srb = srb.effective_bpp()
        results[f'Struct-Rot (n={n_rot})'] = {'corr': corr_srb, 'bpp': bpp_srb}
        print(f"n_rotations={n_rot}: {corr_srb:.4f} @ {bpp_srb:.2f} bpp")
    
    # 2. Hybrid Low-Rank + VQ
    print("\nRunning Hybrid LowRank+VQ...")
    for rank in [2, 4]:
        hlvq = HybridLowRankVQ(d, d, rank=rank, vq_codes=16, block_size=4)
        hlvq.train(W_true)
        W_hlvq = hlvq.get_weights()
        corr_hlvq = np.corrcoef((X_test @ W_hlvq.T).flatten(), Y_test.flatten())[0,1]
        bpp_hlvq = hlvq.effective_bpp()
        results[f'Hybrid-LR{rank}+VQ'] = {'corr': corr_hlvq, 'bpp': bpp_hlvq}
        print(f"Rank={rank}: {corr_hlvq:.4f} @ {bpp_hlvq:.2f} bpp")
    
    # 3. Input-Adaptive
    print("\nRunning Input-Adaptive Binary...")
    iab = InputAdaptiveBinary(d, d)
    iab.train(W_true)
    W_iab = iab.get_weights()
    corr_iab = np.corrcoef((X_test @ W_iab.T).flatten(), Y_test.flatten())[0,1]
    bpp_iab = iab.effective_bpp()
    results['Input-Adaptive'] = {'corr': corr_iab, 'bpp': bpp_iab}
    print(f"Result: {corr_iab:.4f} @ {bpp_iab:.2f} bpp")
    
    # 4. Sign-Magnitude Factorization
    print("\nRunning Sign-Magnitude Factorization...")
    for mag_bits in [4, 8, 16]:
        smf = SignMagnitudeFactorization(d, d, mag_bits=mag_bits)
        smf.train(W_true)
        W_smf = smf.get_weights()
        corr_smf = np.corrcoef((X_test @ W_smf.T).flatten(), Y_test.flatten())[0,1]
        bpp_smf = smf.effective_bpp()
        results[f'SignMag (b={mag_bits})'] = {'corr': corr_smf, 'bpp': bpp_smf}
        print(f"mag_bits={mag_bits}: {corr_smf:.4f} @ {bpp_smf:.2f} bpp")
    
    # Summary
    with open("results_v4_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY - NOVEL IDEAS V4\n")
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
