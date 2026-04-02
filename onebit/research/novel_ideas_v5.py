"""
Novel Ideas V5: Ultimate Experiments - Pushing the Absolute Limits

Based on all previous insights, we test the most promising unexplored directions:

1. Learned Binary Basis (LBB):
   - Instead of sign(W), learn a better binary basis B via gradient descent
   - Optimize: B, scale such that minimize ||X @ B.T @ scale - Y||
   - Hypothesis: Learned binary basis can beat sign quantization

2. Magnitude Clustering Across Weights:
   - Group weights by magnitude pattern, share magnitude within groups
   - Hypothesis: Magnitude has clustering structure we can exploit

3. Bit-Plane Encoding:
   - Store binary weights as multiple bit-planes (MSB, LSB, etc.)
   - Allocate bits to most important planes
   - Hypothesis: Some bit-planes are more important than others
"""

import numpy as np
from typing import Tuple, Dict


# =============================================================================
# 1. LEARNED BINARY BASIS (Coordinate Descent)
# =============================================================================

class LearnedBinaryBasis:
    """
    Learn binary weights B directly for the task, not via sign(W_opt).
    Use coordinate descent: flip bits to improve task loss.
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.B = None
        self.scale = 1.0
        
    def train(self, X: np.ndarray, Y: np.ndarray, n_iter: int = 10):
        """Coordinate descent optimization."""
        # Initialize from sign(W_opt)
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.B = np.sign(W_opt).astype(np.float32)
        self.B[self.B == 0] = 1.0
        
        # Optimal scale
        Y_pred = X @ self.B.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
        
        # Use a smaller subset for the slow coordinate descent part
        n_samples = min(1000, X.shape[0])
        X_sub = X[:n_samples]
        Y_sub = Y[:n_samples]
        
        best_loss = np.mean((X_sub @ (self.B * self.scale).T - Y_sub) ** 2)
        
        print(f"  Initial Loss: {best_loss:.6f}")
        
        # Coordinate descent: try flipping each bit
        for iteration in range(n_iter):
            n_flips = 0
            
            # Random order of indices to try flipping
            indices = np.random.permutation(self.d_out * self.d_in)
            # Limit number of trials per iteration to keep it fast
            trials = indices[:min(500, len(indices))]
            
            for idx in trials:
                i, j = idx // self.d_in, idx % self.d_in
                
                # Flip
                self.B[i, j] *= -1
                
                # Recompute scale (approximate on subset)
                Y_pred = X_sub @ self.B.T
                self.scale = np.sum(Y_pred * Y_sub) / (np.sum(Y_pred ** 2) + 1e-8)
                
                # Evaluate
                new_loss = np.mean((X_sub @ (self.B * self.scale).T - Y_sub) ** 2)
                
                if new_loss < best_loss:
                    best_loss = new_loss
                    n_flips += 1
                else:
                    # Revert
                    self.B[i, j] *= -1
            
            print(f"  Iter {iteration+1}/{n_iter}: {n_flips} flips, Loss {best_loss:.6f}")
            if n_flips == 0:
                break
                
        # Final scale on full data
        Y_pred = X @ self.B.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    
    def get_weights(self) -> np.ndarray:
        return self.B * self.scale
    
    def effective_bpp(self) -> float:
        return 1.0


# =============================================================================
# 2. MAGNITUDE CLUSTERING
# =============================================================================

class MagnitudeClusteringBinary:
    """
    Cluster weights by magnitude, share magnitude within clusters.
    """
    def __init__(self, d_in: int, d_out: int, n_clusters: int = 8):
        self.d_in = d_in
        self.d_out = d_out
        self.n_clusters = n_clusters
        
        self.S = None
        self.cluster_labels = None
        self.cluster_magnitudes = None
        
    def simple_kmeans_1d(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """1D K-means for magnitudes."""
        # Sort-based initialization
        sorted_x = np.sort(x)
        n = len(x)
        centroids = np.array([sorted_x[i * n // k] for i in range(k)])
        
        labels = np.zeros(n, dtype=int)
        
        for _ in range(10):
            # Assign
            dists = np.abs(x[:, None] - centroids[None, :])
            labels = np.argmin(dists, axis=1)
            
            # Update
            new_centroids = np.array([x[labels == i].mean() if np.any(labels == i) 
                                      else centroids[i] for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        return centroids, labels
        
    def train(self, W_target: np.ndarray):
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Magnitudes
        M = np.abs(W_target).flatten()
        
        # Cluster magnitudes
        self.cluster_magnitudes, self.cluster_labels = self.simple_kmeans_1d(
            M, self.n_clusters
        )
    
    def get_weights(self) -> np.ndarray:
        # Assign clustered magnitudes
        M_clustered = self.cluster_magnitudes[self.cluster_labels]
        M_mat = M_clustered.reshape(self.d_out, self.d_in)
        return self.S * M_mat
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        label_bits = n_weights * np.log2(self.n_clusters)
        centroid_bits = self.n_clusters * 32
        return (sign_bits + label_bits + centroid_bits) / n_weights


# =============================================================================
# 3. BIT-PLANE ENCODING
# =============================================================================

class BitPlaneEncoding:
    """
    Represent weights with multiple binary bit-planes.
    weight_ij = sign_ij * sum(2^k * plane_k_ij)
    
    Store only the most important bit-planes.
    """
    def __init__(self, d_in: int, d_out: int, n_planes: int = 2):
        self.d_in = d_in
        self.d_out = d_out
        self.n_planes = n_planes
        
        self.S = None
        self.planes = []  # List of binary planes
        self.plane_scales = []
        
    def train(self, W_target: np.ndarray):
        # Signs
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        # Magnitude
        M = np.abs(W_target)
        
        # Normalize
        M_max = M.max()
        M_norm = M / (M_max + 1e-8)
        
        # Extract bit-planes (like JPEG)
        # Plane k: weights where bit k is set
        self.planes = []
        self.plane_scales = []
        
        Residual = M_norm.copy()
        
        for k in range(self.n_planes):
            # Current bit level
            level = 0.5 ** (k + 1)
            
            # Binary decision: above or below threshold
            plane = (Residual >= level).astype(np.float32)
            
            # Update residual
            Residual = Residual - plane * level
            
            self.planes.append(plane)
            self.plane_scales.append(level * M_max)
    
    def get_weights(self) -> np.ndarray:
        # Reconstruct magnitude from bit-planes
        M = np.zeros((self.d_out, self.d_in), dtype=np.float32)
        
        for plane, scale in zip(self.planes, self.plane_scales):
            M += plane * scale
        
        return self.S * M
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        plane_bits = n_weights * self.n_planes
        scale_bits = self.n_planes * 32
        return (sign_bits + plane_bits + scale_bits) / n_weights


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V5: Ultimate Boundary-Pushing Experiments")
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
    
    X_train = np.random.randn(5000, d).astype(np.float32)
    Y_train = X_train @ W_true.T + np.random.randn(5000, d) * 0.1
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
    
    # 1. Learned Binary Basis
    print("\nRunning Learned Binary Basis (this may take a moment)...")
    lbb = LearnedBinaryBasis(d, d)
    lbb.train(X_train, Y_train, n_iter=20)
    W_lbb = lbb.get_weights()
    corr_lbb = np.corrcoef((X_test @ W_lbb.T).flatten(), Y_test.flatten())[0,1]
    bpp_lbb = lbb.effective_bpp()
    results['Learned-Binary'] = {'corr': corr_lbb, 'bpp': bpp_lbb}
    print(f"Result: {corr_lbb:.4f} @ {bpp_lbb:.2f} bpp")
    
    # 2. Magnitude Clustering
    print("\nRunning Magnitude Clustering...")
    for k in [4, 8, 16]:
        mcb = MagnitudeClusteringBinary(d, d, n_clusters=k)
        mcb.train(W_true)
        W_mcb = mcb.get_weights()
        corr_mcb = np.corrcoef((X_test @ W_mcb.T).flatten(), Y_test.flatten())[0,1]
        bpp_mcb = mcb.effective_bpp()
        results[f'Mag-Cluster (K={k})'] = {'corr': corr_mcb, 'bpp': bpp_mcb}
        print(f"K={k}: {corr_mcb:.4f} @ {bpp_mcb:.2f} bpp")
    
    # 3. Bit-Plane Encoding
    print("\nRunning Bit-Plane Encoding...")
    for n_planes in [1, 2, 3]:
        bpe = BitPlaneEncoding(d, d, n_planes=n_planes)
        bpe.train(W_true)
        W_bpe = bpe.get_weights()
        corr_bpe = np.corrcoef((X_test @ W_bpe.T).flatten(), Y_test.flatten())[0,1]
        bpp_bpe = bpe.effective_bpp()
        results[f'Bit-Plane (n={n_planes})'] = {'corr': corr_bpe, 'bpp': bpp_bpe}
        print(f"n_planes={n_planes}: {corr_bpe:.4f} @ {bpp_bpe:.2f} bpp")
    
    # Summary
    with open("results_v5_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY - NOVEL IDEAS V5\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<25} {'Corr':>8} {'BPP':>8} {'vs Tern':>10}\n")
        f.write("-" * 60 + "\n")
        
        for name, res in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (res['corr'] - corr_tern) / corr_tern * 100
            line = f"{name:<25} {res['corr']:>8.4f} {res['bpp']:>8.2f} {vs_tern:>+9.1f}%\n"
            print(line.strip())
            f.write(line)
    
    print("\n" + "="*80)
    print("Results written to results_v5_utf8.txt")

if __name__ == "__main__":
    run_experiments()
