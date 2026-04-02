"""Low-Rank Binary: Factorize weights as W = U @ V.T with binary U, V.

Key insight: Low-rank approximation is a form of regularization/pruning.
If we can get good quality with low rank, we use fewer bits.

W = U @ V.T where:
- U ∈ {-1,+1}^(d_out × r)
- V ∈ {-1,+1}^(d_in × r)
- scale ∈ R

Storage: r * (d_out + d_in) bits + 32 bits for scale
For d=256, r=64: 0.5 bpp!
"""

import numpy as np
from typing import Tuple


class LowRankBinary:
    """Binary low-rank factorization: W = scale * U @ V.T"""
    
    def __init__(self, d_in: int, d_out: int, rank: int):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        
        self.U = None  # (d_out, rank) binary
        self.V = None  # (d_in, rank) binary
        self.scale = 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        """Train using SVD initialization + refinement."""
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        # SVD to get low-rank approximation
        U_svd, S, Vt_svd = np.linalg.svd(W_opt, full_matrices=False)
        
        # Take top-r components
        U_r = U_svd[:, :self.rank] @ np.diag(np.sqrt(S[:self.rank]))
        V_r = Vt_svd[:self.rank, :].T @ np.diag(np.sqrt(S[:self.rank]))
        
        # Binarize
        self.U = np.sign(U_r).astype(np.float32)
        self.U[self.U == 0] = 1.0
        self.V = np.sign(V_r).astype(np.float32)
        self.V[self.V == 0] = 1.0
        
        # Optimal scale
        W_binary = self.U @ self.V.T
        Y_pred = X @ W_binary.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    
    def get_weights(self) -> np.ndarray:
        return self.scale * (self.U @ self.V.T)
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        factor_bits = self.rank * (self.d_out + self.d_in)
        scale_bits = 32
        return (factor_bits + scale_bits) / n_weights


class LowRankBinaryWithDiag:
    """W = scale * U @ D @ V.T where D is diagonal with learned values."""
    
    def __init__(self, d_in: int, d_out: int, rank: int):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        
        self.U = None
        self.V = None
        self.D = None  # Diagonal values
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        U_svd, S, Vt_svd = np.linalg.svd(W_opt, full_matrices=False)
        
        self.U = np.sign(U_svd[:, :self.rank]).astype(np.float32)
        self.U[self.U == 0] = 1.0
        self.V = np.sign(Vt_svd[:self.rank, :].T).astype(np.float32)
        self.V[self.V == 0] = 1.0
        
        # Learn diagonal to minimize reconstruction error
        # W ≈ U @ D @ V.T
        # Solve for D using least squares
        # Flatten: vec(W) = (V ⊗ U) @ vec(D)
        # For diagonal D, this simplifies
        
        # Use iterative approach
        self.D = S[:self.rank].copy()
        
        for _ in range(10):
            W_approx = self.U @ np.diag(self.D) @ self.V.T
            # Gradient step
            error = W_opt - W_approx
            for k in range(self.rank):
                grad = -2 * np.sum(error * np.outer(self.U[:, k], self.V[:, k]))
                self.D[k] -= 0.01 * grad
    
    def get_weights(self) -> np.ndarray:
        return self.U @ np.diag(self.D) @ self.V.T
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        factor_bits = self.rank * (self.d_out + self.d_in)
        diag_bits = self.rank * 32  # FP32 diagonal
        return (factor_bits + diag_bits) / n_weights


class Binary:
    def __init__(self, d_in, d_out):
        self.W = None; self.scale = 1.0; self.d_in = d_in; self.d_out = d_out
    def train(self, X, Y):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W = np.sign(W_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        Y_pred = X @ self.W.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    def get_weights(self): return self.W * self.scale
    def effective_bpp(self): return 1.0


class Ternary:
    def __init__(self, d_in, d_out, zero_pct=0.3):
        self.W = None; self.scale = 1.0; self.zero_pct = zero_pct
        self.d_in = d_in; self.d_out = d_out
    def train(self, X, Y):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        thresh = np.percentile(np.abs(W_opt), self.zero_pct * 100)
        self.W = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        mask = self.W != 0
        if mask.any(): self.scale = np.mean(np.abs(W_opt[mask]))
    def get_weights(self): return self.W * self.scale
    def effective_bpp(self): return 1.58


def run_experiments():
    print("=" * 80)
    print("LOW-RANK BINARY: Can factorization beat ternary at lower bpp?")
    print("=" * 80)

    for d in [64, 128, 256]:
        print(f"\n{'='*60}")
        print(f"Dimension: {d}x{d}")
        print(f"{'='*60}")

        np.random.seed(42)
        W_true = np.random.randn(d, d).astype(np.float32) * 0.5
        X_train = np.random.randn(5000, d).astype(np.float32)
        Y_train = X_train @ W_true.T + np.random.randn(5000, d) * 0.1
        X_test = np.random.randn(1000, d).astype(np.float32)
        Y_test = X_test @ W_true.T

        results = {}

        # Baselines
        for cls, name in [(Binary, 'binary'), (Ternary, 'ternary')]:
            m = cls(d, d)
            m.train(X_train, Y_train)
            Y_pred = X_test @ m.get_weights().T
            results[name] = {
                'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                'bpp': m.effective_bpp()
            }

        # Low-rank binary at various ranks
        for rank_frac in [0.25, 0.5, 0.75, 1.0]:
            rank = max(1, int(d * rank_frac))
            m = LowRankBinary(d, d, rank)
            m.train(X_train, Y_train)
            Y_pred = X_test @ m.get_weights().T
            results[f'lowrank_{int(rank_frac*100)}'] = {
                'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                'bpp': m.effective_bpp()
            }

        # Low-rank with diagonal
        for rank_frac in [0.25, 0.5, 0.75]:
            rank = max(1, int(d * rank_frac))
            m = LowRankBinaryWithDiag(d, d, rank)
            m.train(X_train, Y_train)
            Y_pred = X_test @ m.get_weights().T
            results[f'lowrank_diag_{int(rank_frac*100)}'] = {
                'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                'bpp': m.effective_bpp()
            }

        # Print sorted by correlation
        print(f"\n{'Method':<20} {'Corr':>10} {'BPP':>10} {'vs Ternary':>12}")
        print("-" * 55)
        ternary_corr = results['ternary']['corr']

        for name, data in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs = (data['corr'] / ternary_corr - 1) * 100
            marker = "✓" if data['bpp'] < 1.58 and data['corr'] >= ternary_corr else ""
            print(f"{name:<20} {data['corr']:>10.4f} {data['bpp']:>10.3f} {vs:>+11.1f}% {marker}")

        # Find best under 1.1 bpp
        sub_1_1 = [(n, d) for n, d in results.items() if d['bpp'] <= 1.1]
        if sub_1_1:
            best = max(sub_1_1, key=lambda x: x[1]['corr'])
            print(f"\n→ Best under 1.1 bpp: {best[0]} ({best[1]['corr']:.4f} @ {best[1]['bpp']:.3f} bpp)")


if __name__ == "__main__":
    run_experiments()

