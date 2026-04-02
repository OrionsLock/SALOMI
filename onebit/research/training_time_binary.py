"""Training-Time Binary: Optimize binary weights directly for the task.

Key insight: Post-hoc quantization loses information.
Training-time quantization lets the model adapt to binary constraints.

Approach:
1. Initialize binary weights randomly
2. Optimize task loss directly with binary weights
3. Use coordinate descent (flip bits) + gradient descent (scale)
"""

import numpy as np
from typing import Tuple
import time


class TrainingTimeBinary:
    """Binary weights optimized directly for the task."""
    
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.W_binary = None
        self.scale = 1.0
    
    def _compute_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y_pred = X @ (self.W_binary * self.scale).T
        return np.mean((Y_pred - Y) ** 2)
    
    def _optimal_scale(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute optimal scale given current binary weights."""
        Y_pred_unit = X @ self.W_binary.T
        return np.sum(Y_pred_unit * Y) / (np.sum(Y_pred_unit ** 2) + 1e-8)
    
    def train(self, X: np.ndarray, Y: np.ndarray, 
              n_epochs: int = 10, verbose: bool = False):
        """Train binary weights using coordinate descent."""
        # Initialize from sign of optimal
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W_binary = np.sign(W_opt).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0
        self.scale = self._optimal_scale(X, Y)
        
        best_loss = self._compute_loss(X, Y)
        if verbose:
            print(f"Initial loss: {best_loss:.6f}")
        
        # Coordinate descent: try flipping each bit
        for epoch in range(n_epochs):
            n_flips = 0
            indices = list(range(self.d_out * self.d_in))
            np.random.shuffle(indices)
            
            for idx in indices:
                i, j = idx // self.d_in, idx % self.d_in
                
                # Flip the bit
                self.W_binary[i, j] *= -1
                self.scale = self._optimal_scale(X, Y)
                new_loss = self._compute_loss(X, Y)
                
                if new_loss < best_loss:
                    best_loss = new_loss
                    n_flips += 1
                else:
                    # Revert
                    self.W_binary[i, j] *= -1
            
            self.scale = self._optimal_scale(X, Y)
            
            if verbose:
                print(f"Epoch {epoch+1}: loss={best_loss:.6f}, flips={n_flips}")
            
            if n_flips == 0:
                break
    
    def get_weights(self) -> np.ndarray:
        return self.W_binary * self.scale
    
    def effective_bpp(self) -> float:
        return 1.0


class TrainingTimeTernary:
    """Ternary weights optimized directly for the task."""
    
    def __init__(self, d_in: int, d_out: int, zero_fraction: float = 0.3):
        self.d_in = d_in
        self.d_out = d_out
        self.zero_fraction = zero_fraction
        self.W_ternary = None  # {-1, 0, +1}
        self.scale = 1.0
    
    def _compute_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y_pred = X @ (self.W_ternary * self.scale).T
        return np.mean((Y_pred - Y) ** 2)
    
    def _optimal_scale(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y_pred_unit = X @ self.W_ternary.T
        denom = np.sum(Y_pred_unit ** 2)
        return np.sum(Y_pred_unit * Y) / (denom + 1e-8) if denom > 0 else 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray,
              n_epochs: int = 10, verbose: bool = False):
        # Initialize from optimal
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        thresh = np.percentile(np.abs(W_opt), self.zero_fraction * 100)
        self.W_ternary = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        self.scale = self._optimal_scale(X, Y)
        
        best_loss = self._compute_loss(X, Y)
        if verbose:
            print(f"Initial loss: {best_loss:.6f}")
        
        # Coordinate descent: try changing each weight to {-1, 0, +1}
        for epoch in range(n_epochs):
            n_changes = 0
            indices = list(range(self.d_out * self.d_in))
            np.random.shuffle(indices)
            
            for idx in indices:
                i, j = idx // self.d_in, idx % self.d_in
                current = self.W_ternary[i, j]
                
                # Try all 3 values
                best_val = current
                for val in [-1, 0, 1]:
                    if val == current:
                        continue
                    self.W_ternary[i, j] = val
                    self.scale = self._optimal_scale(X, Y)
                    new_loss = self._compute_loss(X, Y)
                    
                    if new_loss < best_loss:
                        best_loss = new_loss
                        best_val = val
                        n_changes += 1
                
                self.W_ternary[i, j] = best_val
            
            self.scale = self._optimal_scale(X, Y)
            
            if verbose:
                print(f"Epoch {epoch+1}: loss={best_loss:.6f}, changes={n_changes}")
            
            if n_changes == 0:
                break
    
    def get_weights(self) -> np.ndarray:
        return self.W_ternary * self.scale
    
    def effective_bpp(self) -> float:
        return 1.58


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiments():
    print("=" * 80)
    print("TRAINING-TIME BINARY: Can training-aware binary match ternary?")
    print("=" * 80)

    # Use smaller dimensions for speed (coordinate descent is O(d²) per epoch)
    for d in [32, 48, 64]:
        print(f"\n{'='*60}")
        print(f"Dimension: {d}x{d}")
        print(f"{'='*60}")

        np.random.seed(42)
        W_true = np.random.randn(d, d).astype(np.float32) * 0.5
        X_train = np.random.randn(2000, d).astype(np.float32)
        Y_train = X_train @ W_true.T + np.random.randn(2000, d) * 0.1
        X_test = np.random.randn(500, d).astype(np.float32)
        Y_test = X_test @ W_true.T

        results = {}

        # Post-hoc binary
        print("\nPost-hoc Binary...")
        W_opt = np.linalg.lstsq(X_train, Y_train, rcond=None)[0].T
        W_bin = np.sign(W_opt)
        W_bin[W_bin == 0] = 1.0
        Y_pred_unit = X_train @ W_bin.T
        scale = np.sum(Y_pred_unit * Y_train) / (np.sum(Y_pred_unit ** 2) + 1e-8)
        Y_pred = X_test @ (W_bin * scale).T
        results['posthoc_binary'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.0
        }

        # Post-hoc ternary
        print("Post-hoc Ternary...")
        thresh = np.percentile(np.abs(W_opt), 30)
        W_ter = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        mask = W_ter != 0
        scale = np.mean(np.abs(W_opt[mask])) if mask.any() else 1.0
        Y_pred = X_test @ (W_ter * scale).T
        results['posthoc_ternary'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.58
        }

        # Training-time binary
        print("Training-time Binary (this may take a moment)...")
        t0 = time.time()
        ttb = TrainingTimeBinary(d, d)
        ttb.train(X_train, Y_train, n_epochs=5, verbose=False)
        Y_pred = X_test @ ttb.get_weights().T
        results['trained_binary'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.0,
            'time': time.time() - t0
        }

        # Training-time ternary
        print("Training-time Ternary...")
        t0 = time.time()
        ttt = TrainingTimeTernary(d, d)
        ttt.train(X_train, Y_train, n_epochs=5, verbose=False)
        Y_pred = X_test @ ttt.get_weights().T
        results['trained_ternary'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.58,
            'time': time.time() - t0
        }

        # Print results
        print(f"\n{'Method':<20} {'Corr':>10} {'BPP':>8} {'Time':>8}")
        print("-" * 50)
        for name, data in sorted(results.items(), key=lambda x: -x[1]['corr']):
            t = data.get('time', 0)
            print(f"{name:<20} {data['corr']:>10.4f} {data['bpp']:>8.3f} {t:>7.1f}s")

        # Gap analysis
        posthoc_gap = results['posthoc_ternary']['corr'] - results['posthoc_binary']['corr']
        trained_gap = results['trained_ternary']['corr'] - results['trained_binary']['corr']
        print(f"\nPost-hoc gap (ternary - binary): {posthoc_gap:.4f}")
        print(f"Trained gap (ternary - binary):  {trained_gap:.4f}")
        print(f"Gap reduction: {(1 - trained_gap/posthoc_gap)*100:.1f}%")


if __name__ == "__main__":
    run_experiments()

