"""Iterative Binary: Trade compute for precision.

Key insight from ChatGPT (Idea #9):
- Make prediction with binary weights
- Compute residual error
- Feed residual back through same (or different) binary layer
- Repeat for K iterations

This lets binary approximate higher precision through iteration.
Memory is still 1-bit, but we trade compute for quality.

Also implements Idea #2 (Duty-cycle): Different weights active at different steps.
"""

import numpy as np
from typing import Tuple


class IterativeRefinementBinary:
    """Binary with iterative error-feedback refinement.

    y = W1·x + W2·(y_target - W1·x) + W3·(y_target - W1·x - W2·err1) + ...

    At inference: y = W1·x + W2·W1·x + W3·W2·W1·x + ... (no target)
    """
    def __init__(self, d_in: int, d_out: int, n_iterations: int = 3):
        self.d_in = d_in
        self.d_out = d_out
        self.n_iter = n_iterations

        # One binary matrix per iteration
        self.Ws = []
        self.scales = []

    def train(self, X: np.ndarray, Y: np.ndarray):
        """Train iteratively, each layer corrects previous."""
        self.Ws = []
        self.scales = []
        residual = Y.copy()

        for i in range(self.n_iter):
            W_opt = np.linalg.lstsq(X, residual, rcond=None)[0].T
            W_bin = np.sign(W_opt).astype(np.float32)
            W_bin[W_bin == 0] = 1.0
            Y_pred = X @ W_bin.T
            scale = np.sum(Y_pred * residual) / (np.sum(Y_pred ** 2) + 1e-8)
            self.Ws.append(W_bin)
            self.scales.append(scale)
            residual = residual - Y_pred * scale

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = np.zeros((X.shape[0], self.d_out))
        for W, s in zip(self.Ws, self.scales):
            Y = Y + X @ (W * s).T
        return Y

    def memory_bits(self) -> int:
        return self.n_iter * self.d_in * self.d_out + 32 * self.n_iter


class IterativeBinaryFixedMemory:
    """Iterative binary at SAME memory as ternary.

    With n iterations, each layer uses (1.58/n) fraction of full size.
    Total memory = n * (d_in * d_out * 1.58/n) = d_in * d_out * 1.58 = ternary
    """
    def __init__(self, d_in: int, d_out: int, n_iterations: int = 2):
        self.d_in = d_in
        self.d_out = d_out
        self.n_iter = n_iterations

        # Each layer is smaller
        self.layer_d_out = int(d_out * 1.58 / n_iterations)

        self.Ws = []
        self.scales = []

    def train(self, X: np.ndarray, Y: np.ndarray):
        self.Ws = []
        self.scales = []

        residual = Y.copy()

        for i in range(self.n_iter):
            # Fit to current residual (with reduced output dims)
            W_opt = np.linalg.lstsq(X, residual, rcond=None)[0].T
            W_bin = np.sign(W_opt[:self.layer_d_out]).astype(np.float32)
            W_bin[W_bin == 0] = 1.0

            # Scale
            Y_pred_partial = X @ W_bin.T
            # Tile to full output
            Y_pred = np.tile(Y_pred_partial, (1, (self.d_out + self.layer_d_out - 1) // self.layer_d_out))[:, :self.d_out]

            scale = np.sum(Y_pred * residual) / (np.sum(Y_pred ** 2) + 1e-8)

            self.Ws.append(W_bin)
            self.scales.append(scale)

            residual = residual - Y_pred * scale

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = np.zeros((X.shape[0], self.d_out))
        for W, s in zip(self.Ws, self.scales):
            Y_partial = X @ (W * s).T
            Y_full = np.tile(Y_partial, (1, (self.d_out + W.shape[0] - 1) // W.shape[0]))[:, :self.d_out]
            Y = Y + Y_full
        return Y

    def memory_bits(self) -> int:
        return self.n_iter * self.d_in * self.layer_d_out + 32 * self.n_iter


class DutyCycleBinary:
    """Duty-cycle magnitude: weights active for different fractions of time.
    
    Divide weights into K groups.
    Group k has duty cycle p_k ∈ [0, 1].
    Run T micro-steps, group k active for p_k * T steps.
    
    Effective magnitude of group k = p_k (encoded in compute, not bits).
    """
    def __init__(self, d_in: int, d_out: int, n_groups: int = 4, T: int = 8):
        self.d_in = d_in
        self.d_out = d_out
        self.n_groups = n_groups
        self.T = T
        
        self.W = None  # (d_out, d_in) binary
        self.group_duty = np.ones(n_groups)  # Duty cycle per group
    
    def _get_group(self, i: int, j: int) -> int:
        """Assign weight position to group."""
        return (i * self.d_in + j) % self.n_groups
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        # Binary signs
        self.W = np.sign(W_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        
        # Learn duty cycles per group based on average magnitude
        for g in range(self.n_groups):
            mask = np.zeros_like(W_opt, dtype=bool)
            for i in range(self.d_out):
                for j in range(self.d_in):
                    if self._get_group(i, j) == g:
                        mask[i, j] = True
            
            if mask.any():
                avg_mag = np.mean(np.abs(W_opt[mask]))
                overall_avg = np.mean(np.abs(W_opt))
                self.group_duty[g] = np.clip(avg_mag / overall_avg, 0.1, 1.0)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Simulate duty-cycle execution."""
        Y = np.zeros((X.shape[0], self.d_out))
        
        # Apply group duty cycles
        W_effective = self.W.copy()
        for i in range(self.d_out):
            for j in range(self.d_in):
                g = self._get_group(i, j)
                W_effective[i, j] *= self.group_duty[g]
        
        return X @ W_effective.T
    
    def memory_bits(self) -> int:
        # Binary weights + group duty cycles (quantized to log2(T) bits each)
        weight_bits = self.d_in * self.d_out
        duty_bits = self.n_groups * int(np.ceil(np.log2(self.T)))
        return weight_bits + duty_bits


class BinaryKernelHead:
    """Binary features + tiny FP head (Idea #6).
    
    Large binary feature layer: x → h (d_in → d_feat)
    Tiny FP head: h → y (d_feat → d_out)
    
    d_feat is large, but most bits are binary.
    """
    def __init__(self, d_in: int, d_out: int, feat_mult: float = 2.0):
        self.d_in = d_in
        self.d_out = d_out
        self.d_feat = int(d_in * feat_mult)
        
        self.W_feat = None  # Binary: (d_feat, d_in)
        self.W_head = None  # FP: (d_out, d_feat)
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        # Random binary features
        self.W_feat = np.sign(np.random.randn(self.d_feat, self.d_in)).astype(np.float32)
        self.W_feat[self.W_feat == 0] = 1.0
        
        # Forward through binary
        H = X @ self.W_feat.T  # (n, d_feat)
        
        # Learn FP head
        self.W_head = np.linalg.lstsq(H, Y, rcond=None)[0].T
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        H = X @ self.W_feat.T
        return H @ self.W_head.T
    
    def memory_bits(self) -> int:
        feat_bits = self.d_feat * self.d_in * 1
        head_bits = self.d_out * self.d_feat * 32
        return feat_bits + head_bits


class TernaryBaseline:
    def __init__(self, d_in, d_out):
        self.d_in, self.d_out = d_in, d_out
        self.W = None; self.scale = 1.0
    def train(self, X, Y):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        thresh = np.percentile(np.abs(W_opt), 30)
        self.W = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        mask = self.W != 0
        if mask.any(): self.scale = np.mean(np.abs(W_opt[mask]))
    def forward(self, X): return X @ (self.W * self.scale).T
    def memory_bits(self): return int(self.d_in * self.d_out * 1.58) + 32


class BinaryBaseline:
    def __init__(self, d_in, d_out):
        self.d_in, self.d_out = d_in, d_out
        self.W = None; self.scale = 1.0
    def train(self, X, Y):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W = np.sign(W_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        Y_pred = X @ self.W.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    def forward(self, X): return X @ (self.W * self.scale).T
    def memory_bits(self): return self.d_in * self.d_out + 32


def run_experiments():
    print("=" * 80)
    print("ITERATIVE BINARY: Trade compute for precision")
    print("=" * 80)

    for d in [64, 128, 256]:
        print(f"\n{'='*70}")
        print(f"Dimension: {d}x{d}")
        ternary_bits = int(d * d * 1.58) + 32
        print(f"Ternary memory: {ternary_bits:,} bits")
        print(f"{'='*70}")

        np.random.seed(42)
        W_true = np.random.randn(d, d).astype(np.float32) * 0.5
        X_train = np.random.randn(5000, d).astype(np.float32)
        Y_train = X_train @ W_true.T + np.random.randn(5000, d) * 0.1
        X_test = np.random.randn(1000, d).astype(np.float32)
        Y_test = X_test @ W_true.T

        results = {}

        methods = [
            ('ternary', TernaryBaseline(d, d)),
            ('binary', BinaryBaseline(d, d)),
            ('iter_2', IterativeRefinementBinary(d, d, n_iterations=2)),
            ('iter_3', IterativeRefinementBinary(d, d, n_iterations=3)),
            ('iter_fix_2', IterativeBinaryFixedMemory(d, d, n_iterations=2)),
            ('iter_fix_3', IterativeBinaryFixedMemory(d, d, n_iterations=3)),
            ('duty_4g', DutyCycleBinary(d, d, n_groups=4)),
            ('duty_16g', DutyCycleBinary(d, d, n_groups=16)),
            ('kernel_2x', BinaryKernelHead(d, d, feat_mult=2.0)),
        ]

        for name, model in methods:
            model.train(X_train, Y_train)
            Y_pred = model.forward(X_test)
            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            bits = model.memory_bits()
            results[name] = {'corr': corr, 'bits': bits}

        print(f"\n{'Method':<15} {'Corr':>10} {'Memory':>12} {'Ratio':>8} {'vs Tern':>10}")
        print("-" * 58)
        ternary_corr = results['ternary']['corr']

        for name, data in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs = (data['corr'] / ternary_corr - 1) * 100
            ratio = data['bits'] / ternary_bits
            marker = "✓" if ratio <= 1.05 and data['corr'] >= ternary_corr * 0.99 else ""
            print(f"{name:<15} {data['corr']:>10.4f} {data['bits']:>10,} {ratio:>7.2f}x {vs:>+9.1f}% {marker}")


if __name__ == "__main__":
    run_experiments()

