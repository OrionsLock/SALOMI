"""Fair Fixed Memory Comparison: Binary vs Ternary at SAME memory.

The key is: at fixed memory M bits, binary can have MORE parameters.
But we must count ALL bits fairly (including any FP overhead).

Strategy: Use ONLY 1-bit storage, no FP projections.
"""

import numpy as np
from typing import Tuple


class TernaryBaseline:
    """Ternary: d_in × d_out weights at 1.58 bpp."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W = None
        self.scale = 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        thresh = np.percentile(np.abs(W_opt), 30)
        self.W = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        mask = self.W != 0
        if mask.any():
            self.scale = np.mean(np.abs(W_opt[mask]))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W * self.scale).T
    
    def memory_bits(self) -> int:
        return int(self.d_in * self.d_out * 1.58) + 32  # +32 for scale


class BinaryWider:
    """Binary: d_in × (1.58 × d_out) weights at 1.0 bpp, then reduce output.
    
    Key trick: Use the LAST output dimensions as the "real" output.
    No FP projection - just train for the last d_out outputs.
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.d_wide = int(d_out * 1.58)  # 58% more outputs
        self.W = None  # (d_wide, d_in) binary
        self.scale = 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        # Pad Y to wider output (we only care about first d_out)
        Y_wide = np.zeros((Y.shape[0], self.d_wide))
        Y_wide[:, :self.d_out] = Y
        
        W_opt = np.linalg.lstsq(X, Y_wide, rcond=None)[0].T
        self.W = np.sign(W_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        
        # Scale for the actual output dimensions
        Y_pred = X @ self.W[:self.d_out].T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        # Only use first d_out outputs
        return X @ (self.W[:self.d_out] * self.scale).T
    
    def memory_bits(self) -> int:
        return self.d_in * self.d_wide * 1 + 32  # 1 bit per weight + scale


class BinaryTwoLayer:
    """Binary two-layer: d_in → d_h → d_out, both binary.
    
    Total bits = d_in * d_h + d_h * d_out = same as ternary single layer.
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        # Choose d_h such that total bits ≈ ternary bits
        # ternary_bits = d_in * d_out * 1.58
        # binary_bits = d_in * d_h + d_h * d_out
        # Solve for d_h: d_h = (1.58 * d_in * d_out) / (d_in + d_out)
        self.d_h = int(1.58 * d_in * d_out / (d_in + d_out))
        
        self.W1 = None  # (d_h, d_in)
        self.W2 = None  # (d_out, d_h)
        self.scale1 = 1.0
        self.scale2 = 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        # Random init for W1
        W1_opt = np.random.randn(self.d_h, self.d_in) * 0.1
        self.W1 = np.sign(W1_opt).astype(np.float32)
        self.W1[self.W1 == 0] = 1.0
        
        # Forward through W1
        H = X @ self.W1.T  # (n, d_h)
        H = np.maximum(0, H)  # ReLU
        
        # Optimize W2
        W2_opt = np.linalg.lstsq(H, Y, rcond=None)[0].T
        self.W2 = np.sign(W2_opt).astype(np.float32)
        self.W2[self.W2 == 0] = 1.0
        
        # Scale
        Y_pred = H @ self.W2.T
        self.scale2 = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        H = X @ self.W1.T
        H = np.maximum(0, H)
        return H @ (self.W2 * self.scale2).T
    
    def memory_bits(self) -> int:
        return self.d_in * self.d_h + self.d_h * self.d_out + 64  # +scales


class BinaryResidualStack:
    """Stack of binary layers with residual connections.
    
    y = x @ W1.T + x @ W2.T + ... (same memory budget as ternary)
    """
    def __init__(self, d_in: int, d_out: int, n_layers: int = 2):
        self.d_in = d_in
        self.d_out = d_out
        self.n_layers = n_layers
        # Total bits = n_layers * d_in * d_out ≈ 1.58 * d_in * d_out
        # So each layer gets d_in * d_out * (1.58 / n_layers) effective dims
        self.layer_d_out = int(d_out * 1.58 / n_layers)
        
        self.Ws = []  # List of binary weight matrices
        self.scales = []
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        residual = Y.copy()
        
        self.Ws = []
        self.scales = []
        
        for i in range(self.n_layers):
            # Fit this layer to residual
            W_opt = np.linalg.lstsq(X, residual, rcond=None)[0].T
            
            # Take subset of outputs
            W_layer = np.sign(W_opt[:self.layer_d_out]).astype(np.float32)
            W_layer[W_layer == 0] = 1.0
            
            # Scale
            Y_pred = X @ W_layer.T
            
            # Project back to d_out (simple mean for now)
            if self.layer_d_out != self.d_out:
                # Repeat to match d_out
                n_repeat = (self.d_out + self.layer_d_out - 1) // self.layer_d_out
                Y_pred = np.tile(Y_pred, (1, n_repeat))[:, :self.d_out]
            
            scale = np.sum(Y_pred * residual) / (np.sum(Y_pred ** 2) + 1e-8)
            
            self.Ws.append(W_layer)
            self.scales.append(scale)
            
            # Update residual
            residual = residual - Y_pred * scale
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = np.zeros((X.shape[0], self.d_out))
        
        for W, s in zip(self.Ws, self.scales):
            Y_layer = X @ W.T
            if W.shape[0] != self.d_out:
                n_repeat = (self.d_out + W.shape[0] - 1) // W.shape[0]
                Y_layer = np.tile(Y_layer, (1, n_repeat))[:, :self.d_out]
            Y = Y + Y_layer * s
        
        return Y
    
    def memory_bits(self) -> int:
        return self.n_layers * self.d_in * self.layer_d_out + 32 * self.n_layers


class BinaryStandard:
    """Standard binary (for reference - uses LESS memory than ternary)."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W = None
        self.scale = 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W = np.sign(W_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        Y_pred = X @ self.W.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W * self.scale).T
    
    def memory_bits(self) -> int:
        return self.d_in * self.d_out + 32


class BinaryHashedWider:
    """Use hashed weight sharing to get more width at same memory.

    Idea: Hash many (i,j) positions to fewer actual stored weights.
    This lets us have a WIDER model with the same number of stored bits.
    """
    def __init__(self, d_in: int, d_out: int, width_multiplier: float = 2.0):
        self.d_in = d_in
        self.d_out = d_out
        self.d_wide = int(d_out * width_multiplier)

        # Actual stored weights: same as ternary = d_in * d_out * 1.58
        self.n_stored = int(d_in * d_out * 1.58)

        self.W_stored = None  # (n_stored,) binary values
        self.scale = 1.0

    def _hash(self, i: int, j: int) -> int:
        """Hash position (i,j) to stored weight index."""
        return (i * 31337 + j * 7919) % self.n_stored

    def _get_weight_matrix(self) -> np.ndarray:
        """Expand hashed weights to full matrix."""
        W = np.zeros((self.d_wide, self.d_in), dtype=np.float32)
        for i in range(self.d_wide):
            for j in range(self.d_in):
                idx = self._hash(i, j)
                W[i, j] = self.W_stored[idx]
        return W

    def train(self, X: np.ndarray, Y: np.ndarray):
        # Optimal full weights
        Y_wide = np.zeros((Y.shape[0], self.d_wide))
        Y_wide[:, :self.d_out] = Y
        W_opt = np.linalg.lstsq(X, Y_wide, rcond=None)[0].T

        # Average optimal weights that hash to same index
        self.W_stored = np.zeros(self.n_stored, dtype=np.float32)
        counts = np.zeros(self.n_stored)

        for i in range(self.d_wide):
            for j in range(self.d_in):
                idx = self._hash(i, j)
                self.W_stored[idx] += W_opt[i, j]
                counts[idx] += 1

        self.W_stored = np.sign(self.W_stored / (counts + 1e-8))
        self.W_stored[self.W_stored == 0] = 1.0

        # Scale
        W_full = self._get_weight_matrix()
        Y_pred = X @ W_full[:self.d_out].T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        W_full = self._get_weight_matrix()
        return X @ (W_full[:self.d_out] * self.scale).T

    def memory_bits(self) -> int:
        return self.n_stored + 32


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiments():
    print("=" * 80)
    print("FAIR MEMORY COMPARISON: Binary vs Ternary at EQUAL memory budget")
    print("=" * 80)

    for d in [64, 128, 256]:
        print(f"\n{'='*70}")
        print(f"Dimension: {d}x{d}")
        ternary_bits = int(d * d * 1.58) + 32
        print(f"Memory budget: {ternary_bits:,} bits = {ternary_bits/8/1024:.2f} KB")
        print(f"{'='*70}")

        np.random.seed(42)
        W_true = np.random.randn(d, d).astype(np.float32) * 0.5
        X_train = np.random.randn(5000, d).astype(np.float32)
        Y_train = X_train @ W_true.T + np.random.randn(5000, d) * 0.1
        X_test = np.random.randn(1000, d).astype(np.float32)
        Y_test = X_test @ W_true.T

        results = {}

        # All methods
        methods = [
            ('ternary', TernaryBaseline(d, d)),
            ('binary_std', BinaryStandard(d, d)),
            ('binary_wider', BinaryWider(d, d)),
            ('binary_2layer', BinaryTwoLayer(d, d)),
            ('binary_resid_2', BinaryResidualStack(d, d, n_layers=2)),
            ('binary_resid_3', BinaryResidualStack(d, d, n_layers=3)),
            ('binary_hashed', BinaryHashedWider(d, d, width_multiplier=2.0)),
        ]

        for name, model in methods:
            model.train(X_train, Y_train)
            Y_pred = model.forward(X_test)
            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            bits = model.memory_bits()
            results[name] = {'corr': corr, 'bits': bits}

        # Print results
        print(f"\n{'Method':<18} {'Corr':>10} {'Memory':>12} {'Ratio':>8} {'vs Tern':>10}")
        print("-" * 62)
        ternary_corr = results['ternary']['corr']
        ternary_bits = results['ternary']['bits']

        for name, data in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs = (data['corr'] / ternary_corr - 1) * 100
            ratio = data['bits'] / ternary_bits
            marker = ""
            if ratio <= 1.05 and data['corr'] >= ternary_corr:
                marker = "✓ BEATS!"
            elif ratio <= 1.05 and data['corr'] >= ternary_corr * 0.98:
                marker = "~ close"
            print(f"{name:<18} {data['corr']:>10.4f} {data['bits']:>10,} {ratio:>7.2f}x {vs:>+9.1f}% {marker}")

        # Summary
        print(f"\n→ Ternary correlation: {ternary_corr:.4f}")
        fair_methods = [(n, d) for n, d in results.items()
                        if d['bits'] <= ternary_bits * 1.05 and n != 'ternary']
        if fair_methods:
            best = max(fair_methods, key=lambda x: x[1]['corr'])
            gap = (best[1]['corr'] / ternary_corr - 1) * 100
            print(f"→ Best binary at same memory: {best[0]} ({best[1]['corr']:.4f}, {gap:+.1f}%)")


if __name__ == "__main__":
    run_experiments()

