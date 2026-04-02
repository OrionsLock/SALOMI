"""Structured 1-bit Approaches: Leveraging structure to beat ternary.

Key insights:
1. Structured sparsity encodes zeros cheaply (column/row/block masks)
2. Group scaling gives magnitude variation without per-weight bits
3. The combination might beat ternary at < 1.0 bpp
"""

import numpy as np
from typing import Tuple


# =============================================================================
# APPROACH A: COLUMN-STRUCTURED SPARSITY
# =============================================================================

class ColumnSparseBinary:
    """Binary weights with column-structured zeros.
    
    Zero out entire columns (input features), encode which columns are active.
    Active columns: binary weights {-1, +1}
    
    Bit budget:
    - Column mask: ~0.003 bpp (for 30% zero columns)
    - Binary weights: 0.7 bpp (only active columns)
    - Total: ~0.70 bpp with 30% sparsity!
    """
    
    def __init__(self, d_in: int, d_out: int, keep_ratio: float = 0.7):
        self.d_in = d_in
        self.d_out = d_out
        self.keep_ratio = keep_ratio
        
        self.active_cols = None  # Boolean mask
        self.W_binary = None     # Binary weights for active columns only
        self.scale = 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        """Train by selecting important columns and binary weights."""
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        # Select columns by importance (sum of absolute weights in column)
        col_importance = np.sum(np.abs(W_opt), axis=0)
        n_keep = int(self.d_in * self.keep_ratio)
        top_cols = np.argsort(col_importance)[-n_keep:]
        
        self.active_cols = np.zeros(self.d_in, dtype=bool)
        self.active_cols[top_cols] = True
        
        # Binary weights for active columns
        W_active = W_opt[:, self.active_cols]
        self.W_binary = np.sign(W_active).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0
        
        # Optimal scale
        W_full = self._get_full_weights()
        Y_pred = X @ W_full.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    
    def _get_full_weights(self) -> np.ndarray:
        """Expand sparse weights to full matrix."""
        W_full = np.zeros((self.d_out, self.d_in), dtype=np.float32)
        W_full[:, self.active_cols] = self.W_binary * self.scale
        return W_full
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self._get_full_weights().T
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        n_active = np.sum(self.active_cols)
        
        # Column mask: entropy of keep_ratio
        p = self.keep_ratio
        mask_entropy = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
        mask_bits = self.d_in * mask_entropy
        
        # Binary weights for active columns
        weight_bits = self.d_out * n_active
        
        # Scale
        scale_bits = 32
        
        return (mask_bits + weight_bits + scale_bits) / n_weights


# =============================================================================
# APPROACH B: GROUP-SCALED BINARY
# =============================================================================

class GroupScaledBinary:
    """Binary weights with per-group scaling.
    
    Divide weights into groups, each with its own scale.
    This gives magnitude variation without per-weight magnitude bits.
    
    w[i,j] = sign(w[i,j]) * scale[group(i,j)]
    """
    
    def __init__(self, d_in: int, d_out: int, n_groups: int = 16):
        self.d_in = d_in
        self.d_out = d_out
        self.n_groups = n_groups
        
        self.W_binary = None
        self.group_scales = np.ones(n_groups, dtype=np.float32)
    
    def _get_group_idx(self, i: int, j: int) -> int:
        """Assign weight to group (block-based assignment)."""
        block_h = self.d_out // int(np.sqrt(self.n_groups))
        block_w = self.d_in // int(np.sqrt(self.n_groups))
        gi = min(i // max(block_h, 1), int(np.sqrt(self.n_groups)) - 1)
        gj = min(j // max(block_w, 1), int(np.sqrt(self.n_groups)) - 1)
        return gi * int(np.sqrt(self.n_groups)) + gj
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        """Train binary weights and group scales."""
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        # Binary weights
        self.W_binary = np.sign(W_opt).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0
        
        # Compute optimal scale per group
        n_sqrt = int(np.sqrt(self.n_groups))
        block_h = max(self.d_out // n_sqrt, 1)
        block_w = max(self.d_in // n_sqrt, 1)
        
        for g in range(self.n_groups):
            gi, gj = g // n_sqrt, g % n_sqrt
            i_start, i_end = gi * block_h, min((gi + 1) * block_h, self.d_out)
            j_start, j_end = gj * block_w, min((gj + 1) * block_w, self.d_in)
            
            W_block = W_opt[i_start:i_end, j_start:j_end]
            self.group_scales[g] = np.mean(np.abs(W_block)) if W_block.size > 0 else 1.0
    
    def get_weights(self) -> np.ndarray:
        """Get full weight matrix with group scales applied."""
        n_sqrt = int(np.sqrt(self.n_groups))
        block_h = max(self.d_out // n_sqrt, 1)
        block_w = max(self.d_in // n_sqrt, 1)
        
        W = self.W_binary.copy()
        for g in range(self.n_groups):
            gi, gj = g // n_sqrt, g % n_sqrt
            i_start, i_end = gi * block_h, min((gi + 1) * block_h, self.d_out)
            j_start, j_end = gj * block_w, min((gj + 1) * block_w, self.d_in)
            W[i_start:i_end, j_start:j_end] *= self.group_scales[g]
        
        return W
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        weight_bits = n_weights  # 1 bit per weight
        scale_bits = self.n_groups * 32  # FP32 per group (could be quantized)
        return (weight_bits + scale_bits) / n_weights


# =============================================================================
# APPROACH C: HYBRID STRUCTURED (Column Sparsity + Group Scaling)
# =============================================================================

class HybridStructuredBinary:
    """Best of both: column sparsity + group scaling.

    1. Zero out unimportant columns (like ternary zeros)
    2. Group scales for remaining (like magnitude levels)

    Bit budget:
    - Column mask: ~0.003 bpp
    - Binary weights: 0.7 * 1 = 0.7 bpp
    - Group scales: ~0.008 bpp (16 groups)
    - Total: ~0.71 bpp
    """

    def __init__(self, d_in: int, d_out: int,
                 keep_ratio: float = 0.7, n_groups: int = 16):
        self.d_in = d_in
        self.d_out = d_out
        self.keep_ratio = keep_ratio
        self.n_groups = n_groups

        self.active_cols = None
        self.W_binary = None
        self.group_scales = np.ones(n_groups, dtype=np.float32)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Select columns
        col_importance = np.sum(np.abs(W_opt), axis=0)
        n_keep = int(self.d_in * self.keep_ratio)
        top_cols = np.argsort(col_importance)[-n_keep:]

        self.active_cols = np.zeros(self.d_in, dtype=bool)
        self.active_cols[top_cols] = True

        # Binary weights
        self.W_binary = np.sign(W_opt).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0

        # Group scales (over full matrix)
        n_sqrt = int(np.sqrt(self.n_groups))
        block_h = max(self.d_out // n_sqrt, 1)
        block_w = max(self.d_in // n_sqrt, 1)

        for g in range(self.n_groups):
            gi, gj = g // n_sqrt, g % n_sqrt
            i_start, i_end = gi * block_h, min((gi + 1) * block_h, self.d_out)
            j_start, j_end = gj * block_w, min((gj + 1) * block_w, self.d_in)

            W_block = W_opt[i_start:i_end, j_start:j_end]
            self.group_scales[g] = np.mean(np.abs(W_block)) if W_block.size > 0 else 1.0

    def get_weights(self) -> np.ndarray:
        n_sqrt = int(np.sqrt(self.n_groups))
        block_h = max(self.d_out // n_sqrt, 1)
        block_w = max(self.d_in // n_sqrt, 1)

        W = self.W_binary.copy()

        # Apply group scales
        for g in range(self.n_groups):
            gi, gj = g // n_sqrt, g % n_sqrt
            i_start, i_end = gi * block_h, min((gi + 1) * block_h, self.d_out)
            j_start, j_end = gj * block_w, min((gj + 1) * block_w, self.d_in)
            W[i_start:i_end, j_start:j_end] *= self.group_scales[g]

        # Zero out inactive columns
        W[:, ~self.active_cols] = 0

        return W

    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        n_active = np.sum(self.active_cols)

        p = self.keep_ratio
        mask_entropy = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
        mask_bits = self.d_in * mask_entropy

        weight_bits = self.d_out * n_active  # Only active columns
        scale_bits = self.n_groups * 32

        return (mask_bits + weight_bits + scale_bits) / n_weights


# =============================================================================
# BASELINES
# =============================================================================

class BinaryBaseline:
    def __init__(self, d_in, d_out):
        self.W = None
        self.scale = 1.0
        self.d_in, self.d_out = d_in, d_out

    def train(self, X, Y):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W = np.sign(W_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        Y_pred = X @ self.W.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def get_weights(self): return self.W * self.scale
    def effective_bpp(self): return 1.0


class TernaryBaseline:
    def __init__(self, d_in, d_out, zero_pct=0.3):
        self.W = None
        self.scale = 1.0
        self.zero_pct = zero_pct
        self.d_in, self.d_out = d_in, d_out

    def train(self, X, Y):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        thresh = np.percentile(np.abs(W_opt), self.zero_pct * 100)
        self.W = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        mask = self.W != 0
        if mask.any():
            self.scale = np.mean(np.abs(W_opt[mask]))

    def get_weights(self): return self.W * self.scale
    def effective_bpp(self): return 1.58


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiments(dims=[64, 128, 256]):
    print("=" * 80)
    print("STRUCTURED 1-BIT: Can structured sparsity + group scaling beat ternary?")
    print("=" * 80)

    for d in dims:
        print(f"\n{'='*60}")
        print(f"Dimension: {d}x{d}")
        print(f"{'='*60}")

        # Generate task
        np.random.seed(42)
        W_true = np.random.randn(d, d).astype(np.float32) * 0.5
        X_train = np.random.randn(5000, d).astype(np.float32)
        Y_train = X_train @ W_true.T + np.random.randn(5000, d) * 0.1
        X_test = np.random.randn(1000, d).astype(np.float32)
        Y_test = X_test @ W_true.T

        results = {}

        # Baselines
        binary = BinaryBaseline(d, d)
        binary.train(X_train, Y_train)
        Y_pred = X_test @ binary.get_weights().T
        results['binary'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bpp': binary.effective_bpp()
        }

        ternary = TernaryBaseline(d, d)
        ternary.train(X_train, Y_train)
        Y_pred = X_test @ ternary.get_weights().T
        results['ternary'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bpp': ternary.effective_bpp()
        }

        # Column-sparse binary
        for keep in [0.5, 0.6, 0.7, 0.8]:
            cs = ColumnSparseBinary(d, d, keep_ratio=keep)
            cs.train(X_train, Y_train)
            Y_pred = cs.forward(X_test)
            results[f'col_sparse_{int(keep*100)}'] = {
                'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                'bpp': cs.effective_bpp()
            }

        # Group-scaled binary
        for n_groups in [4, 16, 64]:
            gs = GroupScaledBinary(d, d, n_groups=n_groups)
            gs.train(X_train, Y_train)
            Y_pred = X_test @ gs.get_weights().T
            results[f'group_scale_{n_groups}'] = {
                'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                'bpp': gs.effective_bpp()
            }

        # Hybrid structured
        for keep in [0.6, 0.7, 0.8]:
            for n_groups in [16, 64]:
                hs = HybridStructuredBinary(d, d, keep_ratio=keep, n_groups=n_groups)
                hs.train(X_train, Y_train)
                Y_pred = X_test @ hs.get_weights().T
                results[f'hybrid_{int(keep*100)}_{n_groups}g'] = {
                    'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                    'bpp': hs.effective_bpp()
                }

        # Print results sorted by correlation
        print(f"\n{'Method':<25} {'Corr':>10} {'BPP':>10} {'vs Ternary':>12}")
        print("-" * 60)
        ternary_corr = results['ternary']['corr']

        for name, data in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (data['corr'] / ternary_corr - 1) * 100
            marker = "✓ BEATS!" if data['bpp'] < 1.58 and data['corr'] >= ternary_corr else ""
            print(f"{name:<25} {data['corr']:>10.4f} {data['bpp']:>10.3f} {vs_tern:>+11.1f}% {marker}")

        # Find best under 1.1 bpp
        sub_1_1 = [(n, d) for n, d in results.items() if d['bpp'] <= 1.1]
        if sub_1_1:
            best = max(sub_1_1, key=lambda x: x[1]['corr'])
            print(f"\n→ Best under 1.1 bpp: {best[0]} ({best[1]['corr']:.4f} @ {best[1]['bpp']:.3f} bpp)")
            print(f"  vs Ternary: {(best[1]['corr']/ternary_corr - 1)*100:+.1f}%")


if __name__ == "__main__":
    run_experiments()

