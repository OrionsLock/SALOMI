"""Error-Optimal 1-bit: Minimize quantization error, not just match task.

Key insight: The 10% gap between binary and ternary comes from QUANTIZATION ERROR.
- Binary forces all weights to have magnitude = scale
- Ternary allows magnitude 0 or scale, better matching small weights

Solution: BLOCK-HETEROGENEOUS SCALING
- Binary signs (1 bit per weight)
- Multiple scale levels shared within blocks
- Choose scale per block to minimize error

This gives us more magnitude levels while keeping bpp near 1.0.
"""

import numpy as np
from typing import Tuple, Dict


class BlockHeterogeneousBinary:
    """Binary with block-shared heterogeneous scales.
    
    Divide matrix into blocks. Each block has:
    - Per-weight binary signs: {-1, +1}
    - Shared scale chosen from K levels to minimize error
    
    Bit budget:
    - Signs: 1 bit per weight
    - Scale indices: log2(K) bits per block
    - K scale values: negligible (K floats)
    
    For 4x4 blocks, K=4:
    - Total: 1 + 2/16 = 1.125 bpp
    - Effective: 4 magnitude levels per weight!
    """
    
    def __init__(self, d_in: int, d_out: int, block_size: int = 4, n_scales: int = 4):
        self.d_in = d_in
        self.d_out = d_out
        self.block_size = block_size
        self.n_scales = n_scales
        
        self.W_binary = None          # Binary signs
        self.scale_values = None      # The K scale values
        self.block_scale_idx = None   # Which scale each block uses
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        """Train to minimize reconstruction error."""
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        # Binary signs from optimal
        self.W_binary = np.sign(W_opt).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0
        
        # Determine scale values from magnitude distribution
        magnitudes = np.abs(W_opt)
        # Use percentiles to get scale levels
        percentiles = np.linspace(0, 100, self.n_scales + 1)[1:-1]
        scale_boundaries = [np.percentile(magnitudes, p) for p in percentiles]
        
        # Scale values are the means within each bucket
        self.scale_values = []
        prev = 0
        for boundary in scale_boundaries + [np.inf]:
            mask = (magnitudes >= prev) & (magnitudes < boundary)
            if mask.any():
                self.scale_values.append(np.mean(magnitudes[mask]))
            else:
                self.scale_values.append(boundary)
            prev = boundary
        self.scale_values = np.array(self.scale_values)
        
        # For each block, choose best scale
        n_blocks_h = (self.d_out + self.block_size - 1) // self.block_size
        n_blocks_w = (self.d_in + self.block_size - 1) // self.block_size
        self.block_scale_idx = np.zeros((n_blocks_h, n_blocks_w), dtype=int)
        
        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                i_start = bi * self.block_size
                i_end = min(i_start + self.block_size, self.d_out)
                j_start = bj * self.block_size
                j_end = min(j_start + self.block_size, self.d_in)
                
                block_signs = self.W_binary[i_start:i_end, j_start:j_end]
                block_target = W_opt[i_start:i_end, j_start:j_end]
                
                # Try each scale, pick one with minimum error
                best_scale_idx = 0
                best_error = float('inf')
                
                for k, scale in enumerate(self.scale_values):
                    block_quant = block_signs * scale
                    error = np.mean((block_quant - block_target) ** 2)
                    if error < best_error:
                        best_error = error
                        best_scale_idx = k
                
                self.block_scale_idx[bi, bj] = best_scale_idx
    
    def get_weights(self) -> np.ndarray:
        """Reconstruct weight matrix."""
        W = np.zeros((self.d_out, self.d_in), dtype=np.float32)
        n_blocks_h = (self.d_out + self.block_size - 1) // self.block_size
        n_blocks_w = (self.d_in + self.block_size - 1) // self.block_size
        
        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                i_start = bi * self.block_size
                i_end = min(i_start + self.block_size, self.d_out)
                j_start = bj * self.block_size
                j_end = min(j_start + self.block_size, self.d_in)
                
                scale = self.scale_values[self.block_scale_idx[bi, bj]]
                W[i_start:i_end, j_start:j_end] = self.W_binary[i_start:i_end, j_start:j_end] * scale
        
        return W
    
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        n_blocks = ((self.d_out + self.block_size - 1) // self.block_size) * \
                   ((self.d_in + self.block_size - 1) // self.block_size)
        
        sign_bits = n_weights
        scale_idx_bits = n_blocks * np.log2(self.n_scales)
        scale_value_bits = self.n_scales * 32
        
        return (sign_bits + scale_idx_bits + scale_value_bits) / n_weights
    
    def reconstruction_error(self, W_target: np.ndarray) -> float:
        """Compute reconstruction error."""
        W_quant = self.get_weights()
        return np.mean((W_quant - W_target) ** 2)


class ZeroAwareBinary:
    """Binary with explicit zero selection.

    KEY INSIGHT: Ternary's power comes from ZEROS, not from 3 states.

    Let's explicitly model zeros:
    - Select which weights should be zero (based on magnitude)
    - Non-zero weights are binary {-1, +1} * scale

    This IS ternary, but we encode the zero mask efficiently.
    If zeros are STRUCTURED (block/row/column), encoding is cheap.
    """

    def __init__(self, d_in: int, d_out: int, zero_fraction: float = 0.3,
                 zero_structure: str = 'block'):
        self.d_in = d_in
        self.d_out = d_out
        self.zero_fraction = zero_fraction
        self.zero_structure = zero_structure  # 'none', 'block', 'row', 'column'

        self.W_binary = None
        self.zero_mask = None  # True = zero, False = active
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Binary signs
        self.W_binary = np.sign(W_opt).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0

        # Create zero mask based on structure
        magnitudes = np.abs(W_opt)

        if self.zero_structure == 'none':
            # Per-weight (like ternary)
            threshold = np.percentile(magnitudes, self.zero_fraction * 100)
            self.zero_mask = magnitudes < threshold

        elif self.zero_structure == 'block':
            # Block-structured zeros (4x4 blocks)
            block_size = 4
            n_blocks_h = self.d_out // block_size
            n_blocks_w = self.d_in // block_size

            block_importance = np.zeros((n_blocks_h, n_blocks_w))
            for bi in range(n_blocks_h):
                for bj in range(n_blocks_w):
                    block = magnitudes[bi*block_size:(bi+1)*block_size,
                                       bj*block_size:(bj+1)*block_size]
                    block_importance[bi, bj] = np.mean(block)

            # Zero out least important blocks
            n_zero_blocks = int(n_blocks_h * n_blocks_w * self.zero_fraction)
            threshold = np.sort(block_importance.flatten())[n_zero_blocks]

            self.zero_mask = np.zeros((self.d_out, self.d_in), dtype=bool)
            for bi in range(n_blocks_h):
                for bj in range(n_blocks_w):
                    if block_importance[bi, bj] <= threshold:
                        self.zero_mask[bi*block_size:(bi+1)*block_size,
                                       bj*block_size:(bj+1)*block_size] = True

        elif self.zero_structure == 'row':
            row_importance = np.mean(magnitudes, axis=1)
            n_zero_rows = int(self.d_out * self.zero_fraction)
            threshold = np.sort(row_importance)[n_zero_rows]
            self.zero_mask = np.zeros((self.d_out, self.d_in), dtype=bool)
            for i in range(self.d_out):
                if row_importance[i] <= threshold:
                    self.zero_mask[i, :] = True

        elif self.zero_structure == 'column':
            col_importance = np.mean(magnitudes, axis=0)
            n_zero_cols = int(self.d_in * self.zero_fraction)
            threshold = np.sort(col_importance)[n_zero_cols]
            self.zero_mask = np.zeros((self.d_out, self.d_in), dtype=bool)
            for j in range(self.d_in):
                if col_importance[j] <= threshold:
                    self.zero_mask[:, j] = True

        # Scale from non-zero optimal weights
        active_mask = ~self.zero_mask
        if active_mask.any():
            self.scale = np.mean(np.abs(W_opt[active_mask]))

    def get_weights(self) -> np.ndarray:
        W = self.W_binary * self.scale
        W[self.zero_mask] = 0
        return W

    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        n_active = np.sum(~self.zero_mask)

        # Active weight signs: 1 bit each
        sign_bits = n_active

        # Zero mask encoding depends on structure
        if self.zero_structure == 'none':
            # Entropy of zero_fraction
            p = self.zero_fraction
            mask_entropy = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
            mask_bits = n_weights * mask_entropy
        elif self.zero_structure == 'block':
            # Entropy over blocks
            block_size = 4
            n_blocks = (self.d_out // block_size) * (self.d_in // block_size)
            p = self.zero_fraction
            mask_entropy = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
            mask_bits = n_blocks * mask_entropy
        else:  # row or column
            n_units = self.d_out if self.zero_structure == 'row' else self.d_in
            p = self.zero_fraction
            mask_entropy = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
            mask_bits = n_units * mask_entropy

        scale_bits = 32

        return (sign_bits + mask_bits + scale_bits) / n_weights


# =============================================================================
# BASELINES
# =============================================================================

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


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiments():
    print("=" * 80)
    print("ERROR-OPTIMAL 1-BIT: Can we close the 10% gap?")
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

        # Get optimal weights for reference
        W_opt = np.linalg.lstsq(X_train, Y_train, rcond=None)[0].T

        results = {}

        # Baselines
        for cls, name in [(Binary, 'binary'), (Ternary, 'ternary')]:
            m = cls(d, d)
            m.train(X_train, Y_train)
            Y_pred = X_test @ m.get_weights().T
            results[name] = {
                'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                'bpp': m.effective_bpp(),
                'recon_err': np.mean((m.get_weights() - W_opt) ** 2)
            }

        # Block heterogeneous
        for block_size in [2, 4, 8]:
            for n_scales in [2, 4, 8]:
                m = BlockHeterogeneousBinary(d, d, block_size=block_size, n_scales=n_scales)
                m.train(X_train, Y_train)
                Y_pred = X_test @ m.get_weights().T
                name = f'block{block_size}_s{n_scales}'
                results[name] = {
                    'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                    'bpp': m.effective_bpp(),
                    'recon_err': m.reconstruction_error(W_opt)
                }

        # Zero-aware with different structures
        for zero_frac in [0.2, 0.3, 0.4]:
            for structure in ['none', 'block', 'row', 'column']:
                m = ZeroAwareBinary(d, d, zero_fraction=zero_frac, zero_structure=structure)
                m.train(X_train, Y_train)
                Y_pred = X_test @ m.get_weights().T
                name = f'zero{int(zero_frac*100)}_{structure}'
                results[name] = {
                    'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                    'bpp': m.effective_bpp(),
                    'recon_err': np.mean((m.get_weights() - W_opt) ** 2)
                }

        # Print sorted by correlation
        print(f"\n{'Method':<20} {'Corr':>8} {'BPP':>8} {'ReconErr':>10} {'vsT':>8}")
        print("-" * 60)
        ternary_corr = results['ternary']['corr']

        for name, data in sorted(results.items(), key=lambda x: -x[1]['corr'])[:15]:
            vs = (data['corr'] / ternary_corr - 1) * 100
            marker = "✓" if data['bpp'] <= 1.58 and data['corr'] >= ternary_corr else ""
            print(f"{name:<20} {data['corr']:>8.4f} {data['bpp']:>8.3f} {data['recon_err']:>10.6f} {vs:>+7.1f}% {marker}")


if __name__ == "__main__":
    run_experiments()

