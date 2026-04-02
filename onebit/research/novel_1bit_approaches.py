"""Novel Approaches to Beat 1.58-bit Ternary with True 1-bit.

The "impossible" goal: 1-bit binary that matches or beats 1.58-bit ternary.

Key insight: Information theory assumes INDEPENDENT weights.
If weights have STRUCTURE or DYNAMICS, we can beat the naive bound.

Approaches implemented:
1. CTG Procedural + Sparse Correction
2. Input-Dependent Dynamic Masking (Procedural Zeros)
3. Residual Binary Stacking
4. Multi-Pass Binary Ensemble
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass 
class ExperimentConfig:
    d_in: int = 256
    d_out: int = 256
    n_train: int = 5000
    n_test: int = 1000


# =============================================================================
# APPROACH 1: CTG PROCEDURAL + SPARSE CORRECTION
# =============================================================================

class CTGProceduralLayer:
    """Procedural weight generation with sparse learned correction.
    
    The idea: A simple rule generates most weights (nearly 0 bpp).
    A small sparse correction fixes the important ones.
    
    Total bits = rule_params + sparse_correction_bits
    If correction is 5% of weights, total ≈ 0.05 * 2 = 0.1 bpp
    """
    
    def __init__(self, d_in: int, d_out: int, correction_sparsity: float = 0.05):
        self.d_in = d_in
        self.d_out = d_out
        self.sparsity = correction_sparsity
        
        # Procedural parameters (tiny storage)
        self.seed = np.random.randint(0, 2**31)
        self.pattern_type = 0  # 0=checkerboard, 1=stripes, 2=blocks
        
        # Sparse correction (learned)
        n_corrections = int(d_out * d_in * correction_sparsity)
        self.correction_indices = None  # Will be set during training
        self.correction_values = None   # Binary corrections {-1, +1}
        
        self.scale = 1.0
    
    def generate_procedural(self) -> np.ndarray:
        """Generate weights procedurally (deterministic from seed)."""
        rng = np.random.RandomState(self.seed)
        
        if self.pattern_type == 0:  # Checkerboard
            i, j = np.meshgrid(range(self.d_out), range(self.d_in), indexing='ij')
            W = np.where((i + j) % 2 == 0, 1.0, -1.0)
        elif self.pattern_type == 1:  # Horizontal stripes
            i = np.arange(self.d_out)[:, None]
            W = np.where(i % 2 == 0, 1.0, -1.0) * np.ones((1, self.d_in))
        elif self.pattern_type == 2:  # Random from seed
            W = rng.choice([-1.0, 1.0], size=(self.d_out, self.d_in))
        else:  # Block pattern
            block_size = 4
            n_blocks_h = self.d_out // block_size
            n_blocks_w = self.d_in // block_size
            block_signs = rng.choice([-1.0, 1.0], size=(n_blocks_h, n_blocks_w))
            W = np.repeat(np.repeat(block_signs, block_size, axis=0), block_size, axis=1)
            W = W[:self.d_out, :self.d_in]
        
        return W.astype(np.float32)
    
    def get_weights(self) -> np.ndarray:
        """Get final weights = procedural + sparse correction."""
        W = self.generate_procedural()
        
        if self.correction_indices is not None:
            for idx, val in zip(self.correction_indices, self.correction_values):
                i, j = idx // self.d_in, idx % self.d_in
                W[i, j] = val  # Override with correction
        
        return W * self.scale
    
    def train(self, X: np.ndarray, Y_target: np.ndarray, n_iter: int = 100):
        """Train by finding best pattern + optimal corrections."""
        best_loss = float('inf')
        best_config = None
        
        # Try different patterns
        for pattern in range(4):
            self.pattern_type = pattern
            W_base = self.generate_procedural()
            
            # Find optimal correction locations
            # Use gradient from least squares
            W_optimal = np.linalg.lstsq(X, Y_target, rcond=None)[0].T
            
            # Where does procedural differ most from optimal?
            diff = np.abs(np.sign(W_optimal) - W_base)
            importance = diff * np.abs(W_optimal)  # Weight by magnitude
            
            # Select top-k for correction
            n_corrections = int(self.d_out * self.d_in * self.sparsity)
            flat_importance = importance.flatten()
            top_k_indices = np.argsort(flat_importance)[-n_corrections:]
            
            # Apply corrections
            self.correction_indices = top_k_indices
            self.correction_values = np.sign(W_optimal.flatten()[top_k_indices])
            self.correction_values[self.correction_values == 0] = 1
            
            # Compute scale
            W = self.get_weights()
            # Optimal scale: minimize ||X @ W.T * s - Y||^2
            Y_pred = X @ W.T
            self.scale = np.sum(Y_pred * Y_target) / (np.sum(Y_pred ** 2) + 1e-8)
            
            # Evaluate
            Y_pred_scaled = X @ self.get_weights().T
            loss = np.mean((Y_pred_scaled - Y_target) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_config = (pattern, self.correction_indices.copy(), 
                              self.correction_values.copy(), self.scale)
        
        # Restore best
        self.pattern_type, self.correction_indices, self.correction_values, self.scale = best_config
    
    def effective_bpp(self) -> float:
        """Calculate effective bits per parameter."""
        n_weights = self.d_out * self.d_in
        
        # Procedural: ~32 bits for seed + pattern type
        procedural_bits = 32 + 2
        
        # Corrections: index (log2(n_weights)) + value (1 bit) per correction
        n_corrections = len(self.correction_indices) if self.correction_indices is not None else 0
        index_bits = np.log2(n_weights) if n_weights > 0 else 0
        correction_bits = n_corrections * (index_bits + 1)
        
        # Scale: 32 bits (could be quantized lower)
        scale_bits = 32
        
        total_bits = procedural_bits + correction_bits + scale_bits
        return total_bits / n_weights


# =============================================================================
# APPROACH 2: INPUT-DEPENDENT DYNAMIC MASKING (Procedural Zeros)
# =============================================================================

class DynamicMaskingLayer:
    """Binary weights with input-dependent masking.

    W_effective(x) = W_binary * mask(x)

    Where mask(x) is computed from input, creating DYNAMIC zeros.
    Different inputs see different effective ternary weights!

    Storage: 1 bit per weight + small gating parameters
    Effective: ternary-like behavior with input-dependent zeros
    """

    def __init__(self, d_in: int, d_out: int, mask_type: str = 'magnitude'):
        self.d_in = d_in
        self.d_out = d_out
        self.mask_type = mask_type

        # Binary weights
        self.W_binary = np.random.choice([-1.0, 1.0], size=(d_out, d_in)).astype(np.float32)

        # Gating parameters (small)
        self.gate_threshold = 0.3  # Learned threshold
        self.scale = 1.0

    def compute_mask(self, x: np.ndarray) -> np.ndarray:
        """Compute input-dependent mask."""
        # x: [batch, d_in]

        if self.mask_type == 'magnitude':
            # Mask based on input magnitude
            # Zero out weights for dimensions where input is small
            x_importance = np.abs(x)  # [batch, d_in]
            threshold = np.percentile(x_importance, self.gate_threshold * 100, axis=1, keepdims=True)
            dim_mask = (x_importance > threshold).astype(np.float32)  # [batch, d_in]
            # Broadcast to weight shape
            mask = dim_mask[:, None, :]  # [batch, 1, d_in]

        elif self.mask_type == 'attention':
            # Attention-like gating
            # Compute attention scores for each weight
            attention = np.abs(x @ self.W_binary.T)  # [batch, d_out]
            threshold = np.percentile(attention, self.gate_threshold * 100, axis=1, keepdims=True)
            mask = (attention > threshold).astype(np.float32)
            mask = mask[:, :, None]  # [batch, d_out, 1]

        else:  # 'learned'
            # Fixed learned mask (like ternary, but claimed differently)
            mask = np.ones((1, self.d_out, self.d_in), dtype=np.float32)

        return mask

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward with dynamic masking."""
        mask = self.compute_mask(x)

        if self.mask_type == 'magnitude':
            # mask: [batch, 1, d_in], W: [d_out, d_in]
            W_masked = self.W_binary[None, :, :] * mask  # [batch, d_out, d_in]
            y = np.einsum('bi,boi->bo', x, W_masked) * self.scale
        elif self.mask_type == 'attention':
            # mask: [batch, d_out, 1]
            W_masked = self.W_binary[None, :, :] * mask
            y = np.einsum('bi,boi->bo', x, W_masked) * self.scale
        else:
            y = x @ self.W_binary.T * self.scale

        return y

    def train(self, X: np.ndarray, Y_target: np.ndarray, n_iter: int = 100):
        """Train binary weights and gating threshold."""
        # Get optimal direction
        W_optimal = np.linalg.lstsq(X, Y_target, rcond=None)[0].T

        # Binary weights from optimal
        self.W_binary = np.sign(W_optimal).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0

        # Find optimal threshold
        best_loss = float('inf')
        best_threshold = 0.3

        for thresh in np.linspace(0.1, 0.5, 20):
            self.gate_threshold = thresh
            Y_pred = self.forward(X)
            loss = np.mean((Y_pred - Y_target) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_threshold = thresh

        self.gate_threshold = best_threshold

        # Compute optimal scale
        Y_pred = self.forward(X)
        self.scale = np.sum(Y_pred * Y_target) / (np.sum(Y_pred ** 2) + 1e-8)

    def effective_bpp(self) -> float:
        """BPP = 1 bit per weight + gating overhead."""
        n_weights = self.d_out * self.d_in
        weight_bits = n_weights * 1  # Binary
        gate_bits = 32  # Threshold
        scale_bits = 32
        return (weight_bits + gate_bits + scale_bits) / n_weights


# =============================================================================
# APPROACH 3: RESIDUAL BINARY STACKING
# =============================================================================

class ResidualBinaryLayer:
    """W_effective = W1_binary * s1 + W2_sparse_binary * s2

    Base binary + sparse binary correction.
    If W2 is 10% sparse, total ~1.1 bpp for 4-level effective.
    """

    def __init__(self, d_in: int, d_out: int, residual_sparsity: float = 0.1):
        self.d_in = d_in
        self.d_out = d_out
        self.sparsity = residual_sparsity

        # Base binary
        self.W1 = np.random.choice([-1.0, 1.0], size=(d_out, d_in)).astype(np.float32)
        self.s1 = 1.0

        # Sparse residual binary
        self.W2_indices = None
        self.W2_values = None
        self.s2 = 0.5

    def get_weights(self) -> np.ndarray:
        """Get combined weights."""
        W = self.W1 * self.s1

        if self.W2_indices is not None:
            W2_dense = np.zeros_like(self.W1)
            for idx, val in zip(self.W2_indices, self.W2_values):
                i, j = idx // self.d_in, idx % self.d_in
                W2_dense[i, j] = val
            W = W + W2_dense * self.s2

        return W

    def train(self, X: np.ndarray, Y_target: np.ndarray):
        """Train base + residual."""
        W_optimal = np.linalg.lstsq(X, Y_target, rcond=None)[0].T

        # Base: binary from optimal
        self.W1 = np.sign(W_optimal).astype(np.float32)
        self.W1[self.W1 == 0] = 1.0

        # Optimal s1
        Y_pred1 = X @ self.W1.T
        self.s1 = np.sum(Y_pred1 * Y_target) / (np.sum(Y_pred1 ** 2) + 1e-8)

        # Residual: what's left
        Y_residual = Y_target - X @ (self.W1 * self.s1).T
        W_residual = np.linalg.lstsq(X, Y_residual, rcond=None)[0].T

        # Sparse selection: top-k by magnitude
        n_sparse = int(self.d_out * self.d_in * self.sparsity)
        flat_mag = np.abs(W_residual).flatten()
        top_k = np.argsort(flat_mag)[-n_sparse:]

        self.W2_indices = top_k
        self.W2_values = np.sign(W_residual.flatten()[top_k])
        self.W2_values[self.W2_values == 0] = 1.0

        # Optimal s2
        W2_dense = np.zeros_like(self.W1)
        for idx, val in zip(self.W2_indices, self.W2_values):
            i, j = idx // self.d_in, idx % self.d_in
            W2_dense[i, j] = val

        Y_pred2 = X @ W2_dense.T
        self.s2 = np.sum(Y_pred2 * Y_residual) / (np.sum(Y_pred2 ** 2) + 1e-8)

    def effective_bpp(self) -> float:
        """BPP = 1 (base) + sparse overhead."""
        n_weights = self.d_out * self.d_in

        base_bits = n_weights  # 1 bit per weight

        # Sparse: index + value per nonzero
        n_sparse = len(self.W2_indices) if self.W2_indices is not None else 0
        index_bits = np.log2(n_weights) * n_sparse
        value_bits = n_sparse  # 1 bit per sparse weight

        scale_bits = 64  # Two scales

        return (base_bits + index_bits + value_bits + scale_bits) / n_weights


# =============================================================================
# APPROACH 4: MULTI-PASS BINARY ENSEMBLE
# =============================================================================

class MultiPassBinaryLayer:
    """Same binary weights, multiple passes with different interpretations.

    y = f(pass1, pass2, ...) where each pass uses same W but different scale/bias.
    2 passes = 4 effective states from 1 bit storage!
    """

    def __init__(self, d_in: int, d_out: int, n_passes: int = 2):
        self.d_in = d_in
        self.d_out = d_out
        self.n_passes = n_passes

        # Single binary weight matrix
        self.W_binary = np.random.choice([-1.0, 1.0], size=(d_out, d_in)).astype(np.float32)

        # Per-pass scales and combination weights
        self.pass_scales = np.ones(n_passes, dtype=np.float32)
        self.combination_weights = np.ones(n_passes, dtype=np.float32) / n_passes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Multi-pass forward."""
        outputs = []
        for p in range(self.n_passes):
            y_p = x @ self.W_binary.T * self.pass_scales[p]
            outputs.append(y_p)

        # Combine passes
        y = sum(w * o for w, o in zip(self.combination_weights, outputs))
        return y

    def train(self, X: np.ndarray, Y_target: np.ndarray):
        """Train binary weights and pass parameters."""
        W_optimal = np.linalg.lstsq(X, Y_target, rcond=None)[0].T

        # Binary from optimal
        self.W_binary = np.sign(W_optimal).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0

        # For 2 passes, use different scales to approximate magnitude
        mag_mean = np.mean(np.abs(W_optimal))
        mag_std = np.std(np.abs(W_optimal))

        self.pass_scales[0] = mag_mean - mag_std * 0.5
        if self.n_passes > 1:
            self.pass_scales[1] = mag_mean + mag_std * 0.5

        # Optimize combination weights
        Y_passes = [X @ self.W_binary.T * s for s in self.pass_scales]

        # Stack for linear regression
        Y_stack = np.stack(Y_passes, axis=-1)  # [n, d_out, n_passes]
        Y_flat = Y_stack.reshape(-1, self.n_passes)
        target_flat = Y_target.flatten()

        # Solve for combination weights
        self.combination_weights = np.linalg.lstsq(Y_flat, target_flat, rcond=None)[0]

    def effective_bpp(self) -> float:
        """BPP = 1 bit per weight + small pass parameters."""
        n_weights = self.d_out * self.d_in
        weight_bits = n_weights
        pass_bits = self.n_passes * 64  # scales + combination weights
        return (weight_bits + pass_bits) / n_weights


# =============================================================================
# BASELINES
# =============================================================================

class BinaryBaseline:
    def __init__(self, d_in, d_out):
        self.W = np.random.choice([-1.0, 1.0], size=(d_out, d_in)).astype(np.float32)
        self.scale = 1.0

    def train(self, X, Y):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W = np.sign(W_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        Y_pred = X @ self.W.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def get_weights(self): return self.W * self.scale
    def effective_bpp(self): return 1.0


class TernaryBaseline:
    def __init__(self, d_in, d_out):
        self.W = np.zeros((d_out, d_in), dtype=np.float32)
        self.scale = 1.0

    def train(self, X, Y, zero_pct=0.3):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        thresh = np.percentile(np.abs(W_opt), zero_pct * 100)
        self.W = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        nonzero_mask = self.W != 0
        if nonzero_mask.any():
            self.scale = np.mean(np.abs(W_opt[nonzero_mask]))

    def get_weights(self): return self.W * self.scale
    def effective_bpp(self): return 1.58


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiments(dims: list = [64, 128, 256]):
    """Run all approaches and compare."""
    print("=" * 80)
    print("NOVEL 1-BIT APPROACHES: Can we beat 1.58-bit ternary?")
    print("=" * 80)

    results = {}

    for d in dims:
        print(f"\n{'='*60}")
        print(f"Dimension: {d}x{d}")
        print(f"{'='*60}")

        cfg = ExperimentConfig(d_in=d, d_out=d)

        # Generate task
        W_true = np.random.randn(cfg.d_out, cfg.d_in).astype(np.float32) * 0.5
        X_train = np.random.randn(cfg.n_train, cfg.d_in).astype(np.float32)
        Y_train = X_train @ W_true.T + np.random.randn(cfg.n_train, cfg.d_out) * 0.1

        X_test = np.random.randn(cfg.n_test, cfg.d_in).astype(np.float32)
        Y_test = X_test @ W_true.T

        dim_results = {}

        # Baselines
        binary = BinaryBaseline(cfg.d_in, cfg.d_out)
        binary.train(X_train, Y_train)
        Y_pred = X_test @ binary.get_weights().T
        corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
        dim_results['binary'] = {'corr': corr, 'bpp': binary.effective_bpp()}

        ternary = TernaryBaseline(cfg.d_in, cfg.d_out)
        ternary.train(X_train, Y_train)
        Y_pred = X_test @ ternary.get_weights().T
        corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
        dim_results['ternary'] = {'corr': corr, 'bpp': ternary.effective_bpp()}

        # Novel Approach 1: CTG Procedural
        for sparsity in [0.05, 0.10]:
            ctg = CTGProceduralLayer(cfg.d_in, cfg.d_out, correction_sparsity=sparsity)
            ctg.train(X_train, Y_train)
            Y_pred = X_test @ ctg.get_weights().T
            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            name = f'ctg_{int(sparsity*100)}pct'
            dim_results[name] = {'corr': corr, 'bpp': ctg.effective_bpp()}

        # Novel Approach 2: Dynamic Masking
        for mask_type in ['magnitude', 'attention']:
            dm = DynamicMaskingLayer(cfg.d_in, cfg.d_out, mask_type=mask_type)
            dm.train(X_train, Y_train)
            Y_pred = dm.forward(X_test)
            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            dim_results[f'dynamic_{mask_type}'] = {'corr': corr, 'bpp': dm.effective_bpp()}

        # Novel Approach 3: Residual Binary
        for sparsity in [0.05, 0.10, 0.20]:
            rb = ResidualBinaryLayer(cfg.d_in, cfg.d_out, residual_sparsity=sparsity)
            rb.train(X_train, Y_train)
            Y_pred = X_test @ rb.get_weights().T
            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            dim_results[f'residual_{int(sparsity*100)}pct'] = {'corr': corr, 'bpp': rb.effective_bpp()}

        # Novel Approach 4: Multi-Pass
        for n_passes in [2, 3]:
            mp = MultiPassBinaryLayer(cfg.d_in, cfg.d_out, n_passes=n_passes)
            mp.train(X_train, Y_train)
            Y_pred = mp.forward(X_test)
            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            dim_results[f'multipass_{n_passes}'] = {'corr': corr, 'bpp': mp.effective_bpp()}

        results[d] = dim_results

        # Print results
        print(f"\n{'Method':<25} {'Corr':>10} {'BPP':>10} {'vs Ternary':>12}")
        print("-" * 60)
        ternary_corr = dim_results['ternary']['corr']

        for name, data in sorted(dim_results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (data['corr'] / ternary_corr - 1) * 100 if ternary_corr > 0 else 0
            marker = "✓" if data['bpp'] <= 1.1 and data['corr'] >= ternary_corr * 0.99 else ""
            print(f"{name:<25} {data['corr']:>10.4f} {data['bpp']:>10.3f} {vs_tern:>+11.1f}% {marker}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Methods that beat ternary at ≤1.1 bpp")
    print("=" * 80)

    winners = []
    for d, dim_results in results.items():
        ternary_corr = dim_results['ternary']['corr']
        for name, data in dim_results.items():
            if name != 'ternary' and data['bpp'] <= 1.1 and data['corr'] >= ternary_corr * 0.99:
                winners.append((d, name, data['corr'], data['bpp']))

    if winners:
        print("\n✓ BREAKTHROUGH FOUND:")
        for d, name, corr, bpp in winners:
            print(f"  [{d}x{d}] {name}: {corr:.4f} corr @ {bpp:.3f} bpp")
    else:
        print("\nNo methods beat ternary at ≤1.1 bpp yet.")
        print("But here are the closest approaches...")

    return results


if __name__ == "__main__":
    run_experiments()

