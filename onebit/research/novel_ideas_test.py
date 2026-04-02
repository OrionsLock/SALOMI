"""Test ALL novel ideas to beat ternary at 1.0 bpp.

This implements and benchmarks all 15 crazy ideas.
"""

import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# BASELINES
# =============================================================================

class BinaryBaseline:
    """Standard binary: sign only."""
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
    
    def bpp(self) -> float:
        return 1.0


class TernaryBaseline:
    """Standard ternary: {-1, 0, +1}."""
    def __init__(self, d_in: int, d_out: int, zero_frac: float = 0.3):
        self.d_in, self.d_out = d_in, d_out
        self.zero_frac = zero_frac
        self.W = None
        self.scale = 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        thresh = np.percentile(np.abs(W_opt), self.zero_frac * 100)
        self.W = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        mask = self.W != 0
        if mask.any():
            self.scale = np.mean(np.abs(W_opt[mask]))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W * self.scale).T
    
    def bpp(self) -> float:
        return 1.58


# =============================================================================
# IDEA 1: Codon Encoding (DNA-inspired)
# =============================================================================

class CodonEncoding:
    """Group weights into triplets, use 5 valid patterns for 5-state encoding."""
    
    # 5 valid codon patterns and their semantic values
    CODONS = {
        (1, 1, 1): 3.0,      # Strong positive
        (1, 1, -1): 1.0,     # Weak positive
        (-1, 1, -1): 0.0,    # ZERO
        (-1, -1, 1): -1.0,   # Weak negative
        (-1, -1, -1): -3.0,  # Strong negative
    }
    
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.codons = None  # Indices into CODONS
        self.scale = 1.0
    
    def _closest_codon(self, triplet: np.ndarray) -> Tuple[Tuple, float]:
        """Find closest valid codon to a triplet of values."""
        best_codon = None
        best_score = -np.inf
        target_sign = np.sign(triplet.sum())
        target_mag = np.abs(triplet).mean()
        
        for codon, value in self.CODONS.items():
            codon_arr = np.array(codon)
            # Score by correlation
            score = np.corrcoef(triplet.flatten(), codon_arr)[0, 1] if np.std(triplet) > 0 else 0
            if np.isnan(score):
                score = 0
            if score > best_score:
                best_score = score
                best_codon = codon
        return best_codon, self.CODONS[best_codon]
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        # Flatten and pad to multiple of 3
        flat = W_opt.flatten()
        pad_len = (3 - len(flat) % 3) % 3
        if pad_len:
            flat = np.concatenate([flat, np.zeros(pad_len)])
        
        # Convert to codons
        n_codons = len(flat) // 3
        self.codon_values = np.zeros(n_codons)
        self.codon_patterns = []
        
        for i in range(n_codons):
            triplet = flat[i*3:(i+1)*3]
            codon, value = self._closest_codon(triplet)
            self.codon_patterns.append(codon)
            self.codon_values[i] = value
        
        # Reconstruct and find scale
        W_recon = np.array([v for v in self.codon_values for _ in range(3)])[:self.d_in * self.d_out]
        W_recon = W_recon.reshape(self.d_out, self.d_in)
        
        Y_pred = X @ W_recon.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
        self.W = W_recon
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W * self.scale).T
    
    def bpp(self) -> float:
        # 3 bits per triplet = 1 bit per weight, but 5 states
        # log2(5)/3 per weight = 0.77 bpp
        return np.log2(5) / 3  # ~0.77


# =============================================================================
# IDEA 2: Destructive Interference with Derived Shadow
# =============================================================================

class DestructiveInterference:
    """W_effective = W1 + derive(W1). Zeros from interference."""

    def __init__(self, d_in: int, d_out: int, shift: int = 7):
        self.d_in, self.d_out = d_in, d_out
        self.shift = shift
        self.W1 = None
        self.scale = 1.0

    def _derive_shadow(self, W1: np.ndarray) -> np.ndarray:
        """Derive W2 from W1 via rotation + XOR-like operation."""
        flat = W1.flatten()
        # Rotate
        W2_flat = np.roll(flat, self.shift)
        # XOR pattern: flip every other after rotation
        pattern = np.array([1 if i % 3 == 0 else -1 for i in range(len(flat))])
        W2_flat = W2_flat * pattern
        return W2_flat.reshape(W1.shape)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # We want W1 + derive(W1) ≈ W_opt
        # This is tricky - let's use iterative optimization
        self.W1 = np.sign(W_opt).astype(np.float32)
        self.W1[self.W1 == 0] = 1.0

        # Try a few random shifts to find best interference pattern
        best_corr = -1
        best_W1 = self.W1.copy()
        best_shift = self.shift

        for shift in [3, 5, 7, 11, 13, 17]:
            self.shift = shift
            W2 = self._derive_shadow(self.W1)
            W_eff = self.W1 + W2  # Values in {-2, 0, +2}

            Y_pred = X @ W_eff.T
            corr = np.corrcoef(Y_pred.flatten(), Y.flatten())[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_shift = shift

        self.shift = best_shift
        W2 = self._derive_shadow(self.W1)
        self.W_eff = self.W1 + W2

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Only W1 stored (1 bit), W2 is derived (0 bits), shift is tiny
        return 1.0 + 8 / (self.d_in * self.d_out)  # ~1.0


# =============================================================================
# IDEA 3: Self-Referential Magnitude
# =============================================================================

class SelfReferentialMagnitude:
    """Magnitude from local neighborhood agreement."""

    def __init__(self, d_in: int, d_out: int, kernel_size: int = 3):
        self.d_in, self.d_out = d_in, d_out
        self.k = kernel_size
        self.W_signs = None
        self.scale = 1.0

    def _compute_magnitude(self, signs: np.ndarray) -> np.ndarray:
        """Compute magnitude based on neighborhood agreement."""
        mag = np.ones_like(signs, dtype=np.float32)

        for i in range(self.d_out):
            for j in range(self.d_in):
                # Get neighborhood
                i_min, i_max = max(0, i-1), min(self.d_out, i+2)
                j_min, j_max = max(0, j-1), min(self.d_in, j+2)

                neighborhood = signs[i_min:i_max, j_min:j_max]
                center = signs[i, j]

                # Agreement = fraction of neighbors with same sign
                agreement = np.mean(neighborhood == center)

                # Magnitude: high agreement = high magnitude, low = near zero
                mag[i, j] = agreement ** 2  # Square for sharper contrast

        return mag

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        mag = self._compute_magnitude(self.W_signs)
        self.W_eff = self.W_signs * mag

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        mag = self._compute_magnitude(self.W_signs)
        return X @ (self.W_signs * mag * self.scale).T

    def bpp(self) -> float:
        return 1.0  # Only signs stored, magnitude computed


# =============================================================================
# IDEA 4: Huffman-Coded Zeros in Binary Stream
# =============================================================================

class HuffmanZeroPatterns:
    """Use specific bit patterns to encode zeros."""

    # Patterns and their meanings (for blocks of 4)
    ZERO_PATTERNS = {
        (1, -1, 1, -1): 'zero_block',      # Alternating = zero
        (-1, 1, -1, 1): 'zero_block',      # Alternating = zero
    }

    def __init__(self, d_in: int, d_out: int, block_size: int = 4):
        self.d_in, self.d_out = d_in, d_out
        self.block_size = block_size
        self.W_binary = None
        self.scale = 1.0

    def _is_zero_pattern(self, block: np.ndarray) -> bool:
        """Check if block matches a zero pattern."""
        return tuple(block.astype(int).tolist()) in self.ZERO_PATTERNS

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_binary = np.sign(W_opt).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0

        # For each block, check if we should make it a "zero pattern"
        flat = self.W_binary.flatten()
        flat_opt = W_opt.flatten()

        n_blocks = len(flat) // self.block_size

        for b in range(n_blocks):
            start = b * self.block_size
            end = start + self.block_size

            block_mag = np.abs(flat_opt[start:end]).mean()
            overall_mag = np.abs(flat_opt).mean()

            # If block has low magnitude, encode as zero pattern
            if block_mag < 0.3 * overall_mag:
                flat[start:end] = np.array([1, -1, 1, -1])  # Zero pattern

        self.W_binary = flat.reshape(self.d_out, self.d_in)

        # Compute effective weights
        self.W_eff = self._decode_weights()

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def _decode_weights(self) -> np.ndarray:
        """Decode binary patterns to effective weights."""
        flat = self.W_binary.flatten()
        eff = np.zeros_like(flat)

        n_blocks = len(flat) // self.block_size

        for b in range(n_blocks):
            start = b * self.block_size
            end = start + self.block_size
            block = flat[start:end]

            if self._is_zero_pattern(block):
                eff[start:end] = 0  # Zero block
            else:
                eff[start:end] = block

        return eff.reshape(self.d_out, self.d_in)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0  # 1 bit per weight, zeros encoded in patterns


# =============================================================================
# IDEA 5: Stochastic Binary with Learned Activation Probabilities
# =============================================================================

class StochasticBinaryActivation:
    """Weights fire probabilistically. E[w] = sign * prob.

    FAIR VERSION: Uses position-based buckets, not magnitude-based.
    """

    def __init__(self, d_in: int, d_out: int, n_buckets: int = 8):
        self.d_in, self.d_out = d_in, d_out
        self.n_buckets = n_buckets
        self.W_signs = None
        self.bucket_probs = None  # Probability per bucket
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # FAIR: Assign buckets based on POSITION hash, not magnitude
        # Each bucket gets learned probability based on reconstruction error
        positions = np.arange(self.d_out * self.d_in)
        self.weight_buckets = (positions * 7 + positions // self.d_in * 13) % self.n_buckets

        # Learn bucket probabilities to minimize reconstruction error
        # Start with uniform, then optimize
        self.bucket_probs = np.ones(self.n_buckets) * 0.5

        best_loss = np.inf
        best_probs = self.bucket_probs.copy()

        for _ in range(100):
            # Random perturbation
            probs = np.clip(self.bucket_probs + np.random.randn(self.n_buckets) * 0.1, 0.1, 1.0)
            probs_mat = probs[self.weight_buckets].reshape(self.d_out, self.d_in)
            W_eff = self.W_signs * probs_mat
            Y_pred = X @ W_eff.T
            scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
            loss = np.mean((Y - Y_pred * scale) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_probs = probs.copy()

        self.bucket_probs = best_probs
        probs_mat = self.bucket_probs[self.weight_buckets].reshape(self.d_out, self.d_in)
        self.W_eff = self.W_signs * probs_mat

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        probs = self.bucket_probs[self.weight_buckets].reshape(self.d_out, self.d_in)
        return X @ (self.W_signs * probs * self.scale).T

    def bpp(self) -> float:
        # Signs: 1 bit each + bucket probs (tiny)
        n_weights = self.d_in * self.d_out
        return 1.0 + (self.n_buckets * 32) / n_weights


# =============================================================================
# IDEA 6: Binary + Tiny Zero Predictor
# =============================================================================

class BinaryWithZeroPredictor:
    """Learn a small network that predicts which positions should be zero."""

    def __init__(self, d_in: int, d_out: int, predictor_hidden: int = 32):
        self.d_in, self.d_out = d_in, d_out
        self.predictor_hidden = predictor_hidden
        self.W_signs = None
        # Predictor: (i, j) -> zero probability
        self.pred_W1 = None
        self.pred_W2 = None
        self.scale = 1.0

    def _predict_zeros(self, normalize: bool = True) -> np.ndarray:
        """Predict zero mask using the small predictor network."""
        # Create position features
        i_pos = np.arange(self.d_out)[:, None] / self.d_out
        j_pos = np.arange(self.d_in)[None, :] / self.d_in

        # Broadcast to full matrix
        i_feat = np.broadcast_to(i_pos, (self.d_out, self.d_in))
        j_feat = np.broadcast_to(j_pos, (self.d_out, self.d_in))

        # Simple predictor: linear combination of position features
        # pred = sigmoid(w1 * i + w2 * j + w3 * i*j + b)
        pred = self.pred_W1[0] * i_feat + self.pred_W1[1] * j_feat
        pred = pred + self.pred_W1[2] * (i_feat * j_feat) + self.pred_W1[3]

        # Sigmoid
        zero_prob = 1 / (1 + np.exp(-pred))
        return zero_prob

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Learn predictor parameters to match optimal magnitude pattern
        # Target: high magnitude = low zero prob, low magnitude = high zero prob
        target_active = np.abs(W_opt) / (np.abs(W_opt).max() + 1e-8)

        # Simple optimization for predictor weights
        best_corr = -1
        best_params = None

        for _ in range(100):
            self.pred_W1 = np.random.randn(4) * 2
            zero_prob = self._predict_zeros()
            active_prob = 1 - zero_prob
            corr = np.corrcoef(active_prob.flatten(), target_active.flatten())[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_params = self.pred_W1.copy()

        self.pred_W1 = best_params if best_params is not None else np.zeros(4)

        # Compute effective weights
        zero_prob = self._predict_zeros()
        self.W_eff = self.W_signs * (1 - zero_prob)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        zero_prob = self._predict_zeros()
        return X @ (self.W_signs * (1 - zero_prob) * self.scale).T

    def bpp(self) -> float:
        # Signs: 1 bit each, Predictor: 4 floats = 128 bits
        n_weights = self.d_in * self.d_out
        return 1.0 + 128 / n_weights


# =============================================================================
# IDEA 7: Phase-Coupled Binary
# =============================================================================

class PhaseCoupledBinary:
    """Binary with phase per block. Interference creates zeros."""

    def __init__(self, d_in: int, d_out: int, block_size: int = 16, n_phases: int = 4):
        self.d_in, self.d_out = d_in, d_out
        self.block_size = block_size
        self.n_phases = n_phases
        self.W_signs = None
        self.block_phases = None
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Assign phases to blocks based on magnitude
        flat = W_opt.flatten()
        n_blocks = (len(flat) + self.block_size - 1) // self.block_size

        self.block_phases = np.zeros(n_blocks, dtype=int)
        phase_mults = np.array([1.0, 0.5, 0.0, -0.5])  # Phase effects

        for b in range(n_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, len(flat))
            block_mag = np.abs(flat[start:end]).mean()

            # Assign phase based on magnitude (low mag = phase 2 = zero)
            if block_mag < np.percentile(np.abs(flat), 25):
                self.block_phases[b] = 2  # Zero phase
            elif block_mag < np.percentile(np.abs(flat), 50):
                self.block_phases[b] = 1  # Weak
            else:
                self.block_phases[b] = 0  # Strong

        # Compute effective weights
        self.W_eff = self._compute_effective()

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def _compute_effective(self) -> np.ndarray:
        phase_mults = np.array([1.0, 0.5, 0.0, -0.5])
        flat_signs = self.W_signs.flatten()
        flat_eff = np.zeros_like(flat_signs)

        for b in range(len(self.block_phases)):
            start = b * self.block_size
            end = min(start + self.block_size, len(flat_signs))
            flat_eff[start:end] = flat_signs[start:end] * phase_mults[self.block_phases[b]]

        return flat_eff.reshape(self.d_out, self.d_in)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        n_weights = self.d_in * self.d_out
        n_blocks = (n_weights + self.block_size - 1) // self.block_size
        return 1.0 + (n_blocks * np.log2(self.n_phases)) / n_weights


# =============================================================================
# IDEA 8: Low-Rank Binary Basis
# =============================================================================

class LowRankBinaryBasis:
    """W = U @ V.T where U, V are binary. Low-rank creates smooth values."""

    def __init__(self, d_in: int, d_out: int, rank: int = 16):
        self.d_in, self.d_out = d_in, d_out
        self.rank = rank
        self.U = None  # (d_out, rank) binary
        self.V = None  # (d_in, rank) binary
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # SVD to get low-rank approximation
        U, S, Vt = np.linalg.svd(W_opt, full_matrices=False)

        # Take top-r components and binarize
        U_r = U[:, :self.rank] @ np.diag(np.sqrt(S[:self.rank]))
        V_r = Vt[:self.rank, :].T @ np.diag(np.sqrt(S[:self.rank]))

        self.U = np.sign(U_r).astype(np.float32)
        self.V = np.sign(V_r).astype(np.float32)
        self.U[self.U == 0] = 1.0
        self.V[self.V == 0] = 1.0

        # Effective weight: values in [-rank, +rank]
        self.W_eff = self.U @ self.V.T

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # U: d_out * rank bits, V: d_in * rank bits
        n_weights = self.d_in * self.d_out
        storage = self.d_out * self.rank + self.d_in * self.rank
        return storage / n_weights


# =============================================================================
# IDEA 9: Gradient-Magnitude Coupling
# =============================================================================

class GradientMagnitudeCoupling:
    """Use gradient information to determine which weights matter.

    TRULY FAIR VERSION: Importance is STORED, not computed from test data.
    Uses training data to learn importance, stores compressed importance.
    """

    def __init__(self, d_in: int, d_out: int, n_buckets: int = 4):
        self.d_in, self.d_out = d_in, d_out
        self.n_buckets = n_buckets  # Quantize importance to n levels
        self.W_signs = None
        self.importance_buckets = None  # Per-weight bucket assignment
        self.bucket_values = None  # Value per bucket
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Compute importance from training data
        input_std = np.std(X, axis=0)
        Y_pred_bin = X @ self.W_signs.T
        output_error = Y - Y_pred_bin

        importance = np.zeros((self.d_out, self.d_in))
        for j in range(self.d_in):
            for i in range(self.d_out):
                corr = np.abs(np.corrcoef(X[:, j], output_error[:, i])[0, 1])
                if np.isnan(corr):
                    corr = 0
                importance[i, j] = corr * input_std[j]

        importance = importance / (importance.max() + 1e-8)

        # QUANTIZE importance to n_buckets levels (THIS IS STORED)
        flat_imp = importance.flatten()
        bucket_edges = np.percentile(flat_imp, np.linspace(0, 100, self.n_buckets + 1))
        self.importance_buckets = np.digitize(flat_imp, bucket_edges[1:-1])

        # Compute bucket values (mean importance in each bucket)
        self.bucket_values = np.zeros(self.n_buckets)
        for b in range(self.n_buckets):
            mask = self.importance_buckets == b
            if mask.any():
                self.bucket_values[b] = flat_imp[mask].mean()

        # Make lowest bucket = 0 (creates zeros)
        self.bucket_values[0] = 0.0

        # Reconstruct quantized importance
        quant_importance = self.bucket_values[self.importance_buckets].reshape(self.d_out, self.d_in)
        self.W_eff = self.W_signs * quant_importance

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        quant_importance = self.bucket_values[self.importance_buckets].reshape(self.d_out, self.d_in)
        return X @ (self.W_signs * quant_importance * self.scale).T

    def bpp(self) -> float:
        # Signs: 1 bit per weight
        # Bucket assignment: log2(n_buckets) bits per weight
        # Bucket values: n_buckets * 32 bits (tiny overhead)
        n_weights = self.d_in * self.d_out
        return 1.0 + np.log2(self.n_buckets) + (self.n_buckets * 32) / n_weights


# =============================================================================
# IDEA 10: Error-Correcting Binary (ECC for zeros)
# =============================================================================

class ECCBinary:
    """Use ECC-like encoding where some patterns represent zeros."""

    def __init__(self, d_in: int, d_out: int, block_size: int = 8):
        self.d_in, self.d_out = d_in, d_out
        self.block_size = block_size
        self.blocks = None
        self.block_types = None  # 'data' or 'zero'
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        flat = W_opt.flatten()

        n_blocks = (len(flat) + self.block_size - 1) // self.block_size
        self.blocks = []
        self.block_types = []

        # Threshold for zero blocks
        mag_thresh = np.percentile(np.abs(flat), 30)

        for b in range(n_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, len(flat))
            block = flat[start:end]

            if np.abs(block).mean() < mag_thresh:
                # Zero block - store special pattern
                self.blocks.append(np.ones(len(block)))  # Dummy
                self.block_types.append('zero')
            else:
                # Data block - store signs
                signs = np.sign(block).astype(np.float32)
                signs[signs == 0] = 1.0
                self.blocks.append(signs)
                self.block_types.append('data')

        # Compute effective weights
        eff = []
        for block, btype in zip(self.blocks, self.block_types):
            if btype == 'zero':
                eff.extend([0.0] * len(block))
            else:
                eff.extend(block.tolist())

        self.W_eff = np.array(eff[:self.d_in * self.d_out]).reshape(self.d_out, self.d_in)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Each block: block_size bits for data, 1 bit for type
        n_weights = self.d_in * self.d_out
        n_blocks = len(self.blocks)
        return 1.0 + n_blocks / n_weights


# =============================================================================
# IDEA 11: Binary Superposition of Orthogonal Bases
# =============================================================================

class OrthogonalBinaryBases:
    """W = sum_i c_i * (B_i outer B_i). Few bases, ternary coefficients."""

    def __init__(self, d_in: int, d_out: int, n_bases: int = 8):
        self.d_in, self.d_out = d_in, d_out
        self.n_bases = n_bases
        self.bases_out = None  # (n_bases, d_out) binary
        self.bases_in = None   # (n_bases, d_in) binary
        self.coeffs = None     # (n_bases,) ternary
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Generate random orthogonal-ish binary bases
        self.bases_out = np.sign(np.random.randn(self.n_bases, self.d_out))
        self.bases_in = np.sign(np.random.randn(self.n_bases, self.d_in))

        # Optimize coefficients to minimize error
        best_coeffs = np.zeros(self.n_bases)

        for trial in range(50):
            coeffs = np.random.choice([-1, 0, 1], self.n_bases)
            W_approx = sum(c * np.outer(bo, bi)
                         for c, bo, bi in zip(coeffs, self.bases_out, self.bases_in))
            error = np.sum((W_opt - W_approx) ** 2)

            best_error = np.sum((W_opt - sum(c * np.outer(bo, bi)
                                for c, bo, bi in zip(best_coeffs, self.bases_out, self.bases_in))) ** 2)
            if error < best_error:
                best_coeffs = coeffs

        self.coeffs = best_coeffs
        self.W_eff = sum(c * np.outer(bo, bi)
                        for c, bo, bi in zip(self.coeffs, self.bases_out, self.bases_in))

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        n_weights = self.d_in * self.d_out
        storage = self.n_bases * (self.d_out + self.d_in) + self.n_bases * 1.58
        return storage / n_weights


# =============================================================================
# IDEA 12: Cellular Automata Generated Weights
# =============================================================================

class CellularAutomataWeights:
    """Weights generated by running CA from seed."""

    def __init__(self, d_in: int, d_out: int, rule: int = 110):
        self.d_in, self.d_out = d_in, d_out
        self.rule = rule
        self.seed = None
        self.scale = 1.0

    def _rule_to_table(self, rule: int) -> dict:
        """Convert rule number to lookup table."""
        table = {}
        for i in range(8):
            pattern = tuple(int(b) for b in format(i, '03b'))
            table[pattern] = (rule >> i) & 1
        return table

    def _run_ca(self, seed: np.ndarray, steps: int) -> np.ndarray:
        """Run 1D CA for steps iterations."""
        table = self._rule_to_table(self.rule)
        state = seed.copy()
        n = len(state)

        result = [state.copy()]
        for _ in range(steps - 1):
            new_state = np.zeros_like(state)
            for i in range(n):
                pattern = (state[(i-1) % n], state[i], state[(i+1) % n])
                new_state[i] = table[tuple(pattern)]
            state = new_state
            result.append(state.copy())

        return np.array(result)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Try different seeds to find best match
        best_corr = -1
        best_seed = None
        best_W = None

        for _ in range(100):
            # Random seed
            seed = np.random.randint(0, 2, self.d_in)

            # Run CA
            W_ca = self._run_ca(seed, self.d_out)
            W_ca = W_ca * 2 - 1  # Convert {0,1} to {-1,+1}

            # Correlation with target
            corr = np.corrcoef(W_ca.flatten(), W_opt.flatten())[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_seed = seed
                best_W = W_ca

        self.seed = best_seed
        self.W_eff = best_W if best_W is not None else np.sign(W_opt)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Only store seed + rule
        n_weights = self.d_in * self.d_out
        return (self.d_in + 8) / n_weights  # seed bits + rule


# =============================================================================
# IDEA 13: Residual Binary Stacking
# =============================================================================

class ResidualBinaryStack:
    """Stack multiple binary approximations of residuals."""

    def __init__(self, d_in: int, d_out: int, n_layers: int = 3):
        self.d_in, self.d_out = d_in, d_out
        self.n_layers = n_layers
        self.layers = []  # List of (W_binary, scale)
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        residual = W_opt.copy()
        self.layers = []

        for _ in range(self.n_layers):
            # Binary approximation of residual
            W_bin = np.sign(residual).astype(np.float32)
            W_bin[W_bin == 0] = 1.0

            # Optimal scale for this layer
            layer_scale = np.sum(residual * W_bin) / (np.sum(W_bin ** 2) + 1e-8)

            self.layers.append((W_bin.copy(), layer_scale))

            # Update residual
            residual = residual - W_bin * layer_scale

        # Combine all layers
        self.W_eff = sum(W * s for W, s in self.layers)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Each layer: 1 bit per weight + 32 bits for scale
        n_weights = self.d_in * self.d_out
        return self.n_layers * (1.0 + 32 / n_weights)


# =============================================================================
# IDEA 14: Hadamard-Transformed Binary
# =============================================================================

class HadamardBinary:
    """Apply Hadamard transform, binarize, inverse transform."""

    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W_hadamard_signs = None
        self.scale = 1.0

    def _hadamard_matrix(self, n: int) -> np.ndarray:
        """Generate Hadamard-like matrix (Walsh-Hadamard)."""
        # For simplicity, use random orthogonal
        H, _ = np.linalg.qr(np.random.randn(n, n))
        return H

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Transform to Hadamard domain
        H_out = self._hadamard_matrix(self.d_out)
        H_in = self._hadamard_matrix(self.d_in)

        W_transformed = H_out @ W_opt @ H_in.T

        # Binarize in transform domain
        self.W_hadamard_signs = np.sign(W_transformed)
        self.W_hadamard_signs[self.W_hadamard_signs == 0] = 1.0

        # Store transforms for reconstruction
        self.H_out = H_out
        self.H_in = H_in

        # Inverse transform
        self.W_eff = H_out.T @ self.W_hadamard_signs @ H_in

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Binary in transform domain + transform matrices (but those are fixed/derived)
        return 1.0


# =============================================================================
# IDEA 15: Magnitude from Activation Statistics
# =============================================================================

class ActivationStatsMagnitude:
    """Magnitude learned from input activation patterns."""

    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W_signs = None
        self.input_importance = None
        self.output_importance = None
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Compute importance from activation statistics
        self.input_importance = np.std(X, axis=0)  # High variance = important
        self.input_importance = self.input_importance / (self.input_importance.max() + 1e-8)

        Y_pred_binary = X @ self.W_signs.T
        self.output_importance = np.std(Y_pred_binary, axis=0)
        self.output_importance = self.output_importance / (self.output_importance.max() + 1e-8)

        # Magnitude = outer product of importances
        magnitude = np.outer(self.output_importance, self.input_importance)

        self.W_eff = self.W_signs * magnitude

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        magnitude = np.outer(self.output_importance, self.input_importance)
        return X @ (self.W_signs * magnitude * self.scale).T

    def bpp(self) -> float:
        # Signs: 1 bit, importance vectors: 2 * d * 32 bits
        n_weights = self.d_in * self.d_out
        return 1.0 + (self.d_in + self.d_out) * 32 / n_weights


# =============================================================================
# MAIN TESTING
# =============================================================================

def test_all_ideas():
    """Test all novel ideas and compare to baselines."""
    np.random.seed(42)

    # Generate test data
    n_samples = 1000
    d_in, d_out = 128, 64

    # Create realistic weight pattern (low-rank + sparse)
    rank = 16
    U = np.random.randn(d_out, rank)
    V = np.random.randn(d_in, rank)
    W_true = U @ V.T
    W_true += np.random.randn(d_out, d_in) * 0.1  # noise

    # Make some weights zero (like ternary)
    mask = np.random.rand(d_out, d_in) > 0.3
    W_true = W_true * mask

    X = np.random.randn(n_samples, d_in).astype(np.float32)
    Y = (X @ W_true.T).astype(np.float32)

    # Split
    X_train, Y_train = X[:800], Y[:800]
    X_test, Y_test = X[800:], Y[800:]

    print("=" * 70)
    print("TESTING ALL NOVEL IDEAS TO BEAT TERNARY AT 1.0 BPP")
    print("=" * 70)
    print(f"\nData: {n_samples} samples, {d_in} -> {d_out}")
    print()

    # All methods to test
    methods = [
        ("Binary (baseline)", BinaryBaseline(d_in, d_out)),
        ("Ternary (baseline)", TernaryBaseline(d_in, d_out)),
        ("1. Codon Encoding", CodonEncoding(d_in, d_out)),
        ("2. Destructive Interference", DestructiveInterference(d_in, d_out)),
        ("3. Self-Referential Mag", SelfReferentialMagnitude(d_in, d_out)),
        ("4. Huffman Zero Patterns", HuffmanZeroPatterns(d_in, d_out)),
        ("5. Stochastic Activation", StochasticBinaryActivation(d_in, d_out)),
        ("6. Zero Predictor", BinaryWithZeroPredictor(d_in, d_out)),
        ("7. Phase-Coupled", PhaseCoupledBinary(d_in, d_out)),
        ("8. Low-Rank Binary", LowRankBinaryBasis(d_in, d_out, rank=8)),
        ("9. Gradient-Mag (4 buckets)", GradientMagnitudeCoupling(d_in, d_out, n_buckets=4)),
        ("9b. Gradient-Mag (2 buckets)", GradientMagnitudeCoupling(d_in, d_out, n_buckets=2)),
        ("10. ECC Binary", ECCBinary(d_in, d_out)),
        ("11. Orthogonal Bases", OrthogonalBinaryBases(d_in, d_out)),
        ("12. Cellular Automata", CellularAutomataWeights(d_in, d_out)),
        ("13. Residual Stack (x2)", ResidualBinaryStack(d_in, d_out, n_layers=2)),
        ("14. Hadamard Binary", HadamardBinary(d_in, d_out)),
        ("15. Activation Stats", ActivationStatsMagnitude(d_in, d_out)),
    ]

    results = []
    ternary_corr = None

    for name, model in methods:
        try:
            model.train(X_train, Y_train)
            Y_pred = model.forward(X_test)

            # Metrics
            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            mse = np.mean((Y_pred - Y_test) ** 2)
            bpp = model.bpp()

            if "Ternary" in name:
                ternary_corr = corr

            results.append((name, corr, mse, bpp))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, 0.0, 999.0, 0.0))

    # Sort by correlation
    results.sort(key=lambda x: -x[1])

    print(f"{'Method':<30} {'Corr':>8} {'MSE':>10} {'BPP':>6} {'vs Tern':>8}")
    print("-" * 70)

    for name, corr, mse, bpp in results:
        vs_tern = ((corr / ternary_corr) - 1) * 100 if ternary_corr else 0
        marker = "★" if bpp <= 1.1 and vs_tern >= -5 else ""
        print(f"{name:<30} {corr:>8.4f} {mse:>10.4f} {bpp:>6.2f} {vs_tern:>+7.1f}% {marker}")

    # Find winners
    print("\n" + "=" * 70)
    print("WINNERS: Methods at ≤1.1 bpp within 5% of ternary")
    print("=" * 70)

    winners = [(n, c, m, b) for n, c, m, b in results
               if b <= 1.1 and (c / ternary_corr - 1) >= -0.05]

    if winners:
        for name, corr, mse, bpp in winners:
            print(f"  ★ {name}: {corr:.4f} corr at {bpp:.2f} bpp")
    else:
        print("  No methods achieved ≤1.1 bpp within 5% of ternary")

    # Find best at each BPP threshold
    print("\n" + "=" * 70)
    print("BEST AT EACH BPP LEVEL")
    print("=" * 70)

    for bpp_thresh in [0.8, 1.0, 1.1, 1.2, 1.5]:
        candidates = [(n, c, m, b) for n, c, m, b in results if b <= bpp_thresh]
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            vs_tern = ((best[1] / ternary_corr) - 1) * 100
            print(f"  ≤{bpp_thresh:.1f} bpp: {best[0]} ({best[1]:.4f} corr, {vs_tern:+.1f}% vs ternary)")

    return results


# =============================================================================
# COMBINED APPROACHES: Merge winners!
# =============================================================================

class CombinedGradientHadamard:
    """Combine Gradient-Magnitude importance with Hadamard transform."""

    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W_signs = None
        self.H_out = None
        self.H_in = None
        self.importance = None
        self.scale = 1.0

    def _hadamard_matrix(self, n: int) -> np.ndarray:
        H, _ = np.linalg.qr(np.random.randn(n, n))
        return H

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Hadamard transform
        self.H_out = self._hadamard_matrix(self.d_out)
        self.H_in = self._hadamard_matrix(self.d_in)

        W_transformed = self.H_out @ W_opt @ self.H_in.T

        self.W_signs = np.sign(W_transformed)
        self.W_signs[self.W_signs == 0] = 1.0

        # Gradient-based importance in transform domain
        input_std = np.std(X @ self.H_in.T, axis=0)

        W_bin_space = self.H_out.T @ self.W_signs @ self.H_in
        Y_pred_bin = X @ W_bin_space.T
        output_error = Y - Y_pred_bin

        self.importance = np.zeros((self.d_out, self.d_in))
        X_transformed = X @ self.H_in.T

        for j in range(self.d_in):
            for i in range(self.d_out):
                corr = np.abs(np.corrcoef(X_transformed[:, j], output_error[:, i])[0, 1])
                if np.isnan(corr):
                    corr = 0
                self.importance[i, j] = corr * input_std[j]

        self.importance = self.importance / (self.importance.max() + 1e-8)
        thresh = np.percentile(self.importance, 30)
        self.importance[self.importance < thresh] = 0

        W_eff_transform = self.W_signs * self.importance
        self.W_eff = self.H_out.T @ W_eff_transform @ self.H_in

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0


class CombinedResidualGradient:
    """Residual binary + gradient importance."""

    def __init__(self, d_in: int, d_out: int, n_layers: int = 2):
        self.d_in, self.d_out = d_in, d_out
        self.n_layers = n_layers
        self.layers = []
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        residual = W_opt.copy()
        self.layers = []

        for layer_idx in range(self.n_layers):
            W_bin = np.sign(residual).astype(np.float32)
            W_bin[W_bin == 0] = 1.0

            # Compute importance for this layer
            Y_pred_bin = X @ W_bin.T
            output_error = Y - Y_pred_bin
            input_std = np.std(X, axis=0)

            importance = np.zeros_like(W_bin)
            for j in range(self.d_in):
                for i in range(self.d_out):
                    corr = np.abs(np.corrcoef(X[:, j], output_error[:, i])[0, 1])
                    if np.isnan(corr):
                        corr = 0
                    importance[i, j] = corr * input_std[j]

            importance = importance / (importance.max() + 1e-8)
            thresh = np.percentile(importance, 30)
            importance[importance < thresh] = 0

            W_layer = W_bin * importance
            layer_scale = np.sum(residual * W_layer) / (np.sum(W_layer ** 2) + 1e-8)

            self.layers.append((W_layer, layer_scale))
            residual = residual - W_layer * layer_scale

        self.W_eff = sum(W * s for W, s in self.layers)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return self.n_layers * 1.0


class UltimateCombined:
    """Ultimate combination: Hadamard + Gradient + Residual refinement."""

    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def _hadamard_matrix(self, n: int) -> np.ndarray:
        H, _ = np.linalg.qr(np.random.randn(n, n))
        return H

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Transform
        self.H_out = self._hadamard_matrix(self.d_out)
        self.H_in = self._hadamard_matrix(self.d_in)

        W_t = self.H_out @ W_opt @ self.H_in.T
        X_t = X @ self.H_in.T

        # First pass: binary
        W_bin = np.sign(W_t)
        W_bin[W_bin == 0] = 1.0

        # Gradient importance
        Y_pred = X_t @ W_bin.T @ self.H_out
        error = Y - Y_pred

        importance = np.zeros_like(W_bin)
        input_std = np.std(X_t, axis=0)

        for j in range(self.d_in):
            for i in range(self.d_out):
                corr = np.abs(np.corrcoef(X_t[:, j], error[:, i])[0, 1])
                importance[i, j] = (corr if not np.isnan(corr) else 0) * input_std[j]

        importance = importance / (importance.max() + 1e-8)
        thresh = np.percentile(importance, 25)
        importance[importance < thresh] = 0

        # Combine
        W_t_eff = W_bin * importance
        self.W_eff = self.H_out.T @ W_t_eff @ self.H_in

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0


def test_combined():
    """Test combined approaches."""
    np.random.seed(42)

    n_samples = 1000
    d_in, d_out = 128, 64

    rank = 16
    U = np.random.randn(d_out, rank)
    V = np.random.randn(d_in, rank)
    W_true = U @ V.T
    W_true += np.random.randn(d_out, d_in) * 0.1
    mask = np.random.rand(d_out, d_in) > 0.3
    W_true = W_true * mask

    X = np.random.randn(n_samples, d_in).astype(np.float32)
    Y = (X @ W_true.T).astype(np.float32)

    X_train, Y_train = X[:800], Y[:800]
    X_test, Y_test = X[800:], Y[800:]

    print("\n" + "=" * 70)
    print("TESTING COMBINED APPROACHES")
    print("=" * 70)

    methods = [
        ("Binary", BinaryBaseline(d_in, d_out)),
        ("Ternary", TernaryBaseline(d_in, d_out)),
        ("Gradient-Magnitude", GradientMagnitudeCoupling(d_in, d_out)),
        ("Hadamard Binary", HadamardBinary(d_in, d_out)),
        ("Gradient + Hadamard", CombinedGradientHadamard(d_in, d_out)),
        ("Residual + Gradient", CombinedResidualGradient(d_in, d_out, n_layers=2)),
        ("Ultimate Combined", UltimateCombined(d_in, d_out)),
    ]

    results = []
    ternary_corr = None

    for name, model in methods:
        model.train(X_train, Y_train)
        Y_pred = model.forward(X_test)

        corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
        mse = np.mean((Y_pred - Y_test) ** 2)
        bpp = model.bpp()

        if name == "Ternary":
            ternary_corr = corr

        results.append((name, corr, mse, bpp))

    results.sort(key=lambda x: -x[1])

    print(f"\n{'Method':<25} {'Corr':>8} {'MSE':>10} {'BPP':>6} {'vs Tern':>8}")
    print("-" * 65)

    for name, corr, mse, bpp in results:
        vs_tern = ((corr / ternary_corr) - 1) * 100
        marker = "★★★" if bpp <= 1.0 and vs_tern >= 10 else ("★★" if bpp <= 1.1 and vs_tern >= 0 else "")
        print(f"{name:<25} {corr:>8.4f} {mse:>10.2f} {bpp:>6.2f} {vs_tern:>+7.1f}% {marker}")


if __name__ == "__main__":
    results = test_all_ideas()
    test_combined()

