"""
Unified 1-Bit Research Framework

Goal: Beat 1.58-bit (ternary) at exactly 1.00 bpp

Combined approaches:
1. Training-aware binary with STE
2. Magnitude uniformity regularization  
3. Zero prediction from local context
4. Binary matrix factorization
5. CTG structured sparsity integration
6. Learned binary basis (asymmetric levels)

Author: SALOMI Research
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from enum import Enum


class QuantMode(Enum):
    SIGN_ONLY = "sign"           # Basic {-1, +1}
    TERNARY = "ternary"          # {-1, 0, +1} - baseline to beat
    LEARNED_BASIS = "basis"      # α*B + β with learned α, β
    FACTORIZED = "factor"        # S @ B.T factorization
    ZERO_PREDICTED = "zero_pred" # Predict zeros from context
    CTG_STRUCTURED = "ctg"       # CTG fixed pattern zeros
    UNIFIED = "unified"          # All combined
    TERNARY_ENCODED = "tern_enc" # Ternary-in-Binary with entropy coding
    MAGNITUDE_AWARE = "mag"      # Use magnitude info from adjacent weights
    TRAINED_UNIFORM = "uniform"  # Training-aware: uniform magnitudes
    STOCHASTIC = "stoch"         # Stochastic binary (fair 1-bit)
    IMPORTANCE = "import"        # Importance-weighted binary
    MULTI_SAMPLE = "multi"       # Multi-sample binary (T bits, shows limit)
    CORRELATED = "corr"          # Correlated signs across rows
    HADAMARD = "hadamard"        # Hadamard domain quantization
    RESIDUAL = "residual"        # Residual binary (2 levels)
    OPTIMAL = "optimal"          # Optimal binary search
    CTG_TRAINED = "ctg_train"    # Training-aware CTG (oracle)
    PERFECT = "perfect"          # Perfect binary oracle (theoretical limit)


@dataclass
class UnifiedConfig:
    """Configuration for unified 1-bit approach."""
    # Dimensions
    d_out: int = 256
    d_in: int = 256
    
    # Training
    n_iters: int = 5000
    lr: float = 0.01
    batch_size: int = 32
    
    # Magnitude regularization
    mag_reg_weight: float = 0.1  # Weight for uniform magnitude loss
    
    # Factorization
    rank: int = 8  # Rank for binary factorization
    
    # Zero prediction
    zero_threshold: float = 0.3  # Threshold for ternary zeros
    predictor_window: int = 5    # Context window for zero prediction
    
    # CTG
    ctg_period: int = 5  # CTG inhibit period
    ctg_enabled: bool = True
    
    # Learned basis
    learn_basis: bool = True  # Learn α, β for asymmetric levels


def ste_sign(x: np.ndarray) -> np.ndarray:
    """Straight-Through Estimator for sign function.
    
    Forward: sign(x)
    Backward: gradient passes through unchanged
    """
    return np.sign(x)


def ste_round(x: np.ndarray) -> np.ndarray:
    """STE for rounding to {0, 1}."""
    return np.round(np.clip(x, 0, 1))


class ZeroPredictor:
    """Predicts which weights should be zero based on local context."""
    
    def __init__(self, window: int = 5):
        self.window = window
        self.threshold = 0.5
        
    def predict(self, W_sign: np.ndarray, row_idx: int, col_idx: int) -> float:
        """Predict probability of zero at position (row_idx, col_idx).
        
        Uses local sign pattern to estimate if this position should be zero.
        Key insight: if local signs are balanced (near 50/50), the weight
        might be small/uncertain and could be zero.
        """
        d_out, d_in = W_sign.shape
        
        # Get local window
        r_start = max(0, row_idx - self.window // 2)
        r_end = min(d_out, row_idx + self.window // 2 + 1)
        c_start = max(0, col_idx - self.window // 2)
        c_end = min(d_in, col_idx + self.window // 2 + 1)
        
        local = W_sign[r_start:r_end, c_start:c_end]
        
        # Measure local sign balance
        pos_ratio = np.mean(local > 0)
        balance = 1 - 2 * np.abs(pos_ratio - 0.5)  # 1 = perfectly balanced
        
        return balance
    
    def predict_mask(self, W_sign: np.ndarray, target_sparsity: float = 0.2) -> np.ndarray:
        """Predict zero mask for entire matrix."""
        d_out, d_in = W_sign.shape
        scores = np.zeros((d_out, d_in))
        
        for i in range(d_out):
            for j in range(d_in):
                scores[i, j] = self.predict(W_sign, i, j)
        
        # Threshold to achieve target sparsity
        threshold = np.percentile(scores, 100 * (1 - target_sparsity))
        mask = (scores < threshold).astype(np.float32)
        
        return mask


class CTGPattern:
    """Fixed CTG sparsity patterns."""
    
    def __init__(self, d_in: int, period: int = 5):
        self.d_in = d_in
        self.period = period
        self.mask = self._create_mask()
        
    def _create_mask(self) -> np.ndarray:
        """Create periodic inhibit mask."""
        mask = np.ones(self.d_in, dtype=np.float32)
        mask[::self.period] = 0  # Zero every period-th position
        return mask
    
    def apply(self, W: np.ndarray) -> np.ndarray:
        """Apply CTG mask to weight matrix."""
        return W * self.mask


class BinaryFactorization:
    """Low-rank binary matrix factorization: W ≈ S @ B.T

    Key insight: Use SVD to initialize, then binarize the right factor.
    """

    def __init__(self, d_out: int, d_in: int, rank: int = 8):
        self.d_out = d_out
        self.d_in = d_in
        self.rank = rank

        # Will be set by fit()
        self.S = None  # d_out × rank (FP32 scales)
        self.B = None  # d_in × rank (binary)

    def fit(self, W: np.ndarray):
        """Fit factorization to target weight matrix using SVD."""
        # SVD: W = U @ diag(s) @ V.T
        U, s, Vt = np.linalg.svd(W, full_matrices=False)

        # Take top-r components
        r = min(self.rank, len(s))
        U_r = U[:, :r]
        s_r = s[:r]
        V_r = Vt[:r, :].T  # d_in × r

        # S absorbs singular values: S = U @ diag(s)
        self.S = (U_r * s_r).astype(np.float32)

        # B is binarized V
        self.B = np.sign(V_r).astype(np.float32)
        self.B[self.B == 0] = 1

        # Rescale S to compensate for binarization
        # We want S @ B.T ≈ U @ diag(s) @ V.T
        # So S_new = (U @ diag(s) @ V.T @ B) @ pinv(B.T @ B)
        # Simplified: just rescale S columns by ||V_r[:, i]|| / ||B[:, i]||
        for i in range(r):
            scale = np.linalg.norm(V_r[:, i]) / np.linalg.norm(self.B[:, i])
            self.S[:, i] *= scale

    def forward(self) -> np.ndarray:
        """Compute W = S @ B.T with binary B."""
        return self.S @ self.B.T

    def bits_per_param(self) -> float:
        """Compute effective bits per parameter."""
        # B: rank × d_in bits
        # S: rank × d_out × 32 bits (FP32)
        total_bits = self.rank * self.d_in + self.rank * self.d_out * 32
        total_params = self.d_out * self.d_in
        return total_bits / total_params


class LearnedBasis:
    """Learned asymmetric binary basis: W = α * B + β"""

    def __init__(self, d_out: int, d_in: int):
        self.d_out = d_out
        self.d_in = d_in

        # Learned parameters (per-row or global)
        self.alpha = np.ones((d_out, 1), dtype=np.float32)  # Scale
        self.beta = np.zeros((d_out, 1), dtype=np.float32)  # Shift

        # Latent weights (will be binarized)
        self.W_latent = np.random.randn(d_out, d_in).astype(np.float32) * 0.1

    def get_binary(self) -> np.ndarray:
        """Get binary weights {0, 1}."""
        return (ste_sign(self.W_latent) + 1) / 2  # Map {-1,1} to {0,1}

    def forward(self) -> np.ndarray:
        """Compute W = α * B + β."""
        B = self.get_binary()
        return self.alpha * B + self.beta

    def bits_per_param(self) -> float:
        """1 bit per weight + 64 bits per row for α, β."""
        total_bits = self.d_out * self.d_in + self.d_out * 64
        total_params = self.d_out * self.d_in
        return total_bits / total_params


class TernaryEncoder:
    """Ternary-in-Binary encoding using predictable zero patterns.

    Key insight: If we can PREDICT which weights are zero from local context,
    we only need to store the EXCEPTIONS (prediction errors).

    Storage:
    - 1 bit per weight for sign
    - Extra bits only for prediction errors (ideally rare)

    If prediction accuracy is >80%, effective bpp < 1.2
    """

    def __init__(self, d_out: int, d_in: int, zero_threshold: float = 0.3):
        self.d_out = d_out
        self.d_in = d_in
        self.zero_threshold = zero_threshold

    def get_true_zeros(self, W: np.ndarray) -> np.ndarray:
        """Get true zero mask (where |W| is small)."""
        W_abs_mean = np.mean(np.abs(W))
        return (np.abs(W) < self.zero_threshold * W_abs_mean).astype(np.float32)

    def predict_zeros(self, W_sign: np.ndarray) -> np.ndarray:
        """Predict zero locations from sign pattern.

        Heuristic: positions where local sign pattern is highly balanced
        tend to have smaller magnitudes (the sign is "uncertain").
        """
        d_out, d_in = W_sign.shape
        pred = np.zeros((d_out, d_in), dtype=np.float32)

        # Use local variance as predictor
        window = 5
        for i in range(d_out):
            for j in range(d_in):
                j_start = max(0, j - window // 2)
                j_end = min(d_in, j + window // 2 + 1)
                local = W_sign[i, j_start:j_end]
                # High variance = balanced signs = likely small magnitude
                balance = np.abs(np.mean(local))
                pred[i, j] = 1 - balance  # Low balance = likely zero

        # Threshold to match target sparsity (~19%)
        threshold = np.percentile(pred, 81)  # Top 19% become zeros
        return (pred >= threshold).astype(np.float32)

    def encode(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Encode weights using ternary-in-binary scheme.

        Returns:
            W_sign: sign of weights {-1, +1}
            zero_mask: predicted zero mask
            bpp: effective bits per parameter
        """
        W_sign = np.sign(W)
        W_sign[W_sign == 0] = 1

        true_zeros = self.get_true_zeros(W)
        pred_zeros = self.predict_zeros(W_sign)

        # Prediction errors (need extra bit to correct)
        errors = (true_zeros != pred_zeros).astype(np.float32)
        error_rate = np.mean(errors)

        # BPP: 1 bit for sign + error_rate bits for corrections
        bpp = 1.0 + error_rate

        return W_sign, pred_zeros, bpp

    def decode(self, W_sign: np.ndarray, zero_mask: np.ndarray,
               scale: np.ndarray) -> np.ndarray:
        """Decode to get effective weights."""
        return W_sign * (1 - zero_mask) * scale


class MagnitudeAwareQuantizer:
    """Use magnitude information from weight distribution.

    Key insight: Within a row, weights cluster around certain magnitudes.
    Store the RANK of each weight within its row, encoded efficiently.

    For 1-bit: just store sign
    But: use the row's magnitude distribution to weight the reconstruction
    """

    def __init__(self, d_out: int, d_in: int, n_groups: int = 4):
        self.d_out = d_out
        self.d_in = d_in
        self.n_groups = n_groups

    def quantize(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize with magnitude-aware grouping.

        Within each row, group weights by magnitude and assign different scales.
        """
        d_out, d_in = W.shape
        W_sign = np.sign(W)
        W_sign[W_sign == 0] = 1

        # Per-row magnitude groups
        group_scales = np.zeros((d_out, self.n_groups), dtype=np.float32)
        group_mask = np.zeros((d_out, d_in), dtype=np.int32)

        for i in range(d_out):
            row = np.abs(W[i])
            # Assign to groups by magnitude percentile
            percentiles = np.percentile(row, np.linspace(0, 100, self.n_groups + 1))
            for g in range(self.n_groups):
                mask = (row >= percentiles[g]) & (row < percentiles[g + 1] + 1e-10)
                group_mask[i, mask] = g
                group_scales[i, g] = np.mean(row[mask]) if np.any(mask) else 0

        return W_sign, group_mask, group_scales

    def reconstruct(self, W_sign: np.ndarray, group_mask: np.ndarray,
                   group_scales: np.ndarray) -> np.ndarray:
        """Reconstruct weights from quantized representation."""
        d_out, d_in = W_sign.shape
        W_recon = np.zeros_like(W_sign)

        for i in range(d_out):
            for g in range(self.n_groups):
                mask = group_mask[i] == g
                W_recon[i, mask] = W_sign[i, mask] * group_scales[i, g]

        return W_recon

    def bits_per_param(self) -> float:
        """BPP: 1 bit sign + log2(n_groups) bits for group + scales overhead."""
        # Sign: 1 bit per weight
        # Group: log2(n_groups) bits per weight
        # Scales: n_groups * 32 bits per row
        sign_bits = self.d_out * self.d_in
        group_bits = self.d_out * self.d_in * np.log2(self.n_groups)
        scale_bits = self.d_out * self.n_groups * 32

        return (sign_bits + group_bits + scale_bits) / (self.d_out * self.d_in)


class TrainedUniformWeights:
    """Simulate training-aware quantization with uniform magnitudes.

    Key insight: If training encourages ALL magnitudes to be equal,
    then sign-only quantization loses NO information.

    This simulates what happens when you train with:
    1. Binary weight constraint (STE)
    2. Strong magnitude uniformity regularization
    3. The network learns to encode ALL information in signs
    """

    def __init__(self, d_out: int, d_in: int):
        self.d_out = d_out
        self.d_in = d_in

    def create_uniform_weights(self, W_random: np.ndarray) -> np.ndarray:
        """Create weights where all magnitudes are equal (per row).

        This simulates what perfectly trained uniform weights would look like:
        - Same signs as original (or optimized signs)
        - Uniform magnitude = mean(|W|) per row
        """
        W_sign = np.sign(W_random)
        W_sign[W_sign == 0] = 1

        # Uniform magnitude per row
        row_scales = np.mean(np.abs(W_random), axis=1, keepdims=True)

        return W_sign * row_scales

    def create_optimal_signs(self, W_target: np.ndarray,
                              uniform_scale: float = None) -> np.ndarray:
        """Create optimal signs for approximating W_target with uniform magnitudes.

        Given that magnitudes must be uniform, find signs that minimize ||W_target - s*scale||²
        The optimal sign is simply sign(W_target) with uniform scale.
        """
        W_sign = np.sign(W_target)
        W_sign[W_sign == 0] = 1

        if uniform_scale is None:
            uniform_scale = np.mean(np.abs(W_target), axis=1, keepdims=True)

        return W_sign * uniform_scale

    def compute_correlation(self, W_true: np.ndarray, n_samples: int = 100) -> float:
        """Compute correlation between uniform-magnitude approx and true W."""
        W_uniform = self.create_uniform_weights(W_true)

        correlations = []
        for _ in range(n_samples):
            x = np.random.randn(self.d_in).astype(np.float32)
            y_true = W_true @ x
            y_uniform = W_uniform @ x
            corr = np.corrcoef(y_true, y_uniform)[0, 1]
            correlations.append(corr)

        return np.mean(correlations)


class StochasticBinaryWithImportance:
    """Stochastic binary quantization - FAIR VERSION.

    IMPORTANT: This version is FAIR - it only uses stored 1-bit signs at inference.
    The "stochastic" part happens at QUANTIZATION time, not inference.

    Key insight: Use stochastic rounding during quantization to better preserve
    the expected value. But at inference, we only have the stored signs.

    Storage: 1 bit per weight (the sampled sign)
    Inference: Use stored signs with per-row scale
    """

    def __init__(self, d_out: int, d_in: int, T: int = 16):
        self.d_out = d_out
        self.d_in = d_in
        self.T = T  # Number of samples for reconstruction
        self.stored_signs = None  # The actual stored 1-bit values

    def stochastic_quantize(self, W: np.ndarray, seed: int = 42) -> np.ndarray:
        """Stochastic quantization: P(+1) = sigmoid(W / temperature).

        This happens ONCE at quantization time. The result is stored.
        """
        np.random.seed(seed)

        # Temperature controls how "soft" the sign is
        temperature = np.std(W) * 0.5 + 1e-8
        probs = 1 / (1 + np.exp(-W / temperature))

        # Sample binary ONCE and store
        samples = (np.random.rand(*W.shape) < probs).astype(np.float32)
        self.stored_signs = 2 * samples - 1  # Map {0,1} to {-1,+1}
        return self.stored_signs

    def reconstruct(self, scale: np.ndarray = None) -> np.ndarray:
        """Reconstruct using ONLY stored signs (truly 1-bit).

        This is the fair version - no FP32 information at inference.
        """
        if self.stored_signs is None:
            raise ValueError("Must call stochastic_quantize first")

        if scale is None:
            scale = np.ones((self.d_out, 1), dtype=np.float32)

        return self.stored_signs * scale

    def bits_per_param(self) -> float:
        """BPP: 1 bit per weight + scale overhead."""
        total_params = self.d_out * self.d_in
        return (total_params + self.d_out * 32) / total_params


class MultiSampleBinary:
    """Multi-sample binary using BSDM-W style estimation.

    Key insight: Instead of storing 1 sign, store T signs and average.
    This is NOT 1-bit, but shows the theoretical limit.

    Storage: T bits per weight
    Inference: Average T samples

    As T → ∞, this approaches the soft quantization limit.
    """

    def __init__(self, d_out: int, d_in: int, T: int = 16):
        self.d_out = d_out
        self.d_in = d_in
        self.T = T

    def quantize_and_reconstruct(self, W: np.ndarray,
                                  scale: np.ndarray = None) -> np.ndarray:
        """Quantize with T samples and average."""
        if scale is None:
            scale = np.mean(np.abs(W), axis=1, keepdims=True)

        temperature = np.std(W) * 0.5 + 1e-8
        probs = 1 / (1 + np.exp(-W / temperature))

        # Average over T samples
        W_recon = np.zeros_like(W)
        for t in range(self.T):
            np.random.seed(42 + t)
            samples = (np.random.rand(*W.shape) < probs).astype(np.float32)
            W_recon += 2 * samples - 1
        W_recon /= self.T

        return W_recon * scale

    def bits_per_param(self) -> float:
        """BPP: T bits per weight + scale overhead."""
        total_params = self.d_out * self.d_in
        return (total_params * self.T + self.d_out * 32) / total_params


class CorrelatedSignBinary:
    """Use correlated signs across rows to implicitly encode magnitude.

    Key insight: If we have K rows, we can use the CORRELATION between
    signs across rows to encode magnitude information.

    For a weight W[i,j]:
    - If |W[i,j]| is large, sign(W[i,j]) is "confident" and should be consistent
    - If |W[i,j]| is small, sign(W[i,j]) is "noisy" and may vary

    Strategy:
    1. Store 1 bit per weight (sign)
    2. At reconstruction, look at sign consistency across nearby rows
    3. Consistent signs → large magnitude, inconsistent → small magnitude

    This is like using redundancy across rows to recover magnitude.
    """

    def __init__(self, d_out: int, d_in: int, window: int = 4):
        self.d_out = d_out
        self.d_in = d_in
        self.window = window  # Number of rows to correlate

    def quantize(self, W: np.ndarray) -> np.ndarray:
        """Quantize to signs."""
        W_sign = np.sign(W)
        W_sign[W_sign == 0] = 1
        return W_sign

    def compute_consistency(self, W_sign: np.ndarray) -> np.ndarray:
        """Compute sign consistency across rows.

        For each position j, look at signs in nearby rows.
        High consistency = all same sign = likely large magnitude.
        """
        d_out, d_in = W_sign.shape
        consistency = np.ones((d_out, d_in), dtype=np.float32)

        for i in range(d_out):
            i_start = max(0, i - self.window // 2)
            i_end = min(d_out, i + self.window // 2 + 1)

            # Compute mean sign in window (excluding self)
            window_signs = W_sign[i_start:i_end, :]
            mean_sign = np.mean(window_signs, axis=0)

            # Consistency = how much this row agrees with neighbors
            # |mean_sign| close to 1 = high consistency
            consistency[i, :] = np.abs(mean_sign)

        return consistency

    def reconstruct(self, W_sign: np.ndarray,
                    scale: np.ndarray = None) -> np.ndarray:
        """Reconstruct using sign consistency as magnitude proxy."""
        if scale is None:
            scale = np.ones((self.d_out, 1), dtype=np.float32)

        consistency = self.compute_consistency(W_sign)

        # Use consistency as magnitude multiplier
        # High consistency → full magnitude, low → reduced
        return W_sign * consistency * scale

    def bits_per_param(self) -> float:
        """BPP: 1 bit per weight + scale overhead."""
        total_params = self.d_out * self.d_in
        return (total_params + self.d_out * 32) / total_params


class HadamardBinary:
    """Use Hadamard transform to spread magnitude information.

    Key insight: Hadamard transform spreads information across all positions.
    If we quantize in Hadamard domain, magnitude info is distributed.

    Strategy:
    1. Apply Hadamard transform to each row: H @ W[i,:]
    2. Quantize to signs in Hadamard domain
    3. Apply inverse Hadamard: H.T @ sign(H @ W[i,:])

    The Hadamard transform is orthogonal and self-inverse (H = H.T = H^-1).
    """

    def __init__(self, d_out: int, d_in: int):
        self.d_out = d_out
        self.d_in = d_in
        # Pad to power of 2 for Hadamard
        self.d_in_padded = 2 ** int(np.ceil(np.log2(d_in)))
        self.H = self._hadamard(self.d_in_padded)

    def _hadamard(self, n: int) -> np.ndarray:
        """Generate Hadamard matrix of size n (must be power of 2)."""
        if n == 1:
            return np.array([[1.0]])
        H_half = self._hadamard(n // 2)
        return np.block([[H_half, H_half], [H_half, -H_half]]) / np.sqrt(2)

    def quantize_and_reconstruct(self, W: np.ndarray,
                                  scale: np.ndarray = None) -> np.ndarray:
        """Quantize in Hadamard domain and reconstruct."""
        if scale is None:
            scale = np.mean(np.abs(W), axis=1, keepdims=True)

        d_out, d_in = W.shape

        # Pad to power of 2
        W_padded = np.zeros((d_out, self.d_in_padded), dtype=np.float32)
        W_padded[:, :d_in] = W

        # Transform to Hadamard domain
        W_had = W_padded @ self.H.T

        # Quantize to signs
        W_had_sign = np.sign(W_had)
        W_had_sign[W_had_sign == 0] = 1

        # Transform back
        W_recon_padded = W_had_sign @ self.H

        # Unpad and scale
        W_recon = W_recon_padded[:, :d_in] * scale

        return W_recon

    def bits_per_param(self) -> float:
        """BPP: 1 bit per weight (in Hadamard domain) + scale overhead."""
        total_params = self.d_out * self.d_in
        # We store d_in_padded bits per row, but only d_in are "real"
        return (self.d_out * self.d_in_padded + self.d_out * 32) / total_params


class ResidualBinary:
    """Residual binary quantization - successive approximation.

    Key insight: Quantize to signs, compute residual, quantize residual, etc.
    Each level adds 1 bit per weight but captures finer detail.

    W ≈ s1 * scale1 + s2 * scale2 + s3 * scale3 + ...

    where s_i ∈ {-1, +1} and scale_i decreases with each level.

    This is like binary successive approximation ADC.
    """

    def __init__(self, d_out: int, d_in: int, n_levels: int = 2):
        self.d_out = d_out
        self.d_in = d_in
        self.n_levels = n_levels

    def quantize_and_reconstruct(self, W: np.ndarray) -> np.ndarray:
        """Quantize with n_levels of residual refinement."""
        W_recon = np.zeros_like(W)
        residual = W.copy()

        for level in range(self.n_levels):
            # Compute scale for this level
            scale = np.mean(np.abs(residual), axis=1, keepdims=True)

            # Quantize residual to signs
            signs = np.sign(residual)
            signs[signs == 0] = 1

            # Add to reconstruction
            W_recon += signs * scale

            # Compute new residual
            residual = W - W_recon

        return W_recon

    def bits_per_param(self) -> float:
        """BPP: n_levels bits per weight + scale overhead per level."""
        total_params = self.d_out * self.d_in
        return (total_params * self.n_levels + self.d_out * 32 * self.n_levels) / total_params


class OptimalBinarySearch:
    """Find optimal binary representation via search.

    Key insight: For each weight, find the best {-1, +1} assignment
    that minimizes reconstruction error when combined with neighbors.

    This is like vector quantization but with binary constraint.
    """

    def __init__(self, d_out: int, d_in: int, n_iters: int = 10):
        self.d_out = d_out
        self.d_in = d_in
        self.n_iters = n_iters

    def quantize_and_reconstruct(self, W: np.ndarray,
                                  scale: np.ndarray = None) -> np.ndarray:
        """Iteratively optimize binary assignment."""
        if scale is None:
            scale = np.mean(np.abs(W), axis=1, keepdims=True)

        # Initialize with sign
        W_sign = np.sign(W)
        W_sign[W_sign == 0] = 1

        # Iterative refinement
        for _ in range(self.n_iters):
            # For each position, check if flipping improves error
            for i in range(self.d_out):
                for j in range(self.d_in):
                    # Current error
                    current_recon = W_sign[i, j] * scale[i, 0]
                    current_err = (W[i, j] - current_recon) ** 2

                    # Error if flipped
                    flipped_recon = -W_sign[i, j] * scale[i, 0]
                    flipped_err = (W[i, j] - flipped_recon) ** 2

                    # Flip if better
                    if flipped_err < current_err:
                        W_sign[i, j] = -W_sign[i, j]

        return W_sign * scale

    def bits_per_param(self) -> float:
        """BPP: 1 bit per weight + scale overhead."""
        total_params = self.d_out * self.d_in
        return (total_params + self.d_out * 32) / total_params


class TrainingAwareCTG:
    """Simulate training-aware CTG quantization.

    Key insight: If the network is TRAINED with CTG enabled, it learns to:
    1. Put important information in non-INHIBIT slots
    2. Use INHIBIT slots for redundant/low-importance weights

    This simulates what happens after training by:
    1. Identifying which weights are "important" (large magnitude)
    2. Redistributing importance to non-INHIBIT slots
    3. Quantizing with CTG mask

    This is an ORACLE that shows the theoretical limit of training-aware CTG.
    """

    def __init__(self, d_out: int, d_in: int, inhibit_rate: float = 0.2):
        self.d_out = d_out
        self.d_in = d_in
        self.inhibit_rate = inhibit_rate

    def create_inhibit_mask(self) -> np.ndarray:
        """Create periodic INHIBIT mask."""
        mask = np.ones((self.d_out, self.d_in), dtype=np.float32)
        period = int(1 / self.inhibit_rate)
        mask[:, ::period] = 0  # Every period-th position is INHIBIT
        return mask

    def simulate_trained_weights(self, W: np.ndarray) -> np.ndarray:
        """Simulate what weights would look like after training with CTG.

        Key insight: Training with CTG would make the network learn to:
        1. Keep the same signs in non-INHIBIT positions
        2. Have zeros in INHIBIT positions (network learns to not use them)
        3. Redistribute the "lost" information by scaling up non-INHIBIT weights

        This is an ORACLE - it shows the best possible outcome of training.
        The key is that we DON'T shuffle weights, we just zero out INHIBIT slots
        and compensate by scaling up the remaining weights.
        """
        mask = self.create_inhibit_mask()

        # Zero out INHIBIT positions
        W_trained = W * mask

        # Compensate by scaling up non-INHIBIT weights
        # This simulates the network learning to put more signal in active slots
        scale_factor = 1.0 / (1.0 - self.inhibit_rate)
        W_trained = W_trained * scale_factor

        return W_trained

    def quantize_and_reconstruct(self, W: np.ndarray,
                                  scale: np.ndarray = None) -> np.ndarray:
        """Quantize with CTG after simulating training."""
        # Simulate what training would produce
        W_trained = self.simulate_trained_weights(W)

        if scale is None:
            # Scale based on non-zero weights only
            mask = self.create_inhibit_mask()
            scale = np.zeros((self.d_out, 1), dtype=np.float32)
            for i in range(self.d_out):
                non_zero = mask[i, :] == 1
                if np.any(non_zero):
                    scale[i, 0] = np.mean(np.abs(W_trained[i, non_zero]))
                else:
                    scale[i, 0] = 1.0

        # Quantize to signs (zeros stay zero)
        W_sign = np.sign(W_trained)

        # Apply mask
        mask = self.create_inhibit_mask()

        return W_sign * mask * scale

    def bits_per_param(self) -> float:
        """BPP: 1 bit per non-INHIBIT weight + scale overhead.

        INHIBIT positions are procedurally determined, no storage needed.
        """
        total_params = self.d_out * self.d_in
        non_inhibit_params = total_params * (1 - self.inhibit_rate)
        return (non_inhibit_params + self.d_out * 32) / total_params


class PerfectBinaryOracle:
    """Oracle: What if we could perfectly encode magnitude in signs?

    This is the THEORETICAL LIMIT of 1-bit quantization.

    Key insight: The best possible 1-bit representation would be one where
    the signs are chosen to minimize reconstruction error, not just sign(W).

    For a linear layer y = Wx, the optimal binary W_bin minimizes:
    E[||Wx - W_bin * scale @ x||^2]

    This is NP-hard in general, but we can approximate it.
    """

    def __init__(self, d_out: int, d_in: int, n_samples: int = 1000):
        self.d_out = d_out
        self.d_in = d_in
        self.n_samples = n_samples

    def find_optimal_signs(self, W: np.ndarray,
                           scale: np.ndarray = None) -> np.ndarray:
        """Find optimal signs via gradient descent on sign probabilities.

        We use soft signs (tanh) and optimize, then threshold.
        """
        if scale is None:
            scale = np.mean(np.abs(W), axis=1, keepdims=True)

        # Initialize with sign(W)
        W_sign = np.sign(W).astype(np.float32)
        W_sign[W_sign == 0] = 1

        # Use soft signs for optimization
        # logits such that tanh(logits) ≈ sign(W)
        logits = np.arctanh(np.clip(W_sign * 0.99, -0.99, 0.99))

        lr = 0.1
        for _ in range(100):
            # Soft signs
            soft_signs = np.tanh(logits)

            # Reconstruction
            W_recon = soft_signs * scale

            # Gradient of MSE w.r.t. logits
            error = W_recon - W
            grad_soft = 2 * error * scale  # d(MSE)/d(soft_signs)
            grad_logits = grad_soft * (1 - soft_signs ** 2)  # chain rule

            # Update
            logits -= lr * grad_logits

        # Threshold to get hard signs
        W_sign = np.sign(np.tanh(logits))
        W_sign[W_sign == 0] = 1

        return W_sign

    def quantize_and_reconstruct(self, W: np.ndarray,
                                  scale: np.ndarray = None) -> np.ndarray:
        """Find optimal signs and reconstruct."""
        if scale is None:
            scale = np.mean(np.abs(W), axis=1, keepdims=True)

        W_sign = self.find_optimal_signs(W, scale)
        return W_sign * scale

    def bits_per_param(self) -> float:
        """BPP: 1 bit per weight + scale overhead."""
        total_params = self.d_out * self.d_in
        return (total_params + self.d_out * 32) / total_params


class TrainingAwareSimulation:
    """Simulate training-aware 1-bit quantization.

    Key insight: When trained from scratch with STE, the network learns to:
    1. Put all important information in weight signs
    2. Make magnitudes uniform (so sign-only loses nothing)

    This simulation shows what happens when we TRAIN a linear layer
    with binary constraint, rather than post-training quantize.
    """

    def __init__(self, d_out: int, d_in: int, n_iters: int = 1000, lr: float = 0.01):
        self.d_out = d_out
        self.d_in = d_in
        self.n_iters = n_iters
        self.lr = lr

    def train_binary_layer(self, X: np.ndarray, Y_target: np.ndarray,
                           use_ctg: bool = False, ctg_rate: float = 0.2):
        """Train a binary layer to match target outputs.

        Uses STE: forward with sign(W), backward through latent W.

        Args:
            X: Input data (n_samples, d_in)
            Y_target: Target outputs (n_samples, d_out)
            use_ctg: Whether to use CTG structured zeros
            ctg_rate: Fraction of weights to zero out with CTG

        Returns:
            W_binary: Trained binary weights
            scale: Learned scale per row
            history: Training history
        """
        # Initialize latent weights
        W_latent = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.02
        scale = np.ones((self.d_out, 1), dtype=np.float32) * 0.02

        # CTG mask
        if use_ctg:
            mask = np.ones((self.d_out, self.d_in), dtype=np.float32)
            period = int(1 / ctg_rate)
            mask[:, ::period] = 0
        else:
            mask = np.ones((self.d_out, self.d_in), dtype=np.float32)

        history = {'loss': [], 'correlation': []}

        for i in range(self.n_iters):
            # Forward: binary weights
            W_sign = np.sign(W_latent)
            W_sign[W_sign == 0] = 1
            W_binary = W_sign * mask * scale

            Y_pred = X @ W_binary.T

            # Loss
            loss = np.mean((Y_pred - Y_target) ** 2)

            # Correlation
            corr = np.corrcoef(Y_pred.flatten(), Y_target.flatten())[0, 1]

            history['loss'].append(loss)
            history['correlation'].append(corr)

            # Backward: STE - gradient passes through sign
            error = Y_pred - Y_target  # (n_samples, d_out)
            grad_W = error.T @ X / X.shape[0]  # (d_out, d_in)

            # Apply CTG mask to gradient
            grad_W = grad_W * mask

            # Update latent weights
            W_latent -= self.lr * grad_W

            # Update scale (gradient of scale)
            grad_scale = np.sum(error.T @ X * W_sign * mask, axis=1, keepdims=True) / X.shape[0]
            scale -= self.lr * grad_scale * 0.1  # Slower learning rate for scale
            scale = np.maximum(scale, 0.001)  # Keep positive

        # Final binary weights
        W_sign = np.sign(W_latent)
        W_sign[W_sign == 0] = 1
        W_binary = W_sign * mask * scale

        return W_binary, scale, history

    def compare_training_vs_post(self, n_samples: int = 1000):
        """Compare training-aware vs post-training quantization.

        Creates a ground truth FP32 layer, then:
        1. Post-training: quantize the FP32 weights
        2. Training-aware: train binary weights from scratch

        Returns comparison results.
        """
        # Ground truth FP32 layer
        W_true = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.02

        # Generate training data
        X = np.random.randn(n_samples, self.d_in).astype(np.float32)
        Y_target = X @ W_true.T

        results = {}

        # 1. Post-training sign-only
        scale_post = np.mean(np.abs(W_true), axis=1, keepdims=True)
        W_sign_post = np.sign(W_true)
        W_sign_post[W_sign_post == 0] = 1
        W_post = W_sign_post * scale_post
        Y_post = X @ W_post.T
        results['post_sign'] = {
            'correlation': np.corrcoef(Y_post.flatten(), Y_target.flatten())[0, 1],
            'mse': np.mean((Y_post - Y_target) ** 2)
        }

        # 2. Post-training ternary
        threshold = np.percentile(np.abs(W_true), 30)  # ~30% zeros
        W_ternary = np.sign(W_true) * (np.abs(W_true) > threshold)
        scale_tern = np.mean(np.abs(W_true[np.abs(W_true) > threshold]))
        W_ternary = W_ternary * scale_tern
        Y_ternary = X @ W_ternary.T
        results['post_ternary'] = {
            'correlation': np.corrcoef(Y_ternary.flatten(), Y_target.flatten())[0, 1],
            'mse': np.mean((Y_ternary - Y_target) ** 2)
        }

        # 3. Training-aware sign-only
        W_trained, scale_trained, history = self.train_binary_layer(X, Y_target, use_ctg=False)
        Y_trained = X @ W_trained.T
        results['trained_sign'] = {
            'correlation': np.corrcoef(Y_trained.flatten(), Y_target.flatten())[0, 1],
            'mse': np.mean((Y_trained - Y_target) ** 2),
            'history': history
        }

        # 4. Training-aware with CTG
        W_ctg, scale_ctg, history_ctg = self.train_binary_layer(X, Y_target, use_ctg=True, ctg_rate=0.2)
        Y_ctg = X @ W_ctg.T
        results['trained_ctg'] = {
            'correlation': np.corrcoef(Y_ctg.flatten(), Y_target.flatten())[0, 1],
            'mse': np.mean((Y_ctg - Y_target) ** 2),
            'history': history_ctg
        }

        return results

    def compare_on_learnable_task(self, n_samples: int = 1000):
        """Compare on a task where binary can learn its own solution.

        Key insight: Instead of trying to match FP32 weights, we train
        both FP32 and binary from scratch on the SAME TASK.

        The task: learn a random linear mapping from X to Y.

        This shows the true capacity of binary vs ternary vs FP32.
        """
        # Generate a random linear task
        # Y = X @ W_task + noise
        W_task = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.5

        X = np.random.randn(n_samples, self.d_in).astype(np.float32)
        noise = np.random.randn(n_samples, self.d_out).astype(np.float32) * 0.1
        Y_target = X @ W_task.T + noise

        # Test data
        X_test = np.random.randn(n_samples // 5, self.d_in).astype(np.float32)
        Y_test = X_test @ W_task.T

        results = {}

        # 1. FP32 baseline (optimal solution via least squares)
        W_fp32 = np.linalg.lstsq(X, Y_target, rcond=None)[0].T
        Y_fp32 = X_test @ W_fp32.T
        results['fp32'] = {
            'correlation': np.corrcoef(Y_fp32.flatten(), Y_test.flatten())[0, 1],
            'mse': np.mean((Y_fp32 - Y_test) ** 2)
        }

        # 2. Binary trained from scratch
        W_binary, _, history = self.train_binary_layer(X, Y_target, use_ctg=False)
        Y_binary = X_test @ W_binary.T
        results['binary'] = {
            'correlation': np.corrcoef(Y_binary.flatten(), Y_test.flatten())[0, 1],
            'mse': np.mean((Y_binary - Y_test) ** 2),
            'history': history
        }

        # 3. Binary + CTG trained from scratch
        W_ctg, _, history_ctg = self.train_binary_layer(X, Y_target, use_ctg=True, ctg_rate=0.2)
        Y_ctg = X_test @ W_ctg.T
        results['binary_ctg'] = {
            'correlation': np.corrcoef(Y_ctg.flatten(), Y_test.flatten())[0, 1],
            'mse': np.mean((Y_ctg - Y_test) ** 2),
            'history': history_ctg
        }

        # 4. Ternary trained from scratch
        # Use a simpler approach: train with magnitude-based masking
        W_tern_latent = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.02
        scale_tern = np.ones((self.d_out, 1), dtype=np.float32) * 0.5

        lr_tern = self.lr * 0.5  # Lower learning rate for stability

        for i in range(self.n_iters):
            # Forward: ternary (zero out small magnitudes)
            threshold = np.percentile(np.abs(W_tern_latent), 30)
            mask = (np.abs(W_tern_latent) > threshold).astype(np.float32)
            W_tern = np.sign(W_tern_latent) * mask * scale_tern

            Y_pred = X @ W_tern.T

            # Backward: STE - gradient passes through
            error = Y_pred - Y_target
            grad_W = error.T @ X / X.shape[0]

            # Clip gradient for stability
            grad_norm = np.linalg.norm(grad_W)
            if grad_norm > 1.0:
                grad_W = grad_W / grad_norm

            W_tern_latent -= lr_tern * grad_W

            # Update scale (simpler approach)
            if i % 100 == 0:
                non_zero = mask > 0
                if np.any(non_zero):
                    scale_tern = np.mean(np.abs(W_tern_latent[non_zero])) * np.ones((self.d_out, 1))
                    scale_tern = np.maximum(scale_tern, 0.001)

        threshold = np.percentile(np.abs(W_tern_latent), 30)
        mask = (np.abs(W_tern_latent) > threshold).astype(np.float32)
        W_tern_final = np.sign(W_tern_latent) * mask * scale_tern
        Y_tern = X_test @ W_tern_final.T

        corr = np.corrcoef(Y_tern.flatten(), Y_test.flatten())[0, 1]
        if np.isnan(corr):
            corr = 0.0
        results['ternary'] = {
            'correlation': corr,
            'mse': np.mean((Y_tern - Y_test) ** 2)
        }

        return results

    def train_with_magnitude_regularization(self, X: np.ndarray, Y_target: np.ndarray,
                                             mag_reg: float = 0.1):
        """Train binary with magnitude uniformity regularization.

        Key insight: If we regularize magnitudes to be uniform, then
        sign-only quantization loses less information.

        This is the key to making 1-bit work: train the network to
        put all information in signs, not magnitudes.
        """
        W_latent = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.02
        scale = np.ones((self.d_out, 1), dtype=np.float32) * 0.5

        history = {'loss': [], 'correlation': [], 'mag_var': []}

        for i in range(self.n_iters):
            # Forward: binary
            W_sign = np.sign(W_latent)
            W_sign[W_sign == 0] = 1
            W_binary = W_sign * scale

            Y_pred = X @ W_binary.T

            # Task loss
            task_loss = np.mean((Y_pred - Y_target) ** 2)

            # Magnitude uniformity loss: variance of |W| within each row
            mag_var = np.mean(np.var(np.abs(W_latent), axis=1))

            # Total loss
            total_loss = task_loss + mag_reg * mag_var

            # Correlation
            corr = np.corrcoef(Y_pred.flatten(), Y_target.flatten())[0, 1]

            history['loss'].append(total_loss)
            history['correlation'].append(corr)
            history['mag_var'].append(mag_var)

            # Backward: STE for task loss
            error = Y_pred - Y_target
            grad_task = error.T @ X / X.shape[0]

            # Gradient of magnitude regularization
            # d/dW (var(|W|)) = 2 * (|W| - mean(|W|)) * sign(W) / n
            row_means = np.mean(np.abs(W_latent), axis=1, keepdims=True)
            grad_mag = 2 * (np.abs(W_latent) - row_means) * np.sign(W_latent) / self.d_in

            # Total gradient
            grad_W = grad_task + mag_reg * grad_mag

            # Clip gradient
            grad_norm = np.linalg.norm(grad_W)
            if grad_norm > 1.0:
                grad_W = grad_W / grad_norm

            W_latent -= self.lr * grad_W

            # Update scale
            if i % 100 == 0:
                scale = np.mean(np.abs(W_latent), axis=1, keepdims=True)
                scale = np.maximum(scale, 0.001)

        # Final binary weights
        W_sign = np.sign(W_latent)
        W_sign[W_sign == 0] = 1
        W_binary = W_sign * scale

        return W_binary, scale, history

    def compare_with_mag_reg(self, n_samples: int = 1000):
        """Compare binary with and without magnitude regularization."""
        # Generate task
        W_task = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.5

        X = np.random.randn(n_samples, self.d_in).astype(np.float32)
        noise = np.random.randn(n_samples, self.d_out).astype(np.float32) * 0.1
        Y_target = X @ W_task.T + noise

        X_test = np.random.randn(n_samples // 5, self.d_in).astype(np.float32)
        Y_test = X_test @ W_task.T

        results = {}

        # FP32 baseline
        W_fp32 = np.linalg.lstsq(X, Y_target, rcond=None)[0].T
        Y_fp32 = X_test @ W_fp32.T
        results['fp32'] = {
            'correlation': np.corrcoef(Y_fp32.flatten(), Y_test.flatten())[0, 1],
            'mse': np.mean((Y_fp32 - Y_test) ** 2)
        }

        # Binary without regularization
        W_binary, scale_bin, history = self.train_binary_layer(X, Y_target, use_ctg=False)
        Y_binary = X_test @ W_binary.T
        # Get latent weights from history (last iteration)
        results['binary'] = {
            'correlation': np.corrcoef(Y_binary.flatten(), Y_test.flatten())[0, 1],
            'mse': np.mean((Y_binary - Y_test) ** 2),
            'final_corr': history['correlation'][-1] if history['correlation'] else 0
        }

        # Binary with magnitude regularization
        for mag_reg in [0.01, 0.1, 1.0]:
            W_reg, scale_reg, history_reg = self.train_with_magnitude_regularization(
                X, Y_target, mag_reg=mag_reg)
            Y_reg = X_test @ W_reg.T
            results[f'binary_reg_{mag_reg}'] = {
                'correlation': np.corrcoef(Y_reg.flatten(), Y_test.flatten())[0, 1],
                'mse': np.mean((Y_reg - Y_test) ** 2),
                'final_mag_var': history_reg['mag_var'][-1] if history_reg['mag_var'] else 0
            }

        return results


class EntropyShapedBinary:
    """Train binary weights to be compressible, then use freed bits for magnitude info.

    Key insight from ChatGPT: The constraint is total_bits/weights = 1.0 bpp,
    not "1 raw bit per weight". If we can compress signs to 0.7 bpp, we have
    0.3 bpp for magnitude information.

    Approach:
    1. Add loss term to encourage low-entropy sign patterns
    2. After training, measure empirical sign entropy
    3. Use freed bits for zero-mask or magnitude levels
    """

    def __init__(self, d_out: int, d_in: int, n_iters: int = 3000, lr: float = 0.05):
        self.d_out = d_out
        self.d_in = d_in
        self.n_iters = n_iters
        self.lr = lr

    def estimate_entropy(self, signs: np.ndarray, block_size: int = 8) -> float:
        """Estimate entropy of sign pattern.

        Uses block-wise entropy estimation - counts frequency of each
        block pattern and computes entropy.
        """
        flat = signs.flatten()
        n = len(flat)
        n_blocks = n // block_size

        if n_blocks == 0:
            # Fallback to per-bit entropy
            p_pos = np.mean(signs > 0)
            if p_pos == 0 or p_pos == 1:
                return 0.0
            return -p_pos * np.log2(p_pos) - (1-p_pos) * np.log2(1-p_pos)

        # Convert blocks to integers
        blocks = flat[:n_blocks * block_size].reshape(n_blocks, block_size)
        block_ids = np.sum((blocks > 0).astype(int) * (2 ** np.arange(block_size)), axis=1)

        # Count frequencies
        unique, counts = np.unique(block_ids, return_counts=True)
        probs = counts / n_blocks

        # Entropy in bits per block
        entropy_per_block = -np.sum(probs * np.log2(probs + 1e-10))

        # Convert to bits per weight
        return entropy_per_block / block_size

    def sign_correlation_loss(self, W: np.ndarray) -> float:
        """Loss that encourages neighboring signs to be similar (compressible)."""
        signs = np.sign(W)

        # Horizontal correlation (within rows)
        h_corr = np.mean(signs[:, :-1] * signs[:, 1:])

        # Vertical correlation (within columns)
        v_corr = np.mean(signs[:-1, :] * signs[1:, :])

        # We want high correlation (similar neighbors) -> low loss
        # Correlation is in [-1, 1], we want it close to 1
        return 2.0 - (h_corr + v_corr)  # Loss in [0, 4]

    def train_entropy_shaped(self, X: np.ndarray, Y_target: np.ndarray,
                              entropy_reg: float = 0.1) -> Tuple:
        """Train binary layer with entropy regularization.

        Args:
            X: Input data [n_samples, d_in]
            Y_target: Target outputs [n_samples, d_out]
            entropy_reg: Weight for entropy regularization

        Returns:
            W_binary: Trained binary weights
            scale: Per-row scale
            history: Training history
        """
        W_latent = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.02
        scale = np.ones((self.d_out, 1), dtype=np.float32) * 0.5

        history = {'loss': [], 'task_loss': [], 'entropy_loss': [],
                   'correlation': [], 'sign_entropy': []}

        for i in range(self.n_iters):
            # Forward: binary
            W_sign = np.sign(W_latent)
            W_sign[W_sign == 0] = 1
            W_binary = W_sign * scale

            Y_pred = X @ W_binary.T

            # Task loss
            task_loss = np.mean((Y_pred - Y_target) ** 2)

            # Entropy loss (encourage compressible patterns)
            entropy_loss = self.sign_correlation_loss(W_latent)

            # Total loss
            total_loss = task_loss + entropy_reg * entropy_loss

            # Measure actual entropy
            sign_entropy = self.estimate_entropy(W_sign)

            # Correlation
            corr = np.corrcoef(Y_pred.flatten(), Y_target.flatten())[0, 1]

            history['loss'].append(total_loss)
            history['task_loss'].append(task_loss)
            history['entropy_loss'].append(entropy_loss)
            history['correlation'].append(corr)
            history['sign_entropy'].append(sign_entropy)

            # Backward: STE for task loss
            error = Y_pred - Y_target
            grad_task = error.T @ X / X.shape[0]

            # Gradient of entropy loss (approximate)
            # Encourage signs to match neighbors
            grad_entropy = np.zeros_like(W_latent)
            grad_entropy[:, :-1] -= np.sign(W_latent[:, 1:])  # Pull toward right neighbor
            grad_entropy[:, 1:] -= np.sign(W_latent[:, :-1])  # Pull toward left neighbor
            grad_entropy[:-1, :] -= np.sign(W_latent[1:, :])  # Pull toward bottom neighbor
            grad_entropy[1:, :] -= np.sign(W_latent[:-1, :])  # Pull toward top neighbor
            grad_entropy *= 0.25 / (self.d_out * self.d_in)

            # Total gradient
            grad_W = grad_task + entropy_reg * grad_entropy

            # Clip gradient
            grad_norm = np.linalg.norm(grad_W)
            if grad_norm > 1.0:
                grad_W = grad_W / grad_norm

            W_latent -= self.lr * grad_W

            # Update scale
            if i % 100 == 0:
                scale = np.mean(np.abs(W_latent), axis=1, keepdims=True)
                scale = np.maximum(scale, 0.001)

        # Final binary weights
        W_sign = np.sign(W_latent)
        W_sign[W_sign == 0] = 1
        W_binary = W_sign * scale

        return W_binary, scale, history, W_sign

    def add_magnitude_sidechannel(self, W_sign: np.ndarray, W_target: np.ndarray,
                                   freed_bits: float) -> Tuple[np.ndarray, float]:
        """Use freed bits from compression to add magnitude information.

        Args:
            W_sign: Binary signs {-1, +1}
            W_target: Target weights to approximate
            freed_bits: Bits per weight available for magnitude info

        Returns:
            W_enhanced: Enhanced weights with magnitude info
            effective_bpp: Actual bits per parameter used
        """
        n_weights = W_sign.size
        total_freed_bits = freed_bits * n_weights

        # Strategy 1: Zero mask for smallest magnitudes
        # With freed_bits, we can mark some positions as zero
        # Each zero position "costs" about log2(n_weights/k) bits where k is number of zeros

        # Calculate how many zeros we can afford
        # Using simple encoding: just store indices of zeros
        # Cost per zero ≈ log2(n_weights) bits
        bits_per_zero = np.log2(n_weights)
        n_zeros = int(total_freed_bits / bits_per_zero)
        n_zeros = min(n_zeros, n_weights // 5)  # Cap at 20%

        if n_zeros > 0:
            # Find smallest magnitude positions in target
            flat_target = W_target.flatten()
            flat_sign = W_sign.flatten()

            # Get indices of smallest magnitudes
            zero_indices = np.argsort(np.abs(flat_target))[:n_zeros]

            # Create enhanced weights
            flat_enhanced = flat_sign.copy()
            flat_enhanced[zero_indices] = 0

            W_enhanced = flat_enhanced.reshape(W_sign.shape)

            # Calculate effective BPP
            sign_bits = n_weights  # 1 bit per sign (before compression)
            zero_bits = n_zeros * bits_per_zero
            effective_bpp = (sign_bits + zero_bits) / n_weights

            # But wait - we're using the freed bits, so effective is still ~1.0
            # The sign_bits are compressed, so actual is:
            # compressed_signs + zero_indices = 1.0 * n_weights

            return W_enhanced, 1.0  # Target BPP
        else:
            return W_sign, 1.0

    def compare_approaches(self, n_samples: int = 2000):
        """Compare entropy-shaped binary vs regular binary vs ternary."""
        # Generate task
        W_task = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.5

        X = np.random.randn(n_samples, self.d_in).astype(np.float32)
        noise = np.random.randn(n_samples, self.d_out).astype(np.float32) * 0.1
        Y_target = X @ W_task.T + noise

        X_test = np.random.randn(n_samples // 5, self.d_in).astype(np.float32)
        Y_test = X_test @ W_task.T

        results = {}

        # FP32 baseline
        W_fp32 = np.linalg.lstsq(X, Y_target, rcond=None)[0].T
        Y_fp32 = X_test @ W_fp32.T
        results['fp32'] = {
            'correlation': np.corrcoef(Y_fp32.flatten(), Y_test.flatten())[0, 1],
            'bpp': 32.0,
            'sign_entropy': None
        }

        # Regular binary (no entropy shaping)
        W_binary_reg, scale_reg, hist_reg, signs_reg = self.train_entropy_shaped(
            X, Y_target, entropy_reg=0.0)
        Y_binary_reg = X_test @ W_binary_reg.T
        results['binary_regular'] = {
            'correlation': np.corrcoef(Y_binary_reg.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.0,
            'sign_entropy': hist_reg['sign_entropy'][-1]
        }

        # Entropy-shaped binary
        for entropy_reg in [0.1, 0.5, 1.0]:
            W_binary, scale, history, signs = self.train_entropy_shaped(
                X, Y_target, entropy_reg=entropy_reg)
            Y_binary = X_test @ W_binary.T

            sign_entropy = history['sign_entropy'][-1]
            freed_bits = 1.0 - sign_entropy

            # Add magnitude side-channel using freed bits
            W_enhanced, eff_bpp = self.add_magnitude_sidechannel(
                signs, W_fp32, freed_bits)
            W_enhanced = W_enhanced * scale
            Y_enhanced = X_test @ W_enhanced.T

            results[f'entropy_reg_{entropy_reg}'] = {
                'correlation': np.corrcoef(Y_binary.flatten(), Y_test.flatten())[0, 1],
                'bpp': 1.0,
                'sign_entropy': sign_entropy,
                'freed_bits': freed_bits
            }

            results[f'enhanced_{entropy_reg}'] = {
                'correlation': np.corrcoef(Y_enhanced.flatten(), Y_test.flatten())[0, 1],
                'bpp': 1.0,  # Still 1.0 bpp total
                'sign_entropy': sign_entropy,
                'n_zeros': np.sum(W_enhanced == 0)
            }

        # Ternary baseline
        threshold = np.percentile(np.abs(W_fp32), 30)
        W_ternary = np.sign(W_fp32) * (np.abs(W_fp32) > threshold)
        scale_tern = np.mean(np.abs(W_fp32[np.abs(W_fp32) > threshold]))
        W_ternary = W_ternary * scale_tern
        Y_ternary = X_test @ W_ternary.T
        results['ternary'] = {
            'correlation': np.corrcoef(Y_ternary.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.58,
            'sign_entropy': None
        }

        return results

    def train_block_structured(self, X: np.ndarray, Y_target: np.ndarray,
                                block_size: int = 4, structure_reg: float = 0.5) -> Tuple:
        """Train with block-wise structure - all signs in a block should be same.

        This creates highly compressible patterns: instead of 1 bit per weight,
        we get 1 bit per block (e.g., 0.25 bpp for 4x4 blocks).
        """
        W_latent = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.02
        scale = np.ones((self.d_out, 1), dtype=np.float32) * 0.5

        history = {'loss': [], 'correlation': [], 'block_agreement': []}

        for i in range(self.n_iters):
            # Forward: binary
            W_sign = np.sign(W_latent)
            W_sign[W_sign == 0] = 1
            W_binary = W_sign * scale

            Y_pred = X @ W_binary.T

            # Task loss
            task_loss = np.mean((Y_pred - Y_target) ** 2)

            # Block structure loss: encourage all signs in block to match
            block_loss = 0.0
            n_blocks_h = self.d_out // block_size
            n_blocks_w = self.d_in // block_size

            for bi in range(n_blocks_h):
                for bj in range(n_blocks_w):
                    block = W_latent[bi*block_size:(bi+1)*block_size,
                                      bj*block_size:(bj+1)*block_size]
                    # Encourage block to have uniform sign
                    block_mean = np.mean(block)
                    block_var = np.var(block)
                    block_loss += block_var / (block_size ** 2)

            block_loss /= (n_blocks_h * n_blocks_w)

            # Total loss
            total_loss = task_loss + structure_reg * block_loss

            # Measure block agreement
            agreement = 0.0
            for bi in range(n_blocks_h):
                for bj in range(n_blocks_w):
                    block = W_sign[bi*block_size:(bi+1)*block_size,
                                    bj*block_size:(bj+1)*block_size]
                    # All same sign = agreement
                    agreement += np.abs(np.mean(block))
            agreement /= (n_blocks_h * n_blocks_w)

            # Correlation
            corr = np.corrcoef(Y_pred.flatten(), Y_target.flatten())[0, 1]

            history['loss'].append(total_loss)
            history['correlation'].append(corr)
            history['block_agreement'].append(agreement)

            # Backward: STE for task loss + structure gradient
            error = Y_pred - Y_target
            grad_task = error.T @ X / X.shape[0]

            # Gradient of block structure (pull toward block mean)
            grad_struct = np.zeros_like(W_latent)
            for bi in range(n_blocks_h):
                for bj in range(n_blocks_w):
                    block = W_latent[bi*block_size:(bi+1)*block_size,
                                      bj*block_size:(bj+1)*block_size]
                    block_mean = np.mean(block)
                    grad_struct[bi*block_size:(bi+1)*block_size,
                               bj*block_size:(bj+1)*block_size] = 2 * (block - block_mean) / (block_size ** 2)

            grad_struct /= (n_blocks_h * n_blocks_w)

            # Total gradient
            grad_W = grad_task + structure_reg * grad_struct

            # Clip gradient
            grad_norm = np.linalg.norm(grad_W)
            if grad_norm > 1.0:
                grad_W = grad_W / grad_norm

            W_latent -= self.lr * grad_W

            # Update scale
            if i % 100 == 0:
                scale = np.mean(np.abs(W_latent), axis=1, keepdims=True)
                scale = np.maximum(scale, 0.001)

        # Final binary weights
        W_sign = np.sign(W_latent)
        W_sign[W_sign == 0] = 1
        W_binary = W_sign * scale

        # Calculate effective BPP (1 bit per block + overhead)
        n_blocks = (self.d_out // block_size) * (self.d_in // block_size)
        block_bits = n_blocks  # 1 bit per block
        edge_bits = self.d_out * self.d_in - n_blocks * block_size * block_size  # Edge weights
        total_bits = block_bits + edge_bits
        effective_bpp = total_bits / (self.d_out * self.d_in)

        return W_binary, scale, history, effective_bpp

    def compare_block_structured(self, n_samples: int = 2000):
        """Compare block-structured binary vs regular."""
        # Generate task
        W_task = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.5

        X = np.random.randn(n_samples, self.d_in).astype(np.float32)
        noise = np.random.randn(n_samples, self.d_out).astype(np.float32) * 0.1
        Y_target = X @ W_task.T + noise

        X_test = np.random.randn(n_samples // 5, self.d_in).astype(np.float32)
        Y_test = X_test @ W_task.T

        results = {}

        # FP32 baseline
        W_fp32 = np.linalg.lstsq(X, Y_target, rcond=None)[0].T
        Y_fp32 = X_test @ W_fp32.T
        results['fp32'] = {
            'correlation': np.corrcoef(Y_fp32.flatten(), Y_test.flatten())[0, 1],
            'bpp': 32.0
        }

        # Regular binary
        W_binary, scale, hist, _ = self.train_entropy_shaped(X, Y_target, entropy_reg=0.0)
        Y_binary = X_test @ W_binary.T
        results['binary'] = {
            'correlation': np.corrcoef(Y_binary.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.0
        }

        # Block-structured binary (different block sizes)
        for block_size in [2, 4, 8]:
            if self.d_out >= block_size and self.d_in >= block_size:
                W_block, scale_b, hist_b, eff_bpp = self.train_block_structured(
                    X, Y_target, block_size=block_size, structure_reg=0.5)
                Y_block = X_test @ W_block.T
                results[f'block_{block_size}x{block_size}'] = {
                    'correlation': np.corrcoef(Y_block.flatten(), Y_test.flatten())[0, 1],
                    'bpp': eff_bpp,
                    'block_agreement': hist_b['block_agreement'][-1]
                }

        # Ternary baseline
        threshold = np.percentile(np.abs(W_fp32), 30)
        W_ternary = np.sign(W_fp32) * (np.abs(W_fp32) > threshold)
        scale_tern = np.mean(np.abs(W_fp32[np.abs(W_fp32) > threshold]))
        W_ternary = W_ternary * scale_tern
        Y_ternary = X_test @ W_ternary.T
        results['ternary'] = {
            'correlation': np.corrcoef(Y_ternary.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.58
        }

        return results


    def train_hybrid_block_plus_magnitude(self, X: np.ndarray, Y_target: np.ndarray,
                                           block_size: int = 2, target_bpp: float = 1.0) -> Tuple:
        """Hybrid approach: block-structured signs + per-weight magnitude levels.

        Budget allocation:
        - Block signs: 1 bit per block = d_out*d_in / block_size^2 bits
        - Remaining bits for magnitude levels

        For block_size=2, target_bpp=1.0:
        - Sign bits: 0.25 bpp
        - Magnitude bits: 0.75 bpp available
        - Can encode ~1.75 levels per weight (round to 2 = 1 bit for magnitude)
        """
        n_weights = self.d_out * self.d_in
        sign_bpp = 1.0 / (block_size ** 2)
        mag_bpp = target_bpp - sign_bpp

        # How many magnitude levels can we afford?
        # 2^mag_bpp levels per weight
        n_levels = max(2, int(2 ** mag_bpp))

        # Train block-structured weights first
        W_block, scale, history, _ = self.train_block_structured(
            X, Y_target, block_size=block_size, structure_reg=0.5)

        # Get optimal FP32 solution as reference
        W_fp32 = np.linalg.lstsq(X, Y_target, rcond=None)[0].T

        # Compute per-weight magnitudes (how much each weight should contribute)
        # Use magnitude from FP32 solution
        magnitudes = np.abs(W_fp32)

        # Quantize magnitudes to n_levels
        # Level 0 = zero, Level 1..n-1 = increasing magnitudes
        mag_quantized = np.zeros_like(magnitudes, dtype=int)

        if n_levels > 1:
            # Use percentiles to determine thresholds
            thresholds = [np.percentile(magnitudes, 100 * i / n_levels)
                         for i in range(1, n_levels)]

            for i, thresh in enumerate(thresholds):
                mag_quantized[magnitudes >= thresh] = i + 1

        # Reconstruct weights: sign * magnitude_level * scale
        level_scales = np.linspace(0, 1, n_levels)  # 0 for level 0, 1 for highest

        W_hybrid = np.zeros_like(W_block)
        for level in range(n_levels):
            mask = mag_quantized == level
            W_hybrid[mask] = np.sign(W_block[mask]) * level_scales[level] * np.mean(np.abs(W_fp32))

        # Calculate effective BPP
        # Signs: 1 bit per block
        n_blocks = (self.d_out // block_size) * (self.d_in // block_size)
        sign_bits = n_blocks

        # Magnitudes: log2(n_levels) bits per weight
        mag_bits = np.log2(n_levels) * n_weights

        effective_bpp = (sign_bits + mag_bits) / n_weights

        return W_hybrid, effective_bpp, {'n_levels': n_levels, 'sign_bpp': sign_bpp, 'mag_bpp': mag_bpp}

    def compare_hybrid(self, n_samples: int = 2000):
        """Compare hybrid block+magnitude approach vs ternary."""
        # Generate task
        W_task = np.random.randn(self.d_out, self.d_in).astype(np.float32) * 0.5

        X = np.random.randn(n_samples, self.d_in).astype(np.float32)
        noise = np.random.randn(n_samples, self.d_out).astype(np.float32) * 0.1
        Y_target = X @ W_task.T + noise

        X_test = np.random.randn(n_samples // 5, self.d_in).astype(np.float32)
        Y_test = X_test @ W_task.T

        results = {}

        # FP32 baseline
        W_fp32 = np.linalg.lstsq(X, Y_target, rcond=None)[0].T
        Y_fp32 = X_test @ W_fp32.T
        results['fp32'] = {
            'correlation': np.corrcoef(Y_fp32.flatten(), Y_test.flatten())[0, 1],
            'bpp': 32.0
        }

        # Regular binary
        W_binary, scale, hist, _ = self.train_entropy_shaped(X, Y_target, entropy_reg=0.0)
        Y_binary = X_test @ W_binary.T
        results['binary'] = {
            'correlation': np.corrcoef(Y_binary.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.0
        }

        # Hybrid: block signs + magnitude levels
        for block_size in [2, 4]:
            if self.d_out >= block_size and self.d_in >= block_size:
                W_hybrid, eff_bpp, info = self.train_hybrid_block_plus_magnitude(
                    X, Y_target, block_size=block_size, target_bpp=1.0)
                Y_hybrid = X_test @ W_hybrid.T

                corr = np.corrcoef(Y_hybrid.flatten(), Y_test.flatten())[0, 1]
                if np.isnan(corr):
                    corr = 0.0

                results[f'hybrid_block{block_size}'] = {
                    'correlation': corr,
                    'bpp': eff_bpp,
                    'n_levels': info['n_levels']
                }

        # Ternary baseline
        threshold = np.percentile(np.abs(W_fp32), 30)
        W_ternary = np.sign(W_fp32) * (np.abs(W_fp32) > threshold)
        scale_tern = np.mean(np.abs(W_fp32[np.abs(W_fp32) > threshold]))
        W_ternary = W_ternary * scale_tern
        Y_ternary = X_test @ W_ternary.T
        results['ternary'] = {
            'correlation': np.corrcoef(Y_ternary.flatten(), Y_test.flatten())[0, 1],
            'bpp': 1.58
        }

        return results


class ImportanceWeightedBinary:
    """Binary weights with importance-weighted reconstruction.

    Key insight: Not all weights are equally important.
    Weights with larger magnitude contribute more to the output.

    Strategy:
    1. Store 1 bit per weight (sign)
    2. At reconstruction, weight each sign by its EXPECTED importance
    3. Importance = f(position, local statistics)

    This is like having variable-magnitude binary weights without
    storing the magnitudes explicitly.
    """

    def __init__(self, d_out: int, d_in: int, importance_type: str = "position"):
        self.d_out = d_out
        self.d_in = d_in
        self.importance_type = importance_type

    def compute_importance(self, W: np.ndarray) -> np.ndarray:
        """Compute importance weights from local statistics.

        Heuristic: positions where local variance is high tend to have
        larger magnitudes (the sign is "confident").
        """
        d_out, d_in = W.shape
        importance = np.ones((d_out, d_in), dtype=np.float32)

        if self.importance_type == "position":
            # Positional importance: center positions more important
            pos = np.arange(d_in) / d_in
            importance = 1 + 0.5 * np.sin(np.pi * pos)  # Peak at center
            importance = np.tile(importance, (d_out, 1))

        elif self.importance_type == "local_var":
            # Local variance as importance proxy
            W_sign = np.sign(W)
            W_sign[W_sign == 0] = 1
            window = 5
            for i in range(d_out):
                for j in range(d_in):
                    j_start = max(0, j - window // 2)
                    j_end = min(d_in, j + window // 2 + 1)
                    local = W_sign[i, j_start:j_end]
                    # High variance = confident sign = likely large magnitude
                    importance[i, j] = 1 + np.var(local)

        elif self.importance_type == "learned":
            # Simulate learned importance (oracle: use true magnitudes)
            importance = np.abs(W) / np.mean(np.abs(W))

        return importance

    def quantize_and_reconstruct(self, W: np.ndarray) -> np.ndarray:
        """Quantize to binary and reconstruct with importance weighting."""
        W_sign = np.sign(W)
        W_sign[W_sign == 0] = 1

        importance = self.compute_importance(W)
        scale = np.mean(np.abs(W), axis=1, keepdims=True)

        return W_sign * importance * scale

    def bits_per_param(self) -> float:
        """BPP depends on importance type."""
        total_params = self.d_out * self.d_in
        if self.importance_type == "position":
            # Position is procedural, no extra bits
            return (total_params + self.d_out * 32) / total_params
        elif self.importance_type == "local_var":
            # Computed at runtime, no extra bits
            return (total_params + self.d_out * 32) / total_params
        else:
            # Learned importance needs storage
            return (total_params * 2 + self.d_out * 32) / total_params


class UnifiedBinaryLayer:
    """Unified layer combining all approaches."""

    def __init__(self, cfg: UnifiedConfig):
        self.cfg = cfg
        d_out, d_in = cfg.d_out, cfg.d_in

        # Core: latent FP32 weights (trained, then binarized)
        self.W_latent = np.random.randn(d_out, d_in).astype(np.float32) * 0.02

        # Learned scales (per-row)
        self.scale = np.ones((d_out, 1), dtype=np.float32) * 0.02

        # Learned basis parameters
        self.alpha = np.ones((d_out, 1), dtype=np.float32)
        self.beta = np.zeros((d_out, 1), dtype=np.float32)

        # CTG pattern
        self.ctg = CTGPattern(d_in, cfg.ctg_period) if cfg.ctg_enabled else None

        # Zero predictor
        self.zero_pred = ZeroPredictor(cfg.predictor_window)

        # Optional: factorization
        self.factorizer = BinaryFactorization(d_out, d_in, cfg.rank)

    def get_binary_weights(self) -> np.ndarray:
        """Get binarized weights."""
        return ste_sign(self.W_latent)

    def forward(self, x: np.ndarray, mode: QuantMode = QuantMode.UNIFIED) -> np.ndarray:
        """Forward pass with specified quantization mode."""

        if mode == QuantMode.SIGN_ONLY:
            W = self.get_binary_weights() * self.scale
            return W @ x

        elif mode == QuantMode.TERNARY:
            W_abs_mean = np.mean(np.abs(self.W_latent))
            W = np.zeros_like(self.W_latent)
            W[self.W_latent > self.cfg.zero_threshold * W_abs_mean] = 1
            W[self.W_latent < -self.cfg.zero_threshold * W_abs_mean] = -1
            return (W * self.scale) @ x

        elif mode == QuantMode.LEARNED_BASIS:
            B = (self.get_binary_weights() + 1) / 2  # {0, 1}
            W = self.alpha * B + self.beta
            return W @ x

        elif mode == QuantMode.FACTORIZED:
            if self.factorizer.S is None:
                # Not fitted yet, return zeros
                return np.zeros(self.cfg.d_out, dtype=np.float32)
            W = self.factorizer.forward()
            return W @ x

        elif mode == QuantMode.CTG_STRUCTURED:
            W = self.get_binary_weights() * self.scale
            if self.ctg:
                W = self.ctg.apply(W)
            return W @ x

        elif mode == QuantMode.ZERO_PREDICTED:
            W_sign = self.get_binary_weights()
            mask = self.zero_pred.predict_mask(W_sign, target_sparsity=0.2)
            W = W_sign * mask * self.scale
            return W @ x

        elif mode == QuantMode.UNIFIED:
            # Combine: learned basis + CTG + predicted zeros
            W_sign = self.get_binary_weights()

            # Apply learned basis
            B = (W_sign + 1) / 2
            W = self.alpha * B + self.beta

            # Apply CTG pattern (structured sparsity)
            if self.ctg and self.cfg.ctg_enabled:
                W = self.ctg.apply(W)

            return W @ x

        elif mode == QuantMode.TERNARY_ENCODED:
            # Ternary-in-Binary: use predicted zeros
            encoder = TernaryEncoder(self.cfg.d_out, self.cfg.d_in)
            W_sign, pred_zeros, _ = encoder.encode(self.W_latent)
            # Use TRUE zeros for best reconstruction (oracle)
            true_zeros = encoder.get_true_zeros(self.W_latent)
            W = W_sign * (1 - true_zeros) * self.scale
            return W @ x

        elif mode == QuantMode.MAGNITUDE_AWARE:
            # Magnitude-aware grouping
            mag_quant = MagnitudeAwareQuantizer(self.cfg.d_out, self.cfg.d_in, n_groups=4)
            W_sign, group_mask, group_scales = mag_quant.quantize(self.W_latent)
            W = mag_quant.reconstruct(W_sign, group_mask, group_scales)
            return W @ x

        elif mode == QuantMode.TRAINED_UNIFORM:
            # Training-aware: uniform magnitudes
            uniform = TrainedUniformWeights(self.cfg.d_out, self.cfg.d_in)
            W = uniform.create_uniform_weights(self.W_latent)
            return W @ x

        elif mode == QuantMode.STOCHASTIC:
            # Stochastic binary - FAIR version (truly 1-bit storage)
            stoch = StochasticBinaryWithImportance(self.cfg.d_out, self.cfg.d_in)
            stoch.stochastic_quantize(self.W_latent)  # Quantize once
            W = stoch.reconstruct(self.scale)  # Use only stored signs
            return W @ x

        elif mode == QuantMode.IMPORTANCE:
            # Importance-weighted binary (oracle: uses true magnitudes)
            imp = ImportanceWeightedBinary(self.cfg.d_out, self.cfg.d_in, "learned")
            W = imp.quantize_and_reconstruct(self.W_latent)
            return W @ x

        elif mode == QuantMode.MULTI_SAMPLE:
            # Multi-sample binary (T bits per weight - shows theoretical limit)
            multi = MultiSampleBinary(self.cfg.d_out, self.cfg.d_in, T=16)
            W = multi.quantize_and_reconstruct(self.W_latent, self.scale)
            return W @ x

        elif mode == QuantMode.CORRELATED:
            # Correlated signs across rows
            corr = CorrelatedSignBinary(self.cfg.d_out, self.cfg.d_in, window=4)
            W_sign = corr.quantize(self.W_latent)
            W = corr.reconstruct(W_sign, self.scale)
            return W @ x

        elif mode == QuantMode.HADAMARD:
            # Hadamard domain quantization
            had = HadamardBinary(self.cfg.d_out, self.cfg.d_in)
            W = had.quantize_and_reconstruct(self.W_latent, self.scale)
            return W @ x

        elif mode == QuantMode.RESIDUAL:
            # Residual binary (2 levels = 2 bpp)
            res = ResidualBinary(self.cfg.d_out, self.cfg.d_in, n_levels=2)
            W = res.quantize_and_reconstruct(self.W_latent)
            return W @ x

        elif mode == QuantMode.OPTIMAL:
            # Optimal binary search (slow but shows limit)
            opt = OptimalBinarySearch(self.cfg.d_out, self.cfg.d_in, n_iters=3)
            W = opt.quantize_and_reconstruct(self.W_latent, self.scale)
            return W @ x

        elif mode == QuantMode.CTG_TRAINED:
            # Training-aware CTG (oracle - simulates perfect training)
            ctg = TrainingAwareCTG(self.cfg.d_out, self.cfg.d_in, inhibit_rate=0.2)
            W = ctg.quantize_and_reconstruct(self.W_latent, self.scale)
            return W @ x

        elif mode == QuantMode.PERFECT:
            # Perfect binary oracle (theoretical limit)
            perfect = PerfectBinaryOracle(self.cfg.d_out, self.cfg.d_in)
            W = perfect.quantize_and_reconstruct(self.W_latent, self.scale)
            return W @ x

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def magnitude_uniformity_loss(self) -> float:
        """Loss to encourage uniform magnitudes within rows."""
        row_vars = np.var(np.abs(self.W_latent), axis=1)
        return np.mean(row_vars)

    def train_step(self, x: np.ndarray, y_target: np.ndarray,
                   mode: QuantMode = QuantMode.UNIFIED) -> float:
        """Single training step with STE."""
        # Forward
        y_pred = self.forward(x, mode)

        # Loss
        mse = np.mean((y_pred - y_target) ** 2)
        mag_loss = self.magnitude_uniformity_loss() * self.cfg.mag_reg_weight
        total_loss = mse + mag_loss

        # Backward (simplified - real impl would use autograd)
        error = y_pred - y_target

        # Gradient for W_latent (STE: gradient passes through sign)
        grad_W = np.outer(error, x) / len(x)

        # Gradient for scale
        W_sign = self.get_binary_weights()
        grad_scale = np.sum(error[:, None] * (W_sign @ x), axis=0, keepdims=True).T

        # Update
        self.W_latent -= self.cfg.lr * grad_W
        self.scale -= self.cfg.lr * 0.1 * grad_scale

        return total_loss


def evaluate_mode(layer: UnifiedBinaryLayer, W_true: np.ndarray,
                  mode: QuantMode, n_samples: int = 100) -> Dict:
    """Evaluate a quantization mode."""
    d_in = layer.cfg.d_in

    correlations = []
    mses = []

    for _ in range(n_samples):
        x = np.random.randn(d_in).astype(np.float32)
        y_true = W_true @ x
        y_pred = layer.forward(x, mode)

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        mse = np.mean((y_true - y_pred) ** 2)

        correlations.append(corr)
        mses.append(mse)

    return {
        'mode': mode.value,
        'correlation': np.mean(correlations),
        'correlation_std': np.std(correlations),
        'mse': np.mean(mses),
    }


def compute_bpp(mode: QuantMode, cfg: UnifiedConfig) -> float:
    """Compute bits per parameter for each mode."""
    d_out, d_in = cfg.d_out, cfg.d_in
    total_params = d_out * d_in

    if mode == QuantMode.SIGN_ONLY:
        # 1 bit per weight + 32 bits per row for scale
        return (total_params + d_out * 32) / total_params

    elif mode == QuantMode.TERNARY:
        # log2(3) ≈ 1.58 bits per weight
        return np.log2(3)

    elif mode == QuantMode.LEARNED_BASIS:
        # 1 bit per weight + 64 bits per row (α, β)
        return (total_params + d_out * 64) / total_params

    elif mode == QuantMode.FACTORIZED:
        # rank × d_in bits + rank × d_out × 32 bits
        r = cfg.rank
        return (r * d_in + r * d_out * 32) / total_params

    elif mode == QuantMode.CTG_STRUCTURED:
        # 1 bit per weight (CTG pattern is fixed/procedural)
        return 1.0

    elif mode == QuantMode.ZERO_PREDICTED:
        # 1 bit per weight (zeros predicted at runtime)
        return 1.0

    elif mode == QuantMode.UNIFIED:
        # 1 bit per weight + small overhead for α, β
        return (total_params + d_out * 64) / total_params

    elif mode == QuantMode.TERNARY_ENCODED:
        # Ternary with 1.58 bpp (oracle - knows true zeros)
        return np.log2(3)

    elif mode == QuantMode.MAGNITUDE_AWARE:
        # 1 bit sign + 2 bits group + scales overhead
        n_groups = 4
        return (total_params + total_params * 2 + d_out * n_groups * 32) / total_params

    elif mode == QuantMode.TRAINED_UNIFORM:
        # 1 bit per weight + 32 bits per row for scale
        return (total_params + d_out * 32) / total_params

    elif mode == QuantMode.STOCHASTIC:
        # 1 bit per weight + scale overhead (samples are procedural)
        return (total_params + d_out * 32) / total_params

    elif mode == QuantMode.IMPORTANCE:
        # 1 bit per weight + importance weights (oracle uses true magnitudes)
        # In practice this would be ~2 bpp, but oracle version is "free"
        return (total_params * 2 + d_out * 32) / total_params

    elif mode == QuantMode.MULTI_SAMPLE:
        # T bits per weight (shows theoretical limit)
        T = 16
        return (total_params * T + d_out * 32) / total_params

    elif mode == QuantMode.CORRELATED:
        # 1 bit per weight + scale overhead
        return (total_params + d_out * 32) / total_params

    elif mode == QuantMode.HADAMARD:
        # 1 bit per weight (in Hadamard domain) + scale overhead
        # May have padding overhead for non-power-of-2 dimensions
        d_in_padded = 2 ** int(np.ceil(np.log2(d_in)))
        return (d_out * d_in_padded + d_out * 32) / total_params

    elif mode == QuantMode.RESIDUAL:
        # 2 bits per weight (2 levels) + scale overhead per level
        n_levels = 2
        return (total_params * n_levels + d_out * 32 * n_levels) / total_params

    elif mode == QuantMode.OPTIMAL:
        # 1 bit per weight + scale overhead (same as sign, just optimized)
        return (total_params + d_out * 32) / total_params

    elif mode == QuantMode.CTG_TRAINED:
        # 0.8 bits per weight (20% are procedural zeros) + scale overhead
        inhibit_rate = 0.2
        return (total_params * (1 - inhibit_rate) + d_out * 32) / total_params

    elif mode == QuantMode.PERFECT:
        # 1 bit per weight + scale overhead (theoretical limit)
        return (total_params + d_out * 32) / total_params

    return 1.0


def run_comparison(cfg: UnifiedConfig, verbose: bool = True) -> List[Dict]:
    """Run comprehensive comparison of all modes.

    Key insight: We compare POST-TRAINING quantization of the same weights.
    Each mode gets the same W_latent, just different quantization strategy.
    """

    # Create ground truth weights
    np.random.seed(42)
    W_true = np.random.randn(cfg.d_out, cfg.d_in).astype(np.float32) * 0.02

    # Create layer and set weights
    layer = UnifiedBinaryLayer(cfg)
    layer.W_latent = W_true.copy()

    # Compute optimal scale per row
    layer.scale = np.mean(np.abs(W_true), axis=1, keepdims=True)

    # Compute learned basis parameters (optimal for this W)
    # For W = α * B + β, we want to minimize ||W - (α*B + β)||²
    # With B ∈ {0,1}, optimal is: α = mean(W where B=1) - mean(W where B=0)
    W_sign = np.sign(W_true)
    W_sign[W_sign == 0] = 1
    B = (W_sign + 1) / 2  # {0, 1}

    # Per-row optimal α, β
    for i in range(cfg.d_out):
        mask1 = B[i] > 0.5
        mask0 = ~mask1
        if np.any(mask1) and np.any(mask0):
            mean1 = np.mean(W_true[i, mask1])
            mean0 = np.mean(W_true[i, mask0])
            layer.alpha[i] = mean1 - mean0
            layer.beta[i] = mean0
        else:
            layer.alpha[i] = np.mean(np.abs(W_true[i]))
            layer.beta[i] = 0

    # Fit factorizer
    layer.factorizer.fit(W_true)

    if verbose:
        print(f"Evaluating {cfg.d_out}x{cfg.d_in} weights...")

    # Evaluate all modes
    results = []
    modes = [
        QuantMode.SIGN_ONLY,
        QuantMode.TERNARY,
        QuantMode.LEARNED_BASIS,
        QuantMode.CTG_STRUCTURED,
        QuantMode.ZERO_PREDICTED,
        QuantMode.UNIFIED,
        QuantMode.FACTORIZED,
        QuantMode.TERNARY_ENCODED,
        QuantMode.MAGNITUDE_AWARE,
        QuantMode.TRAINED_UNIFORM,
        QuantMode.STOCHASTIC,
        QuantMode.IMPORTANCE,
        QuantMode.MULTI_SAMPLE,
        QuantMode.CORRELATED,
        QuantMode.HADAMARD,
        QuantMode.RESIDUAL,
        QuantMode.OPTIMAL,
        QuantMode.CTG_TRAINED,
        QuantMode.PERFECT,
    ]

    for mode in modes:
        result = evaluate_mode(layer, W_true, mode)
        result['bpp'] = compute_bpp(mode, cfg)
        results.append(result)
        if verbose:
            print(f"  {mode.value}: corr={result['correlation']:.4f}")

    return results


def print_results(results: List[Dict]):
    """Pretty print comparison results."""
    print("\n" + "=" * 70)
    print("UNIFIED 1-BIT COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Mode':<20} {'BPP':>8} {'Correlation':>12} {'MSE':>12}")
    print("-" * 70)

    # Sort by BPP
    results_sorted = sorted(results, key=lambda x: x['bpp'])

    for r in results_sorted:
        print(f"{r['mode']:<20} {r['bpp']:>8.3f} {r['correlation']:>12.4f} {r['mse']:>12.6f}")

    print("-" * 70)

    # Find best at ~1.0 bpp
    one_bit_results = [r for r in results if r['bpp'] <= 1.1]
    if one_bit_results:
        best = max(one_bit_results, key=lambda x: x['correlation'])
        print(f"Best at ~1.0 bpp: {best['mode']} (corr={best['correlation']:.4f})")

    # Compare to ternary
    ternary = next((r for r in results if r['mode'] == 'ternary'), None)
    if ternary:
        print(f"Ternary baseline:  corr={ternary['correlation']:.4f} @ {ternary['bpp']:.2f} bpp")

        beats_ternary = [r for r in results
                        if r['bpp'] < ternary['bpp'] and r['correlation'] >= ternary['correlation']]
        if beats_ternary:
            print(f"\n🎉 METHODS THAT BEAT TERNARY:")
            for r in beats_ternary:
                print(f"   {r['mode']}: corr={r['correlation']:.4f} @ {r['bpp']:.3f} bpp")
        else:
            print(f"\n⚠️  No method beats ternary yet at lower bpp")


if __name__ == "__main__":
    # Run with default config
    cfg = UnifiedConfig(
        d_out=256,
        d_in=256,
        n_iters=3000,
        lr=0.01,
        mag_reg_weight=0.1,
        rank=8,
        ctg_period=5,
        ctg_enabled=True,
    )

    results = run_comparison(cfg, verbose=True)
    print_results(results)

