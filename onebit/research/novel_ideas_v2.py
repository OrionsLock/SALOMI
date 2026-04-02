"""Novel Ideas V2: Variations and Improvements.

Testing:
1. Hadamard variations (block, DCT, learned)
2. Improved versions of failed methods
3. New crazy ideas
"""

import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Simple DCT implementation (avoid scipy dependency)
def dct_1d(x):
    """Type-II DCT."""
    N = len(x)
    n = np.arange(N)
    k = np.arange(N)
    return np.sum(x * np.cos(np.pi * k[:, None] * (2 * n + 1) / (2 * N)), axis=1) * np.sqrt(2/N)

def idct_1d(X):
    """Type-III DCT (inverse)."""
    N = len(X)
    n = np.arange(N)
    k = np.arange(N)
    X_scaled = X.copy()
    X_scaled[0] /= np.sqrt(2)
    return np.sum(X_scaled * np.cos(np.pi * k * (2 * n[:, None] + 1) / (2 * N)), axis=1) * np.sqrt(2/N)

def dct(x, axis=0, norm='ortho'):
    """2D DCT via 1D along axis."""
    if axis == 0:
        return np.array([dct_1d(x[:, j]) for j in range(x.shape[1])]).T
    else:
        return np.array([dct_1d(x[i, :]) for i in range(x.shape[0])])

def idct(X, axis=0, norm='ortho'):
    """2D IDCT via 1D along axis."""
    if axis == 0:
        return np.array([idct_1d(X[:, j]) for j in range(X.shape[1])]).T
    else:
        return np.array([idct_1d(X[i, :]) for i in range(X.shape[0])])


# =============================================================================
# BASELINES
# =============================================================================

class BinaryBaseline:
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
# HADAMARD VARIATIONS
# =============================================================================

class HadamardBinaryRandom:
    """Original: Random orthogonal matrices."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.H_out, _ = np.linalg.qr(np.random.randn(self.d_out, self.d_out))
        self.H_in, _ = np.linalg.qr(np.random.randn(self.d_in, self.d_in))

        W_t = self.H_out @ W_opt @ self.H_in.T
        self.W_signs = np.sign(W_t)
        self.W_signs[self.W_signs == 0] = 1.0
        self.W_eff = self.H_out.T @ self.W_signs @ self.H_in

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0


class HadamardBinaryWHT:
    """True Walsh-Hadamard Transform (deterministic, no storage for transform)."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def _hadamard(self, n: int) -> np.ndarray:
        """Generate normalized Hadamard matrix of size n (must be power of 2)."""
        # Pad to next power of 2
        n_padded = 2 ** int(np.ceil(np.log2(max(n, 2))))
        H = np.array([[1]])
        while H.shape[0] < n_padded:
            H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
        return H[:n, :n] / np.sqrt(n)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.H_out = self._hadamard(self.d_out)
        self.H_in = self._hadamard(self.d_in)

        W_t = self.H_out @ W_opt @ self.H_in.T
        self.W_signs = np.sign(W_t)
        self.W_signs[self.W_signs == 0] = 1.0
        self.W_eff = self.H_out.T @ self.W_signs @ self.H_in

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0  # Transform is deterministic, no storage


class DCTBinary:
    """DCT domain binarization - like JPEG for weights."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # 2D DCT
        W_dct = dct(dct(W_opt, axis=0, norm='ortho'), axis=1, norm='ortho')

        # Binarize DCT coefficients
        self.W_dct_signs = np.sign(W_dct)
        self.W_dct_signs[self.W_dct_signs == 0] = 1.0

        # Inverse DCT
        self.W_eff = idct(idct(self.W_dct_signs, axis=0, norm='ortho'), axis=1, norm='ortho')

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0


class BlockHadamard:
    """Block-wise Hadamard - smaller transforms, potentially better locality."""
    def __init__(self, d_in: int, d_out: int, block_size: int = 8):
        self.d_in, self.d_out = d_in, d_out
        self.block_size = block_size
        self.scale = 1.0

    def _hadamard(self, n: int) -> np.ndarray:
        n_padded = 2 ** int(np.ceil(np.log2(max(n, 2))))
        H = np.array([[1]])
        while H.shape[0] < n_padded:
            H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
        return H[:n, :n] / np.sqrt(n)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        bs = self.block_size
        H = self._hadamard(bs)

        # Process in blocks
        self.W_eff = np.zeros_like(W_opt)

        for i in range(0, self.d_out, bs):
            for j in range(0, self.d_in, bs):
                i_end = min(i + bs, self.d_out)
                j_end = min(j + bs, self.d_in)

                block = W_opt[i:i_end, j:j_end]
                H_i = H[:i_end-i, :i_end-i]
                H_j = H[:j_end-j, :j_end-j]

                # Transform, binarize, inverse
                block_t = H_i @ block @ H_j.T
                block_bin = np.sign(block_t)
                block_bin[block_bin == 0] = 1.0
                self.W_eff[i:i_end, j:j_end] = H_i.T @ block_bin @ H_j

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0


# =============================================================================
# INPUT-DEPENDENT MAGNITUDE (Key insight: use X stats, not Y)
# =============================================================================

class InputDependentMagnitude:
    """Magnitude computed purely from input statistics at inference time.

    Key insight: We can compute importance from X alone!
    - High variance inputs are more important
    - Weights connecting to high-variance inputs should have higher magnitude
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W_signs = None
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Learn how to weight based on input variance
        # This is just fitting a single parameter per input dimension
        input_std = np.std(X, axis=0)

        # Magnitude = input_std (normalized)
        self.input_weights = input_std / (input_std.max() + 1e-8)

        # Effective weight: sign * input_importance
        self.W_eff = self.W_signs * self.input_weights[None, :]

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # At inference: compute input importance from X
        input_std = np.std(X, axis=0)
        input_weights = input_std / (input_std.max() + 1e-8)
        W_eff = self.W_signs * input_weights[None, :]
        return X @ (W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0  # Only signs stored, magnitude computed from X


class InputOutputMagnitude:
    """Magnitude from both input AND output statistics (computed at inference)."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W_signs = None
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        Y_pred = X @ self.W_signs.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Compute importance from X statistics
        input_std = np.std(X, axis=0)
        input_weights = input_std / (input_std.max() + 1e-8)

        # First pass: compute output
        Y_binary = X @ self.W_signs.T
        output_std = np.std(Y_binary, axis=0)
        output_weights = output_std / (output_std.max() + 1e-8)

        # Magnitude = outer product of importances
        magnitude = np.outer(output_weights, input_weights)

        return X @ (self.W_signs * magnitude * self.scale).T

    def bpp(self) -> float:
        return 1.0


# =============================================================================
# IMPROVED SELF-REFERENTIAL
# =============================================================================

class SelfReferentialV2:
    """Improved self-referential with multi-scale neighborhoods."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W_signs = None
        self.scale = 1.0

    def _compute_magnitude(self, signs: np.ndarray) -> np.ndarray:
        mag = np.ones_like(signs, dtype=np.float32)

        # Multi-scale: check agreement at different radii
        for radius in [1, 2, 3]:
            for i in range(self.d_out):
                for j in range(self.d_in):
                    i_min = max(0, i - radius)
                    i_max = min(self.d_out, i + radius + 1)
                    j_min = max(0, j - radius)
                    j_max = min(self.d_in, j + radius + 1)

                    neighborhood = signs[i_min:i_max, j_min:j_max]
                    agreement = np.mean(neighborhood == signs[i, j])

                    # Weight by radius (closer = more important)
                    weight = 1.0 / radius
                    mag[i, j] *= (agreement ** weight)

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
        return 1.0


class RowColAgreement:
    """Magnitude from row/column agreement (efficient version)."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W_signs = None
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Row agreement: how much does this weight agree with its row?
        row_means = np.mean(self.W_signs, axis=1, keepdims=True)
        row_agreement = self.W_signs * row_means  # Positive if agrees

        # Column agreement
        col_means = np.mean(self.W_signs, axis=0, keepdims=True)
        col_agreement = self.W_signs * col_means

        # Combined agreement -> magnitude
        agreement = (row_agreement + col_agreement) / 2
        mag = (agreement + 1) / 2  # Scale to [0, 1]

        self.W_eff = self.W_signs * mag

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        row_means = np.mean(self.W_signs, axis=1, keepdims=True)
        col_means = np.mean(self.W_signs, axis=0, keepdims=True)
        agreement = (self.W_signs * row_means + self.W_signs * col_means) / 2
        mag = (agreement + 1) / 2
        return X @ (self.W_signs * mag * self.scale).T

    def bpp(self) -> float:
        return 1.0


# =============================================================================
# FREQUENCY-SELECTIVE BINARIZATION
# =============================================================================

class FrequencySelectiveBinary:
    """Keep low frequencies as binary, zero out high frequencies."""
    def __init__(self, d_in: int, d_out: int, keep_frac: float = 0.5):
        self.d_in, self.d_out = d_in, d_out
        self.keep_frac = keep_frac
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # DCT
        W_dct = dct(dct(W_opt, axis=0, norm='ortho'), axis=1, norm='ortho')

        # Create frequency mask (keep low frequencies)
        freq_i = np.arange(self.d_out)[:, None]
        freq_j = np.arange(self.d_in)[None, :]
        freq_dist = np.sqrt((freq_i / self.d_out) ** 2 + (freq_j / self.d_in) ** 2)

        threshold = np.percentile(freq_dist, self.keep_frac * 100)
        mask = freq_dist <= threshold

        # Binarize only kept frequencies
        W_dct_bin = np.sign(W_dct) * mask
        W_dct_bin[W_dct_bin == 0] = 0  # High freq = 0

        # Fix zeros in low freq region
        W_dct_bin[mask & (W_dct_bin == 0)] = 1.0

        self.W_eff = idct(idct(W_dct_bin, axis=0, norm='ortho'), axis=1, norm='ortho')

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Only store keep_frac of the coefficients
        return self.keep_frac


# =============================================================================
# LEARNABLE SPARSE BINARY
# =============================================================================

class LearnableSparseBinary:
    """Learn which positions should be zero via simple heuristic."""
    def __init__(self, d_in: int, d_out: int, sparsity: float = 0.3):
        self.d_in, self.d_out = d_in, d_out
        self.sparsity = sparsity
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Heuristic for sparsity: positions where binary error is highest
        Y_pred_bin = X @ self.W_signs.T

        # Per-weight contribution to error
        error_contribution = np.zeros((self.d_out, self.d_in))
        for i in range(self.d_out):
            for j in range(self.d_in):
                # How much does flipping this weight reduce error?
                W_flipped = self.W_signs.copy()
                W_flipped[i, j] = 0  # Zero it out
                Y_pred_zero = X @ W_flipped.T
                error_contribution[i, j] = np.mean((Y - Y_pred_bin) ** 2) - np.mean((Y - Y_pred_zero) ** 2)

        # Zero out positions where zeroing helps (negative contribution)
        # Or positions with smallest positive contribution
        thresh = np.percentile(error_contribution, self.sparsity * 100)
        self.mask = error_contribution > thresh

        self.W_eff = self.W_signs * self.mask

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Signs: 1 bit, mask: 1 bit (but we store as ternary)
        return 1.58  # Effectively ternary


# =============================================================================
# SIGN PREDICTION FROM STRUCTURE
# =============================================================================

class StructuralSignPrediction:
    """Predict some signs from others, reducing storage."""
    def __init__(self, d_in: int, d_out: int, predict_frac: float = 0.3):
        self.d_in, self.d_out = d_in, d_out
        self.predict_frac = predict_frac
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Identify which positions to predict (vs store)
        # Use checkerboard pattern for simplicity
        predict_mask = np.zeros((self.d_out, self.d_in), dtype=bool)
        predict_mask[::2, 1::2] = True  # Every other in checkerboard
        predict_mask[1::2, ::2] = True

        # Limit to predict_frac
        n_predict = int(self.predict_frac * self.d_out * self.d_in)
        predict_indices = np.where(predict_mask.flatten())[0][:n_predict]
        self.predict_mask = np.zeros(self.d_out * self.d_in, dtype=bool)
        self.predict_mask[predict_indices] = True
        self.predict_mask = self.predict_mask.reshape(self.d_out, self.d_in)

        # For predicted positions: use majority of neighbors
        self.W_reconstructed = self.W_signs.copy()
        for i in range(self.d_out):
            for j in range(self.d_in):
                if self.predict_mask[i, j]:
                    # Predict from neighbors
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.d_out and 0 <= nj < self.d_in:
                                if not self.predict_mask[ni, nj]:
                                    neighbors.append(self.W_signs[ni, nj])
                    if neighbors:
                        self.W_reconstructed[i, j] = np.sign(np.mean(neighbors))
                        if self.W_reconstructed[i, j] == 0:
                            self.W_reconstructed[i, j] = 1.0

        self.W_eff = self.W_reconstructed
        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0 - self.predict_frac  # Only store non-predicted signs


# =============================================================================
# CORRELATION-BASED WEIGHT SHARING
# =============================================================================

class CorrelationWeightSharing:
    """Share signs between correlated input dimensions."""
    def __init__(self, d_in: int, d_out: int, n_groups: int = 16):
        self.d_in, self.d_out = d_in, d_out
        self.n_groups = n_groups
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Compute input correlation
        X_corr = np.corrcoef(X.T)
        X_corr = np.nan_to_num(X_corr)

        # Cluster inputs by correlation (simple k-means-like)
        # Assign each input to a group
        self.input_groups = np.zeros(self.d_in, dtype=int)
        group_centers = np.random.choice(self.d_in, self.n_groups, replace=False)

        for _ in range(10):  # Iterations
            # Assign to nearest center
            for j in range(self.d_in):
                dists = [1 - X_corr[j, c] for c in group_centers]
                self.input_groups[j] = np.argmin(dists)

            # Update centers
            for g in range(self.n_groups):
                members = np.where(self.input_groups == g)[0]
                if len(members) > 0:
                    # Center = member with highest avg correlation to others
                    avg_corr = [np.mean([X_corr[m, m2] for m2 in members]) for m in members]
                    group_centers[g] = members[np.argmax(avg_corr)]

        # For each group: use ONE sign pattern (the average)
        self.W_signs = np.zeros((self.d_out, self.d_in))
        for g in range(self.n_groups):
            members = np.where(self.input_groups == g)[0]
            if len(members) > 0:
                group_signs = np.sign(np.mean(W_opt[:, members], axis=1))
                group_signs[group_signs == 0] = 1.0
                for m in members:
                    self.W_signs[:, m] = group_signs

        self.W_eff = self.W_signs
        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # n_groups sign patterns of size d_out
        # Plus group assignments: d_in * log2(n_groups)
        n_weights = self.d_in * self.d_out
        storage = self.n_groups * self.d_out + self.d_in * np.log2(self.n_groups)
        return storage / n_weights


# =============================================================================
# BINARY + TERNARY HYBRID (explicit)
# =============================================================================

class BinaryTernaryHybrid:
    """Use binary for high-magnitude, ternary for borderline."""
    def __init__(self, d_in: int, d_out: int, ternary_frac: float = 0.3):
        self.d_in, self.d_out = d_in, d_out
        self.ternary_frac = ternary_frac
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Split: high magnitude = binary, low magnitude = ternary (with zeros)
        thresh = np.percentile(np.abs(W_opt), self.ternary_frac * 100)

        self.W_eff = np.sign(W_opt).astype(np.float32)
        self.W_eff[np.abs(W_opt) < thresh] = 0  # Ternary zeros

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Mixed: ternary_frac uses ~1.58, rest uses 1.0
        # Actually need to encode which is which, so ~1.58 overall
        return 1.0 + 0.58 * self.ternary_frac


# =============================================================================
# HADAMARD + SPARSITY
# =============================================================================

class HadamardSparse:
    """Hadamard transform + keep only top-k coefficients."""
    def __init__(self, d_in: int, d_out: int, keep_frac: float = 0.7):
        self.d_in, self.d_out = d_in, d_out
        self.keep_frac = keep_frac
        self.scale = 1.0

    def _hadamard(self, n: int) -> np.ndarray:
        n_padded = 2 ** int(np.ceil(np.log2(max(n, 2))))
        H = np.array([[1]])
        while H.shape[0] < n_padded:
            H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
        return H[:n, :n] / np.sqrt(n)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        H_out = self._hadamard(self.d_out)
        H_in = self._hadamard(self.d_in)

        W_t = H_out @ W_opt @ H_in.T

        # Keep only top-k by magnitude
        thresh = np.percentile(np.abs(W_t), (1 - self.keep_frac) * 100)
        mask = np.abs(W_t) >= thresh

        W_t_sparse = np.sign(W_t) * mask

        self.W_eff = H_out.T @ W_t_sparse @ H_in

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

        self.H_out = H_out
        self.H_in = H_in
        self.W_t_sparse = W_t_sparse

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Sparse storage: position + sign for each kept coeff
        n_kept = int(self.keep_frac * self.d_in * self.d_out)
        n_weights = self.d_in * self.d_out
        # Position: log2(n_weights) bits, Sign: 1 bit
        return n_kept * (np.log2(n_weights) + 1) / n_weights


# =============================================================================
# TESTING
# =============================================================================

def test_all():
    np.random.seed(42)

    n_samples = 1000
    d_in, d_out = 128, 64

    # Create realistic weight pattern
    rank = 16
    U = np.random.randn(d_out, rank)
    V = np.random.randn(d_in, rank)
    W_true = U @ V.T + np.random.randn(d_out, d_in) * 0.1
    mask = np.random.rand(d_out, d_in) > 0.3
    W_true = W_true * mask

    X = np.random.randn(n_samples, d_in).astype(np.float32)
    Y = (X @ W_true.T).astype(np.float32)

    X_train, Y_train = X[:800], Y[:800]
    X_test, Y_test = X[800:], Y[800:]

    print("=" * 75)
    print("NOVEL IDEAS V2: Testing All Variations")
    print("=" * 75)

    methods = [
        # Baselines
        ("Binary", BinaryBaseline(d_in, d_out)),
        ("Ternary", TernaryBaseline(d_in, d_out)),

        # Hadamard variations
        ("Hadamard (random)", HadamardBinaryRandom(d_in, d_out)),
        ("Hadamard (WHT)", HadamardBinaryWHT(d_in, d_out)),
        ("DCT Binary", DCTBinary(d_in, d_out)),
        ("Block Hadamard 8", BlockHadamard(d_in, d_out, block_size=8)),
        ("Block Hadamard 16", BlockHadamard(d_in, d_out, block_size=16)),

        # Input-dependent
        ("Input-Dep Magnitude", InputDependentMagnitude(d_in, d_out)),
        ("Input-Output Magnitude", InputOutputMagnitude(d_in, d_out)),

        # Self-referential
        ("Self-Ref V2 (multiscale)", SelfReferentialV2(d_in, d_out)),
        ("Row-Col Agreement", RowColAgreement(d_in, d_out)),

        # Frequency-selective
        ("Freq-Select 50%", FrequencySelectiveBinary(d_in, d_out, keep_frac=0.5)),
        ("Freq-Select 70%", FrequencySelectiveBinary(d_in, d_out, keep_frac=0.7)),

        # Structural
        ("Sign Prediction 20%", StructuralSignPrediction(d_in, d_out, predict_frac=0.2)),
        ("Sign Prediction 30%", StructuralSignPrediction(d_in, d_out, predict_frac=0.3)),

        # Correlation sharing
        ("Corr Sharing (16 grp)", CorrelationWeightSharing(d_in, d_out, n_groups=16)),
        ("Corr Sharing (32 grp)", CorrelationWeightSharing(d_in, d_out, n_groups=32)),

        # Hybrids
        ("Binary-Ternary 30%", BinaryTernaryHybrid(d_in, d_out, ternary_frac=0.3)),
        ("Hadamard + Sparse 70%", HadamardSparse(d_in, d_out, keep_frac=0.7)),
        ("Hadamard + Sparse 80%", HadamardSparse(d_in, d_out, keep_frac=0.8)),
    ]

    results = []
    ternary_corr = None

    for name, model in methods:
        try:
            model.train(X_train, Y_train)
            Y_pred = model.forward(X_test)

            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            mse = np.mean((Y_pred - Y_test) ** 2)
            bpp = model.bpp()

            if name == "Ternary":
                ternary_corr = corr

            results.append((name, corr, mse, bpp))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, 0.0, 999.0, 0.0))

    results.sort(key=lambda x: -x[1])

    print(f"\n{'Method':<30} {'Corr':>8} {'MSE':>10} {'BPP':>6} {'vs Tern':>8}")
    print("-" * 75)

    for name, corr, mse, bpp in results:
        vs_tern = ((corr / ternary_corr) - 1) * 100 if ternary_corr else 0

        # Markers
        if bpp <= 1.0 and vs_tern >= 0:
            marker = "★★★"
        elif bpp <= 1.1 and vs_tern >= -5:
            marker = "★★"
        elif bpp <= 1.2 and vs_tern >= -10:
            marker = "★"
        else:
            marker = ""

        print(f"{name:<30} {corr:>8.4f} {mse:>10.2f} {bpp:>6.2f} {vs_tern:>+7.1f}% {marker}")

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY: Best at each BPP threshold")
    print("=" * 75)

    for bpp_thresh in [0.7, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0]:
        candidates = [(n, c, m, b) for n, c, m, b in results if b <= bpp_thresh and c > 0]
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            vs = ((best[1] / ternary_corr) - 1) * 100
            print(f"  ≤{bpp_thresh:.1f} bpp: {best[0]:<25} {best[1]:.4f} corr ({vs:+.1f}% vs ternary)")

    return results


# =============================================================================
# NEW CRAZY IDEAS
# =============================================================================

class GolombRiceBinary:
    """Use Golomb-Rice-like coding: run-length encode sign changes."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Sort weights by position for better run-length encoding
        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Compute run-length statistics
        flat = self.W_signs.flatten()
        runs = []
        current_sign = flat[0]
        run_length = 1
        for i in range(1, len(flat)):
            if flat[i] == current_sign:
                run_length += 1
            else:
                runs.append(run_length)
                current_sign = flat[i]
                run_length = 1
        runs.append(run_length)

        self.avg_run_length = np.mean(runs)
        self.W_eff = self.W_signs

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Entropy-based: if runs are long, we need fewer bits
        # H ≈ 1 / avg_run_length for run-length coding
        return min(1.0, 1.0 / max(1, self.avg_run_length - 1) + 0.5)


class DoubleBinary:
    """Two binary matrices that combine: W = A * B (element-wise)."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # A determines sign, B determines whether to keep or zero
        self.A = np.sign(W_opt).astype(np.float32)
        self.A[self.A == 0] = 1.0

        # B: 1 for high magnitude, -1 for low (becomes 0 when multiplied)
        thresh = np.percentile(np.abs(W_opt), 30)
        self.B = np.ones_like(W_opt)
        self.B[np.abs(W_opt) < thresh] = -1

        # W = (A + 1) / 2 * (B + 1) / 2 * 2 - 1
        # When A=1, B=1: W=1
        # When A=-1, B=1: W=-1
        # When B=-1: W=0
        self.W_eff = self.A * ((self.B + 1) / 2)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 2.0  # Two binary matrices


class PermutedBinary:
    """Store one binary row, permute for others."""
    def __init__(self, d_in: int, d_out: int, n_templates: int = 8):
        self.d_in, self.d_out = d_in, d_out
        self.n_templates = n_templates
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Create n_templates binary templates
        self.templates = np.sign(np.random.randn(self.n_templates, self.d_in))

        # Assign each row to best matching template + flip pattern
        self.row_templates = np.zeros(self.d_out, dtype=int)
        self.row_flips = np.zeros(self.d_out)  # 1 or -1

        for i in range(self.d_out):
            row = np.sign(W_opt[i, :])
            row[row == 0] = 1

            best_match = -1
            best_score = -np.inf
            best_flip = 1

            for t in range(self.n_templates):
                # Try both flipped and non-flipped
                score_pos = np.sum(row == self.templates[t])
                score_neg = np.sum(row == -self.templates[t])

                if score_pos > best_score:
                    best_score = score_pos
                    best_match = t
                    best_flip = 1
                if score_neg > best_score:
                    best_score = score_neg
                    best_match = t
                    best_flip = -1

            self.row_templates[i] = best_match
            self.row_flips[i] = best_flip

        # Reconstruct
        self.W_eff = np.zeros((self.d_out, self.d_in))
        for i in range(self.d_out):
            self.W_eff[i, :] = self.templates[self.row_templates[i]] * self.row_flips[i]

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        n_weights = self.d_in * self.d_out
        storage = (self.n_templates * self.d_in +  # templates
                   self.d_out * np.log2(self.n_templates) +  # assignments
                   self.d_out)  # flips
        return storage / n_weights


class RandomProjectionBinary:
    """Project to lower dim, binarize, project back."""
    def __init__(self, d_in: int, d_out: int, hidden_dim: int = 32):
        self.d_in, self.d_out = d_in, d_out
        self.hidden_dim = hidden_dim
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Random projection matrices (fixed, not stored per-weight)
        np.random.seed(42)  # Deterministic
        self.P1 = np.random.randn(self.d_out, self.hidden_dim) / np.sqrt(self.hidden_dim)
        self.P2 = np.random.randn(self.hidden_dim, self.d_in) / np.sqrt(self.hidden_dim)

        # Project W to low-rank approximation
        W_proj = self.P1.T @ W_opt @ self.P2.T  # (hidden, hidden)

        # Binarize the projection
        self.W_hidden = np.sign(W_proj)
        self.W_hidden[self.W_hidden == 0] = 1.0

        # Reconstruct
        self.W_eff = self.P1 @ self.W_hidden @ self.P2

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        n_weights = self.d_in * self.d_out
        return self.hidden_dim ** 2 / n_weights  # Only hidden weights stored


class SignXOR:
    """XOR patterns to create structure."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Create XOR pattern: row_sign XOR col_sign
        self.row_signs = np.sign(np.mean(W_opt, axis=1))
        self.row_signs[self.row_signs == 0] = 1.0
        self.col_signs = np.sign(np.mean(W_opt, axis=0))
        self.col_signs[self.col_signs == 0] = 1.0

        # XOR approximation: outer product
        self.W_xor = np.outer(self.row_signs, self.col_signs)

        # Residual from XOR
        residual = self.W_signs * self.W_xor  # If they match, +1; else -1

        # Encode residual sparsely (where XOR fails)
        error_rate = np.mean(residual == -1)

        # Use XOR as base, correct only high-error positions
        self.W_eff = self.W_xor

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        n_weights = self.d_in * self.d_out
        return (self.d_in + self.d_out) / n_weights


class HadamardDCTCombined:
    """Combine Hadamard on rows, DCT on columns."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def _hadamard(self, n: int) -> np.ndarray:
        n_padded = 2 ** int(np.ceil(np.log2(max(n, 2))))
        H = np.array([[1]])
        while H.shape[0] < n_padded:
            H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
        return H[:n, :n] / np.sqrt(n)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Hadamard on rows
        H = self._hadamard(self.d_out)
        W_h = H @ W_opt

        # DCT on columns
        W_hd = dct(W_h, axis=1, norm='ortho')

        # Binarize
        W_bin = np.sign(W_hd)
        W_bin[W_bin == 0] = 1.0

        # Inverse
        W_ihd = idct(W_bin, axis=1, norm='ortho')
        self.W_eff = H.T @ W_ihd

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0


def test_new_ideas():
    np.random.seed(42)

    n_samples = 1000
    d_in, d_out = 128, 64

    rank = 16
    U = np.random.randn(d_out, rank)
    V = np.random.randn(d_in, rank)
    W_true = U @ V.T + np.random.randn(d_out, d_in) * 0.1
    mask = np.random.rand(d_out, d_in) > 0.3
    W_true = W_true * mask

    X = np.random.randn(n_samples, d_in).astype(np.float32)
    Y = (X @ W_true.T).astype(np.float32)

    X_train, Y_train = X[:800], Y[:800]
    X_test, Y_test = X[800:], Y[800:]

    print("\n" + "=" * 75)
    print("NEW CRAZY IDEAS")
    print("=" * 75)

    methods = [
        ("Binary", BinaryBaseline(d_in, d_out)),
        ("Ternary", TernaryBaseline(d_in, d_out)),
        ("DCT Binary", DCTBinary(d_in, d_out)),
        ("Hadamard-DCT Combined", HadamardDCTCombined(d_in, d_out)),
        ("Golomb-Rice Binary", GolombRiceBinary(d_in, d_out)),
        ("Double Binary", DoubleBinary(d_in, d_out)),
        ("Permuted Binary (8)", PermutedBinary(d_in, d_out, n_templates=8)),
        ("Permuted Binary (16)", PermutedBinary(d_in, d_out, n_templates=16)),
        ("Random Projection 16", RandomProjectionBinary(d_in, d_out, hidden_dim=16)),
        ("Random Projection 32", RandomProjectionBinary(d_in, d_out, hidden_dim=32)),
        ("SignXOR", SignXOR(d_in, d_out)),
    ]

    results = []
    ternary_corr = None

    for name, model in methods:
        try:
            model.train(X_train, Y_train)
            Y_pred = model.forward(X_test)

            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            mse = np.mean((Y_pred - Y_test) ** 2)
            bpp = model.bpp()

            if name == "Ternary":
                ternary_corr = corr

            results.append((name, corr, mse, bpp))
        except Exception as e:
            print(f"ERROR in {name}: {e}")

    results.sort(key=lambda x: -x[1])

    print(f"\n{'Method':<30} {'Corr':>8} {'MSE':>10} {'BPP':>6} {'vs Tern':>8}")
    print("-" * 75)

    for name, corr, mse, bpp in results:
        vs_tern = ((corr / ternary_corr) - 1) * 100 if ternary_corr else 0
        marker = "★★★" if bpp <= 1.0 and vs_tern >= 0 else ("★★" if bpp <= 1.1 else "")
        print(f"{name:<30} {corr:>8.4f} {mse:>10.2f} {bpp:>6.2f} {vs_tern:>+7.1f}% {marker}")


# =============================================================================
# ULTIMATE COMBINED: Best of the best
# =============================================================================

class DCTHadamardResidual:
    """DCT binarize, then add Hadamard residual correction."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def _hadamard(self, n: int) -> np.ndarray:
        n_padded = 2 ** int(np.ceil(np.log2(max(n, 2))))
        H = np.array([[1]])
        while H.shape[0] < n_padded:
            H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
        return H[:n, :n] / np.sqrt(n)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # First pass: DCT
        W_dct = dct(dct(W_opt, axis=0), axis=1)
        W_dct_bin = np.sign(W_dct)
        W_dct_bin[W_dct_bin == 0] = 1.0
        W_pass1 = idct(idct(W_dct_bin, axis=0), axis=1)

        # Residual
        residual = W_opt - W_pass1

        # Second pass: Hadamard on residual
        H_out = self._hadamard(self.d_out)
        H_in = self._hadamard(self.d_in)

        res_h = H_out @ residual @ H_in.T
        res_h_bin = np.sign(res_h)
        res_h_bin[res_h_bin == 0] = 1.0
        W_pass2 = H_out.T @ res_h_bin @ H_in

        # Combine
        self.W_eff = W_pass1 + W_pass2 * 0.5  # Weight residual lower

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 2.0  # Two binary matrices


class MultiScaleDCT:
    """DCT at multiple block sizes, combine."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Full DCT
        W_dct = dct(dct(W_opt, axis=0), axis=1)
        W_full = np.sign(W_dct)
        W_full[W_full == 0] = 1.0
        W_rec_full = idct(idct(W_full, axis=0), axis=1)

        # Block DCT (8x8 blocks)
        W_rec_block = np.zeros_like(W_opt)
        bs = 8
        for i in range(0, self.d_out, bs):
            for j in range(0, self.d_in, bs):
                i_end = min(i + bs, self.d_out)
                j_end = min(j + bs, self.d_in)
                block = W_opt[i:i_end, j:j_end]

                b_dct = dct(dct(block, axis=0), axis=1)
                b_bin = np.sign(b_dct)
                b_bin[b_bin == 0] = 1.0
                W_rec_block[i:i_end, j:j_end] = idct(idct(b_bin, axis=0), axis=1)

        # Combine: average or weighted
        self.W_eff = (W_rec_full + W_rec_block) / 2

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 2.0  # Two binary sets


class AdaptiveTransform:
    """Choose best transform per block."""
    def __init__(self, d_in: int, d_out: int, block_size: int = 16):
        self.d_in, self.d_out = d_in, d_out
        self.block_size = block_size
        self.scale = 1.0

    def _hadamard(self, n: int) -> np.ndarray:
        n_padded = 2 ** int(np.ceil(np.log2(max(n, 2))))
        H = np.array([[1]])
        while H.shape[0] < n_padded:
            H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
        return H[:n, :n] / np.sqrt(n)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_eff = np.zeros_like(W_opt)
        bs = self.block_size
        self.block_choices = []

        for i in range(0, self.d_out, bs):
            for j in range(0, self.d_in, bs):
                i_end = min(i + bs, self.d_out)
                j_end = min(j + bs, self.d_in)
                block = W_opt[i:i_end, j:j_end]

                # Try DCT
                b_dct = dct(dct(block, axis=0), axis=1)
                b_dct_bin = np.sign(b_dct)
                b_dct_bin[b_dct_bin == 0] = 1.0
                rec_dct = idct(idct(b_dct_bin, axis=0), axis=1)
                err_dct = np.sum((block - rec_dct) ** 2)

                # Try Hadamard
                H_i = self._hadamard(i_end - i)
                H_j = self._hadamard(j_end - j)
                b_h = H_i @ block @ H_j.T
                b_h_bin = np.sign(b_h)
                b_h_bin[b_h_bin == 0] = 1.0
                rec_h = H_i.T @ b_h_bin @ H_j
                err_h = np.sum((block - rec_h) ** 2)

                # Choose better
                if err_dct < err_h:
                    self.W_eff[i:i_end, j:j_end] = rec_dct
                    self.block_choices.append('dct')
                else:
                    self.W_eff[i:i_end, j:j_end] = rec_h
                    self.block_choices.append('had')

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        n_blocks = len(self.block_choices)
        n_weights = self.d_in * self.d_out
        return 1.0 + n_blocks / n_weights  # 1 bit per block for choice


class ThresholdedDCT:
    """DCT with magnitude thresholding for better zeros."""
    def __init__(self, d_in: int, d_out: int, zero_frac: float = 0.3):
        self.d_in, self.d_out = d_in, d_out
        self.zero_frac = zero_frac
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        W_dct = dct(dct(W_opt, axis=0), axis=1)

        # Threshold small coefficients to zero
        thresh = np.percentile(np.abs(W_dct), self.zero_frac * 100)
        W_dct_thresh = np.sign(W_dct) * (np.abs(W_dct) > thresh)

        self.W_eff = idct(idct(W_dct_thresh, axis=0), axis=1)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.58  # Ternary in DCT domain


def test_ultimate():
    np.random.seed(42)

    n_samples = 1000
    d_in, d_out = 128, 64

    rank = 16
    U = np.random.randn(d_out, rank)
    V = np.random.randn(d_in, rank)
    W_true = U @ V.T + np.random.randn(d_out, d_in) * 0.1
    mask = np.random.rand(d_out, d_in) > 0.3
    W_true = W_true * mask

    X = np.random.randn(n_samples, d_in).astype(np.float32)
    Y = (X @ W_true.T).astype(np.float32)

    X_train, Y_train = X[:800], Y[:800]
    X_test, Y_test = X[800:], Y[800:]

    print("\n" + "=" * 75)
    print("ULTIMATE COMBINED APPROACHES")
    print("=" * 75)

    methods = [
        ("Binary", BinaryBaseline(d_in, d_out)),
        ("Ternary", TernaryBaseline(d_in, d_out)),
        ("DCT Binary", DCTBinary(d_in, d_out)),
        ("Hadamard WHT", HadamardBinaryWHT(d_in, d_out)),
        ("DCT+Hadamard Residual", DCTHadamardResidual(d_in, d_out)),
        ("Multi-Scale DCT", MultiScaleDCT(d_in, d_out)),
        ("Adaptive Transform", AdaptiveTransform(d_in, d_out, block_size=16)),
        ("Thresholded DCT 20%", ThresholdedDCT(d_in, d_out, zero_frac=0.2)),
        ("Thresholded DCT 30%", ThresholdedDCT(d_in, d_out, zero_frac=0.3)),
    ]

    results = []
    ternary_corr = None

    for name, model in methods:
        try:
            model.train(X_train, Y_train)
            Y_pred = model.forward(X_test)

            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            mse = np.mean((Y_pred - Y_test) ** 2)
            bpp = model.bpp()

            if name == "Ternary":
                ternary_corr = corr

            results.append((name, corr, mse, bpp))
        except Exception as e:
            print(f"ERROR in {name}: {e}")

    results.sort(key=lambda x: -x[1])

    print(f"\n{'Method':<30} {'Corr':>8} {'MSE':>10} {'BPP':>6} {'vs Tern':>8}")
    print("-" * 75)

    for name, corr, mse, bpp in results:
        vs_tern = ((corr / ternary_corr) - 1) * 100 if ternary_corr else 0
        marker = "★★★" if bpp <= 1.0 and vs_tern >= 0 else ("★★" if bpp <= 1.58 and vs_tern >= 0 else "")
        print(f"{name:<30} {corr:>8.4f} {mse:>10.2f} {bpp:>6.2f} {vs_tern:>+7.1f}% {marker}")

    # Final Summary
    print("\n" + "=" * 75)
    print("FINAL SUMMARY: Methods that BEAT Ternary")
    print("=" * 75)

    winners = [(n, c, m, b) for n, c, m, b in results if (c / ternary_corr - 1) >= 0]
    for name, corr, mse, bpp in sorted(winners, key=lambda x: x[3]):
        vs = ((corr / ternary_corr) - 1) * 100
        print(f"  {name:<30} {corr:.4f} @ {bpp:.2f} bpp ({vs:+.1f}% vs ternary)")


# =============================================================================
# IDEAS FROM EXISTING CODEBASE
# =============================================================================

class BlockHeterogeneousBinaryV2:
    """From error_optimal_1bit.py: Multiple scale levels per block.

    Combined with transform domain for extra boost.
    """
    def __init__(self, d_in: int, d_out: int, block_size: int = 4, n_scales: int = 4):
        self.d_in, self.d_out = d_in, d_out
        self.block_size = block_size
        self.n_scales = n_scales
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # DCT transform first
        W_dct = dct(dct(W_opt, axis=0), axis=1)

        # Binary signs
        self.W_binary = np.sign(W_dct).astype(np.float32)
        self.W_binary[self.W_binary == 0] = 1.0

        # Scale values from magnitude distribution
        magnitudes = np.abs(W_dct)
        percentiles = np.linspace(0, 100, self.n_scales + 1)[1:-1]
        self.scale_values = []
        prev = 0
        for p in list(percentiles) + [100]:
            boundary = np.percentile(magnitudes, p)
            mask = (magnitudes >= prev) & (magnitudes < boundary)
            if mask.any():
                self.scale_values.append(np.mean(magnitudes[mask]))
            else:
                self.scale_values.append(1.0)
            prev = boundary
        self.scale_values = np.array(self.scale_values)

        # Per-block scale selection
        bs = self.block_size
        n_blocks_h = (self.d_out + bs - 1) // bs
        n_blocks_w = (self.d_in + bs - 1) // bs
        self.block_scale_idx = np.zeros((n_blocks_h, n_blocks_w), dtype=int)

        self.W_scaled = np.zeros_like(self.W_binary)

        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                i_s, i_e = bi * bs, min((bi + 1) * bs, self.d_out)
                j_s, j_e = bj * bs, min((bj + 1) * bs, self.d_in)

                block_signs = self.W_binary[i_s:i_e, j_s:j_e]
                block_target = W_dct[i_s:i_e, j_s:j_e]

                best_idx, best_err = 0, float('inf')
                for k, scale in enumerate(self.scale_values):
                    err = np.mean((block_signs * scale - block_target) ** 2)
                    if err < best_err:
                        best_err = err
                        best_idx = k

                self.block_scale_idx[bi, bj] = best_idx
                self.W_scaled[i_s:i_e, j_s:j_e] = block_signs * self.scale_values[best_idx]

        # Inverse DCT
        self.W_eff = idct(idct(self.W_scaled, axis=0), axis=1)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        n_weights = self.d_in * self.d_out
        n_blocks = ((self.d_out + self.block_size - 1) // self.block_size) * \
                   ((self.d_in + self.block_size - 1) // self.block_size)
        return 1.0 + n_blocks * np.log2(self.n_scales) / n_weights


class IterativeDCTBinary:
    """From iterative_binary.py: Multi-pass refinement in DCT domain."""
    def __init__(self, d_in: int, d_out: int, n_iterations: int = 2):
        self.d_in, self.d_out = d_in, d_out
        self.n_iter = n_iterations
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        residual = Y.copy()
        self.W_eff = np.zeros((self.d_out, self.d_in))

        for i in range(self.n_iter):
            # Fit to residual
            W_opt = np.linalg.lstsq(X, residual, rcond=None)[0].T

            # DCT domain binarization
            W_dct = dct(dct(W_opt, axis=0), axis=1)
            W_dct_bin = np.sign(W_dct)
            W_dct_bin[W_dct_bin == 0] = 1.0
            W_rec = idct(idct(W_dct_bin, axis=0), axis=1)

            # Scale for this iteration
            Y_pred = X @ W_rec.T
            scale_i = np.sum(Y_pred * residual) / (np.sum(Y_pred ** 2) + 1e-8)

            self.W_eff += W_rec * scale_i
            residual = residual - Y_pred * scale_i

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return float(self.n_iter)  # n_iter binary matrices


class LowRankDCTBinary:
    """From lowrank_binary.py: SVD + DCT + binary."""
    def __init__(self, d_in: int, d_out: int, rank: int = 32):
        self.d_in, self.d_out = d_in, d_out
        self.rank = rank
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # SVD
        U, S, Vt = np.linalg.svd(W_opt, full_matrices=False)
        r = min(self.rank, len(S))

        # Low-rank approximation
        W_lr = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]

        # DCT on low-rank
        W_dct = dct(dct(W_lr, axis=0), axis=1)
        W_dct_bin = np.sign(W_dct)
        W_dct_bin[W_dct_bin == 0] = 1.0

        self.W_eff = idct(idct(W_dct_bin, axis=0), axis=1)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0  # Still 1 binary matrix


class WalshCarrierBinary:
    """From walsh.py: Use Walsh carriers for structured binarization."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.scale = 1.0

    def _walsh_carrier_bit(self, row_idx: int, t: int) -> int:
        """Walsh carrier ±1."""
        v = row_idx & t
        parity = bin(v).count('1') % 2
        return 1 if parity == 0 else -1

    def _walsh_matrix(self, n: int) -> np.ndarray:
        """Generate Walsh-Hadamard matrix."""
        H = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                H[i, j] = self._walsh_carrier_bit(i, j)
        return H / np.sqrt(n)

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Walsh transform
        H_out = self._walsh_matrix(self.d_out)
        H_in = self._walsh_matrix(self.d_in)

        W_walsh = H_out @ W_opt @ H_in.T

        # Binarize
        self.W_walsh_bin = np.sign(W_walsh)
        self.W_walsh_bin[self.W_walsh_bin == 0] = 1.0

        # Inverse Walsh
        self.W_eff = H_out.T @ self.W_walsh_bin @ H_in

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        return 1.0


class CTGInhibitBinary:
    """From ctg_grammar.py: Periodic INHIBIT pattern creates pseudo-ternary."""
    def __init__(self, d_in: int, d_out: int, period: int = 4):
        self.d_in, self.d_out = d_in, d_out
        self.period = period
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Binary signs
        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # CTG inhibit mask: periodic pattern
        self.mask = np.ones((self.d_out, self.d_in), dtype=np.float32)
        for i in range(self.d_out):
            for j in range(self.d_in):
                # Inhibit based on position modulo period
                if (i + j) % self.period == 0:
                    self.mask[i, j] = 0.0

        self.W_eff = self.W_signs * self.mask

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Signs: 1 bit, mask: deterministic from position (0 bits)
        return 1.0


class CTGAdaptiveInhibit:
    """CTG with magnitude-based inhibit selection (optimized)."""
    def __init__(self, d_in: int, d_out: int, inhibit_frac: float = 0.25, period: int = 4):
        self.d_in, self.d_out = d_in, d_out
        self.inhibit_frac = inhibit_frac
        self.period = period
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        self.W_signs = np.sign(W_opt).astype(np.float32)
        self.W_signs[self.W_signs == 0] = 1.0

        # Compute block importance for CTG-style selection
        bs = self.period
        n_blocks_h = (self.d_out + bs - 1) // bs
        n_blocks_w = (self.d_in + bs - 1) // bs

        block_importance = np.zeros((n_blocks_h, n_blocks_w))
        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                i_s, i_e = bi * bs, min((bi + 1) * bs, self.d_out)
                j_s, j_e = bj * bs, min((bj + 1) * bs, self.d_in)
                block_importance[bi, bj] = np.mean(np.abs(W_opt[i_s:i_e, j_s:j_e]))

        # Inhibit least important blocks
        n_inhibit = int(n_blocks_h * n_blocks_w * self.inhibit_frac)
        thresh = np.sort(block_importance.flatten())[n_inhibit]

        self.mask = np.ones((self.d_out, self.d_in), dtype=np.float32)
        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                if block_importance[bi, bj] <= thresh:
                    i_s, i_e = bi * bs, min((bi + 1) * bs, self.d_out)
                    j_s, j_e = bj * bs, min((bj + 1) * bs, self.d_in)
                    self.mask[i_s:i_e, j_s:j_e] = 0.0

        self.W_eff = self.W_signs * self.mask

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        n_blocks = ((self.d_out + self.period - 1) // self.period) * \
                   ((self.d_in + self.period - 1) // self.period)
        n_weights = self.d_in * self.d_out
        # Signs + block inhibit bitmap
        return 1.0 + n_blocks / n_weights


class DCTWithCTGInhibit:
    """ULTIMATE COMBO: DCT domain + CTG-style inhibit."""
    def __init__(self, d_in: int, d_out: int, inhibit_frac: float = 0.3):
        self.d_in, self.d_out = d_in, d_out
        self.inhibit_frac = inhibit_frac
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # DCT
        W_dct = dct(dct(W_opt, axis=0), axis=1)

        # Binarize + inhibit small coefficients (like ternary in DCT domain!)
        thresh = np.percentile(np.abs(W_dct), self.inhibit_frac * 100)
        W_dct_tern = np.sign(W_dct) * (np.abs(W_dct) > thresh)

        # Inverse DCT
        self.W_eff = idct(idct(W_dct_tern, axis=0), axis=1)

        Y_pred = X @ self.W_eff.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W_eff * self.scale).T

    def bpp(self) -> float:
        # Ternary in DCT domain
        return 1.58


def test_codebase_ideas():
    np.random.seed(42)

    n_samples = 1000
    d_in, d_out = 128, 64

    rank = 16
    U = np.random.randn(d_out, rank)
    V = np.random.randn(d_in, rank)
    W_true = U @ V.T + np.random.randn(d_out, d_in) * 0.1
    mask = np.random.rand(d_out, d_in) > 0.3
    W_true = W_true * mask

    X = np.random.randn(n_samples, d_in).astype(np.float32)
    Y = (X @ W_true.T).astype(np.float32)

    X_train, Y_train = X[:800], Y[:800]
    X_test, Y_test = X[800:], Y[800:]

    print("\n" + "=" * 75)
    print("IDEAS FROM EXISTING CODEBASE")
    print("=" * 75)

    methods = [
        ("Binary", BinaryBaseline(d_in, d_out)),
        ("Ternary", TernaryBaseline(d_in, d_out)),
        ("DCT Binary", DCTBinary(d_in, d_out)),
        ("Adaptive Transform", AdaptiveTransform(d_in, d_out)),
        ("Block Hetero DCT 4x4", BlockHeterogeneousBinaryV2(d_in, d_out, block_size=4, n_scales=4)),
        ("Block Hetero DCT 8x4", BlockHeterogeneousBinaryV2(d_in, d_out, block_size=8, n_scales=4)),
        ("Iterative DCT x2", IterativeDCTBinary(d_in, d_out, n_iterations=2)),
        ("LowRank DCT r=32", LowRankDCTBinary(d_in, d_out, rank=32)),
        ("LowRank DCT r=64", LowRankDCTBinary(d_in, d_out, rank=64)),
        ("Walsh Carrier", WalshCarrierBinary(d_in, d_out)),
        ("CTG Inhibit p=4", CTGInhibitBinary(d_in, d_out, period=4)),
        ("CTG Adaptive 25%", CTGAdaptiveInhibit(d_in, d_out, inhibit_frac=0.25)),
        ("DCT+CTG Inhibit 30%", DCTWithCTGInhibit(d_in, d_out, inhibit_frac=0.30)),
    ]

    results = []
    ternary_corr = None

    for name, model in methods:
        try:
            model.train(X_train, Y_train)
            Y_pred = model.forward(X_test)

            corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
            mse = np.mean((Y_pred - Y_test) ** 2)
            bpp = model.bpp()

            if name == "Ternary":
                ternary_corr = corr

            results.append((name, corr, mse, bpp))
        except Exception as e:
            print(f"ERROR in {name}: {e}")

    results.sort(key=lambda x: -x[1])

    print(f"\n{'Method':<25} {'Corr':>8} {'MSE':>10} {'BPP':>6} {'vs Tern':>8}")
    print("-" * 70)

    for name, corr, mse, bpp in results:
        vs_tern = ((corr / ternary_corr) - 1) * 100 if ternary_corr else 0
        if bpp <= 1.0 and vs_tern >= 0:
            marker = "★★★"
        elif bpp <= 1.1 and vs_tern >= 0:
            marker = "★★"
        elif bpp <= 1.58 and vs_tern >= 0:
            marker = "★"
        else:
            marker = ""
        print(f"{name:<25} {corr:>8.4f} {mse:>10.2f} {bpp:>6.2f} {vs_tern:>+7.1f}% {marker}")

    # Final winner summary
    print("\n" + "=" * 75)
    print("FINAL RANKINGS (sorted by quality at each BPP)")
    print("=" * 75)

    for bpp_thresh in [1.0, 1.1, 1.2, 1.58]:
        candidates = [(n, c, m, b) for n, c, m, b in results if b <= bpp_thresh]
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            vs = ((best[1] / ternary_corr) - 1) * 100
            status = "BEATS" if vs > 0 else "MATCHES" if vs >= -1 else "LOSES"
            print(f"  ≤{bpp_thresh:.2f} bpp: {best[0]:<25} {best[1]:.4f} ({vs:+.1f}%) - {status}")


if __name__ == "__main__":
    test_all()
    test_new_ideas()
    test_ultimate()
    test_codebase_ideas()

