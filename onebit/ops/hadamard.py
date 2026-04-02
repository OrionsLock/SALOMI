"""Hadamard Binary Quantization (H1BQ) + Self-Indexed Sparsity (SIS).

This module implements 1-bit quantization with dynamic ternary simulation.

H1BQ: Quantize in Hadamard domain for better binary quality.
SIS: Use bit patterns to create dynamic zeros at 1.00 bpp.

Theory:
    y = W @ x
      = (W @ H) @ (H @ x)    # H is self-inverse: H @ H = I
      = W_h @ x_h

    We quantize W_h to binary: W_bits = sign(W_h)
    At inference: y ≈ scale * W_bits @ x_h

Self-Indexed Sparsity (SIS):
    For each block of G bits, compute popcount.
    If popcount is exactly G/2 (balanced), treat as "uncertain" → zero.
    This creates ~C(G, G/2) / 2^G sparsity without extra storage.

    For G=8: sparsity = C(8,4)/256 = 70/256 ≈ 27%
    For G=4: sparsity = C(4,2)/16 = 6/16 = 37.5%

Benefits:
    - Exactly 1.00 bpp storage
    - Dynamic zeros from bit patterns (no extra bits)
    - Ternary-like behavior at binary storage cost
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def fast_walsh_hadamard(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (in-place friendly).

    Computes H @ x where H is the normalized Hadamard matrix.
    H @ H = I (self-inverse when normalized by 1/sqrt(d)).

    Args:
        x: Input array of shape (..., d) where d is a power of 2
        normalize: If True, normalize by 1/sqrt(d) to make it self-inverse

    Returns:
        Transformed array of same shape
    """
    x = np.asarray(x, dtype=np.float64).copy()  # Use float64 for precision

    *batch_dims, d = x.shape

    # Check power of 2
    if d & (d - 1) != 0:
        raise ValueError(f"Dimension {d} must be a power of 2")

    # In-place Walsh-Hadamard via butterfly operations
    # Using vectorized operations for efficiency
    h = 1
    while h < d:
        # Process all butterflies at this stage
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                a = x[..., j].copy()
                b = x[..., j + h].copy()
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2

    if normalize:
        # Normalize to make H self-inverse: H @ H = I
        x = x / np.sqrt(d)

    return x.astype(np.float32)


def hadamard_quantize(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Quantize weights in Hadamard domain.

    Args:
        W: Weight matrix of shape (d_out, d_in) where d_in is power of 2

    Returns:
        W_bits: Binary weights sign(W @ H), packed as uint32
        scale: Per-row scaling factors (d_out,)
        d_original: Original input dimension
    """
    d_out, d_in = W.shape
    d_original = d_in

    # Pad to power of 2 if needed
    d_pad = 1 << (d_in - 1).bit_length()
    if d_pad != d_in:
        W_padded = np.zeros((d_out, d_pad), dtype=np.float32)
        W_padded[:, :d_in] = W
        W = W_padded
        d_in = d_pad

    # Transform to Hadamard domain (apply to each row)
    W_h = np.zeros_like(W)
    for i in range(d_out):
        W_h[i] = fast_walsh_hadamard(W[i], normalize=True)

    # Compute per-row scales (mean absolute value before quantization)
    scale = np.mean(np.abs(W_h), axis=1).astype(np.float32)

    # Quantize to binary: sign(W_h)
    # Pack bits: bit=1 for >=0, bit=0 for <0
    W_sign = (W_h >= 0).astype(np.uint8)

    # Pack to uint32
    n_words = (d_in + 31) // 32
    W_bits = np.zeros((d_out, n_words), dtype=np.uint32)

    for j in range(d_in):
        word_idx = j // 32
        bit_idx = j % 32
        W_bits[:, word_idx] |= W_sign[:, j].astype(np.uint32) << bit_idx

    return W_bits, scale, d_original


def hadamard_matmul(W_bits: np.ndarray, scale: np.ndarray,
                    x: np.ndarray, d_original: int) -> np.ndarray:
    """Matrix multiply using Hadamard-quantized weights.

    Args:
        W_bits: Packed binary weights (d_out, n_words)
        scale: Per-row scales (d_out,)
        x: Input vector (d_in,) or (batch, d_in)
        d_original: Original input dimension (before padding)

    Returns:
        y: Output (d_out,) or (batch, d_out)
    """
    d_out, n_words = W_bits.shape
    d_in = n_words * 32

    # Handle batch dimension
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]

    batch_size = x.shape[0]

    # Pad input if needed
    if x.shape[1] < d_in:
        x_padded = np.zeros((batch_size, d_in), dtype=np.float32)
        x_padded[:, :x.shape[1]] = x
        x = x_padded

    # Transform input to Hadamard domain
    x_h = fast_walsh_hadamard(x, normalize=True)

    # Unpack bits and compute dot product
    # W_unpacked[i,j] = 2 * bit - 1 = +1 or -1
    y = np.zeros((batch_size, d_out), dtype=np.float32)

    for i in range(d_out):
        for b in range(batch_size):
            dot = 0.0
            for w in range(n_words):
                word = W_bits[i, w]
                for bit in range(32):
                    j = w * 32 + bit
                    if j >= d_in:
                        break
                    sign = 1.0 if (word >> bit) & 1 else -1.0
                    dot += sign * x_h[b, j]
            y[b, i] = dot * scale[i]

    if single:
        y = y[0]

    return y




def popcount8(byte: int) -> int:
    """Count bits in a byte."""
    byte = byte - ((byte >> 1) & 0x55)
    byte = (byte & 0x33) + ((byte >> 2) & 0x33)
    return (byte + (byte >> 4)) & 0x0F


def sis_matmul(W_bits: np.ndarray, scale: np.ndarray, x: np.ndarray,
               d_original: int, block_size: int = 8) -> np.ndarray:
    """Matrix multiply with Self-Indexed Sparsity.

    Uses bit patterns to create dynamic zeros:
    - For each block of `block_size` bits, compute popcount
    - If popcount == block_size/2 (balanced), treat as zero (uncertain)
    - Otherwise, vote: popcount > block_size/2 → +1, else → -1

    This simulates ternary {-1, 0, +1} at 1.00 bpp.

    Args:
        W_bits: Packed binary weights (d_out, n_words)
        scale: Per-row scales (d_out,)
        x: Input vector (d_in,) or (batch, d_in)
        d_original: Original input dimension
        block_size: Bits per block for sparsity decision (default 8)

    Returns:
        y: Output with ternary-like behavior
    """
    d_out, n_words = W_bits.shape
    d_in = n_words * 32

    # Handle batch dimension
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]

    batch_size = x.shape[0]

    # Pad input if needed
    if x.shape[1] < d_in:
        x_padded = np.zeros((batch_size, d_in), dtype=np.float32)
        x_padded[:, :x.shape[1]] = x
        x = x_padded

    # Transform input to Hadamard domain
    x_h = fast_walsh_hadamard(x, normalize=True)

    # Compute output with SIS
    y = np.zeros((batch_size, d_out), dtype=np.float32)
    half_block = block_size // 2

    for i in range(d_out):
        for b in range(batch_size):
            dot = 0.0
            j = 0

            while j < d_in:
                # Extract block_size bits
                word_idx = j // 32
                bit_offset = j % 32

                if word_idx < n_words:
                    word = W_bits[i, word_idx]
                    block_bits = (word >> bit_offset) & ((1 << block_size) - 1)

                    # Handle block crossing word boundary
                    if bit_offset + block_size > 32 and word_idx + 1 < n_words:
                        remaining = bit_offset + block_size - 32
                        next_word = W_bits[i, word_idx + 1]
                        block_bits |= (next_word & ((1 << remaining) - 1)) << (32 - bit_offset)

                    # Count ones in block
                    pc = bin(block_bits & 0xFF).count('1')

                    # SIS decision
                    if pc == half_block:
                        # Balanced block → treat as zero
                        pass  # dot += 0
                    else:
                        # Sum contributions from this block
                        block_sum = 0.0
                        for k in range(min(block_size, d_in - j)):
                            wrd = W_bits[i, (j + k) // 32]
                            bit = (wrd >> ((j + k) % 32)) & 1
                            sign = 1.0 if bit else -1.0
                            block_sum += sign * x_h[b, j + k]
                        dot += block_sum

                j += block_size

            y[b, i] = dot * scale[i]

    if single:
        y = y[0]

    return y


def sis_matmul_v2(W_bits: np.ndarray, scale: np.ndarray, x: np.ndarray,
                  d_original: int, threshold: float = 0.4) -> np.ndarray:
    """SIS v2: Input-magnitude gating combined with Hadamard.

    Instead of using bit patterns, gate based on |x_h|:
    - Compute x_h = H @ x
    - Zero dimensions where |x_h| < threshold * mean(|x_h|)
    - This creates dynamic sparsity based on input

    Args:
        W_bits: Packed binary weights (d_out, n_words)
        scale: Per-row scales (d_out,)
        x: Input vector (d_in,) or (batch, d_in)
        d_original: Original input dimension
        threshold: Gating threshold (default 0.4 → ~33% sparsity)

    Returns:
        y: Output with input-gated sparsity
    """
    d_out, n_words = W_bits.shape
    d_in = n_words * 32

    # Handle batch dimension
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]

    batch_size = x.shape[0]

    # Pad input if needed
    if x.shape[1] < d_in:
        x_padded = np.zeros((batch_size, d_in), dtype=np.float32)
        x_padded[:, :x.shape[1]] = x
        x = x_padded

    # Transform input to Hadamard domain
    x_h = fast_walsh_hadamard(x, normalize=True)

    # Create gating mask based on |x_h|
    x_h_abs = np.abs(x_h)
    x_h_mean = np.mean(x_h_abs, axis=1, keepdims=True)
    gate = (x_h_abs >= threshold * x_h_mean).astype(np.float32)

    # Apply gate
    x_h_gated = x_h * gate

    # Compute output
    y = np.zeros((batch_size, d_out), dtype=np.float32)

    for i in range(d_out):
        for b in range(batch_size):
            dot = 0.0
            for w in range(n_words):
                word = W_bits[i, w]
                for bit in range(32):
                    j = w * 32 + bit
                    if j >= d_in:
                        break
                    sign = 1.0 if (word >> bit) & 1 else -1.0
                    dot += sign * x_h_gated[b, j]
            y[b, i] = dot * scale[i]

    if single:
        y = y[0]

    return y


# =============================================================================
# Learned Inhibit Mask (LIM) - Structured Sparsity at ~1.00 bpp
# =============================================================================

def lim_quantize(W: np.ndarray, sparsity: float = 0.33,
                 mode: str = "column") -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Quantize with Learned Inhibit Mask.

    Identifies the smallest-magnitude weights and creates a structured
    sparsity mask. The mask is stored at row/column level for compression.

    Args:
        W: Weight matrix (d_out, d_in)
        sparsity: Target sparsity ratio (default 0.33 = ~ternary)
        mode: "column" - per-column mask, "row" - per-row mask

    Returns:
        W_bits: Binary weights (d_out, n_words)
        scale: Per-row scaling factors (d_out,)
        mask: Inhibit mask - which columns/rows to zero
        d_original: Original dimension
    """
    d_out, d_in = W.shape
    d_original = d_in

    # Pad to power of 2 for Hadamard
    d_pad = 1 << (d_in - 1).bit_length()
    if d_pad != d_in:
        W_padded = np.zeros((d_out, d_pad), dtype=np.float32)
        W_padded[:, :d_in] = W
        W = W_padded
        d_in = d_pad

    # Transform to Hadamard domain
    W_h = np.zeros_like(W)
    for i in range(d_out):
        W_h[i] = fast_walsh_hadamard(W[i], normalize=True)

    # Compute importance per column (mean absolute value)
    if mode == "column":
        importance = np.mean(np.abs(W_h), axis=0)  # (d_in,)
        n_zero = int(d_in * sparsity)
        threshold_idx = np.argsort(importance)[n_zero]
        threshold = importance[threshold_idx]
        mask = (importance >= threshold).astype(np.uint8)  # 1 = keep, 0 = zero
    else:
        # Row-wise mask (less common)
        importance = np.mean(np.abs(W_h), axis=1)  # (d_out,)
        n_zero = int(d_out * sparsity)
        threshold_idx = np.argsort(importance)[n_zero]
        threshold = importance[threshold_idx]
        mask = (importance >= threshold).astype(np.uint8)

    # Scale only from non-masked weights
    if mode == "column":
        W_h_masked = W_h * mask[np.newaxis, :]
    else:
        W_h_masked = W_h * mask[:, np.newaxis]

    scale = np.mean(np.abs(W_h_masked), axis=1).astype(np.float32)
    scale[scale == 0] = 1e-8  # Avoid div by zero

    # Quantize to binary
    W_sign = (W_h >= 0).astype(np.uint8)

    # Pack to uint32
    n_words = (d_in + 31) // 32
    W_bits = np.zeros((d_out, n_words), dtype=np.uint32)
    for j in range(d_in):
        word_idx = j // 32
        bit_idx = j % 32
        W_bits[:, word_idx] |= W_sign[:, j].astype(np.uint32) << bit_idx

    return W_bits, scale, mask, d_original


def lim_matmul(W_bits: np.ndarray, scale: np.ndarray, mask: np.ndarray,
               x: np.ndarray, d_original: int) -> np.ndarray:
    """Matrix multiply with Learned Inhibit Mask.

    Args:
        W_bits: Packed binary weights (d_out, n_words)
        scale: Per-row scales (d_out,)
        mask: Column/row inhibit mask
        x: Input vector (d_in,) or (batch, d_in)
        d_original: Original input dimension

    Returns:
        y: Output with structured sparsity applied
    """
    d_out, n_words = W_bits.shape
    d_in = n_words * 32

    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]

    batch_size = x.shape[0]

    # Pad input if needed
    if x.shape[1] < d_in:
        x_padded = np.zeros((batch_size, d_in), dtype=np.float32)
        x_padded[:, :x.shape[1]] = x
        x = x_padded

    # Transform input to Hadamard domain
    x_h = fast_walsh_hadamard(x, normalize=True)

    # Apply mask to input (column-wise sparsity)
    x_h_masked = x_h * mask.astype(np.float32)

    # Compute output
    y = np.zeros((batch_size, d_out), dtype=np.float32)

    for i in range(d_out):
        for b in range(batch_size):
            dot = 0.0
            for w in range(n_words):
                word = W_bits[i, w]
                for bit in range(32):
                    j = w * 32 + bit
                    if j >= d_in:
                        break
                    if mask[j]:  # Only compute if not masked
                        sign = 1.0 if (word >> bit) & 1 else -1.0
                        dot += sign * x_h_masked[b, j]
            y[b, i] = dot * scale[i]

    if single:
        y = y[0]

    return y


def lim_storage_overhead(d_in: int, d_out: int, mode: str = "column") -> float:
    """Calculate storage overhead of LIM mask.

    Returns bits per parameter overhead.
    """
    n_weights = d_out * d_in
    if mode == "column":
        mask_bits = d_in  # 1 bit per column
    else:
        mask_bits = d_out  # 1 bit per row

    return mask_bits / n_weights