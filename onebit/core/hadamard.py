"""Hadamard transform utilities for on-the-fly row generation and Fast Walsh-Hadamard Transform.

Convention: LSB-first bit packing within 32-bit words.
"""
from __future__ import annotations

import numpy as np


def gray_index(i: int) -> int:
    """Standard Gray code: i XOR (i >> 1).
    
    Args:
        i: Non-negative integer index
    
    Returns:
        Gray code of i
    
    Examples:
        >>> gray_index(0)
        0
        >>> gray_index(1)
        1
        >>> gray_index(2)
        3
        >>> gray_index(3)
        2
    """
    return i ^ (i >> 1)


def gray_inverse(g: int) -> int:
    """Inverse Gray code.

    Args:
        g: Gray code value

    Returns:
        Original index i such that gray_index(i) == g
    """
    i = g
    j = 1
    while j < 64:  # Sufficient for 64-bit integers
        i ^= i >> j
        j <<= 1
    return i


def build_col_masks(d: int, word_bits: int = 32) -> np.ndarray:
    """Build column masks for Hadamard row generation.
    
    For each dimension j in [0, d), we need to know which bit positions
    in the Gray-coded Hadamard row contribute to that column.
    
    Convention: LSB-first packing within 32-bit words.
    For column j, the bit set inside mask word is determined by:
        (gray_index(j) >> bit_position) & 1
    
    Args:
        d: Dimension (must be power of 2 for standard Hadamard)
        word_bits: Bits per word (default: 32)
    
    Returns:
        Array of shape [d] with dtype uint32.
        col_masks[j] contains the mask for column j.
    
    Example:
        For d=8, word_bits=32:
        - Column 0: gray_index(0) = 0b000 → mask = 0x00000000
        - Column 1: gray_index(1) = 0b001 → mask = 0x00000001
        - Column 2: gray_index(2) = 0b011 → mask = 0x00000003
        - Column 3: gray_index(3) = 0b010 → mask = 0x00000002
        etc.
    """
    col_masks = np.zeros(d, dtype=np.uint32)
    
    for j in range(d):
        gray_j = gray_index(j)
        col_masks[j] = np.uint32(gray_j)
    
    return col_masks


def hadamard_sign_word(v_id: int, word_idx: int, word_bits: int = 32) -> np.uint32:
    """Compute Hadamard row bits for token v_id over one word.
    
    Generates word_bits dimensions starting at word_idx * word_bits.
    Uses Gray-coded Hadamard construction:
        H[v_id, j] = (-1)^popcount(v_id & gray_index(j))
    
    Convention: LSB-first packing. Bit k in the returned word corresponds
    to dimension (word_idx * word_bits + k).
    
    Args:
        v_id: Token/row index
        word_idx: Word index (0-based)
        word_bits: Bits per word (default: 32)
    
    Returns:
        32-bit word with Hadamard row signs.
        Bit k = 1 if H[v_id, word_idx*32 + k] = +1
        Bit k = 0 if H[v_id, word_idx*32 + k] = -1
    
    Example:
        For v_id=5 (0b101), word_idx=0:
        - Dimension 0: gray(0)=0, popcount(5&0)=0 (even) → +1 → bit=1
        - Dimension 1: gray(1)=1, popcount(5&1)=1 (odd) → -1 → bit=0
        - Dimension 2: gray(2)=3, popcount(5&3)=2 (even) → +1 → bit=1
        - Dimension 3: gray(3)=2, popcount(5&2)=1 (odd) → -1 → bit=0
        etc.
    """
    word = np.uint32(0)
    
    base_dim = word_idx * word_bits
    
    for k in range(word_bits):
        j = base_dim + k
        gray_j = gray_index(j)
        
        # Compute popcount(v_id & gray_j)
        masked = v_id & gray_j
        pc = bin(masked).count('1')
        
        # If popcount is even, sign is +1 → bit = 1
        # If popcount is odd, sign is -1 → bit = 0
        if pc % 2 == 0:
            word |= np.uint32(1 << k)
    
    return word


def hadamard_row_full(v_id: int, d: int) -> np.ndarray:
    """Generate full Hadamard row as ±1 values (for testing).
    
    Args:
        v_id: Token/row index
        d: Dimension
    
    Returns:
        Array of shape [d] with dtype int8, values in {-1, +1}
    """
    row = np.zeros(d, dtype=np.int8)
    
    for j in range(d):
        gray_j = gray_index(j)
        masked = v_id & gray_j
        pc = bin(masked).count('1')
        
        # Even popcount → +1, odd → -1
        row[j] = 1 if pc % 2 == 0 else -1
    
    return row


def hadamard_row_bits(v_id: int, d: int, word_bits: int = 32) -> np.ndarray:
    """Generate full Hadamard row as packed bits (for testing).
    
    Args:
        v_id: Token/row index
        d: Dimension
        word_bits: Bits per word (default: 32)
    
    Returns:
        Array of shape [d_words] with dtype uint32.
        LSB-first packing: bit k in word w corresponds to dimension w*32 + k.
    """
    d_words = (d + word_bits - 1) // word_bits
    row_bits = np.zeros(d_words, dtype=np.uint32)
    
    for word_idx in range(d_words):
        row_bits[word_idx] = hadamard_sign_word(v_id, word_idx, word_bits)
    
    return row_bits


def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (sequency-ordered/Gray-code NOT guaranteed).
    
    Standard Sylvester construction (natural order).
    If you need Gray-ordered (sequency) output, you must permute the input/output.
    
    For HCL, as long as the encoder (W_code generation) and decoder (FWHT)
    use the same ordering, it works.
    
    Complexity: O(N log N)
    
    Args:
        a: Input array of shape [..., n], where n is power of 2.
    
    Returns:
        Transformed array (same shape).
    """
    a = np.asarray(a, dtype=np.float32)
    n = a.shape[-1]
    if (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2")
    
    # Use separate implementation for loop logic to avoid view issues
    # Iterative FWHT (in-place on copy)
    a = a.copy()
    
    h = 1
    while h < n:
        # We can reshape to (..., n//(2h), 2, h)
        shape = a.shape[:-1] + (n // (2 * h), 2, h)
        a_view = a.reshape(shape)
        
        x = a_view[..., 0, :]
        y = a_view[..., 1, :]
        
        sum_xy = x + y
        diff_xy = x - y
        
        a_view[..., 0, :] = sum_xy
        a_view[..., 1, :] = diff_xy
        
        h *= 2
        
    return a


def inverse_fwht(a: np.ndarray) -> np.ndarray:
    """Inverse Fast Walsh-Hadamard Transform.
    
    IFWHT(x) = FWHT(x) / N
    
    Args:
        a: Input array of shape [..., n]
        
    Returns:
        Inverse transformed array.
    """
    n = a.shape[-1]
    return fwht(a) / float(n)
