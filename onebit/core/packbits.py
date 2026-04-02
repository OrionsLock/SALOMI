from __future__ import annotations

import numpy as np


def pack_signs_rowmajor(W_f: np.ndarray) -> np.ndarray:
    """
    Pack the signs of a 2D float array row-major into uint32 words.

    Mapping: bit=1 for non-negative (>=0), bit=0 for negative.
    Bit order within each 32-bit word is least-significant-bit first.

    Args:
        W_f: float array of shape [M, K]

    Returns:
        np.ndarray[uint32] of shape [M, ceil(K/32)]
    """
    if not isinstance(W_f, np.ndarray):
        W_f = np.asarray(W_f)
    if W_f.ndim != 2:
        raise ValueError("W_f must be 2D [M, K]")

    M, K = W_f.shape
    Kw = (K + 31) // 32
    out = np.zeros((M, Kw), dtype=np.uint32)

    # Vectorized packing
    signs = (W_f >= 0).astype(np.uint32)
    
    for w in range(Kw):
        start = w * 32
        end = min(start + 32, K)
        
        # Process this word for all rows at once
        chunk = signs[:, start:end]
        
        # Pack bits
        for b in range(end - start):
            out[:, w] |= (chunk[:, b] << b)
            
    return out


def pack_signs_colmajor(W_f: np.ndarray) -> np.ndarray:
    """
    Pack the signs of a 2D float array COLUMN-major into uint32 words.
    
    This is equivalent to: pack_signs_rowmajor(W_f.T)
    Useful when we need to access columns efficiently (e.g. W @ x accesses W rows,
    but W.T @ x or x @ W accesses W columns).

    Args:
        W_f: float array of shape [M, K]

    Returns:
        np.ndarray[uint32] of shape [K, ceil(M/32)]
    """
    if not isinstance(W_f, np.ndarray):
        W_f = np.asarray(W_f)
    if W_f.ndim != 2:
        raise ValueError("W_f must be 2D [M, K]")
        
    return pack_signs_rowmajor(W_f.T)



def pack_input_signs(x_f: np.ndarray) -> np.ndarray:
    """
    Pack the signs of a 1D float vector into uint32 words.

    Mapping: bit=1 for non-negative (>=0), bit=0 for negative.
    Bit order within each 32-bit word is least-significant-bit first.

    Args:
        x_f: float array of shape [K]

    Returns:
        np.ndarray[uint32] of shape [ceil(K/32)]
    """
    if not isinstance(x_f, np.ndarray):
        x_f = np.asarray(x_f)
    if x_f.ndim != 1:
        x_f = x_f.reshape(-1)
    K = int(x_f.shape[0])
    Kw = (K + 31) // 32
    out = np.zeros((Kw,), dtype=np.uint32)

    for w in range(Kw):
        start = w * 32
        end = min(start + 32, K)
        chunk = (x_f[start:end] >= 0)
        for b in range(end - start):
            out[w] |= (np.uint32(chunk[b]) << b)
    return out


def pack_float_to_stream(x_f: np.ndarray, k: int) -> np.ndarray:
    """
    Pack float vector(s) into a stochastic bitstream using Sigma-Delta modulation.
    Supports single vector [d_in] or batch [Batch, d_in].
    
    Args:
        x_f: float array of shape [d_in] or [Batch, d_in], values assumed in [-1, 1]
        k: number of ticks (time steps)
        
    Returns:
        x_stream: uint32 array of shape [k, Kw] (if 1D) or [Batch, k, Kw] (if 2D)
    """
    if not isinstance(x_f, np.ndarray):
        x_f = np.asarray(x_f)
        
    ndim = x_f.ndim
    if ndim == 1:
        B = 1
        d_in = x_f.shape[0]
        x_f = x_f[None, :] # [1, d_in]
    elif ndim == 2:
        B, d_in = x_f.shape
    else:
        raise ValueError("x_f must be 1D [d_in] or 2D [B, d_in]")
        
    Kw = (d_in + 31) // 32
    x_stream = np.zeros((B, k, Kw), dtype=np.uint32)
    
    # Error state for Sigma-Delta (per element) [B, d_in]
    error = np.zeros((B, d_in), dtype=np.float32)
    
    # Ensure input is float32
    u = x_f.astype(np.float32)
    
    for t in range(k):
        # Sigma-Delta Step (Order 1)
        # y = sign(u + error)
        # error_new = error + u - y
        # But we output bits 0/1. 1 -> +1, 0 -> -1.
        
        val = u + error
        y_sign = (val >= 0) # Boolean [B, d_in]
        y_val = np.where(y_sign, 1.0, -1.0)
        
        # Update error
        error += (u - y_val)
        
        # Pack bits
        # Vectorized packing for words
        for w in range(Kw):
            start = w * 32
            end = min(start + 32, d_in)
            
            # Extract chunk for this word [B, bits]
            chunk = y_sign[:, start:end]
            
            # Pack boolean chunk into uint32 word
            # Using dot product with powers of 2
            bits = chunk.astype(np.uint32)
            powers = (1 << np.arange(bits.shape[1], dtype=np.uint32))
            word_val = np.sum(bits * powers, axis=1) # [B]
            
            x_stream[:, t, w] = word_val
    
    # If input was 1D, squeeze Batch dim
    if ndim == 1:
        return x_stream[0] # [k, Kw]
    else:
        return x_stream # [B, k, Kw]
