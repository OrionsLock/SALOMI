"""Run-length encoding for bit streams.

Encodes/decodes bit streams using run-length encoding with LEB128 variable-length integers.
Used for compressing y_bits_main/twin in logs and serialization.

Wire format:
  - First byte: start_bit (0 or 1)
  - Remaining bytes: LEB128-encoded run lengths
  - Sign alternates implicitly between runs
"""

from typing import Tuple, Iterator
import numpy as np


def _leb128_encode_u(n: int) -> bytes:
    """Encode unsigned integer as LEB128 (Little Endian Base 128).
    
    Args:
        n: Non-negative integer to encode
        
    Returns:
        Bytes representing the LEB128 encoding
    """
    if n < 0:
        raise ValueError("LEB128 encoding requires non-negative integer")
    
    result = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n != 0:
            byte |= 0x80  # Set continuation bit
        result.append(byte)
        if n == 0:
            break
    return bytes(result)


def _leb128_decode_u(blob: bytes, start_idx: int) -> Tuple[int, int]:
    """Decode LEB128 unsigned integer from bytes.
    
    Args:
        blob: Byte array containing LEB128 data
        start_idx: Starting index in blob
        
    Returns:
        Tuple of (decoded_value, next_index)
        
    Raises:
        ValueError: If LEB128 chain exceeds 10 bytes or is truncated
    """
    value = 0
    shift = 0
    idx = start_idx
    max_bytes = 10  # Reject overlong chains
    
    for _ in range(max_bytes):
        if idx >= len(blob):
            raise ValueError("Truncated LEB128 encoding")
        
        byte = blob[idx]
        idx += 1
        
        value |= (byte & 0x7F) << shift
        shift += 7
        
        if (byte & 0x80) == 0:
            # No continuation bit, done
            return value, idx
    
    raise ValueError("LEB128 chain exceeds 10 bytes (malformed)")


def _iter_bits_u32(bits_u32: np.ndarray, total_bits: int) -> Iterator[int]:
    """Iterate over bits in LSB-first order from uint32 array.
    
    Args:
        bits_u32: uint32 array with LSB-first bit packing
        total_bits: Number of meaningful bits to iterate
        
    Yields:
        Bits (0 or 1) in LSB-first order
    """
    for bit_idx in range(total_bits):
        word_idx = bit_idx >> 5  # bit_idx // 32
        bit_pos = bit_idx & 31   # bit_idx % 32
        bit = (bits_u32[word_idx] >> bit_pos) & 1
        yield int(bit)


def pack_runs(bits_u32: np.ndarray, total_bits: int) -> Tuple[bytes, int]:
    """Encode bit stream as run-length encoded bytes.
    
    Args:
        bits_u32: uint32 array with LSB-first bit packing
        total_bits: Exact number of meaningful bits
        
    Returns:
        Tuple of (encoded_bytes, num_runs)
        
    Raises:
        ValueError: If total_bits < 1
    """
    if total_bits < 1:
        raise ValueError("total_bits must be >= 1")
    
    bits_u32 = np.asarray(bits_u32, dtype=np.uint32)
    
    # Iterate bits and compute runs
    bit_iter = _iter_bits_u32(bits_u32, total_bits)
    
    # Get first bit
    start_bit = next(bit_iter)
    current_bit = start_bit
    current_run_len = 1
    runs = []
    
    for bit in bit_iter:
        if bit == current_bit:
            current_run_len += 1
        else:
            # Run ended, save it
            runs.append(current_run_len)
            current_bit = bit
            current_run_len = 1
    
    # Save final run
    runs.append(current_run_len)
    
    # Encode as bytes
    result = bytearray()
    result.append(start_bit)  # First byte is start_bit
    
    for run_len in runs:
        result.extend(_leb128_encode_u(run_len))
    
    return bytes(result), len(runs)


def unpack_runs(blob: bytes, total_bits: int) -> np.ndarray:
    """Decode run-length encoded bytes to uint32 bit array.
    
    Args:
        blob: Encoded bytes (start_bit + LEB128 run lengths)
        total_bits: Expected number of bits to decode
        
    Returns:
        uint32 array with LSB-first bit packing
        
    Raises:
        ValueError: If blob is malformed or decoded bits != total_bits
    """
    if total_bits < 1:
        raise ValueError("total_bits must be >= 1")
    
    if len(blob) < 1:
        raise ValueError("Blob must contain at least start_bit")
    
    # Decode start_bit
    start_bit = blob[0]
    if start_bit not in (0, 1):
        raise ValueError(f"Invalid start_bit: {start_bit}")
    
    # Decode runs
    runs = []
    idx = 1
    while idx < len(blob):
        run_len, idx = _leb128_decode_u(blob, idx)
        runs.append(run_len)
    
    # Verify total
    decoded_bits = sum(runs)
    if decoded_bits != total_bits:
        raise ValueError(
            f"Decoded {decoded_bits} bits but expected {total_bits}"
        )
    
    # Reconstruct bit array
    n_words = (total_bits + 31) >> 5  # Ceiling division by 32
    bits_u32 = np.zeros(n_words, dtype=np.uint32)
    
    bit_idx = 0
    current_bit = start_bit
    
    for run_len in runs:
        for _ in range(run_len):
            if current_bit == 1:
                word_idx = bit_idx >> 5
                bit_pos = bit_idx & 31
                bits_u32[word_idx] |= (1 << bit_pos)
            bit_idx += 1
        # Alternate bit value for next run
        current_bit = 1 - current_bit
    
    return bits_u32

