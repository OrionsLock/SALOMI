"""Bit packing utilities for golden logs with consistent endianness.

Packing convention for y_bits:
- Most-recent tick in the least significant bit of each word
- Hex rendered big-endian for JSONL logs
- RLE encoding available for compression (optional)
"""
from __future__ import annotations

import base64
import numpy as np

from onebit.codec.rle import pack_runs, unpack_runs


def pack_y_bits_to_hex(y_bits: np.ndarray, num_bits: int) -> str:
    """Pack y_bits array to big-endian hex string.
    
    Args:
        y_bits: uint32 array with LSB-first bit packing
        num_bits: number of valid bits (may be < len(y_bits)*32)
        
    Returns:
        Big-endian hex string (uppercase, no 0x prefix)
    """
    if y_bits.dtype != np.uint32:
        raise TypeError("y_bits must be uint32")
    
    # Convert to bytes (little-endian on host)
    byte_arr = y_bits.tobytes()
    
    # Truncate to exact byte count needed
    num_bytes = (num_bits + 7) // 8
    byte_arr = byte_arr[:num_bytes]
    
    # Reverse to big-endian and convert to hex
    # (LSB-first within words, but hex rendered big-endian)
    hex_str = byte_arr[::-1].hex().upper()
    
    return hex_str


def unpack_y_bits_from_hex(hex_str: str, num_bits: int) -> np.ndarray:
    """Unpack big-endian hex string to y_bits uint32 array.
    
    Args:
        hex_str: Big-endian hex string (uppercase, no 0x prefix)
        num_bits: number of valid bits
        
    Returns:
        uint32 array with LSB-first bit packing
    """
    num_bytes = (num_bits + 7) // 8
    num_words = (num_bits + 31) // 32
    
    # Decode hex and reverse to little-endian
    byte_arr = bytes.fromhex(hex_str)[::-1]
    
    # Pad to word boundary
    if len(byte_arr) < num_words * 4:
        byte_arr += b'\x00' * (num_words * 4 - len(byte_arr))
    
    # Convert to uint32 array
    y_bits = np.frombuffer(byte_arr, dtype=np.uint32, count=num_words)
    
    return y_bits.copy()


def pack_y_bits_array(y_values: list[int], num_bits: int) -> np.ndarray:
    """Pack list of y values (+1/-1 or 1/0) into uint32 array.

    Args:
        y_values: list of y values (1 for +1, 0 for -1)
        num_bits: number of bits to pack

    Returns:
        uint32 array with LSB-first bit packing
    """
    num_words = (num_bits + 31) // 32
    y_bits = np.zeros(num_words, dtype=np.uint32)

    for i, y in enumerate(y_values[:num_bits]):
        if y > 0:  # +1 -> bit 1
            y_bits[i >> 5] |= np.uint32(1) << (i & 31)

    return y_bits


def pack_y_bits_rle(y_bits: np.ndarray, total_bits: int) -> dict:
    """Pack y_bits array to RLE format for JSONL logs.

    Args:
        y_bits: uint32 array with LSB-first bit packing
        total_bits: exact number of meaningful bits

    Returns:
        Dictionary with keys:
            - coding: "rle-v1"
            - b64: base64-encoded RLE blob
            - total_bits: number of bits
            - start_bit: first bit value (0 or 1)
    """
    if y_bits.dtype != np.uint32:
        raise TypeError("y_bits must be uint32")

    blob, num_runs = pack_runs(y_bits, total_bits)

    return {
        "coding": "rle-v1",
        "b64": base64.b64encode(blob).decode('ascii'),
        "total_bits": total_bits,
        "start_bit": int(blob[0]),
        "num_runs": num_runs,
    }


def unpack_y_bits_rle(rle_obj: dict) -> np.ndarray:
    """Unpack RLE-encoded y_bits from JSONL log format.

    Args:
        rle_obj: Dictionary with keys: coding, b64, total_bits

    Returns:
        uint32 array with LSB-first bit packing

    Raises:
        ValueError: If coding is not "rle-v1" or data is malformed
    """
    if rle_obj.get("coding") != "rle-v1":
        raise ValueError(f"Unsupported coding: {rle_obj.get('coding')}")

    blob = base64.b64decode(rle_obj["b64"])
    total_bits = rle_obj["total_bits"]

    return unpack_runs(blob, total_bits)

