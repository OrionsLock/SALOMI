"""Tests for RLE integration in golden_bits module."""

import pytest
import numpy as np

from onebit.core.golden_bits import pack_y_bits_rle, unpack_y_bits_rle, pack_y_bits_array


def test_pack_unpack_rle_roundtrip():
    """Test pack/unpack RLE roundtrip."""
    # Create test bit pattern
    y_values = [1, 1, 1, 0, 0, 0, 1, 1]
    total_bits = len(y_values)
    y_bits = pack_y_bits_array(y_values, total_bits)
    
    # Pack to RLE
    rle_obj = pack_y_bits_rle(y_bits, total_bits)
    
    # Verify structure
    assert rle_obj["coding"] == "rle-v1"
    assert "b64" in rle_obj
    assert rle_obj["total_bits"] == total_bits
    assert rle_obj["start_bit"] in (0, 1)
    
    # Unpack
    decoded = unpack_y_bits_rle(rle_obj)
    
    # Verify bits match
    for i in range(total_bits):
        word_idx = i >> 5
        bit_pos = i & 31
        orig_bit = (y_bits[word_idx] >> bit_pos) & 1
        decoded_bit = (decoded[word_idx] >> bit_pos) & 1
        assert orig_bit == decoded_bit


def test_rle_format_fields():
    """Test that RLE format has all required fields."""
    y_bits = np.array([0xFFFFFFFF], dtype=np.uint32)
    total_bits = 32
    
    rle_obj = pack_y_bits_rle(y_bits, total_bits)
    
    assert "coding" in rle_obj
    assert "b64" in rle_obj
    assert "total_bits" in rle_obj
    assert "start_bit" in rle_obj
    assert "num_runs" in rle_obj
    
    assert rle_obj["coding"] == "rle-v1"
    assert isinstance(rle_obj["b64"], str)
    assert rle_obj["total_bits"] == 32
    assert rle_obj["start_bit"] == 1  # All ones
    assert rle_obj["num_runs"] == 1  # Single run


def test_rle_invalid_coding():
    """Test that invalid coding raises ValueError."""
    rle_obj = {
        "coding": "invalid",
        "b64": "AQEB",
        "total_bits": 8,
    }
    
    with pytest.raises(ValueError, match="Unsupported coding"):
        unpack_y_bits_rle(rle_obj)


def test_rle_various_patterns():
    """Test RLE with various bit patterns."""
    patterns = [
        [1] * 100,  # All ones
        [0] * 100,  # All zeros
        [1, 0] * 50,  # Alternating
        [1] * 50 + [0] * 50,  # Two runs
    ]
    
    for pattern in patterns:
        total_bits = len(pattern)
        y_bits = pack_y_bits_array(pattern, total_bits)
        
        rle_obj = pack_y_bits_rle(y_bits, total_bits)
        decoded = unpack_y_bits_rle(rle_obj)
        
        # Verify all bits match
        for i in range(total_bits):
            word_idx = i >> 5
            bit_pos = i & 31
            orig_bit = (y_bits[word_idx] >> bit_pos) & 1
            decoded_bit = (decoded[word_idx] >> bit_pos) & 1
            assert orig_bit == decoded_bit, f"Mismatch at bit {i}"


def test_rle_partial_word():
    """Test RLE with partial last word."""
    # 35 bits (1 full word + 3 bits)
    y_values = [1] * 20 + [0] * 15
    total_bits = 35
    y_bits = pack_y_bits_array(y_values, total_bits)
    
    rle_obj = pack_y_bits_rle(y_bits, total_bits)
    decoded = unpack_y_bits_rle(rle_obj)
    
    # Verify all 35 bits
    for i in range(total_bits):
        word_idx = i >> 5
        bit_pos = i & 31
        orig_bit = (y_bits[word_idx] >> bit_pos) & 1
        decoded_bit = (decoded[word_idx] >> bit_pos) & 1
        assert orig_bit == decoded_bit


def test_rle_base64_valid():
    """Test that base64 encoding is valid."""
    import base64
    
    y_bits = np.array([0x12345678], dtype=np.uint32)
    total_bits = 32
    
    rle_obj = pack_y_bits_rle(y_bits, total_bits)
    
    # Should be valid base64
    try:
        decoded_blob = base64.b64decode(rle_obj["b64"])
        assert len(decoded_blob) > 0
    except Exception as e:
        pytest.fail(f"Invalid base64: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

