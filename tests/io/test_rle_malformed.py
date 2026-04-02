"""Tests for malformed RLE inputs."""

import pytest
import numpy as np

from onebit.codec.rle import pack_runs, unpack_runs, _leb128_encode_u, _leb128_decode_u


def test_truncated_leb128():
    """Test that truncated LEB128 raises ValueError."""
    # Create a blob with continuation bit set but no following byte
    blob = bytes([0x01, 0x85])  # start_bit=1, then 0x85 (continuation bit set)
    
    with pytest.raises(ValueError, match="Truncated LEB128"):
        unpack_runs(blob, 10)


def test_overlong_leb128_chain():
    """Test that LEB128 chain >10 bytes raises ValueError."""
    # Create a blob with 11 continuation bytes
    blob = bytearray([0x01])  # start_bit
    for _ in range(11):
        blob.append(0x80)  # All continuation bits set
    
    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        unpack_runs(bytes(blob), 100)


def test_sum_mismatch():
    """Test that sum of runs != total_bits raises ValueError."""
    # Encode 10 bits, but claim total_bits=15
    blob = bytes([0x01, 0x05, 0x05])  # start_bit=1, runs=[5,5] = 10 bits
    
    with pytest.raises(ValueError, match="Decoded 10 bits but expected 15"):
        unpack_runs(blob, 15)
    
    # Encode 10 bits, but claim total_bits=5
    with pytest.raises(ValueError, match="Decoded 10 bits but expected 5"):
        unpack_runs(blob, 5)


def test_invalid_start_bit():
    """Test that invalid start_bit raises ValueError."""
    blob = bytes([0x02, 0x05])  # start_bit=2 (invalid)
    
    with pytest.raises(ValueError, match="Invalid start_bit"):
        unpack_runs(blob, 5)
    
    blob = bytes([0xFF, 0x05])  # start_bit=255 (invalid)
    
    with pytest.raises(ValueError, match="Invalid start_bit"):
        unpack_runs(blob, 5)


def test_empty_blob():
    """Test that empty blob raises ValueError."""
    with pytest.raises(ValueError, match="at least start_bit"):
        unpack_runs(bytes([]), 10)


def test_total_bits_zero():
    """Test that total_bits < 1 raises ValueError."""
    bits_u32 = np.array([0], dtype=np.uint32)
    
    with pytest.raises(ValueError, match="total_bits must be >= 1"):
        pack_runs(bits_u32, 0)
    
    with pytest.raises(ValueError, match="total_bits must be >= 1"):
        unpack_runs(bytes([0x01, 0x05]), 0)


def test_leb128_negative():
    """Test that LEB128 encoding negative number raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        _leb128_encode_u(-1)


def test_leb128_decode_at_end():
    """Test LEB128 decode when starting at end of blob."""
    blob = bytes([0x01, 0x05])
    
    # Try to decode starting past the end
    with pytest.raises(ValueError, match="Truncated"):
        _leb128_decode_u(blob, 10)


def test_zero_length_run():
    """Test that zero-length run is handled (though shouldn't occur in practice)."""
    # Manually construct blob with zero-length run
    blob = bytes([0x01, 0x00, 0x0A])  # start_bit=1, runs=[0, 10]
    
    # This should decode to 10 bits total
    decoded = unpack_runs(blob, 10)
    
    # Should be all 0s (since first run is 0 length of bit=1, then 10 of bit=0)
    bits = []
    for i in range(10):
        word_idx = i >> 5
        bit_pos = i & 31
        bit = (decoded[word_idx] >> bit_pos) & 1
        bits.append(int(bit))
    
    assert bits == [0] * 10


def test_very_long_run():
    """Test encoding/decoding very long run (>127, requires multi-byte LEB128)."""
    # 300 ones
    total_bits = 300
    n_words = (total_bits + 31) >> 5
    bits_u32 = np.zeros(n_words, dtype=np.uint32)

    # Set all bits to 1 for first 300 bits
    for i in range(total_bits):
        word_idx = i >> 5
        bit_pos = i & 31
        bits_u32[word_idx] |= (1 << bit_pos)

    blob, num_runs = pack_runs(bits_u32, total_bits)

    # Should be single run of 300
    assert num_runs == 1
    assert blob[0] == 1

    # LEB128(300) = 0xAC 0x02 (300 = 0b100101100)
    # 300 & 0x7F = 0x2C, set continuation -> 0xAC
    # 300 >> 7 = 2, no continuation -> 0x02
    assert blob == bytes([0x01, 0xAC, 0x02])

    decoded = unpack_runs(blob, total_bits)

    # Verify all 300 bits are 1
    for i in range(total_bits):
        word_idx = i >> 5
        bit_pos = i & 31
        bit = (decoded[word_idx] >> bit_pos) & 1
        assert bit == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

