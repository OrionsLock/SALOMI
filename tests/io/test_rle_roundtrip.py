"""Round-trip tests for run-length encoding."""

import pytest
import numpy as np

from onebit.codec.rle import pack_runs, unpack_runs


def bits_to_u32(bits_list, total_bits):
    """Helper: convert list of bits to uint32 array (LSB-first)."""
    n_words = (total_bits + 31) >> 5
    arr = np.zeros(n_words, dtype=np.uint32)
    for i, bit in enumerate(bits_list[:total_bits]):
        if bit:
            word_idx = i >> 5
            bit_pos = i & 31
            arr[word_idx] |= (1 << bit_pos)
    return arr


def u32_to_bits(arr, total_bits):
    """Helper: convert uint32 array to list of bits (LSB-first)."""
    bits = []
    for i in range(total_bits):
        word_idx = i >> 5
        bit_pos = i & 31
        bit = (arr[word_idx] >> bit_pos) & 1
        bits.append(int(bit))
    return bits


def test_all_zeros_various_lengths():
    """Test all-zeros bit streams of various lengths."""
    for total_bits in [1, 7, 32, 33, 1000]:
        bits = [0] * total_bits
        bits_u32 = bits_to_u32(bits, total_bits)
        
        blob, num_runs = pack_runs(bits_u32, total_bits)
        
        # Should be single run
        assert num_runs == 1
        assert blob[0] == 0  # start_bit = 0
        
        # Decode
        decoded = unpack_runs(blob, total_bits)
        decoded_bits = u32_to_bits(decoded, total_bits)
        
        assert decoded_bits == bits


def test_all_ones_various_lengths():
    """Test all-ones bit streams of various lengths."""
    for total_bits in [1, 7, 32, 33, 1000]:
        bits = [1] * total_bits
        bits_u32 = bits_to_u32(bits, total_bits)
        
        blob, num_runs = pack_runs(bits_u32, total_bits)
        
        # Should be single run
        assert num_runs == 1
        assert blob[0] == 1  # start_bit = 1
        
        # Decode
        decoded = unpack_runs(blob, total_bits)
        decoded_bits = u32_to_bits(decoded, total_bits)
        
        assert decoded_bits == bits


def test_strict_alternation():
    """Test strict alternating pattern 010101..."""
    for total_bits in [2, 10, 33, 100]:
        bits = [i % 2 for i in range(total_bits)]
        bits_u32 = bits_to_u32(bits, total_bits)
        
        blob, num_runs = pack_runs(bits_u32, total_bits)
        
        # Should have total_bits runs (each length 1)
        assert num_runs == total_bits
        
        # Decode
        decoded = unpack_runs(blob, total_bits)
        decoded_bits = u32_to_bits(decoded, total_bits)
        
        assert decoded_bits == bits


def test_two_long_runs():
    """Test two long runs: 11111...00000..."""
    total_bits = 100
    bits = [1] * 60 + [0] * 40
    bits_u32 = bits_to_u32(bits, total_bits)
    
    blob, num_runs = pack_runs(bits_u32, total_bits)
    
    # Should have 2 runs
    assert num_runs == 2
    assert blob[0] == 1  # start_bit = 1
    
    # Decode
    decoded = unpack_runs(blob, total_bits)
    decoded_bits = u32_to_bits(decoded, total_bits)
    
    assert decoded_bits == bits


def test_random_patterns():
    """Test 100 random bit patterns."""
    np.random.seed(42)
    
    for trial in range(100):
        total_bits = np.random.randint(1, 5001)
        
        # Generate random bits with some run structure
        bits = []
        while len(bits) < total_bits:
            run_len = np.random.randint(1, 50)
            bit_val = np.random.randint(0, 2)
            bits.extend([bit_val] * run_len)
        bits = bits[:total_bits]
        
        bits_u32 = bits_to_u32(bits, total_bits)
        
        # Encode
        blob, num_runs = pack_runs(bits_u32, total_bits)
        
        # Decode
        decoded = unpack_runs(blob, total_bits)
        decoded_bits = u32_to_bits(decoded, total_bits)
        
        assert decoded_bits == bits, f"Mismatch on trial {trial}"


def test_test_vectors():
    """Test specific vectors from spec."""
    
    # Vector 1: total_bits=10, bits: 1111100000
    bits = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    bits_u32 = bits_to_u32(bits, 10)
    blob, num_runs = pack_runs(bits_u32, 10)
    
    assert blob[0] == 1  # start_bit = 1
    assert num_runs == 2
    # Runs: [5, 5] -> LEB128: 0x05, 0x05
    assert blob == bytes([0x01, 0x05, 0x05])
    
    decoded = unpack_runs(blob, 10)
    assert u32_to_bits(decoded, 10) == bits
    
    # Vector 2: total_bits=9, bits: 101010101
    bits = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    bits_u32 = bits_to_u32(bits, 9)
    blob, num_runs = pack_runs(bits_u32, 9)
    
    assert blob[0] == 1  # start_bit = 1
    assert num_runs == 9
    # Runs: [1,1,1,1,1,1,1,1,1] -> 9 bytes of 0x01
    assert blob == bytes([0x01] + [0x01] * 9)
    
    decoded = unpack_runs(blob, 9)
    assert u32_to_bits(decoded, 9) == bits
    
    # Vector 3: total_bits=33, bits: 1×33
    bits = [1] * 33
    bits_u32 = bits_to_u32(bits, 33)
    blob, num_runs = pack_runs(bits_u32, 33)
    
    assert blob[0] == 1  # start_bit = 1
    assert num_runs == 1
    # Run: [33] -> LEB128(33) = 0x21 (33 = 0b100001, fits in 1 byte)
    assert blob == bytes([0x01, 0x21])
    
    decoded = unpack_runs(blob, 33)
    assert u32_to_bits(decoded, 33) == bits


def test_partial_last_word():
    """Test that partial last word is handled correctly."""
    # 35 bits = 1 full word + 3 bits in second word
    total_bits = 35
    bits = [1] * 20 + [0] * 15
    bits_u32 = bits_to_u32(bits, total_bits)
    
    # Ensure second word has padding bits set to 0
    assert bits_u32.shape[0] == 2
    
    blob, num_runs = pack_runs(bits_u32, total_bits)
    decoded = unpack_runs(blob, total_bits)
    decoded_bits = u32_to_bits(decoded, total_bits)
    
    assert decoded_bits == bits


def test_single_bit():
    """Test edge case of single bit."""
    for bit_val in [0, 1]:
        bits = [bit_val]
        bits_u32 = bits_to_u32(bits, 1)
        
        blob, num_runs = pack_runs(bits_u32, 1)
        assert num_runs == 1
        assert blob[0] == bit_val
        
        decoded = unpack_runs(blob, 1)
        assert u32_to_bits(decoded, 1) == bits


def test_exact_word_boundary():
    """Test patterns at exact 32-bit boundaries."""
    # Exactly 32 bits
    bits = [1] * 16 + [0] * 16
    bits_u32 = bits_to_u32(bits, 32)
    
    blob, num_runs = pack_runs(bits_u32, 32)
    decoded = unpack_runs(blob, 32)
    
    assert u32_to_bits(decoded, 32) == bits
    
    # Exactly 64 bits
    bits = [1] * 32 + [0] * 32
    bits_u32 = bits_to_u32(bits, 64)
    
    blob, num_runs = pack_runs(bits_u32, 64)
    decoded = unpack_runs(blob, 64)
    
    assert u32_to_bits(decoded, 64) == bits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

