"""Tests for Hadamard transform utilities."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.core.hadamard import (
    gray_index,
    gray_inverse,
    build_col_masks,
    hadamard_sign_word,
    hadamard_row_full,
    hadamard_row_bits,
)


def test_gray_roundtrip():
    """Test that gray_index and gray_inverse are inverses."""
    for i in range(256):
        g = gray_index(i)
        i_recovered = gray_inverse(g)
        assert i_recovered == i, f"Gray roundtrip failed for i={i}: got {i_recovered}"
    
    # Test some specific values
    assert gray_index(0) == 0
    assert gray_index(1) == 1
    assert gray_index(2) == 3
    assert gray_index(3) == 2
    assert gray_index(4) == 6
    assert gray_index(5) == 7
    assert gray_index(6) == 5
    assert gray_index(7) == 4


def test_row_orthogonality_small():
    """Test that Hadamard rows are orthogonal for m=32."""
    m = 32
    d = 32
    
    # Generate rows as ±1 values
    rows = np.zeros((m, d), dtype=np.int8)
    for v_id in range(m):
        rows[v_id] = hadamard_row_full(v_id, d)
    
    # Check orthogonality: rows[i] · rows[j] = 0 for i ≠ j, = d for i = j
    for i in range(m):
        for j in range(m):
            dot = np.dot(rows[i], rows[j])
            if i == j:
                assert dot == d, f"Self-dot for row {i} should be {d}, got {dot}"
            else:
                assert dot == 0, f"Rows {i} and {j} should be orthogonal, got dot={dot}"


def test_word_determinism():
    """Test that hadamard_sign_word is deterministic."""
    v_id = 42
    word_idx = 3
    
    # Run multiple times
    word1 = hadamard_sign_word(v_id, word_idx)
    word2 = hadamard_sign_word(v_id, word_idx)
    word3 = hadamard_sign_word(v_id, word_idx)
    
    assert word1 == word2, "hadamard_sign_word should be deterministic"
    assert word2 == word3, "hadamard_sign_word should be deterministic"
    
    # Check that different v_id gives different word
    word_other = hadamard_sign_word(v_id + 1, word_idx)
    assert word1 != word_other, "Different v_id should give different word"


def test_col_masks_structure():
    """Test that col_masks have correct structure."""
    d = 64
    col_masks = build_col_masks(d)
    
    assert col_masks.shape == (d,), f"col_masks should have shape ({d},)"
    assert col_masks.dtype == np.uint32, "col_masks should be uint32"
    
    # Check that col_masks[j] == gray_index(j)
    for j in range(d):
        assert col_masks[j] == gray_index(j), f"col_masks[{j}] should equal gray_index({j})"


def test_hadamard_row_bits_consistency():
    """Test that hadamard_row_bits matches hadamard_row_full."""
    d = 128
    v_id = 17
    
    # Generate row as ±1 values
    row_full = hadamard_row_full(v_id, d)
    
    # Generate row as packed bits
    row_bits = hadamard_row_bits(v_id, d)
    
    # Unpack bits and compare
    for j in range(d):
        word_idx = j // 32
        bit_idx = j % 32
        bit_val = (row_bits[word_idx] >> bit_idx) & 1
        
        # bit_val = 1 → +1, bit_val = 0 → -1
        expected_sign = 1 if bit_val == 1 else -1
        
        assert row_full[j] == expected_sign, f"Mismatch at dimension {j}: full={row_full[j]}, bits={expected_sign}"


def test_hadamard_sign_word_specific():
    """Test specific values of hadamard_sign_word."""
    # For v_id=0, all dimensions should be +1 (all bits set)
    word = hadamard_sign_word(0, 0)
    assert word == 0xFFFFFFFF, f"v_id=0 should give all +1 (0xFFFFFFFF), got {word:08x}"
    
    # For v_id=1, pattern depends on Gray code
    # Dimension 0: gray(0)=0, popcount(1&0)=0 (even) → +1 → bit=1
    # Dimension 1: gray(1)=1, popcount(1&1)=1 (odd) → -1 → bit=0
    # Dimension 2: gray(2)=3, popcount(1&3)=1 (odd) → -1 → bit=0
    # Dimension 3: gray(3)=2, popcount(1&2)=0 (even) → +1 → bit=1
    # etc.
    word = hadamard_sign_word(1, 0)
    # Check first 4 bits: should be 0b1001 = 0x9 in lowest nibble
    assert (word & 0xF) == 0x9, f"v_id=1 first 4 bits should be 0x9, got {word & 0xF:x}"


def test_hadamard_orthogonality_via_bits():
    """Test orthogonality using bit-packed representation."""
    m = 16
    d = 64
    
    # Generate rows as packed bits
    rows_bits = np.zeros((m, d // 32), dtype=np.uint32)
    for v_id in range(m):
        rows_bits[v_id] = hadamard_row_bits(v_id, d)
    
    # Check orthogonality via XNOR-popcount
    for i in range(m):
        for j in range(m):
            # Compute dot product via XNOR-popcount
            pc = 0
            for word_idx in range(d // 32):
                xnor = ~(rows_bits[i, word_idx] ^ rows_bits[j, word_idx])
                pc += bin(xnor & 0xFFFFFFFF).count('1')
            
            dot = 2 * pc - d
            
            if i == j:
                assert dot == d, f"Self-dot for row {i} should be {d}, got {dot}"
            else:
                assert dot == 0, f"Rows {i} and {j} should be orthogonal, got dot={dot}"

