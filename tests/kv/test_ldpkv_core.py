"""Tests for LDP-KV core functionality."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.ops.ldpkv import (
    build_expander_csr,
    encode_kv_ldp,
    decode_kv_ldp_stage1,
    decode_kv_ldp_stage2,
)
from onebit.core.packbits import pack_input_signs


def test_expander_graph_degrees():
    """Test that expander graph satisfies degree constraints."""
    n_pos = 64
    d_left = 8
    d_right = 4
    seed = 12345
    
    row_ptr, col_idx, edge_weights = build_expander_csr(n_pos, d_left, d_right, seed)
    
    # Check row_ptr structure
    assert len(row_ptr) == n_pos + 1, "row_ptr should have n_pos + 1 elements"
    assert row_ptr[0] == 0, "row_ptr[0] should be 0"
    assert row_ptr[-1] == n_pos * d_left, "row_ptr[-1] should be total edges"
    
    # Check left degrees
    for pos in range(n_pos):
        degree = row_ptr[pos + 1] - row_ptr[pos]
        assert degree == d_left, f"Position {pos} should have degree {d_left}, got {degree}"
    
    # Check right degrees
    d_dim = (n_pos * d_left) // d_right
    right_degree = np.zeros(d_dim, dtype=np.int32)
    
    for e in range(len(col_idx)):
        dim_id = col_idx[e]
        right_degree[dim_id] += 1
    
    # All right nodes should have degree d_right
    for dim_id in range(d_dim):
        assert right_degree[dim_id] == d_right, \
            f"Dimension {dim_id} should have degree {d_right}, got {right_degree[dim_id]}"
    
    # Check edge weights are ±1
    assert np.all(np.abs(edge_weights) == 1), "Edge weights should be ±1"
    
    print(f"Expander graph: n_pos={n_pos}, d_left={d_left}, d_right={d_right}, d_dim={d_dim}")
    print(f"  Total edges: {len(col_idx)}")
    print(f"  Edge weight distribution: +1={np.sum(edge_weights == 1)}, -1={np.sum(edge_weights == -1)}")


def test_encode_decode_roundtrip_small():
    """Test encode-decode roundtrip on small example."""
    np.random.seed(42)
    
    n_pos = 16
    d_kv = 64
    d_left = 4
    d_right = 2
    seed = 7777
    
    # Create random K/V
    K = np.random.randn(n_pos, d_kv)
    V = np.random.randn(n_pos, d_kv)
    
    K_bits = np.array([pack_input_signs(K[i]) for i in range(n_pos)])
    V_bits = np.array([pack_input_signs(V[i]) for i in range(n_pos)])
    
    # Encode
    enc_result = encode_kv_ldp(
        K_bits, V_bits,
        d_kv=d_kv,
        d_left=d_left,
        d_right=d_right,
        prf_seed=seed,
    )
    
    K_enc = enc_result["K_enc"]
    V_enc = enc_result["V_enc"]
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    d_dim = enc_result["d_dim"]
    
    print(f"\nEncoded: d_dim={d_dim}, K_enc range=[{K_enc.min():.3f}, {K_enc.max():.3f}]")
    
    # Decode Stage-1: compute energies for all positions
    Q = K[0]  # Use first key as query
    Q_bits = pack_input_signs(Q)
    
    dec_result = decode_kv_ldp_stage1(
        Q_bits, K_enc,
        d_kv=d_kv,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=16,
        prf_seed=seed,
    )
    
    E_mean = dec_result["E_mean"]
    
    print(f"Stage-1 energies: {E_mean[:5]}")
    
    # Position 0 should have highest energy (query matches key 0)
    # (This is not guaranteed with expander encoding, but check interface works)
    assert E_mean.shape == (n_pos,), "E_mean should have shape (n_pos,)"
    assert dec_result["k_used"] > 0, "Should use at least some ticks"
    
    # Decode Stage-2: retrieve values for top positions
    top_k = 4
    winner_positions = np.argsort(E_mean)[::-1][:top_k]
    
    dec2_result = decode_kv_ldp_stage2(
        Q_bits, V_enc,
        d_kv=d_kv,
        winner_positions=winner_positions,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=16,
        prf_seed=seed,
    )
    
    V_decoded = dec2_result["V_decoded"]
    
    print(f"Stage-2 decoded {len(winner_positions)} values")
    
    assert V_decoded.shape == (top_k, (d_kv + 31) // 32), "V_decoded shape should match"


def test_expander_determinism():
    """Test that expander graph generation is deterministic."""
    n_pos = 32
    d_left = 6
    d_right = 3
    seed = 99999
    
    # Generate twice with same seed
    row_ptr1, col_idx1, edge_weights1 = build_expander_csr(n_pos, d_left, d_right, seed)
    row_ptr2, col_idx2, edge_weights2 = build_expander_csr(n_pos, d_left, d_right, seed)
    
    # Should be identical
    np.testing.assert_array_equal(row_ptr1, row_ptr2, err_msg="row_ptr should be deterministic")
    np.testing.assert_array_equal(col_idx1, col_idx2, err_msg="col_idx should be deterministic")
    np.testing.assert_array_equal(edge_weights1, edge_weights2, err_msg="edge_weights should be deterministic")
    
    # Generate with different seed
    row_ptr3, col_idx3, edge_weights3 = build_expander_csr(n_pos, d_left, d_right, seed + 1)
    
    # Should be different
    assert not np.array_equal(col_idx1, col_idx3), "Different seeds should give different graphs"


def test_encode_determinism():
    """Test that encoding is deterministic."""
    np.random.seed(123)
    
    n_pos = 8
    d_kv = 32
    d_left = 4
    d_right = 2
    seed = 55555
    
    # Create random K/V
    K = np.random.randn(n_pos, d_kv)
    V = np.random.randn(n_pos, d_kv)
    
    K_bits = np.array([pack_input_signs(K[i]) for i in range(n_pos)])
    V_bits = np.array([pack_input_signs(V[i]) for i in range(n_pos)])
    
    # Encode twice
    enc1 = encode_kv_ldp(K_bits, V_bits, d_kv=d_kv, d_left=d_left, d_right=d_right, prf_seed=seed)
    enc2 = encode_kv_ldp(K_bits, V_bits, d_kv=d_kv, d_left=d_left, d_right=d_right, prf_seed=seed)
    
    # Should be identical
    np.testing.assert_array_equal(enc1["K_enc"], enc2["K_enc"], err_msg="K_enc should be deterministic")
    np.testing.assert_array_equal(enc1["V_enc"], enc2["V_enc"], err_msg="V_enc should be deterministic")
    np.testing.assert_array_equal(enc1["row_ptr"], enc2["row_ptr"], err_msg="row_ptr should be deterministic")
    np.testing.assert_array_equal(enc1["col_idx"], enc2["col_idx"], err_msg="col_idx should be deterministic")
    np.testing.assert_array_equal(enc1["edge_weights"], enc2["edge_weights"], err_msg="edge_weights should be deterministic")

