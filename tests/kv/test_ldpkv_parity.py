"""Tests for LDP-KV CPU-OpenCL parity."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.ops.ldpkv import (
    build_expander_csr,
    encode_kv_ldp,
    decode_kv_ldp_stage1,
)
from onebit.core.packbits import pack_input_signs


@pytest.mark.opencl
def test_ldpkv_stage1_cpu_opencl_parity():
    """Test CPU-OpenCL parity for LDP-KV Stage-1 decode."""
    np.random.seed(777)
    
    n_pos = 32
    d_kv = 128
    d_left = 8
    d_right = 4
    seed = 12345
    k_ticks = 16
    
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
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    
    # Query
    Q = K[5]  # Use key 5 as query
    Q_bits = pack_input_signs(Q)
    
    # CPU decode
    cpu_result = decode_kv_ldp_stage1(
        Q_bits, K_enc,
        d_kv=d_kv,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=k_ticks,
        prf_seed=seed,
        early_exit_enable=False,
    )
    
    # OpenCL decode
    from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    gemm = OpenCLBinGemm()
    
    ocl_result = gemm.run_ldpkv_decode_stage1(
        K_enc,
        row_ptr,
        col_idx,
        edge_weights,
        n_pos=n_pos,
        T=k_ticks,
        early_exit_enable=False,
    )
    
    # Check parity
    np.testing.assert_array_almost_equal(
        cpu_result["E_mean"],
        ocl_result["E_mean"],
        decimal=5,
        err_msg="E_mean mismatch between CPU and OpenCL"
    )
    
    # Check k_used
    assert cpu_result["k_used"] == k_ticks, "CPU k_used should equal k_ticks"
    assert np.all(ocl_result["T_eff"] == k_ticks), "OpenCL T_eff should all equal k_ticks"
    
    print(f"\nCPU E_mean: {cpu_result['E_mean'][:5]}")
    print(f"OCL E_mean: {ocl_result['E_mean'][:5]}")
    print(f"Max diff: {np.max(np.abs(cpu_result['E_mean'] - ocl_result['E_mean'])):.6f}")


def test_ldpkv_stage1_determinism():
    """Test that Stage-1 decode is deterministic."""
    np.random.seed(888)
    
    n_pos = 16
    d_kv = 64
    d_left = 4
    d_right = 2
    seed = 99999
    k_ticks = 8
    
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
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    
    # Query
    Q = K[0]
    Q_bits = pack_input_signs(Q)
    
    # Decode twice
    result1 = decode_kv_ldp_stage1(
        Q_bits, K_enc,
        d_kv=d_kv,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=k_ticks,
        prf_seed=seed,
        early_exit_enable=False,
    )
    
    result2 = decode_kv_ldp_stage1(
        Q_bits, K_enc,
        d_kv=d_kv,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=k_ticks,
        prf_seed=seed,
        early_exit_enable=False,
    )
    
    # Should be identical
    np.testing.assert_array_equal(result1["E_mean"], result2["E_mean"], err_msg="E_mean should be deterministic")
    assert result1["k_used"] == result2["k_used"], "k_used should be deterministic"

