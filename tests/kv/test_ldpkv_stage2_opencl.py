"""Tests for LDP-KV Stage-2 OpenCL kernel."""
import pytest
import numpy as np

from onebit.ops.ldpkv import build_expander_csr, encode_kv_ldp, decode_kv_ldp_stage2
from onebit.backends.opencl.host_opencl import OpenCLBinGemm


@pytest.mark.parametrize("seed", [42, 7, 99999])
@pytest.mark.parametrize("n_pos,d_kv", [(32, 512), (64, 1024), (96, 1536)])
def test_ldpkv_stage2_opencl_parity_vs_cpu(seed, n_pos, d_kv):
    """Test that LDP-KV Stage-2 OpenCL kernel matches CPU implementation."""
    np.random.seed(seed)
    
    # Generate test data
    d_kv_words = (d_kv + 31) // 32
    K_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    V_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    
    # Encode
    d_left = 6
    d_right = 3
    prf_seed = seed
    
    enc_result = encode_kv_ldp(
        K_bits, V_bits,
        d_kv=d_kv,
        d_left=d_left,
        d_right=d_right,
        prf_seed=prf_seed,
    )
    
    V_enc = enc_result["V_enc"]
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    
    # Select winner positions
    n_winners = 4
    winner_positions = np.array([0, 5, 10, 15], dtype=np.int32)
    
    # CPU Stage-2
    cpu_result = decode_kv_ldp_stage2(
        None, V_enc,
        d_kv=d_kv,
        winner_positions=winner_positions,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=16,
        prf_seed=prf_seed,
    )
    
    # OpenCL Stage-2
    gemm = OpenCLBinGemm()
    opencl_result = gemm.run_ldpkv_decode_stage2(
        V_enc,
        row_ptr,
        col_idx,
        edge_weights,
        winner_positions,
        d_kv=d_kv,
    )
    
    # Check parity
    np.testing.assert_array_equal(
        cpu_result["V_decoded"],
        opencl_result["V_decoded"],
        err_msg="V_decoded should match between CPU and OpenCL"
    )


@pytest.mark.parametrize("group_idx", [0, 1])
def test_ldpkv_stage2_opencl_repair_mode(group_idx):
    """Test LDP-KV Stage-2 OpenCL kernel in repair mode."""
    seed = 12345
    np.random.seed(seed)

    n_pos = 128
    d_kv = 512
    d_kv_words = (d_kv + 31) // 32
    
    K_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    V_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    
    # Encode
    d_left = 6
    d_right = 3
    prf_seed = seed
    
    enc_result = encode_kv_ldp(
        K_bits, V_bits,
        d_kv=d_kv,
        d_left=d_left,
        d_right=d_right,
        prf_seed=prf_seed,
    )
    
    K_enc = enc_result["K_enc"]
    V_enc = enc_result["V_enc"]
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    
    # CPU repair
    K_bits_cpu = K_bits.copy()
    cpu_result = decode_kv_ldp_stage2(
        None, V_enc,
        d_kv=d_kv,
        winner_positions=np.array([], dtype=np.int32),
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=16,
        prf_seed=prf_seed,
        repair_pass=True,
        group_idx=group_idx,
        K_enc=K_enc,
        K_bits_inout=K_bits_cpu,
        group_size=64,
    )
    
    # OpenCL repair
    K_bits_opencl = K_bits.copy()
    gemm = OpenCLBinGemm()
    opencl_result = gemm.run_ldpkv_decode_stage2(
        V_enc,
        row_ptr,
        col_idx,
        edge_weights,
        np.array([], dtype=np.int32),
        d_kv=d_kv,
        repair_pass=True,
        group_idx=group_idx,
        group_size=64,
        K_enc=K_enc,
        K_bits_inout=K_bits_opencl,
    )
    
    # Check that repair was performed
    assert cpu_result["repaired"]
    assert opencl_result["repaired"]
    
    # Check that repaired K_bits match
    np.testing.assert_array_equal(
        K_bits_cpu,
        K_bits_opencl,
        err_msg="Repaired K_bits should match between CPU and OpenCL"
    )


def test_ldpkv_stage2_opencl_determinism():
    """Test that LDP-KV Stage-2 OpenCL kernel is deterministic."""
    seed = 7777
    np.random.seed(seed)
    
    n_pos = 64
    d_kv = 768
    d_kv_words = (d_kv + 31) // 32
    
    K_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    V_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    
    # Encode
    d_left = 6
    d_right = 3
    prf_seed = seed
    
    enc_result = encode_kv_ldp(
        K_bits, V_bits,
        d_kv=d_kv,
        d_left=d_left,
        d_right=d_right,
        prf_seed=prf_seed,
    )
    
    V_enc = enc_result["V_enc"]
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    
    winner_positions = np.array([0, 10, 20, 30], dtype=np.int32)
    
    gemm = OpenCLBinGemm()
    
    # Run twice
    result1 = gemm.run_ldpkv_decode_stage2(
        V_enc,
        row_ptr,
        col_idx,
        edge_weights,
        winner_positions,
        d_kv=d_kv,
    )
    
    result2 = gemm.run_ldpkv_decode_stage2(
        V_enc,
        row_ptr,
        col_idx,
        edge_weights,
        winner_positions,
        d_kv=d_kv,
    )
    
    # Should be identical
    np.testing.assert_array_equal(
        result1["V_decoded"],
        result2["V_decoded"],
        err_msg="Results should be deterministic"
    )


@pytest.mark.skipif(
    True,  # Skip by default, run manually for performance testing
    reason="Performance test, run manually"
)
def test_ldpkv_stage2_opencl_speedup():
    """Test that LDP-KV Stage-2 OpenCL kernel is faster than CPU."""
    import time
    
    seed = 42
    np.random.seed(seed)
    
    n_pos = 512
    d_kv = 4096
    d_kv_words = (d_kv + 31) // 32
    
    K_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    V_bits = np.random.randint(0, 2, size=(n_pos, d_kv_words), dtype=np.uint32)
    
    # Encode
    d_left = 6
    d_right = 3
    prf_seed = seed
    
    enc_result = encode_kv_ldp(
        K_bits, V_bits,
        d_kv=d_kv,
        d_left=d_left,
        d_right=d_right,
        prf_seed=prf_seed,
    )
    
    V_enc = enc_result["V_enc"]
    row_ptr = enc_result["row_ptr"]
    col_idx = enc_result["col_idx"]
    edge_weights = enc_result["edge_weights"]
    
    winner_positions = np.arange(16, dtype=np.int32)
    
    # Warmup
    for _ in range(3):
        decode_kv_ldp_stage2(
            None, V_enc,
            d_kv=d_kv,
            winner_positions=winner_positions,
            row_ptr=row_ptr,
            col_idx=col_idx,
            edge_weights=edge_weights,
            k_ticks=16,
            prf_seed=prf_seed,
        )
    
    gemm = OpenCLBinGemm()
    for _ in range(3):
        gemm.run_ldpkv_decode_stage2(
            V_enc, row_ptr, col_idx, edge_weights, winner_positions, d_kv=d_kv
        )
    
    # Benchmark CPU
    runs = 10
    t0 = time.perf_counter()
    for _ in range(runs):
        decode_kv_ldp_stage2(
            None, V_enc,
            d_kv=d_kv,
            winner_positions=winner_positions,
            row_ptr=row_ptr,
            col_idx=col_idx,
            edge_weights=edge_weights,
            k_ticks=16,
            prf_seed=prf_seed,
        )
    t_cpu = (time.perf_counter() - t0) / runs
    
    # Benchmark OpenCL
    t0 = time.perf_counter()
    for _ in range(runs):
        gemm.run_ldpkv_decode_stage2(
            V_enc, row_ptr, col_idx, edge_weights, winner_positions, d_kv=d_kv
        )
    t_opencl = (time.perf_counter() - t0) / runs
    
    speedup = t_cpu / t_opencl
    print(f"\nCPU: {t_cpu*1000:.2f}ms, OpenCL: {t_opencl*1000:.2f}ms, Speedup: {speedup:.2f}x")
    
    # Soft gate: expect at least 2.0x speedup
    assert speedup >= 2.0, f"Expected speedup >= 2.0x, got {speedup:.2f}x"


if __name__ == "__main__":
    # Run parity tests
    test_ldpkv_stage2_opencl_parity_vs_cpu(42, 32, 512)
    test_ldpkv_stage2_opencl_repair_mode(0)
    test_ldpkv_stage2_opencl_determinism()
    print("✅ All LDP-KV Stage-2 OpenCL tests passed!")

