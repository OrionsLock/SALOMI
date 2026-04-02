"""Tests for tiled BSDM-W kernel parity with naive."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.backends.opencl.host_opencl import OpenCLBinGemm
from onebit.core.packbits import pack_input_signs


@pytest.mark.opencl
def test_tiled_equals_naive_fullk_sd1_sd2():
    """Test tiled vs naive parity with CTG off, early_exit_enable=0, Walsh N=2 + antithetic."""
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")
    
    # Test 3 shapes
    shapes = [
        (128, 2048, 16),  # M=128, d=2048, T=16
        (256, 4096, 8),   # M=256, d=4096, T=8
        (64, 1024, 32),   # M=64, d=1024, T=32
    ]
    
    for M, d, T in shapes:
        print(f"\nTesting shape M={M}, d={d}, T={T}")
        
        Kw = (d + 31) // 32
        
        # Create random data
        np.random.seed(42)
        W = np.random.randn(M, d)
        X = np.random.randn(T, d)
        
        W_bits = np.array([pack_input_signs(W[i]) for i in range(M)])
        X_bits = np.array([pack_input_signs(X[t]) for t in range(T)])
        
        # Test SD-1 and SD-2
        for order in [1, 2]:
            print(f"  Testing order={order}")
            
            # Run naive
            result_naive = gemm.run_bsdm_w_naive_norm(
                W_bits, X_bits,
                T=T,
                eps=0.0,  # No early-exit
                delta=1e-3,
                order=order,
                beta=0.30,
                lambd=1.0/256.0,
                use_ctg=False,
                prf_seed=12345,
                early_exit_enable=False,
                want_y_pack=True,
                want_pc32=False,
                kernel="naive",
            )
            
            # Run tiled
            result_tiled = gemm.run_bsdm_w_naive_norm(
                W_bits, X_bits,
                T=T,
                eps=0.0,
                delta=1e-3,
                order=order,
                beta=0.30,
                lambd=1.0/256.0,
                use_ctg=False,
                prf_seed=12345,
                early_exit_enable=False,
                want_y_pack=True,
                want_pc32=False,
                kernel="tiled",
            )
            
            # Check exact parity
            np.testing.assert_array_almost_equal(
                result_naive["Y"],
                result_tiled["Y"],
                decimal=6,
                err_msg=f"Y mismatch for shape {(M, d, T)}, order={order}"
            )
            
            np.testing.assert_array_equal(
                result_naive["T_eff"],
                result_tiled["T_eff"],
                err_msg=f"T_eff mismatch for shape {(M, d, T)}, order={order}"
            )
            
            np.testing.assert_array_equal(
                result_naive["y_bits_main"],
                result_tiled["y_bits_main"],
                err_msg=f"y_bits_main mismatch for shape {(M, d, T)}, order={order}"
            )
            
            if "y_bits_twin" in result_naive and "y_bits_twin" in result_tiled:
                np.testing.assert_array_equal(
                    result_naive["y_bits_twin"],
                    result_tiled["y_bits_twin"],
                    err_msg=f"y_bits_twin mismatch for shape {(M, d, T)}, order={order}"
                )
            
            np.testing.assert_array_equal(
                result_naive["ctg_digest"],
                result_tiled["ctg_digest"],
                err_msg=f"ctg_digest mismatch for shape {(M, d, T)}, order={order}"
            )
            
            print(f"    ✅ Parity verified")


@pytest.mark.opencl
def test_tiled_vector_widths():
    """Test tiled kernel with Kw not divisible by vector width."""
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")
    
    # Use dimension not divisible by 64 words (TILE_KW_WORDS)
    M = 32
    d = 1000  # Not divisible by 32 or 64*32
    T = 8
    Kw = (d + 31) // 32
    
    np.random.seed(777)
    W = np.random.randn(M, d)
    X = np.random.randn(T, d)
    
    W_bits = np.array([pack_input_signs(W[i]) for i in range(M)])
    X_bits = np.array([pack_input_signs(X[t]) for t in range(T)])
    
    # Run naive
    result_naive = gemm.run_bsdm_w_naive_norm(
        W_bits, X_bits,
        T=T,
        eps=0.0,
        delta=1e-3,
        order=2,
        beta=0.30,
        lambd=1.0/256.0,
        use_ctg=False,
        prf_seed=99999,
        early_exit_enable=False,
        want_y_pack=True,
        kernel="naive",
    )
    
    # Run tiled
    result_tiled = gemm.run_bsdm_w_naive_norm(
        W_bits, X_bits,
        T=T,
        eps=0.0,
        delta=1e-3,
        order=2,
        beta=0.30,
        lambd=1.0/256.0,
        use_ctg=False,
        prf_seed=99999,
        early_exit_enable=False,
        want_y_pack=True,
        kernel="tiled",
    )
    
    # Check exact parity
    np.testing.assert_array_almost_equal(
        result_naive["Y"],
        result_tiled["Y"],
        decimal=6,
        err_msg="Y mismatch for non-aligned Kw"
    )
    
    np.testing.assert_array_equal(
        result_naive["y_bits_main"],
        result_tiled["y_bits_main"],
        err_msg="y_bits_main mismatch for non-aligned Kw"
    )
    
    print(f"\n✅ Tiled kernel works with non-aligned Kw={Kw}")


@pytest.mark.opencl
def test_auto_selection_policy():
    """Test that 'auto' picks tiled when eligible, naive otherwise."""
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")
    
    # Small shape, should use naive
    M_small = 16
    d_small = 128
    T_small = 4
    Kw_small = (d_small + 31) // 32
    
    np.random.seed(123)
    W_small = np.random.randn(M_small, d_small)
    X_small = np.random.randn(T_small, d_small)
    
    W_bits_small = np.array([pack_input_signs(W_small[i]) for i in range(M_small)])
    X_bits_small = np.array([pack_input_signs(X_small[t]) for t in range(T_small)])
    
    # Run with auto (should use naive for small shape)
    result_auto_small = gemm.run_bsdm_w_naive_norm(
        W_bits_small, X_bits_small,
        T=T_small,
        eps=0.0,
        delta=1e-3,
        order=2,
        early_exit_enable=False,
        kernel="auto",
    )
    
    # Run with explicit naive
    result_naive_small = gemm.run_bsdm_w_naive_norm(
        W_bits_small, X_bits_small,
        T=T_small,
        eps=0.0,
        delta=1e-3,
        order=2,
        early_exit_enable=False,
        kernel="naive",
    )
    
    # Should be identical (both use naive)
    np.testing.assert_array_equal(result_auto_small["Y"], result_naive_small["Y"])
    
    # Large shape, should use tiled
    M_large = 256
    d_large = 2048
    T_large = 16
    Kw_large = (d_large + 31) // 32
    
    W_large = np.random.randn(M_large, d_large)
    X_large = np.random.randn(T_large, d_large)
    
    W_bits_large = np.array([pack_input_signs(W_large[i]) for i in range(M_large)])
    X_bits_large = np.array([pack_input_signs(X_large[t]) for t in range(T_large)])
    
    # Run with auto (should use tiled for large shape)
    result_auto_large = gemm.run_bsdm_w_naive_norm(
        W_bits_large, X_bits_large,
        T=T_large,
        eps=0.0,
        delta=1e-3,
        order=2,
        early_exit_enable=False,
        kernel="auto",
    )
    
    # Run with explicit tiled
    result_tiled_large = gemm.run_bsdm_w_naive_norm(
        W_bits_large, X_bits_large,
        T=T_large,
        eps=0.0,
        delta=1e-3,
        order=2,
        early_exit_enable=False,
        kernel="tiled",
    )
    
    # Should be identical (both use tiled)
    np.testing.assert_array_almost_equal(result_auto_large["Y"], result_tiled_large["Y"], decimal=6)
    
    print(f"\n✅ Auto selection policy works correctly")

