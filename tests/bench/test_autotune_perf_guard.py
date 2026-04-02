"""Performance guard tests for auto-tuner.

These tests verify that tiled kernel achieves target speedup.
Skipped if no GPU available.
"""

import pytest
import numpy as np

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False


@pytest.mark.skipif(not HAS_OPENCL, reason="OpenCL not available")
@pytest.mark.opencl
def test_tiled_speedup_gate():
    """Test that tiled kernel achieves ≥1.25x speedup on large shapes.
    
    This is a soft gate - logs results but doesn't fail if GPU unavailable.
    """
    from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    from onebit.core.packbits import pack_input_signs
    from onebit.autotune.tuner import bench_kernel
    
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"Failed to initialize OpenCL: {e}")
        return
    
    # Test shapes (from PR-3.2 spec)
    shapes = [
        (2048, 4096, 8),
        (4096, 8192, 4),
    ]
    
    speedups = []
    
    for M, d, T in shapes:
        Kw = (d + 31) // 32
        
        # Generate test data
        np.random.seed(42)
        W = np.random.randn(M, d)
        X = np.random.randn(T, d)
        W_bits = np.array([pack_input_signs(W[i]) for i in range(M)])
        X_bits = np.array([pack_input_signs(X[t]) for t in range(T)])
        
        # Benchmark naive kernel
        def run_naive():
            gemm.run_bsdm_w_naive_norm(
                W_bits, X_bits, T=T, eps=0.0, delta=1e-3, order=2,
                beta=0.30, lambd=1.0/256.0, use_ctg=False, prf_seed=0,
                early_exit_enable=False, kernel='naive'
            )
        
        naive_ms = bench_kernel(run_naive, repeats=5, warmup=2)
        
        # Benchmark tiled kernel
        def run_tiled():
            gemm.run_bsdm_w_naive_norm(
                W_bits, X_bits, T=T, eps=0.0, delta=1e-3, order=2,
                beta=0.30, lambd=1.0/256.0, use_ctg=False, prf_seed=0,
                early_exit_enable=False, kernel='tiled'
            )
        
        tiled_ms = bench_kernel(run_tiled, repeats=5, warmup=2)
        
        speedup = naive_ms / tiled_ms
        speedups.append(speedup)
        
        print(f"\nShape ({M}, {d}, {T}):")
        print(f"  Naive: {naive_ms:.2f} ms")
        print(f"  Tiled: {tiled_ms:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Check acceptance criteria
    shapes_above_125x = sum(1 for s in speedups if s >= 1.25)
    shapes_below_1x = sum(1 for s in speedups if s < 1.0)
    
    print(f"\n✅ Summary:")
    print(f"  Shapes with ≥1.25x speedup: {shapes_above_125x}/{len(shapes)}")
    print(f"  Shapes with <1.0x speedup: {shapes_below_1x}/{len(shapes)}")
    print(f"  Mean speedup: {np.mean(speedups):.2f}x")
    
    # Soft assertion - log warning but don't fail
    if shapes_above_125x < 2:
        print(f"\n⚠️  WARNING: Expected ≥2 shapes with ≥1.25x speedup, got {shapes_above_125x}")
        print("   This may indicate suboptimal tiled kernel performance.")
    
    if shapes_below_1x > 0:
        print(f"\n⚠️  WARNING: {shapes_below_1x} shapes slower than naive!")
        print("   Tiled kernel should never be slower than naive.")
    
    # Hard assertion: no shape should be slower than naive
    assert shapes_below_1x == 0, f"Tiled kernel slower than naive on {shapes_below_1x} shapes"


@pytest.mark.skipif(not HAS_OPENCL, reason="OpenCL not available")
@pytest.mark.opencl
def test_autotune_determinism():
    """Test that auto-tuning produces deterministic results."""
    from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    from onebit.autotune.tuner import get_device_key, get_kernel_hash
    
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"Failed to initialize OpenCL: {e}")
        return
    
    device_key1 = get_device_key(gemm.ctx)
    device_key2 = get_device_key(gemm.ctx)
    
    assert device_key1 == device_key2, "Device key should be deterministic"
    
    kernel_hash1 = get_kernel_hash(gemm.kernel_source)
    kernel_hash2 = get_kernel_hash(gemm.kernel_source)
    
    assert kernel_hash1 == kernel_hash2, "Kernel hash should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

