"""Smoke test for bake-off harness and storage guard."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.cli.bench_bakeoff import compare_backends
from onebit.tools.export_guard import verify_bpp, verify_model_bpp
from onebit.core.packbits import pack_input_signs


def test_bakeoff_smoke():
    """Smoke test for bake-off harness."""
    np.random.seed(42)

    n_keys = 16
    d = 64
    kA = 8

    # Create synthetic data
    Q = np.random.randn(d)
    K = np.random.randn(n_keys, d)

    Q_bits = pack_input_signs(Q)
    K_bits = np.array([pack_input_signs(K[i]) for i in range(n_keys)])

    # Run comparison
    result = compare_backends(
        Q_bits, K_bits,
        kA=kA,
        prf_seed=12345,
    )

    # Check structure
    assert "cpu" in result, "Result should have cpu"
    assert "opencl" in result, "Result should have opencl"
    assert "max_diff" in result, "Result should have max_diff"
    assert "speedup" in result, "Result should have speedup"

    # Check CPU result
    cpu_result = result["cpu"]
    assert "E_mean" in cpu_result, "CPU result should have E_mean"
    assert "idx_top" in cpu_result, "CPU result should have idx_top"
    assert "T_sel" in cpu_result, "CPU result should have T_sel"
    assert "elapsed_ms" in cpu_result, "CPU result should have elapsed_ms"

    assert cpu_result["E_mean"].shape == (n_keys,), "E_mean should have shape (n_keys,)"
    assert cpu_result["elapsed_ms"] > 0, "elapsed_ms should be positive"

    # Check OpenCL result (if available)
    if result["opencl"] is not None:
        ocl_result = result["opencl"]
        assert "E_mean" in ocl_result, "OpenCL result should have E_mean"
        assert "idx_top" in ocl_result, "OpenCL result should have idx_top"
        assert "T_sel" in ocl_result, "OpenCL result should have T_sel"
        assert "elapsed_ms" in ocl_result, "OpenCL result should have elapsed_ms"

        assert ocl_result["E_mean"].shape == (n_keys,), "E_mean should have shape (n_keys,)"
        assert ocl_result["elapsed_ms"] > 0, "elapsed_ms should be positive"

        # Check parity
        assert result["max_diff"] is not None, "max_diff should be set"
        assert result["speedup"] is not None, "speedup should be set"

        # Parity should be good
        assert result["max_diff"] < 1e-4, f"CPU-OpenCL parity should be good, got max_diff={result['max_diff']}"

    print(f"\nBake-off smoke test passed!")
    print(f"  CPU time: {cpu_result['elapsed_ms']:.2f} ms")
    if result["opencl"] is not None:
        print(f"  OpenCL time: {ocl_result['elapsed_ms']:.2f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
        print(f"  Max diff: {result['max_diff']:.6e}")


def test_storage_guard_single_layer():
    """Test storage guard for single layer."""
    n_params = 1024
    n_words = (n_params + 31) // 32
    
    # Create packed weights (exactly 1.00 bpp)
    W_bits = np.random.randint(0, 2**32, size=n_words, dtype=np.uint32)
    
    # Verify BPP
    result = verify_bpp(W_bits, n_params=n_params)
    
    # Check structure
    assert "n_params" in result, "Result should have n_params"
    assert "n_bits" in result, "Result should have n_bits"
    assert "bpp" in result, "Result should have bpp"
    assert "pass" in result, "Result should have pass"
    
    # Check values
    assert result["n_params"] == n_params, "n_params should match"
    assert result["n_bits"] == n_words * 32, "n_bits should be n_words * 32"
    assert abs(result["bpp"] - 1.0) < 1e-9, f"bpp should be 1.00, got {result['bpp']}"
    assert result["pass"], "Should pass BPP check"
    
    print(f"\nStorage guard single layer test passed!")
    print(f"  n_params: {result['n_params']}")
    print(f"  n_bits: {result['n_bits']}")
    print(f"  bpp: {result['bpp']:.10f}")


def test_storage_guard_model():
    """Test storage guard for multi-layer model."""
    # Create synthetic model
    layers = {}
    for i in range(4):
        n_params = 512 * (i + 1)
        n_words = (n_params + 31) // 32
        W_bits = np.random.randint(0, 2**32, size=n_words, dtype=np.uint32)
        layers[f"layer_{i}"] = W_bits
    
    # Verify BPP
    result = verify_model_bpp(layers)
    
    # Check structure
    assert "total_params" in result, "Result should have total_params"
    assert "total_bits" in result, "Result should have total_bits"
    assert "bpp" in result, "Result should have bpp"
    assert "pass" in result, "Result should have pass"
    assert "layers" in result, "Result should have layers"
    
    # Check values
    assert result["total_params"] > 0, "total_params should be positive"
    assert result["total_bits"] > 0, "total_bits should be positive"
    assert abs(result["bpp"] - 1.0) < 1e-9, f"bpp should be 1.00, got {result['bpp']}"
    assert result["pass"], "Should pass BPP check"
    
    # Check per-layer results
    assert len(result["layers"]) == 4, "Should have 4 layers"
    for name, layer_result in result["layers"].items():
        assert layer_result["pass"], f"Layer {name} should pass BPP check"
        assert abs(layer_result["bpp"] - 1.0) < 1e-9, f"Layer {name} bpp should be 1.00"
    
    print(f"\nStorage guard model test passed!")
    print(f"  total_params: {result['total_params']}")
    print(f"  total_bits: {result['total_bits']}")
    print(f"  bpp: {result['bpp']:.10f}")
    print(f"  layers: {len(result['layers'])}")

