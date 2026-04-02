"""Tests for auto-tune cache functionality."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from onebit.autotune.tuner import (
    get_device_key,
    get_kernel_hash,
    load_autotune_cache,
    save_autotune_cache,
    get_best_config,
    bench_kernel,
    tune_bsdm_w,
)


def test_kernel_hash_deterministic():
    """Test that kernel hash is deterministic."""
    source1 = "kernel void foo() { }"
    source2 = "kernel void foo() { }"
    source3 = "kernel void bar() { }"
    
    hash1 = get_kernel_hash(source1)
    hash2 = get_kernel_hash(source2)
    hash3 = get_kernel_hash(source3)
    
    assert hash1 == hash2, "Same source should produce same hash"
    assert hash1 != hash3, "Different source should produce different hash"
    assert len(hash1) == 16, "Hash should be 16 characters"


def test_cache_roundtrip():
    """Test cache save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override cache path
        import onebit.autotune.tuner as tuner_module
        original_get_cache_path = tuner_module.get_cache_path
        
        def mock_get_cache_path():
            return Path(tmpdir) / "autotune.json"
        
        tuner_module.get_cache_path = mock_get_cache_path
        
        try:
            # Create test cache
            test_cache = {
                "device1": {
                    "kernel_hash": "abc123",
                    "tile_kw": 64,
                    "local_size": 256,
                    "mean_speedup": 1.5,
                },
                "device2": {
                    "kernel_hash": "def456",
                    "tile_kw": 128,
                    "local_size": 512,
                    "mean_speedup": 2.0,
                },
            }
            
            # Save
            save_autotune_cache(test_cache)
            
            # Load
            loaded = load_autotune_cache()
            
            assert loaded == test_cache, "Loaded cache should match saved cache"
            
            # Verify file exists
            cache_path = mock_get_cache_path()
            assert cache_path.exists(), "Cache file should exist"
            
            # Verify JSON format
            with open(cache_path, 'r') as f:
                data = json.load(f)
            assert data == test_cache, "File content should match"
            
        finally:
            tuner_module.get_cache_path = original_get_cache_path


def test_get_best_config_from_cache():
    """Test getting best config from cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import onebit.autotune.tuner as tuner_module
        original_get_cache_path = tuner_module.get_cache_path
        
        def mock_get_cache_path():
            return Path(tmpdir) / "autotune.json"
        
        tuner_module.get_cache_path = mock_get_cache_path
        
        try:
            # Create cache with entry
            cache = {
                "test_device": {
                    "kernel_hash": "test_hash",
                    "tile_kw": 128,
                    "local_size": 512,
                }
            }
            save_autotune_cache(cache)
            
            # Get config with matching hash
            tile_kw, local_size = get_best_config("test_device", "test_hash")
            assert tile_kw == 128
            assert local_size == 512
            
            # Get config with mismatched hash (should return defaults)
            tile_kw, local_size = get_best_config("test_device", "wrong_hash")
            assert tile_kw == 64  # default
            assert local_size == 256  # default
            
            # Get config for unknown device (should return defaults)
            tile_kw, local_size = get_best_config("unknown_device", "test_hash")
            assert tile_kw == 64
            assert local_size == 256
            
        finally:
            tuner_module.get_cache_path = original_get_cache_path


def test_get_best_config_env_override():
    """Test environment variable override."""
    import os
    import onebit.autotune.tuner as tuner_module
    
    # Set environment variables
    os.environ['ONEBIT_TILE'] = '32'
    os.environ['ONEBIT_LS'] = '128'
    
    try:
        tile_kw, local_size = get_best_config("any_device", "any_hash")
        assert tile_kw == 32
        assert local_size == 128
    finally:
        # Clean up
        os.environ.pop('ONEBIT_TILE', None)
        os.environ.pop('ONEBIT_LS', None)


def test_bench_kernel_timing():
    """Test kernel benchmarking."""
    import time
    
    # Mock kernel that sleeps for 10ms
    def mock_kernel():
        time.sleep(0.01)
    
    # Benchmark with minimal repeats
    ms_mean = bench_kernel(mock_kernel, repeats=3, warmup=1)
    
    # Should be around 10ms (allow 5ms tolerance for overhead)
    assert 8.0 <= ms_mean <= 15.0, f"Expected ~10ms, got {ms_mean}ms"


def test_tune_bsdm_w_selection():
    """Test tuning selects best configuration."""
    
    # Mock benchmark function
    # Config (64, 256) is fastest, (128, 512) is slowest
    def mock_bench_fn(M, Kw, T, tile_kw, local_size):
        if tile_kw == 0:  # naive
            return 100.0
        elif tile_kw == 64 and local_size == 256:
            return 50.0  # 2x speedup
        elif tile_kw == 128 and local_size == 512:
            return 80.0  # 1.25x speedup
        else:
            return 90.0  # 1.11x speedup
    
    candidates = [(64, 256), (128, 512), (32, 128)]
    shapes = [(128, 1024, 8)]
    
    result = tune_bsdm_w(
        device_key="test_device",
        kernel_hash="test_hash",
        candidates=candidates,
        shapes=shapes,
        bench_fn=mock_bench_fn,
    )
    
    # Should select (64, 256) as it has best speedup
    assert result['tile_kw'] == 64
    assert result['local_size'] == 256
    assert result['mean_speedup'] == 2.0
    assert result['min_speedup'] == 2.0


def test_tune_bsdm_w_prefers_safe_configs():
    """Test tuning prefers configs with min_speedup >= 1.0."""
    
    # Config (64, 256) has higher mean but one shape is slower than naive
    # Config (128, 512) has lower mean but all shapes faster than naive
    def mock_bench_fn(M, Kw, T, tile_kw, local_size):
        if tile_kw == 0:  # naive
            return 100.0
        elif tile_kw == 64 and local_size == 256:
            if M == 128:
                return 40.0  # 2.5x speedup
            else:
                return 120.0  # 0.83x speedup (slower!)
        elif tile_kw == 128 and local_size == 512:
            return 80.0  # 1.25x speedup (consistent)
        else:
            return 90.0
    
    candidates = [(64, 256), (128, 512)]
    shapes = [(128, 1024, 8), (256, 2048, 8)]
    
    result = tune_bsdm_w(
        device_key="test_device",
        kernel_hash="test_hash",
        candidates=candidates,
        shapes=shapes,
        bench_fn=mock_bench_fn,
    )
    
    # Should select (128, 512) because it has min_speedup >= 1.0
    # even though (64, 256) has higher mean_speedup
    assert result['tile_kw'] == 128
    assert result['local_size'] == 512
    assert result['min_speedup'] >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

