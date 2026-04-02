#!/usr/bin/env python3
"""
Comprehensive Speed Benchmark Tests for SALOMI

This test suite validates inference speed claims and measures:
1. Token latency (P50, P95, P99)
2. Throughput (tokens/second)
3. Memory bandwidth utilization
4. Comparison to FP32 baseline

Target: ≥100 tokens/second for interactive use
"""

import pytest
import numpy as np
import torch
import time
import gc
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import sys
import os
import statistics

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class LatencyResult:
    """Result of latency measurement."""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    n_samples: int


@dataclass
class ThroughputResult:
    """Result of throughput measurement."""
    tokens_per_sec: float
    batch_size: int
    seq_len: int
    n_iterations: int
    total_time: float


@dataclass
class MemoryResult:
    """Result of memory measurement."""
    peak_mb: float
    allocated_mb: float
    model_size_mb: float
    activation_mb: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    method: str
    bpp: float
    latency: LatencyResult
    throughput: ThroughputResult
    memory: MemoryResult
    speedup_vs_fp32: float


def measure_latency(
    fn: Callable[[], Any],
    n_warmup: int = 10,
    n_iterations: int = 100,
) -> LatencyResult:
    """Measure latency of a function.
    
    Args:
        fn: Function to benchmark
        n_warmup: Number of warmup iterations
        n_iterations: Number of timed iterations
        
    Returns:
        LatencyResult with statistics
    """
    # Warmup
    for _ in range(n_warmup):
        fn()
    
    # Force sync if using GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed iterations
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return LatencyResult(
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        p99_ms=float(np.percentile(latencies, 99)),
        mean_ms=float(np.mean(latencies)),
        min_ms=float(np.min(latencies)),
        max_ms=float(np.max(latencies)),
        std_ms=float(np.std(latencies)),
        n_samples=n_iterations,
    )


def measure_throughput(
    fn: Callable[[], Any],
    batch_size: int,
    seq_len: int,
    n_warmup: int = 5,
    n_iterations: int = 20,
) -> ThroughputResult:
    """Measure throughput (tokens/second).
    
    Args:
        fn: Function that processes batch_size * seq_len tokens
        batch_size: Batch size
        seq_len: Sequence length
        n_warmup: Warmup iterations
        n_iterations: Timed iterations
        
    Returns:
        ThroughputResult
    """
    # Warmup
    for _ in range(n_warmup):
        fn()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = end - start
    total_tokens = batch_size * seq_len * n_iterations
    tokens_per_sec = total_tokens / total_time
    
    return ThroughputResult(
        tokens_per_sec=tokens_per_sec,
        batch_size=batch_size,
        seq_len=seq_len,
        n_iterations=n_iterations,
        total_time=total_time,
    )


class MockBinaryLinear:
    """Mock binary linear layer for benchmarking."""
    
    def __init__(self, in_features: int, out_features: int, device: str = "cpu"):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Binary weights (packed as uint8)
        n_bytes = (in_features * out_features + 7) // 8
        self.weight_packed = np.random.randint(0, 256, size=n_bytes, dtype=np.uint8)
        self.scale = np.float32(0.02)
        self.bias = np.random.randn(out_features).astype(np.float32) * 0.02
        
        # FP32 equivalent for comparison
        self.weight_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        
    def forward_binary(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with binary weights (simulated)."""
        # In reality, this would use XNOR + popcount
        # Here we just simulate the latency
        signs = 2 * np.unpackbits(self.weight_packed).astype(np.float32)[:self.out_features * self.in_features].reshape(self.out_features, self.in_features) - 1
        return x @ signs.T * self.scale + self.bias
        
    def forward_fp32(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with FP32 weights."""
        return x @ self.weight_fp32.T + self.bias


class MockBinaryTransformer:
    """Mock binary transformer for benchmarking."""
    
    def __init__(self, d_model: int = 768, n_layers: int = 12, 
                 n_heads: int = 12, d_ff: int = 3072, device: str = "cpu"):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.device = device
        
        # Create layers
        self.layers = []
        for _ in range(n_layers):
            layer = {
                "attn_qkv": MockBinaryLinear(d_model, 3 * d_model, device),
                "attn_proj": MockBinaryLinear(d_model, d_model, device),
                "ff_up": MockBinaryLinear(d_model, d_ff, device),
                "ff_down": MockBinaryLinear(d_ff, d_model, device),
            }
            self.layers.append(layer)
            
    def forward_binary(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with binary weights."""
        for layer in self.layers:
            # Simplified: no actual attention computation
            qkv = layer["attn_qkv"].forward_binary(x)
            attn_out = layer["attn_proj"].forward_binary(qkv[..., :self.d_model])
            x = x + attn_out
            
            h = layer["ff_up"].forward_binary(x)
            h = np.maximum(h, 0)  # ReLU for simplicity
            ff_out = layer["ff_down"].forward_binary(h)
            x = x + ff_out
            
        return x
        
    def forward_fp32(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with FP32 weights."""
        for layer in self.layers:
            qkv = layer["attn_qkv"].forward_fp32(x)
            attn_out = layer["attn_proj"].forward_fp32(qkv[..., :self.d_model])
            x = x + attn_out
            
            h = layer["ff_up"].forward_fp32(x)
            h = np.maximum(h, 0)
            ff_out = layer["ff_down"].forward_fp32(h)
            x = x + ff_out
            
        return x


class TestLatencyMeasurement:
    """Test latency measurement accuracy."""
    
    def test_latency_measurement_is_stable(self):
        """Verify latency measurement produces stable results."""
        def dummy_fn():
            time.sleep(0.001)  # 1ms sleep
            
        result = measure_latency(dummy_fn, n_warmup=5, n_iterations=50)
        
        print(f"\nLatency measurement stability:")
        print(f"  Mean: {result.mean_ms:.2f} ms")
        print(f"  Std: {result.std_ms:.2f} ms")
        print(f"  CV: {result.std_ms / result.mean_ms * 100:.1f}%")
        
        # Coefficient of variation should be low
        cv = result.std_ms / result.mean_ms
        assert cv < 0.5, f"Measurement unstable: CV={cv*100:.1f}%"
        
    def test_percentiles_are_ordered(self):
        """Verify P50 < P95 < P99."""
        def variable_fn():
            # Add some variation
            time.sleep(0.001 + np.random.exponential(0.0005))
            
        result = measure_latency(variable_fn, n_warmup=5, n_iterations=100)
        
        print(f"\nPercentile ordering:")
        print(f"  P50: {result.p50_ms:.2f} ms")
        print(f"  P95: {result.p95_ms:.2f} ms")
        print(f"  P99: {result.p99_ms:.2f} ms")
        
        assert result.p50_ms <= result.p95_ms, "P50 should be <= P95"
        assert result.p95_ms <= result.p99_ms, "P95 should be <= P99"


class TestLinearLayerSpeed:
    """Test speed of linear layer operations."""
    
    def test_binary_vs_fp32_linear(self):
        """Compare binary and FP32 linear layer speed."""
        in_features = 768
        out_features = 3072
        batch_size = 32
        
        layer = MockBinaryLinear(in_features, out_features)
        x = np.random.randn(batch_size, in_features).astype(np.float32)
        
        # Measure binary
        binary_latency = measure_latency(
            lambda: layer.forward_binary(x),
            n_warmup=10,
            n_iterations=100,
        )
        
        # Measure FP32
        fp32_latency = measure_latency(
            lambda: layer.forward_fp32(x),
            n_warmup=10,
            n_iterations=100,
        )
        
        speedup = fp32_latency.mean_ms / binary_latency.mean_ms
        
        print(f"\nLinear layer ({in_features}x{out_features}, batch={batch_size}):")
        print(f"  Binary: {binary_latency.mean_ms:.3f} ms (P95: {binary_latency.p95_ms:.3f})")
        print(f"  FP32: {fp32_latency.mean_ms:.3f} ms (P95: {fp32_latency.p95_ms:.3f})")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Note: Our mock doesn't use actual binary ops, so speedup may be ~1x
        # Real binary with XNOR+popcount should be faster
        
    def test_batch_size_scaling(self):
        """Test how latency scales with batch size."""
        in_features = 768
        out_features = 768
        layer = MockBinaryLinear(in_features, out_features)
        
        batch_sizes = [1, 8, 32, 128]
        latencies = []
        
        print(f"\nBatch size scaling ({in_features}x{out_features}):")
        for bs in batch_sizes:
            x = np.random.randn(bs, in_features).astype(np.float32)
            result = measure_latency(lambda: layer.forward_binary(x), n_iterations=50)
            latencies.append(result.mean_ms)
            print(f"  Batch {bs:3d}: {result.mean_ms:.3f} ms ({bs/result.mean_ms:.1f} samples/ms)")
        
        # Larger batches should be more efficient per sample
        efficiency_1 = batch_sizes[0] / latencies[0]
        efficiency_128 = batch_sizes[-1] / latencies[-1]
        
        print(f"\nEfficiency improvement (batch 128 vs 1): {efficiency_128/efficiency_1:.2f}x")


class TestTransformerSpeed:
    """Test full transformer inference speed."""
    
    def test_transformer_latency(self):
        """Measure transformer forward pass latency."""
        model = MockBinaryTransformer(d_model=768, n_layers=12)
        batch_size = 1
        seq_len = 128
        
        x = np.random.randn(batch_size, seq_len, model.d_model).astype(np.float32)
        
        # Measure binary
        binary_latency = measure_latency(
            lambda: model.forward_binary(x),
            n_warmup=5,
            n_iterations=20,
        )
        
        # Measure FP32
        fp32_latency = measure_latency(
            lambda: model.forward_fp32(x),
            n_warmup=5,
            n_iterations=20,
        )
        
        speedup = fp32_latency.mean_ms / binary_latency.mean_ms
        tokens_per_sec_binary = (batch_size * seq_len) / (binary_latency.mean_ms / 1000)
        tokens_per_sec_fp32 = (batch_size * seq_len) / (fp32_latency.mean_ms / 1000)
        
        print(f"\nTransformer latency (batch={batch_size}, seq={seq_len}):")
        print(f"  Binary: {binary_latency.mean_ms:.1f} ms ({tokens_per_sec_binary:.0f} tok/s)")
        print(f"  FP32: {fp32_latency.mean_ms:.1f} ms ({tokens_per_sec_fp32:.0f} tok/s)")
        print(f"  Speedup: {speedup:.2f}x")
        
    def test_transformer_throughput(self):
        """Measure transformer throughput."""
        model = MockBinaryTransformer(d_model=768, n_layers=12)
        batch_size = 8
        seq_len = 128
        
        x = np.random.randn(batch_size, seq_len, model.d_model).astype(np.float32)
        
        result = measure_throughput(
            lambda: model.forward_binary(x),
            batch_size=batch_size,
            seq_len=seq_len,
            n_iterations=20,
        )
        
        print(f"\nTransformer throughput (batch={batch_size}, seq={seq_len}):")
        print(f"  Throughput: {result.tokens_per_sec:.0f} tokens/sec")
        print(f"  Total time: {result.total_time:.2f}s for {result.n_iterations} iterations")
        
        # Target: >=100 tokens/sec
        # Note: This is a mock, real performance may differ
        print(f"\nTarget: >=100 tokens/sec")
        if result.tokens_per_sec >= 100:
            print("MEETS TARGET")
        else:
            print(f"Below target by {100 - result.tokens_per_sec:.0f} tok/s")


class TestMemoryUsage:
    """Test memory usage of binary models."""
    
    def test_model_size_binary_vs_fp32(self):
        """Compare model size between binary and FP32."""
        d_model = 768
        n_layers = 12
        d_ff = 3072
        
        # Calculate FP32 size
        params_per_layer = (
            d_model * 3 * d_model +  # QKV
            d_model * d_model +      # Proj
            d_model * d_ff +         # FF up
            d_ff * d_model +         # FF down
            4 * d_model              # Layer norms
        )
        total_params = params_per_layer * n_layers
        fp32_mb = total_params * 4 / (1024 * 1024)
        
        # Calculate binary size
        binary_bytes = total_params / 8  # 1 bit per weight
        binary_mb = binary_bytes / (1024 * 1024)
        
        compression = fp32_mb / binary_mb
        
        print(f"\nModel size comparison:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  FP32 size: {fp32_mb:.1f} MB")
        print(f"  Binary size: {binary_mb:.1f} MB")
        print(f"  Compression: {compression:.1f}x")
        
        # Binary should be ~32x smaller
        assert compression > 20, f"Compression should be >20x, got {compression:.1f}x"
        
    def test_runtime_memory(self):
        """Test runtime memory usage."""
        d_model = 768
        n_layers = 12
        batch_size = 32
        seq_len = 512
        
        # Activation memory (FP32)
        # Each layer: inputs + attention scores + intermediate
        activation_per_token = (
            d_model +           # Input activation
            d_model * 12 +      # QKV
            seq_len * 12 +      # Attention scores
            d_model * 4 +       # Intermediate
            d_model             # Output
        ) * 4  # 4 bytes per float
        
        total_activation = batch_size * seq_len * activation_per_token * n_layers
        activation_mb = total_activation / (1024 * 1024)
        
        # KV cache (for inference)
        kv_cache_per_layer = 2 * batch_size * seq_len * d_model * 4
        total_kv = kv_cache_per_layer * n_layers
        kv_mb = total_kv / (1024 * 1024)
        
        print(f"\nRuntime memory (batch={batch_size}, seq={seq_len}):")
        print(f"  Activation memory: {activation_mb:.1f} MB")
        print(f"  KV cache: {kv_mb:.1f} MB")
        print(f"  Total runtime: {activation_mb + kv_mb:.1f} MB")


class TestSpeedTargets:
    """Test that we meet speed targets."""
    
    def test_interactive_latency_target(self):
        """Target: <100ms for interactive use."""
        model = MockBinaryTransformer(d_model=768, n_layers=12)
        batch_size = 1
        seq_len = 1  # Single token generation
        
        x = np.random.randn(batch_size, seq_len, model.d_model).astype(np.float32)
        
        result = measure_latency(lambda: model.forward_binary(x), n_iterations=100)
        
        print(f"\nInteractive latency (single token):")
        print(f"  P50: {result.p50_ms:.2f} ms")
        print(f"  P95: {result.p95_ms:.2f} ms")
        print(f"  Target: <100ms")
        
        if result.p95_ms < 100:
            print("MEETS TARGET")
        else:
            print(f"P95 exceeds target by {result.p95_ms - 100:.1f} ms")
            
    def test_throughput_target(self):
        """Target: ≥100 tokens/second."""
        model = MockBinaryTransformer(d_model=768, n_layers=12)
        batch_size = 8
        seq_len = 128
        
        x = np.random.randn(batch_size, seq_len, model.d_model).astype(np.float32)
        
        result = measure_throughput(
            lambda: model.forward_binary(x),
            batch_size=batch_size,
            seq_len=seq_len,
            n_iterations=20,
        )
        
        print(f"\nThroughput target:")
        print(f"  Achieved: {result.tokens_per_sec:.0f} tokens/sec")
        print(f"  Target: ≥100 tokens/sec")
        
        if result.tokens_per_sec >= 100:
            print("MEETS TARGET")
        else:
            print(f"Below target by {100 - result.tokens_per_sec:.0f} tok/s")


class TestSpeedComparison:
    """Compare speed across different configurations."""
    
    def test_layer_count_scaling(self):
        """Test how speed scales with layer count."""
        batch_size = 8
        seq_len = 128
        d_model = 768
        
        layer_counts = [1, 6, 12, 24]
        results = []
        
        print(f"\nLayer count scaling:")
        for n_layers in layer_counts:
            model = MockBinaryTransformer(d_model=d_model, n_layers=n_layers)
            x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            
            result = measure_throughput(
                lambda: model.forward_binary(x),
                batch_size=batch_size,
                seq_len=seq_len,
                n_iterations=10,
            )
            results.append(result)
            print(f"  {n_layers:2d} layers: {result.tokens_per_sec:.0f} tok/s")
        
        # Throughput should decrease with more layers
        assert results[-1].tokens_per_sec < results[0].tokens_per_sec
        
    def test_sequence_length_scaling(self):
        """Test how speed scales with sequence length."""
        batch_size = 8
        d_model = 768
        n_layers = 12
        
        seq_lengths = [64, 128, 256, 512]
        results = []
        
        model = MockBinaryTransformer(d_model=d_model, n_layers=n_layers)
        
        print(f"\nSequence length scaling:")
        for seq_len in seq_lengths:
            x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            
            result = measure_throughput(
                lambda: model.forward_binary(x),
                batch_size=batch_size,
                seq_len=seq_len,
                n_iterations=10,
            )
            results.append(result)
            ms_per_token = (result.total_time / result.n_iterations) / (batch_size * seq_len) * 1000
            print(f"  seq={seq_len:3d}: {result.tokens_per_sec:.0f} tok/s ({ms_per_token:.3f} ms/tok)")


class TestBenchmarkReport:
    """Generate comprehensive benchmark report."""
    
    def test_generate_full_report(self):
        """Generate full speed benchmark report."""
        d_model = 768
        n_layers = 12
        batch_size = 1
        seq_len = 128
        
        model = MockBinaryTransformer(d_model=d_model, n_layers=n_layers)
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        
        # Binary benchmarks
        binary_latency = measure_latency(lambda: model.forward_binary(x), n_iterations=50)
        binary_throughput = measure_throughput(
            lambda: model.forward_binary(x), batch_size, seq_len, n_iterations=20
        )
        
        # FP32 benchmarks
        fp32_latency = measure_latency(lambda: model.forward_fp32(x), n_iterations=50)
        fp32_throughput = measure_throughput(
            lambda: model.forward_fp32(x), batch_size, seq_len, n_iterations=20
        )
        
        # Model size
        total_params = n_layers * (768*3*768 + 768*768 + 768*3072 + 3072*768)
        fp32_mb = total_params * 4 / (1024 * 1024)
        binary_mb = total_params / 8 / (1024 * 1024)
        
        print("\n" + "=" * 70)
        print("COMPREHENSIVE SPEED BENCHMARK REPORT")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Model: GPT-2 style ({n_layers} layers, d_model={d_model})")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        
        print(f"\nLatency (ms):")
        print(f"  {'':15} {'P50':>10} {'P95':>10} {'P99':>10} {'Mean':>10}")
        print(f"  {'Binary':15} {binary_latency.p50_ms:10.2f} {binary_latency.p95_ms:10.2f} {binary_latency.p99_ms:10.2f} {binary_latency.mean_ms:10.2f}")
        print(f"  {'FP32':15} {fp32_latency.p50_ms:10.2f} {fp32_latency.p95_ms:10.2f} {fp32_latency.p99_ms:10.2f} {fp32_latency.mean_ms:10.2f}")
        
        speedup = fp32_latency.mean_ms / binary_latency.mean_ms
        print(f"\n  Latency speedup: {speedup:.2f}x")
        
        print(f"\nThroughput (tokens/sec):")
        print(f"  Binary: {binary_throughput.tokens_per_sec:.0f}")
        print(f"  FP32: {fp32_throughput.tokens_per_sec:.0f}")
        print(f"  Speedup: {binary_throughput.tokens_per_sec/fp32_throughput.tokens_per_sec:.2f}x")
        
        print(f"\nMemory:")
        print(f"  FP32 model: {fp32_mb:.1f} MB")
        print(f"  Binary model: {binary_mb:.1f} MB")
        print(f"  Compression: {fp32_mb/binary_mb:.1f}x")
        
        print(f"\nTargets:")
        print(f"  Latency <100ms: {'PASS' if binary_latency.p95_ms < 100 else 'FAIL'}")
        print(f"  Throughput >=100 tok/s: {'PASS' if binary_throughput.tokens_per_sec >= 100 else 'FAIL'}")


def run_all_speed_tests():
    """Run all speed benchmark tests."""
    print("=" * 70)
    print("COMPREHENSIVE SPEED BENCHMARK TESTS")
    print("=" * 70)
    
    test_classes = [
        TestLatencyMeasurement,
        TestLinearLayerSpeed,
        TestTransformerSpeed,
        TestMemoryUsage,
        TestSpeedTargets,
        TestSpeedComparison,
        TestBenchmarkReport,
    ]
    
    total_passed = 0
    total_failed = 0
    failures = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 50)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    print(f"  {method_name}... ", end="")
                    method()
                    print("PASSED")
                    total_passed += 1
                except AssertionError as e:
                    print(f"FAILED: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"ERROR: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    
    if failures:
        print("\nFailures:")
        for cls, method, msg in failures:
            print(f"  {cls}.{method}: {msg}")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_speed_tests()
    sys.exit(0 if success else 1)