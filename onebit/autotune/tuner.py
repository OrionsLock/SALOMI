"""Auto-tuner for BSDM-W tiled kernel parameters.

Selects optimal (TILE_KW_WORDS, local_size) per device.
Never changes math - only kernel launch parameters.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np


def get_device_key(ctx) -> str:
    """Generate unique device key from OpenCL context.
    
    Args:
        ctx: PyOpenCL context
        
    Returns:
        Device key string: "vendor_device_driver"
    """
    device = ctx.devices[0]
    vendor = device.vendor.strip().replace(" ", "_")
    name = device.name.strip().replace(" ", "_")
    driver = device.driver_version.strip().replace(" ", "_")
    return f"{vendor}_{name}_{driver}"


def get_kernel_hash(kernel_source: str) -> str:
    """Compute SHA256 hash of kernel source.
    
    Args:
        kernel_source: OpenCL kernel source code
        
    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(kernel_source.encode('utf-8')).hexdigest()[:16]


def get_cache_path() -> Path:
    """Get path to autotune cache file.
    
    Returns:
        Path to ~/.onebit/autotune.json
    """
    cache_dir = Path.home() / ".onebit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "autotune.json"


def load_autotune_cache() -> Dict:
    """Load autotune cache from disk.
    
    Returns:
        Cache dictionary, empty if file doesn't exist
    """
    cache_path = get_cache_path()
    if not cache_path.exists():
        return {}
    
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_autotune_cache(cache: Dict) -> None:
    """Save autotune cache to disk.
    
    Args:
        cache: Cache dictionary to save
    """
    cache_path = get_cache_path()
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"Warning: Failed to save autotune cache: {e}")


def bench_kernel(
    fn: Callable[[], None],
    *,
    repeats: int = 10,
    warmup: int = 3
) -> float:
    """Benchmark a kernel function.
    
    Args:
        fn: Function to benchmark (should include queue.finish())
        repeats: Number of timing runs
        warmup: Number of warmup runs
        
    Returns:
        Mean execution time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # Convert to ms
    
    return float(np.mean(times))


def tune_bsdm_w(
    device_key: str,
    kernel_hash: str,
    candidates: List[Tuple[int, int]],
    shapes: List[Tuple[int, int, int]],
    bench_fn: Callable[[int, int, int, int, int], float],
) -> Dict:
    """Tune BSDM-W kernel for given device.
    
    Args:
        device_key: Unique device identifier
        kernel_hash: Hash of kernel source
        candidates: List of (TILE_KW_WORDS, local_size) tuples
        shapes: List of (M, Kw, T) tuples to benchmark
        bench_fn: Function(M, Kw, T, tile_kw, local_size) -> ms_mean
        
    Returns:
        Best config dict with keys: tile_kw, local_size, speedup, timings
    """
    # Benchmark naive kernel for baseline
    print(f"Benchmarking naive kernel on {len(shapes)} shapes...")
    naive_times = {}
    for M, Kw, T in shapes:
        ms = bench_fn(M, Kw, T, tile_kw=0, local_size=0)  # 0 = naive
        naive_times[(M, Kw, T)] = ms
        print(f"  Shape ({M},{Kw},{T}): {ms:.2f} ms")
    
    # Benchmark all candidates
    print(f"\nBenchmarking {len(candidates)} tiled configurations...")
    results = []
    
    for tile_kw, local_size in candidates:
        print(f"\n  Config: TILE_KW={tile_kw}, local_size={local_size}")
        tiled_times = {}
        speedups = []
        
        for M, Kw, T in shapes:
            try:
                ms = bench_fn(M, Kw, T, tile_kw=tile_kw, local_size=local_size)
                tiled_times[(M, Kw, T)] = ms
                speedup = naive_times[(M, Kw, T)] / ms
                speedups.append(speedup)
                print(f"    Shape ({M},{Kw},{T}): {ms:.2f} ms, speedup: {speedup:.2f}x")
            except Exception as e:
                print(f"    Shape ({M},{Kw},{T}): FAILED - {e}")
                speedups.append(0.0)
                tiled_times[(M, Kw, T)] = float('inf')
        
        # Compute metrics
        valid_speedups = [s for s in speedups if s > 0]
        if not valid_speedups:
            continue
        
        mean_speedup = float(np.mean(valid_speedups))
        min_speedup = float(np.min(valid_speedups))
        
        results.append({
            'tile_kw': tile_kw,
            'local_size': local_size,
            'mean_speedup': mean_speedup,
            'min_speedup': min_speedup,
            'speedups': speedups,
            'timings': {f"{M}_{Kw}_{T}": ms for (M, Kw, T), ms in tiled_times.items()},
        })
    
    if not results:
        raise RuntimeError("No valid tiled configurations found")
    
    # Select best: highest mean speedup, with min_speedup >= 1.0 preferred
    results.sort(key=lambda r: (r['min_speedup'] >= 1.0, r['mean_speedup']), reverse=True)
    best = results[0]
    
    print(f"\n✅ Best config: TILE_KW={best['tile_kw']}, local_size={best['local_size']}")
    print(f"   Mean speedup: {best['mean_speedup']:.2f}x")
    print(f"   Min speedup: {best['min_speedup']:.2f}x")
    
    return {
        'device_key': device_key,
        'kernel_hash': kernel_hash,
        'tile_kw': best['tile_kw'],
        'local_size': best['local_size'],
        'mean_speedup': best['mean_speedup'],
        'min_speedup': best['min_speedup'],
        'timings': best['timings'],
        'all_results': results,
    }


def get_best_config(
    device_key: str,
    kernel_hash: str,
    default_tile_kw: int = 64,
    default_local_size: int = 256,
) -> Tuple[int, int]:
    """Get best config from cache or return defaults.
    
    Args:
        device_key: Unique device identifier
        kernel_hash: Hash of kernel source
        default_tile_kw: Default tile size if not cached
        default_local_size: Default local size if not cached
        
    Returns:
        Tuple of (tile_kw, local_size)
    """
    # Check environment overrides
    env_tile = os.environ.get('ONEBIT_TILE')
    env_ls = os.environ.get('ONEBIT_LS')
    
    if env_tile and env_ls:
        try:
            return int(env_tile), int(env_ls)
        except ValueError:
            pass
    
    # Check cache
    cache = load_autotune_cache()
    
    if device_key in cache:
        entry = cache[device_key]
        
        # Validate kernel hash
        if entry.get('kernel_hash') == kernel_hash:
            return entry['tile_kw'], entry['local_size']
        else:
            print(f"Warning: Kernel hash mismatch for {device_key}, using defaults")
    
    return default_tile_kw, default_local_size


def should_autotune() -> bool:
    """Check if auto-tuning should run.
    
    Returns:
        True if ONEBIT_AUTOTUNE=1, False otherwise
    """
    return os.environ.get('ONEBIT_AUTOTUNE', '0') == '1'

