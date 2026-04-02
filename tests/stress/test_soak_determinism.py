"""
Determinism soak test.

Repeat identical case K times with same seed.
K={100 (smoke), 1000 (nightly)}.

Backends: opencl: {naive,tiled}; cpu reference where applicable.

Assert all digests equal across repeats. Flakes must be 0.
"""

import pytest
import numpy as np
import hashlib


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 and return first 16 hex chars."""
    return hashlib.sha256(data).hexdigest()[:16]


@pytest.mark.stress
@pytest.mark.opencl
@pytest.mark.parametrize("kernel", ["naive", "tiled"])
def test_opencl_determinism_soak(kernel):
    """Test OpenCL determinism by repeating identical case K times."""
    import os
    K = 1000 if os.getenv("ONEBIT_NIGHTLY") == "1" else 100
    
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")
    
    from onebit.core.packbits import pack_input_signs
    
    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")
    
    # Fixed test case
    seed = 1000001
    M, Kw, k = 33, 1024, 16
    d = Kw * 32
    
    # Generate test data once
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    # Pack to bits
    W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
    X_bits = pack_input_signs(X).reshape(1, -1)
    X_bits_tiled = np.tile(X_bits, (k, 1))
    
    # Run K times and collect digests
    digests = []
    
    for i in range(K):
        result = backend.run_bsdm_w_naive_norm(
            W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3,
            order=2, beta=0.30, lambd=1.0/256.0,
            walsh_N=2, antithetic=True, use_ctg=False,
            prf_seed=seed, early_exit_enable=False,
            want_y_pack=True, want_pc32=True, kernel=kernel, instr_on=False
        )
        
        digest = {
            "y_mean": sha256_hex(result["Y"].tobytes()),
            "y_main": sha256_hex(result["y_bits_main"].tobytes()),
            "y_twin": sha256_hex(result["y_bits_twin"].tobytes()),
            "pc32_main": sha256_hex(result["pc32_main"].tobytes()),
            "pc32_twin": sha256_hex(result["pc32_twin"].tobytes()),
        }
        
        digests.append(digest)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{K}")
    
    # Check all digests are identical
    first_digest = digests[0]
    flakes = 0
    
    for i, digest in enumerate(digests[1:], start=1):
        if digest != first_digest:
            flakes += 1
            print(f"  FLAKE at iteration {i}:")
            print(f"    Expected: {first_digest}")
            print(f"    Got: {digest}")
    
    if flakes > 0:
        pytest.fail(f"Determinism soak failed: {flakes}/{K} flakes (kernel={kernel})")
    
    print(f"[PASS] Determinism soak: {K} runs, 0 flakes (kernel={kernel})")


@pytest.mark.stress
def test_cpu_determinism_soak():
    """Test CPU determinism by repeating identical case K times."""
    import os
    K = 1000 if os.getenv("ONEBIT_NIGHTLY") == "1" else 100
    
    from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig
    
    # Fixed test case
    seed = 1000001
    M, Kw, k = 4, 128, 16
    d = Kw * 32
    
    # Generate test data once
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
    
    # Run K times and collect digests
    digests = []
    
    for i in range(K):
        results = []
        for row_idx in range(M):
            est, diags = bsdm_w_dot(
                W[row_idx], X, k=k, cfg=cfg, seed=seed,
                want_pc32=True, want_y_pack=True, instr_on=False
            )
            results.append({
                "y_mean": est,
                "y_main": diags["y_bits_main"],
                "y_twin": diags["y_bits_twin"],
                "pc32_main": diags["pc32_main"],
                "pc32_twin": diags["pc32_twin"],
            })
        
        # Compute digests
        y_mean_arr = np.array([r["y_mean"] for r in results], dtype=np.float32)
        y_main_all = np.concatenate([r["y_main"] for r in results])
        y_twin_all = np.concatenate([r["y_twin"] for r in results])
        pc32_main_all = np.concatenate([r["pc32_main"] for r in results])
        pc32_twin_all = np.concatenate([r["pc32_twin"] for r in results])
        
        digest = {
            "y_mean": sha256_hex(y_mean_arr.tobytes()),
            "y_main": sha256_hex(y_main_all.tobytes()),
            "y_twin": sha256_hex(y_twin_all.tobytes()),
            "pc32_main": sha256_hex(pc32_main_all.tobytes()),
            "pc32_twin": sha256_hex(pc32_twin_all.tobytes()),
        }
        
        digests.append(digest)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{K}")
    
    # Check all digests are identical
    first_digest = digests[0]
    flakes = 0
    
    for i, digest in enumerate(digests[1:], start=1):
        if digest != first_digest:
            flakes += 1
            print(f"  FLAKE at iteration {i}:")
            print(f"    Expected: {first_digest}")
            print(f"    Got: {digest}")
    
    if flakes > 0:
        pytest.fail(f"CPU determinism soak failed: {flakes}/{K} flakes")
    
    print(f"[PASS] CPU determinism soak: {K} runs, 0 flakes")


@pytest.mark.stress
@pytest.mark.opencl
@pytest.mark.nightly
def test_opencl_determinism_with_ctg():
    """Test OpenCL determinism with CTG enabled (nightly only)."""
    import os
    if os.getenv("ONEBIT_NIGHTLY") != "1":
        pytest.skip("Nightly test only")
    
    K = 1000
    
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")
    
    from onebit.core.packbits import pack_input_signs
    
    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")
    
    # Fixed test case
    seed = 1000001
    M, Kw, k = 33, 1024, 16
    d = Kw * 32
    
    # Generate test data once
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    # Pack to bits
    W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
    X_bits = pack_input_signs(X).reshape(1, -1)
    X_bits_tiled = np.tile(X_bits, (k, 1))
    
    # Run K times with CTG enabled
    digests = []
    
    for i in range(K):
        result = backend.run_bsdm_w_naive_norm(
            W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3,
            order=2, beta=0.30, lambd=1.0/256.0,
            walsh_N=2, antithetic=True, use_ctg=True,
            prf_seed=seed, early_exit_enable=False,
            want_y_pack=True, want_pc32=True, kernel="tiled", instr_on=False
        )
        
        digest = {
            "y_mean": sha256_hex(result["Y"].tobytes()),
            "ctg_digest": sha256_hex(result["ctg_digest"].tobytes()),
        }
        
        digests.append(digest)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{K}")
    
    # Check all digests are identical
    first_digest = digests[0]
    flakes = 0
    
    for i, digest in enumerate(digests[1:], start=1):
        if digest != first_digest:
            flakes += 1
            print(f"  FLAKE at iteration {i}:")
            print(f"    Expected: {first_digest}")
            print(f"    Got: {digest}")
    
    if flakes > 0:
        pytest.fail(f"CTG determinism soak failed: {flakes}/{K} flakes")
    
    print(f"[PASS] CTG determinism soak: {K} runs, 0 flakes")

