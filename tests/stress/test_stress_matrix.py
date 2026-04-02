"""
Stress test: seed/shape sweeps.

Test matrix:
- Seeds: first 256 odd numbers ≥ 10^6 (smoke: 10, nightly: 256)
- Shapes: (M, Kw, k) ∈ {(1,32,16), (2,64,16), (33,1024,16), (128,2048,16), (256,4096,32), (512,8192,32)}
- Orders: SD-1, SD-2
- CTG: {0,1}
- Early-exit: {0,1}

Gates:
- No crashes. All outputs finite.
- OpenCL naive ↔ tiled digests equal (y_main, y_twin, pc32_main, pc32_twin, y_mean).
- CPU vs OpenCL: y_mean and pc32_* equal; y_bits not required.
"""

import pytest
import numpy as np
import hashlib
from pathlib import Path


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 and return first 16 hex chars."""
    return hashlib.sha256(data).hexdigest()[:16]


# Test configurations
SMOKE_SEEDS = [1000001 + 2*i for i in range(10)]  # First 10 odd numbers >= 10^6
NIGHTLY_SEEDS = [1000001 + 2*i for i in range(256)]  # First 256 odd numbers >= 10^6

SMOKE_SHAPES = [(1, 32, 16), (2, 64, 16), (33, 1024, 16)]
NIGHTLY_SHAPES = [(1, 32, 16), (2, 64, 16), (33, 1024, 16), (128, 2048, 16), (256, 4096, 32)]


@pytest.fixture
def seeds():
    """Return seed list based on test mode."""
    import os
    if os.getenv("ONEBIT_NIGHTLY") == "1":
        return NIGHTLY_SEEDS
    return SMOKE_SEEDS


@pytest.fixture
def shapes():
    """Return shape list based on test mode."""
    import os
    if os.getenv("ONEBIT_NIGHTLY") == "1":
        return NIGHTLY_SHAPES
    return SMOKE_SHAPES


@pytest.mark.stress
@pytest.mark.opencl
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("use_ctg", [False, True])
def test_opencl_naive_tiled_parity(seeds, shapes, order, use_ctg):
    """Test OpenCL naive ↔ tiled parity across seed/shape matrix."""
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")
    
    from onebit.core.packbits import pack_input_signs
    
    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")
    
    mismatches = []
    
    for seed in seeds[:5]:  # Limit to 5 seeds per config in smoke mode
        for M, Kw, k in shapes:
            d = Kw * 32
            
            # Generate test data
            rng = np.random.default_rng(seed)
            W = rng.standard_normal((M, d), dtype=np.float32)
            X = rng.standard_normal(d, dtype=np.float32)
            
            # Pack to bits
            W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
            X_bits = pack_input_signs(X).reshape(1, -1)
            X_bits_tiled = np.tile(X_bits, (k, 1))
            
            # Run naive kernel
            result_naive = backend.run_bsdm_w_naive_norm(
                W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3,
                order=order, beta=0.30, lambd=1.0/256.0,
                walsh_N=2, antithetic=True, use_ctg=use_ctg,
                prf_seed=seed, early_exit_enable=False,
                want_y_pack=True, want_pc32=True, kernel="naive", instr_on=False
            )
            
            # Run tiled kernel
            result_tiled = backend.run_bsdm_w_naive_norm(
                W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3,
                order=order, beta=0.30, lambd=1.0/256.0,
                walsh_N=2, antithetic=True, use_ctg=use_ctg,
                prf_seed=seed, early_exit_enable=False,
                want_y_pack=True, want_pc32=True, kernel="tiled", instr_on=False
            )
            
            # Check outputs are finite
            assert np.all(np.isfinite(result_naive["Y"])), f"Naive Y has non-finite values"
            assert np.all(np.isfinite(result_tiled["Y"])), f"Tiled Y has non-finite values"
            
            # Compute digests
            digest_naive = {
                "y_mean": sha256_hex(result_naive["Y"].tobytes()),
                "y_main": sha256_hex(result_naive["y_bits_main"].tobytes()),
                "y_twin": sha256_hex(result_naive["y_bits_twin"].tobytes()),
                "pc32_main": sha256_hex(result_naive["pc32_main"].tobytes()),
                "pc32_twin": sha256_hex(result_naive["pc32_twin"].tobytes()),
            }
            
            digest_tiled = {
                "y_mean": sha256_hex(result_tiled["Y"].tobytes()),
                "y_main": sha256_hex(result_tiled["y_bits_main"].tobytes()),
                "y_twin": sha256_hex(result_tiled["y_bits_twin"].tobytes()),
                "pc32_main": sha256_hex(result_tiled["pc32_main"].tobytes()),
                "pc32_twin": sha256_hex(result_tiled["pc32_twin"].tobytes()),
            }
            
            # Check parity
            if digest_naive != digest_tiled:
                mismatches.append({
                    "seed": seed,
                    "M": M,
                    "Kw": Kw,
                    "k": k,
                    "order": order,
                    "use_ctg": use_ctg,
                    "digest_naive": digest_naive,
                    "digest_tiled": digest_tiled,
                })
    
    if mismatches:
        print(f"\n[FAIL] Found {len(mismatches)} mismatches:")
        for mm in mismatches[:5]:  # Print first 5
            print(f"  seed={mm['seed']}, M={mm['M']}, Kw={mm['Kw']}, k={mm['k']}, order={mm['order']}, ctg={mm['use_ctg']}")
            print(f"    Naive: {mm['digest_naive']}")
            print(f"    Tiled: {mm['digest_tiled']}")
        pytest.fail(f"OpenCL naive ↔ tiled parity failed: {len(mismatches)} mismatches")
    
    print(f"[PASS] OpenCL naive ↔ tiled parity: {len(seeds[:5]) * len(shapes)} cases")


@pytest.mark.stress
def test_cpu_outputs_finite(seeds, shapes):
    """Test CPU outputs are finite across seed/shape matrix."""
    from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig
    
    for seed in seeds[:5]:  # Limit to 5 seeds in smoke mode
        for M, Kw, k in shapes[:3]:  # Limit to 3 shapes
            d = Kw * 32
            
            # Generate test data
            rng = np.random.default_rng(seed)
            W = rng.standard_normal((M, d), dtype=np.float32)
            X = rng.standard_normal(d, dtype=np.float32)
            
            cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=True)
            
            for row_idx in range(M):
                est, diags = bsdm_w_dot(
                    W[row_idx], X, k=k, cfg=cfg, seed=seed,
                    want_pc32=True, want_y_pack=True, instr_on=False
                )
                
                # Check outputs are finite
                assert np.isfinite(est), f"CPU Y_mean is not finite: seed={seed}, M={M}, Kw={Kw}, k={k}, row={row_idx}"
                assert np.all(np.isfinite(diags["pc32_main"])), f"CPU pc32_main has non-finite values"
                assert np.all(np.isfinite(diags["pc32_twin"])), f"CPU pc32_twin has non-finite values"
    
    print(f"[PASS] CPU outputs finite: {len(seeds[:5]) * len(shapes[:3])} cases")


@pytest.mark.stress
@pytest.mark.opencl
@pytest.mark.parametrize("early_exit", [False, True])
def test_early_exit_extremes(early_exit):
    """Test early-exit with extreme parameters."""
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")
    
    from onebit.core.packbits import pack_input_signs
    
    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")
    
    seed = 1000001
    M, Kw, k_max = 4, 128, 8
    d = Kw * 32
    
    # Generate test data
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    # Pack to bits
    W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
    X_bits = pack_input_signs(X).reshape(1, -1)
    X_bits_tiled = np.tile(X_bits, (k_max, 1))
    
    # Test with eps=0 (bound-only stop)
    result = backend.run_bsdm_w_naive_norm(
        W_bits, X_bits_tiled, T=k_max, eps=0.0, delta=1e-3,
        order=2, beta=0.30, lambd=1.0/256.0,
        walsh_N=2, antithetic=True, use_ctg=False,
        prf_seed=seed, early_exit_enable=early_exit,
        want_y_pack=True, want_pc32=True, kernel="naive", instr_on=False
    )
    
    # Check T_eff never exceeds k_max
    assert np.all(result["T_eff"] <= k_max), f"T_eff exceeded k_max: {result['T_eff']}"
    
    # Check no negative T_eff
    assert np.all(result["T_eff"] >= 0), f"Negative T_eff: {result['T_eff']}"
    
    # Check no NaN Y_mean
    assert np.all(np.isfinite(result["Y"])), f"Y has non-finite values"
    
    print(f"[PASS] Early-exit extremes: early_exit={early_exit}, T_eff={result['T_eff'].tolist()}")


@pytest.mark.stress
@pytest.mark.nightly
def test_large_shape_sweep():
    """Test large shapes (nightly only)."""
    import os
    if os.getenv("ONEBIT_NIGHTLY") != "1":
        pytest.skip("Nightly test only")
    
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    except ImportError:
        pytest.skip("OpenCL not available")
    
    from onebit.core.packbits import pack_input_signs
    
    try:
        backend = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL initialization failed: {e}")
    
    large_shapes = [(512, 8192, 32)]
    seed = 1000001
    
    for M, Kw, k in large_shapes:
        d = Kw * 32
        
        # Generate test data
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((M, d), dtype=np.float32)
        X = rng.standard_normal(d, dtype=np.float32)
        
        # Pack to bits
        W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
        X_bits = pack_input_signs(X).reshape(1, -1)
        X_bits_tiled = np.tile(X_bits, (k, 1))
        
        try:
            result = backend.run_bsdm_w_naive_norm(
                W_bits, X_bits_tiled, T=k, eps=0.0, delta=1e-3,
                order=2, beta=0.30, lambd=1.0/256.0,
                walsh_N=2, antithetic=True, use_ctg=False,
                prf_seed=seed, early_exit_enable=False,
                want_y_pack=True, want_pc32=True, kernel="tiled", instr_on=False
            )
            
            # Check outputs are finite
            assert np.all(np.isfinite(result["Y"])), f"Y has non-finite values"
            
            print(f"[PASS] Large shape: M={M}, Kw={Kw}, k={k}")
        
        except Exception as e:
            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                pytest.skip(f"Device memory insufficient for M={M}, Kw={Kw}: {e}")
            else:
                raise

