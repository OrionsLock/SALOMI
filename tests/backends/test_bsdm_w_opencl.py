import numpy as np
import pytest

from onebit.backends.opencl.host_opencl import OpenCLBinGemm
from onebit.core.packbits import pack_input_signs
from onebit.ops.bsdm_w import SDConfig, bsdm_w_dot


def _rand_pm1(K, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=K, dtype=np.int8)
    return (x * 2 - 1).astype(np.int8)


def _pack_pm1(x):
    return pack_input_signs(x.astype(np.float32))


@pytest.mark.opencl
def test_bsdm_w_opencl_parity_sd1_sd2_small():
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")

    K = 1024
    Kw = (K + 31) // 32
    M = 4
    T = 16

    # Single input vector repeated across ticks
    a_rows = np.stack([_rand_pm1(K, 100 + i) for i in range(M)], axis=0)
    b_vec = _rand_pm1(K, 777)
    W_bits = np.stack([_pack_pm1(a_rows[i]) for i in range(M)], axis=0)
    X0 = _pack_pm1(b_vec)
    X_bits = np.tile(X0[None, :], (T, 1))

    # Disable early exit via negative eps
    eps = -1.0
    delta = 0.5

    # SD-1
    out = gemm.run_bsdm_w_naive_norm(W_bits, X_bits, T=T, eps=eps, delta=delta,
                                      order=1, beta=0.30, lambd=1.0/256.0, use_ctg=False, prf_seed=0,
                                      local_size=128, want_y_pack=False)
    Y_cl = out["Y"].astype(np.float64)

    # CPU baseline: walsh_N=1, antithetic=False, seed=None (no dithering)
    Y_cpu = []
    cfg1 = SDConfig(order=1, walsh_N=1, antithetic=False)
    for i in range(M):
        est, _ = bsdm_w_dot(W_bits[i], X0, T, cfg1, seed=None)
        Y_cpu.append(est)
    Y_cpu = np.asarray(Y_cpu, dtype=np.float64)

    assert np.allclose(Y_cl, Y_cpu, atol=1e-5, rtol=0.0)

    # SD-2
    out2 = gemm.run_bsdm_w_naive_norm(W_bits, X_bits, T=T, eps=eps, delta=delta,
                                       order=2, beta=0.30, lambd=1.0/256.0, use_ctg=False, prf_seed=0,
                                       local_size=128, want_y_pack=False)
    Y_cl2 = out2["Y"].astype(np.float64)

    cfg2 = SDConfig(order=2, beta=0.30, walsh_N=1, antithetic=False)
    Y_cpu2 = []
    for i in range(M):
        est, _ = bsdm_w_dot(W_bits[i], X0, T, cfg2, seed=None)
        Y_cpu2.append(est)
    Y_cpu2 = np.asarray(Y_cpu2, dtype=np.float64)

    assert np.allclose(Y_cl2, Y_cpu2, atol=1e-5, rtol=0.0)


@pytest.mark.opencl
def test_bsdm_w_opencl_early_exit_sanity():
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")

    K = 2048
    M = 64
    T = 32
    Kw = (K + 31) // 32

    # Random rows and random tick vectors to induce near-zero means
    rng = np.random.default_rng(42)
    W_bits = np.stack([_pack_pm1(_rand_pm1(K, int(rng.integers(0, 1<<31)))) for _ in range(M)], axis=0)
    X_ticks = np.stack([_pack_pm1(_rand_pm1(K, int(rng.integers(0, 1<<31)))) for _ in range(T)], axis=0)

    eps = 0.05
    delta = 1e-3

    out = gemm.run_bsdm_w_naive_norm(W_bits, X_ticks, T=T, eps=eps, delta=delta,
                                      order=2, beta=0.30, lambd=1.0/256.0, use_ctg=False, prf_seed=1234,
                                      local_size=128, want_y_pack=False)
    Y = out["Y"].astype(np.float64)
    Teff = out["T_eff"].astype(np.int32)

    # Some early exits expected
    assert np.any(Teff < T)

    # Bound check: |mean| <= eps + thr for each row at its own Teff
    ok = 0
    for i in range(M):
        t_i = int(max(1, Teff[i]))
        mean_i = float(Y[i])
        thr_i = np.sqrt(0.5 * np.log(2.0/delta) / float(t_i))
        if abs(mean_i) <= eps + thr_i + 1e-6:
            ok += 1
    assert ok >= M  # all rows should satisfy their own bound



@pytest.mark.opencl
def test_bsdm_w_opencl_throughput_smoke():
    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not available: {e}")

    K = 2048
    M = 256
    T = 8
    rng = np.random.default_rng(7)
    W_bits = np.stack([_pack_pm1(_rand_pm1(K, int(rng.integers(0, 1<<31)))) for _ in range(M)], axis=0)
    X_ticks = np.stack([_pack_pm1(_rand_pm1(K, int(rng.integers(0, 1<<31)))) for _ in range(T)], axis=0)

    # time OpenCL
    import time
    t0 = time.time()
    _ = gemm.run_bsdm_w_naive_norm(W_bits, X_ticks, T=T, eps=-1.0, delta=0.5,
                                    order=2, beta=0.30, lambd=1.0/256.0, use_ctg=False,
                                    prf_seed=0, local_size=256, want_y_pack=False)
    t1 = time.time()

    # time CPU reference per row (just a few rows to keep CI time reasonable)
    cfg = SDConfig(order=2, walsh_N=1, antithetic=False)
    t0c = time.time()
    for i in range(min(16, M)):
        _ = bsdm_w_dot(W_bits[i], X_ticks[0], T, cfg, seed=None)
    t1c = time.time()

    assert (t1 - t0) * 4 < (t1c - t0c) + 1e-6
