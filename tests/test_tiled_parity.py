import numpy as np
import pytest

from onebit.core import pack_signs_rowmajor, pack_input_signs

try:
    from onebit.backends.opencl import OpenCLBinGemm
    import pyopencl as cl  # type: ignore
    _HAVE_OCL = bool(cl.get_platforms())
except Exception:
    OpenCLBinGemm = None  # type: ignore
    _HAVE_OCL = False

pytestmark = pytest.mark.skipif(not _HAVE_OCL, reason="OpenCL/pyopencl not available")


def _make_bits(M, K, T, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, K), dtype=np.float32)
    X = rng.standard_normal((T, K), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    Kw = W_bits.shape[1]
    X_bits = np.zeros((T, Kw), dtype=np.uint32)
    for t in range(T):
        X_bits[t] = pack_input_signs(X[t])
    return W_bits, X_bits


def test_tiled_equals_naive_eps0_small():
    M, K, T = 32, 256, 8
    W_bits, X_bits = _make_bits(M, K, T)
    cv = np.array([1.0, 0.0], dtype=np.float32)

    gemm = OpenCLBinGemm()
    out_naive = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.0, return_teff="per_row", kernel="naive")
    out_tiled = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.0, return_teff="per_row", kernel="tiled")

    assert np.array_equal(out_naive["T_eff"], out_tiled["T_eff"])  # both == T
    assert np.allclose(out_naive["Y"], out_tiled["Y"], atol=0, rtol=0)


def test_tiled_equals_naive_eps0_large():
    M, K, T = 256, 2048, 8
    W_bits, X_bits = _make_bits(M, K, T)
    cv = np.array([1.0, 0.0], dtype=np.float32)

    gemm = OpenCLBinGemm()
    out_naive = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.0, return_teff="per_row", kernel="naive")
    out_tiled = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.0, return_teff="per_row", kernel="tiled")

    assert np.array_equal(out_naive["T_eff"], out_tiled["T_eff"])  # both == T
    assert np.allclose(out_naive["Y"], out_tiled["Y"], atol=0, rtol=0)

