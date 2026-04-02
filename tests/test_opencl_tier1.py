import numpy as np
import pytest

from onebit.core import pack_signs_rowmajor, pack_input_signs

try:
    from onebit.backends.opencl import OpenCLBinGemm
    import pyopencl as cl  # type: ignore

    _PLATS = cl.get_platforms()
    _HAVE_OCL = bool(_PLATS)
except Exception:  # pyopencl missing or no platform
    OpenCLBinGemm = None  # type: ignore
    _HAVE_OCL = False


pytestmark = pytest.mark.skipif(not _HAVE_OCL, reason="OpenCL/pyopencl not available")


def _python_ref_y(W_bits: np.ndarray, X_bits: np.ndarray, cv: np.ndarray) -> np.ndarray:
    M, Kw = W_bits.shape
    T = X_bits.shape[0]
    out = np.zeros((M,), dtype=np.float32)
    for i in range(M):
        acc_lo = 0.0
        acc_hi = 0.0
        for t in range(T):
            pc = 0
            for w in range(Kw):
                xnor = ~(int(W_bits[i, w]) ^ int(X_bits[t, w])) & 0xFFFFFFFF
                pc += int(xnor).bit_count()
            Kbits = Kw * 32
            dot = float((pc << 1) - Kbits)
            dot = dot * float(cv[0]) + float(cv[1])
            if (t & 1) == 0:
                acc_lo += dot
            else:
                acc_hi += dot
        out[i] = (acc_lo + acc_hi) / float(T if T > 0 else 1)
    return out


def test_t1_parity_T1():
    rng = np.random.default_rng(0)
    M, K, T = 16, 128, 1  # K multiple of 32
    W = rng.standard_normal((M, K), dtype=np.float32)
    x = rng.standard_normal((K,), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    X_bits = np.expand_dims(pack_input_signs(x), axis=0)
    cv = np.array([1.0, 0.0], dtype=np.float32)

    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not usable: {e}")

    out = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.0, return_teff="none")
    Y = out["Y"]
    Y_ref = _python_ref_y(W_bits, X_bits, cv)
    assert np.allclose(Y, Y_ref, atol=1e-5, rtol=0)


def test_no_nans_up_to_T64():
    rng = np.random.default_rng(1)
    M, K, T = 8, 128, 64
    W = rng.standard_normal((M, K), dtype=np.float32)
    X = rng.standard_normal((T, K), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    Kw = W_bits.shape[1]
    X_bits = np.zeros((T, Kw), dtype=np.uint32)
    for t in range(T):
        X_bits[t] = pack_input_signs(X[t])
    cv = np.array([1.0, 0.0], dtype=np.float32)

    try:
        gemm = OpenCLBinGemm()
    except Exception as e:
        pytest.skip(f"OpenCL not usable: {e}")

    out = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.0, return_teff="none")
    Y = out["Y"]
    assert not np.isnan(Y).any()

