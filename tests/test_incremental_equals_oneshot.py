import numpy as np
import pytest

from onebit.core import pack_signs_rowmajor, pack_input_signs

try:
    from onebit.backends.opencl import OpenCLBinGemm, OpenCLBinGemmIncremental
    import pyopencl as cl  # type: ignore
    _HAVE_OCL = bool(cl.get_platforms())
except Exception:
    OpenCLBinGemm = None  # type: ignore
    OpenCLBinGemmIncremental = None  # type: ignore
    _HAVE_OCL = False

pytestmark = pytest.mark.skipif(not _HAVE_OCL, reason="OpenCL/pyopencl not available")


def test_incremental_equals_one_shot():
    rng = np.random.default_rng(2)
    M, K, T = 16, 256, 12
    W = rng.standard_normal((M, K), dtype=np.float32)
    X = rng.standard_normal((T, K), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    Kw = W_bits.shape[1]
    X_bits = np.zeros((T, Kw), dtype=np.uint32)
    for t in range(T):
        X_bits[t] = pack_input_signs(X[t])
    cv = np.array([1.0, 0.0], dtype=np.float32)

    gemm = OpenCLBinGemm()
    one_shot = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.0, return_teff="per_row")

    inc = OpenCLBinGemmIncremental(gemm)
    inc.start(W_bits, cv)
    t1 = T // 3
    t2 = T - t1
    inc.extend(X_bits[:t1], T_chunk=t1, eps_margin=0.0)
    inc.extend(X_bits[t1:], T_chunk=t2, eps_margin=0.0)
    out = inc.finalize()

    assert np.allclose(out["Y"], one_shot["Y"], atol=1e-6, rtol=0)
    assert np.array_equal(out["T_eff"], one_shot["T_eff"]) 

