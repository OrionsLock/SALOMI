import numpy as np

from onebit.core import pack_signs_rowmajor, pack_input_signs


def test_packbits_determinism_matrix():
    rng = np.random.default_rng(2024)
    M, K = 4, 96
    W = rng.standard_normal((M, K)).astype(np.float32)
    bits1 = pack_signs_rowmajor(W)
    bits2 = pack_signs_rowmajor(W.copy())
    assert np.array_equal(bits1, bits2)


def test_packbits_determinism_vector():
    rng = np.random.default_rng(2025)
    K = 33
    x = rng.standard_normal(K).astype(np.float32)
    b1 = pack_input_signs(x)
    b2 = pack_input_signs(x.copy())
    assert np.array_equal(b1, b2)




def test_opencl_teff_determinism():
    try:
        from onebit.backends.opencl import OpenCLBinGemm
        import pyopencl as cl  # type: ignore
        if not cl.get_platforms():
            return
    except Exception:
        return
    rng = np.random.default_rng(7)
    M, K, T = 4, 128, 16
    W = rng.standard_normal((M, K), dtype=np.float32)
    X = rng.standard_normal((T, K), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    Kw = W_bits.shape[1]
    X_bits = np.zeros((T, Kw), dtype=np.uint32)
    for t in range(T):
        X_bits[t] = pack_input_signs(X[t])
    cv = np.array([1.0, 0.0], dtype=np.float32)

    gemm = OpenCLBinGemm()
    out1 = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.5, return_teff="per_row")
    out2 = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.5, return_teff="per_row")
    assert np.array_equal(out1["T_eff"], out2["T_eff"])
    assert np.allclose(out1["Y"], out2["Y"], atol=0, rtol=0)
