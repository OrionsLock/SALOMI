import numpy as np

from onebit.core import pack_signs_rowmajor, pack_input_signs


def _unpack_rowmajor(W_bits: np.ndarray, K: int) -> np.ndarray:
    M, Kw = W_bits.shape
    out = np.zeros((M, K), dtype=bool)
    for i in range(M):
        for w in range(Kw):
            word = int(W_bits[i, w])
            start = w * 32
            end = min(start + 32, K)
            for b in range(end - start):
                out[i, start + b] = bool((word >> b) & 1)
    return out


def _unpack_vec(x_bits: np.ndarray, K: int) -> np.ndarray:
    Kw = x_bits.shape[0]
    out = np.zeros((K,), dtype=bool)
    for w in range(Kw):
        word = int(x_bits[w])
        start = w * 32
        end = min(start + 32, K)
        for b in range(end - start):
            out[start + b] = bool((word >> b) & 1)
    return out


def test_pack_unpack_roundtrip_rowmajor():
    rng = np.random.default_rng(42)
    M, K = 5, 70
    W = rng.standard_normal((M, K)).astype(np.float32)
    bits = pack_signs_rowmajor(W)
    recovered = _unpack_rowmajor(bits, K)
    assert np.array_equal(recovered, W >= 0)


def test_pack_unpack_input_vector():
    rng = np.random.default_rng(123)
    K = 77
    x = rng.standard_normal(K).astype(np.float32)
    bits = pack_input_signs(x)
    recovered = _unpack_vec(bits, K)
    assert np.array_equal(recovered, x >= 0)

