import numpy as np
import pytest

from onebit.core import pack_signs_rowmajor, assert_bpp_one


def test_bpp_guard_exact_1bpp():
    rng = np.random.default_rng(0)
    M, K = 3, 64  # K multiple of 32 so storage is exactly 1.00 bpp
    W = rng.standard_normal((M, K)).astype(np.float32)
    W_bits = pack_signs_rowmajor(W)
    assert_bpp_one(W_bits, M * K)  # should not raise


def test_bpp_guard_detects_mismatch():
    rng = np.random.default_rng(1)
    M, K = 2, 32
    W = rng.standard_normal((M, K)).astype(np.float32)
    W_bits = pack_signs_rowmajor(W)
    with pytest.raises(AssertionError):
        assert_bpp_one(W_bits, M * K + 1)

