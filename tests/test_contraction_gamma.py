import numpy as np
import pytest

from onebit.core import (
    pack_signs_rowmajor,
    hutch_pp_norm_estimator,
    choose_gamma,
    apply_block_rescale,
)


def _prf_rademacher(master_seed: int):
    rng = np.random.default_rng(master_seed)
    def sample_fn(K: int, i: int) -> np.ndarray:
        # Deterministic per-probe seed
        r = np.random.default_rng((master_seed + i) & 0xFFFFFFFF)
        s = r.integers(0, 2, size=(K,), dtype=np.int8)
        return np.where(s > 0, 1.0, -1.0).astype(np.float32)
    return sample_fn


def test_estimator_monotone_in_probes():
    rng = np.random.default_rng(0)
    M, K = 64, 256
    W = rng.standard_normal((M, K), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    sample_fn = _prf_rademacher(123)
    v8 = hutch_pp_norm_estimator(W_bits, sample_fn, probes=8)
    v16 = hutch_pp_norm_estimator(W_bits, sample_fn, probes=16)
    assert v16 >= v8


def test_choose_gamma_shrinks_with_L():
    g1 = choose_gamma(kappa_hat=10.0, L=1)
    g8 = choose_gamma(kappa_hat=10.0, L=8)
    assert g8 < g1


def test_runtime_mock_no_blowup():
    rng = np.random.default_rng(1)
    M, K = 128, 256
    W = rng.standard_normal((M, K), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    sample_fn = _prf_rademacher(456)

    # Estimate and choose gamma for L layers
    kappa = hutch_pp_norm_estimator(W_bits, sample_fn, probes=8)
    L = 8
    gamma = choose_gamma(kappa, L)

    # Run a simple mock over many steps
    steps = 10000  # reduced from 1M to keep tests fast
    y = np.zeros((M,), dtype=np.float32)
    for _ in range(steps):
        x = rng.standard_normal((K,), dtype=np.float32)
        y = y + apply_block_rescale(x[:M], gamma)  # mock block map + rescale
        if not np.isfinite(y).all():
            pytest.fail("Non-finite detected during mock run")
    assert np.isfinite(y).all()

