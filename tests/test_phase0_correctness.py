"""Phase-0 correctness tests (fast, synthetic tensors, no model / dataset downloads)."""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from onebit.core.bpp_guard import BPPCalculator
from onebit.eval.harness import BitReport
from onebit.quantization import baselines as B
from onebit.quantization.gptq import gptq_layer
from onebit.quantization.hessian_vq import HessianVQ
from onebit.quantization.lowrank_residual import LowRankResidual, ResidualHessianVQ


def _mse(a, b):
    return float(((a - b) ** 2).mean())


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_rtn_bits_accounting():
    w = torch.randn(64, 256)
    _, bits = B.rtn(w, 4, group=128)
    assert bits == 64 * 256 * 4 + (64 * 2) * 2 * 16  # 2 groups/row, fp16 scale + zero
    _, bits_sym = B.rtn(w, 4, group=128, sym=True)
    assert bits_sym == 64 * 256 * 4 + (64 * 2) * 16
    _, bits_pc = B.rtn(w, 3, group=None)
    assert bits_pc == 64 * 256 * 3 + 64 * 2 * 16


def test_rtn_error_decreases_with_bits():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    errs = [_mse(B.rtn(w, b, group=128)[0], w) for b in (2, 3, 4, 8)]
    assert all(e1 > e2 for e1, e2 in zip(errs, errs[1:])), errs


def test_rtn_g128_beats_per_tensor_absmax_with_outliers():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    w[torch.randint(0, 64, (8,)), torch.randint(0, 256, (8,))] = 1.0  # a few large outliers
    # per-tensor symmetric absmax INT4 (what the old comprehensive_real_test did)
    s = w.abs().max() / 7
    w_pt = torch.clamp(torch.round(w / s), -7, 7) * s
    w_g = B.rtn(w, 4, group=128)[0]
    assert _mse(w_g, w) < 0.1 * _mse(w_pt, w)


def test_ternary_binary_bits():
    w = torch.randn(32, 256)
    wq, bits = B.ternary_absmean(w, group=None)
    assert bits == pytest.approx(32 * 256 * 1.6 + 32 * 16)
    assert torch.unique(torch.round(wq / wq.abs().amax(1, keepdim=True))).numel() <= 3
    wq, bits = B.binary_absmean(w, group=None)
    assert bits == 32 * 256 + 32 * 16
    assert torch.all(wq.abs() > 0)
    _, bits = B.binary_absmean(w, group=128)
    assert bits == 32 * 256 + 32 * 2 * 16


def test_ternary_twn_bits_and_mse():
    w = torch.randn(64, 512)
    wq, bits = B.ternary_twn(w, group=None)
    assert bits == pytest.approx(64 * 512 * 1.6 + 64 * 16)
    # at most 3 distinct values per row
    assert all(torch.unique(row).numel() <= 3 for row in wq)
    w_abs, _ = B.ternary_absmean(w, group=None)
    assert _mse(wq, w) < _mse(w_abs, w)


# ---------------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------------

def test_gptq_beats_rtn_on_correlated_inputs():
    g = torch.Generator().manual_seed(0)
    out_f, in_f, n = 64, 256, 2048
    W = torch.randn(out_f, in_f, generator=g)
    mix = torch.randn(in_f, in_f, generator=g) / math.sqrt(in_f)
    X = torch.randn(n, in_f, generator=g) @ (torch.eye(in_f) + 2.0 * mix)
    H = 2.0 / n * X.t() @ X
    Q_gptq = gptq_layer(W, H, bits=3, group_size=128)
    Q_rtn = B.rtn(W, 3, group=128)[0]
    err_gptq = torch.linalg.norm((W - Q_gptq) @ X.t())
    err_rtn = torch.linalg.norm((W - Q_rtn) @ X.t())
    assert err_gptq < 0.9 * err_rtn, (float(err_gptq), float(err_rtn))


# ---------------------------------------------------------------------------
# HessianVQ
# ---------------------------------------------------------------------------

def test_hvq_weighted_distance_matches_bruteforce_and_chunking():
    rng = np.random.default_rng(0)
    X = rng.random((50, 16))
    Wt = rng.random((50, 16)) * 5
    C = rng.random((8, 16))
    brute = np.array([[np.sum(Wt[b] * (X[b] - C[c]) ** 2) for c in range(8)] for b in range(50)])
    old = HessianVQ._CHUNK
    try:
        HessianVQ._CHUNK = 7  # force several chunks
        d = HessianVQ._weighted_dists(X, Wt, C)
        a = HessianVQ._assign(X, Wt, C)
    finally:
        HessianVQ._CHUNK = old
    np.testing.assert_allclose(d, brute, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(a, brute.argmin(1))


def test_hvq_weighted_assignment_differs_from_unweighted():
    # One block x = (1, 0); c0 = (1, 2), c1 = (0, 0).
    x = np.array([[1.0, 0.0]])
    C = np.array([[1.0, 2.0], [0.0, 0.0]])
    w_nonuniform = np.array([[10.0, 0.1]])
    assert HessianVQ._assign(x, w_nonuniform, C)[0] == 0  # 0.4 vs 10
    assert HessianVQ._assign(x, np.ones_like(x), C)[0] == 1  # 4 vs 1
    # The old implementation's per-block mean weight is equivalent to unweighted.
    w_mean = np.full_like(x, w_nonuniform.mean())
    assert HessianVQ._assign(x, w_mean, C)[0] == 1

    # End-to-end on a matrix with strongly non-uniform H within blocks.
    rng = np.random.default_rng(1)
    W = rng.standard_normal((64, 128))
    h = np.exp(3.0 * rng.standard_normal(128)).astype(np.float32)
    q_w = HessianVQ(n_codes=16, block_size=4, max_iter=10, use_hessian_weight=True, seed=0)
    q_u = HessianVQ(n_codes=16, block_size=4, max_iter=10, use_hessian_weight=False, seed=0)
    Wq_w = q_w.quantize(W, h)
    Wq_u = q_u.quantize(W, h)
    assert not np.array_equal(q_w.indices, q_u.indices)
    # weighted VQ should not do worse on the H-weighted objective it optimises
    err_w = np.sum(h[None, :] * (W - Wq_w) ** 2)
    err_u = np.sum(h[None, :] * (W - Wq_u) ** 2)
    assert err_w < err_u


def test_hvq_deterministic_for_seed():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((32, 64))
    h = rng.random(64).astype(np.float32) + 0.1
    a = HessianVQ(n_codes=16, max_iter=5, seed=3).quantize(W, h)
    np.random.seed(123)  # global RNG must not matter
    b = HessianVQ(n_codes=16, max_iter=5, seed=3).quantize(W, h)
    np.testing.assert_array_equal(a, b)
    c = HessianVQ(n_codes=16, max_iter=5, seed=4).quantize(W, h)
    assert not np.array_equal(a, c)


def test_hvq_no_global_mean_shift_and_signs_preserved():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((16, 32)) + 0.5
    Wq = HessianVQ(n_codes=8, max_iter=5).quantize(W, np.ones(32))
    assert np.all(np.sign(Wq)[Wq != 0] == np.sign(W)[Wq != 0])


def test_hvq_gptq_refine_raises():
    with pytest.raises(NotImplementedError, match="gptq.py"):
        HessianVQ(gptq_refine=True)


def test_hvq_bpp_entropy_and_fixed():
    rng = np.random.default_rng(4)
    W = rng.standard_normal((64, 64))
    q = HessianVQ(n_codes=64, block_size=4, max_iter=5)
    q.quantize(W, np.ones(64))
    n = W.size
    fixed = q.effective_bpp_fixed(n)
    assert fixed == pytest.approx((n + (n // 16) * 6 + 64 * 16 * 32) / n)
    ent = q.effective_bpp(n)
    assert 1.0 < ent <= fixed + 1e-12


# ---------------------------------------------------------------------------
# Low-rank / residual VQ
# ---------------------------------------------------------------------------

def test_residual_hvq_bpp_counts_both_sign_planes():
    rng = np.random.default_rng(5)
    W = rng.standard_normal((256, 256))
    rvq = ResidualHessianVQ(n_codes_coarse=64, n_codes_fine=32, block_size=4, max_iter=3)
    rvq.quantize(W, np.ones(256, dtype=np.float32))
    bpp = rvq.effective_bpp()
    assert bpp > 2.0
    assert bpp == pytest.approx(rvq.vq1.effective_bpp(W.size) + rvq.vq2.effective_bpp(W.size))


def test_lowrank_bpp_counts_factor_and_row_scales():
    rng = np.random.default_rng(6)
    W = rng.standard_normal((96, 64))
    n = W.size
    lr = LowRankResidual(rank=4, factor_bits=8)
    lr.quantize(W, None)
    assert lr.effective_bpp() == pytest.approx((n + 32 + 4 * (96 + 64) * 8 + 2 * 4 * 16) / n)
    lr2 = LowRankResidual(rank=4, factor_bits=8, per_row_scale=True)
    lr2.quantize(W, None)
    assert lr2.effective_bpp() == pytest.approx((n + 96 * 16 + 4 * (96 + 64) * 8 + 2 * 4 * 16) / n)
    lr32 = LowRankResidual(rank=4, factor_bits=32)
    lr32.quantize(W, None)
    assert lr32.effective_bpp() == pytest.approx((n + 32 + 4 * (96 + 64) * 32) / n)


def test_lowrank_per_rank_scales_help_unbalanced_factors():
    rng = np.random.default_rng(7)
    U = rng.standard_normal((128, 4)) * np.array([100.0, 1.0, 0.01, 0.01])
    lr = LowRankResidual(rank=4, factor_bits=4)
    per_col = lr._quantize_factors(U, axis=0)
    max_val = np.abs(U).max()
    glob = np.clip(np.round(U / max_val * 7), -7, 7) / 7 * max_val
    # small-magnitude components survive with per-rank scales, vanish with a global one
    assert np.abs(per_col[:, 2]).sum() > 0
    assert np.abs(glob[:, 2]).sum() == 0


# ---------------------------------------------------------------------------
# BPPCalculator
# ---------------------------------------------------------------------------

def test_bpp_calculator_counts_params_once_per_tensor():
    n = 1024
    calc = BPPCalculator()
    calc.add_signs(np.ones(n), n, tensor="w")
    calc.add_indices(np.zeros(n // 16, dtype=np.int64), n, n_codes=64, tensor="w")
    assert calc.total_params == n
    assert calc.total_bits == n + (n // 16) * 6
    assert calc.calculate_bpp() == pytest.approx(1.0 + 6 / 16)
    with pytest.raises(ValueError):
        calc.add_signs(np.ones(10), 10, tensor="w")


def test_bpp_calculator_index_bits_from_k():
    calc = BPPCalculator()
    calc.add_indices(np.zeros(100, dtype=np.int64), 400, n_codes=32)
    assert calc.total_bits == 100 * 5
    calc.reset()
    calc.add_indices(np.zeros(100, dtype=np.uint8), 400)  # unknown K -> storage width
    assert calc.total_bits == 100 * 8


def test_bpp_calculator_model_counts_once_and_includes_unquantized():
    m = nn.Module()
    m.q = nn.Parameter(torch.zeros(64, 64))
    m.fp = nn.Parameter(torch.zeros(100))  # unquantized fp32
    n = 64 * 64
    m.q.signs = np.ones(n, dtype=np.uint8)
    m.q.indices = np.zeros(n // 16, dtype=np.int64)
    m.q.codebook = np.zeros((16, 16), dtype=np.float16)
    out = BPPCalculator().calculate_for_quantized_model(m)
    assert out["total_params"] == n + 100
    expected_bits = n * 1 + (n // 16) * 4 + 16 * 16 * 16 + 100 * 32
    assert out["total_bits"] == expected_bits
    out16 = BPPCalculator().calculate_for_quantized_model(m, unquantized_bits=16)
    assert out16["total_bits"] == expected_bits - 100 * 16


# ---------------------------------------------------------------------------
# Harness BitReport
# ---------------------------------------------------------------------------

def test_bitreport_whole_model_math():
    r = BitReport(total_params=1000)
    r.add("a", 400, 400 * 2.0)
    r.add("b", 200, 200 * 2.0)
    assert r.linear_params == 600
    assert r.bpp_linear == pytest.approx(2.0)
    assert r.other_params == 400
    assert r.bpp_whole_model == pytest.approx((1200 + 16 * 400) / 1000)
    s = r.summary()
    assert s["other_params_fp16"] == 400 and s["total_params"] == 1000
