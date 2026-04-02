import numpy as np
import pytest

from onebit.core import pack_signs_rowmajor
from onebit.tsr import tsr_pack_input_bits

try:
    from onebit.backends.opencl import OpenCLBinGemm
    import pyopencl as cl  # type: ignore
    _HAVE_OCL = bool(cl.get_platforms())
except Exception:
    OpenCLBinGemm = None  # type: ignore
    _HAVE_OCL = False


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


def test_tsr_bre_slope():
    rng = np.random.default_rng(42)
    M, K = 32, 256
    W = rng.standard_normal((M, K), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    W_signs = np.where(W >= 0.0, 1.0, -1.0).astype(np.float32)
    x = np.clip(rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32), -1.0, 1.0)
    cv = np.array([1.0, 0.0], dtype=np.float32)

    # True target (unbiased): E[sgn(x_t)] = x, so E[W_signs @ sgn(x_t)] = W_signs @ x
    Y_true = (W_signs @ x).astype(np.float32)

    try:
        gemm = OpenCLBinGemm() if _HAVE_OCL else None
    except Exception:
        gemm = None

    T_list = [1, 2, 4, 8, 16, 32, 64]
    R = 8  # replicate streams to stabilize Monte Carlo estimate
    mses = []
    for T in T_list:
        mse_acc = 0.0
        for r in range(R):
            X_bits = tsr_pack_input_bits(x, T=T, master_seed=0, layer=0, stream=r)
            if gemm is not None:
                out = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=0.0, return_teff="none")
                Y = out["Y"]
            else:
                Y = _python_ref_y(W_bits, X_bits, cv)
            err = Y - Y_true
            mse_acc += float(np.mean(err * err))
        mses.append(mse_acc / float(R))

    # Global slope check on log-log scale from T>=4
    idx4 = T_list.index(4)
    ts = np.array(T_list[idx4:], dtype=np.float64)
    ms = np.array(mses[idx4:], dtype=np.float64)
    b = np.polyfit(np.log(ts), np.log(ms + 1e-12), 1)[0]
    # Expect variance to shrink ~1/T => slope ~ -1; allow weaker bound for speed/noise
    assert b <= -0.6

    # Also ensure large-T error is meaningfully below mid-range
    idx8 = T_list.index(8)
    assert mses[-1] <= 0.8 * mses[idx8]

