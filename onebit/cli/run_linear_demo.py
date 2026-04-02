from __future__ import annotations

import argparse
import sys

import numpy as np

from onebit.core import pack_signs_rowmajor, pack_input_signs
from onebit.backends.opencl import OpenCLBinGemm


def python_ref_y(W_bits: np.ndarray, X_bits: np.ndarray, cv: np.ndarray) -> np.ndarray:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--T", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps_margin", type=float, default=0.0)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--return_teff", type=str, default="per_row", choices=["none","scalar","per_row"])\n    p.add_argument("--kernel", type=str, default="auto", choices=["auto","naive","tiled"])
    p.add_argument("--log_jsonl", type=str, default=None)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    M, K, T = args.M, args.K, args.T

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
        print(f"[WARN] OpenCL not available: {e}", file=sys.stderr)
        print("Falling back to Python reference only.")
        Y_ref = python_ref_y(W_bits, X_bits, cv)
        print(f"Y_ref[:4] = {Y_ref[:4]}")
        return 0

    import time, json
    t0 = time.perf_counter()
    out = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=args.eps_margin, delta=args.delta, return_teff=args.return_teff, kernel=args.kernel)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    Y = out["Y"]
    Y_ref = python_ref_y(W_bits, X_bits, cv)

    max_abs = float(np.max(np.abs(Y - Y_ref)))
    print(f"max|Y - Y_ref| = {max_abs:.6g}")

    if T == 1 and not np.allclose(Y, Y_ref, atol=args.tol, rtol=0):
        print("[ERROR] T=1 parity check failed")
        return 2

    if np.isnan(Y).any():
        print("[ERROR] NaNs detected in Y")
        return 3

    # Optional JSONL logging
    if args.log_jsonl is not None:
        teff = out.get("T_eff")
        if teff is None:
            avg_teff = float(T)
            min_teff = int(T)
            max_teff = int(T)
        elif isinstance(teff, np.ndarray):
            avg_teff = float(np.mean(teff))
            min_teff = int(np.min(teff))
            max_teff = int(np.max(teff))
        else:
            avg_teff = float(teff)
            min_teff = int(teff)
            max_teff = int(teff)
        K_feat = K
        gops = (2.0 * float(M) * float(K_feat) * float(avg_teff)) / (dt_ms * 1e6)
        rec = {
            "M": M, "K": K_feat, "T": T,
            "kernel": args.kernel,
            "avg_T_eff": avg_teff, "min_T_eff": min_teff, "max_T_eff": max_teff,
            "time_ms": dt_ms, "GOPS": gops,
        }
        with open(args.log_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

