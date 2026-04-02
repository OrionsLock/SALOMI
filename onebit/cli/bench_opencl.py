from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from onebit.core import pack_signs_rowmajor, pack_input_signs
from onebit.backends.opencl import OpenCLBinGemm


def run_case(M: int, K: int, T: int, eps_margin: float, delta: float = 0.05, kernel: str = "auto"):
    rng = np.random.default_rng(0)
    W = rng.standard_normal((M, K), dtype=np.float32)
    X = rng.standard_normal((T, K), dtype=np.float32)
    W_bits = pack_signs_rowmajor(W)
    Kw = W_bits.shape[1]
    X_bits = np.zeros((T, Kw), dtype=np.uint32)
    for t in range(T):
        X_bits[t] = pack_input_signs(X[t])
    cv = np.array([1.0, 0.0], dtype=np.float32)

    gemm = OpenCLBinGemm()
    t0 = time.perf_counter()
    out = gemm.run(W_bits, X_bits, cv=cv, T=T, eps_margin=eps_margin, delta=delta, return_teff="per_row", kernel=kernel)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    rec_kernel = kernel

    teff = out.get("T_eff")
    if isinstance(teff, np.ndarray):
        avg_teff = float(np.mean(teff))
        min_teff = int(np.min(teff))
        max_teff = int(np.max(teff))
    elif teff is None:
        avg_teff = float(T)
        min_teff = int(T)
        max_teff = int(T)
    else:
        avg_teff = float(teff)
        min_teff = int(teff)
        max_teff = int(teff)

    gops = (2.0 * float(M) * float(K) * avg_teff) / (dt_ms * 1e6)
    return {
        "M": M,
        "K": K,
        "T": T,
        "eps_margin": eps_margin,
        "kernel": rec_kernel,
        "avg_T_eff": avg_teff,
        "min_T_eff": min_teff,
        "max_T_eff": max_teff,
        "time_ms": dt_ms,
        "GOPS": gops,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str, default=None)
    p.add_argument("--csv", action="store_true")
    p.add_argument("--kernel", type=str, default="auto", choices=["auto","naive","tiled","compare"], help="kernel variant or 'compare'")
    p.add_argument("--M", type=int, default=None, help="rows (override sweep)")
    p.add_argument("--K", type=int, default=None, help="features (override sweep)")
    p.add_argument("--T", type=int, default=None, help="passes (override sweep)")
    p.add_argument("--eps", type=float, default=None, help="eps_margin (override sweep)")
    args = p.parse_args()

    try:
        import pyopencl as cl  # type: ignore
        if not cl.get_platforms():
            print("[SKIP] No OpenCL platforms found", file=sys.stderr)
            return 0
    except Exception as e:
        print(f"[SKIP] OpenCL not usable: {e}", file=sys.stderr)
        return 0

    Ms = [args.M] if args.M is not None else [1024, 2048, 4096, 11008]
    Ks = [args.K] if args.K is not None else [1024, 2048, 4096, 8192]
    Ts = [args.T] if args.T is not None else [2, 4, 8, 16]
    epses = [args.eps] if args.eps is not None else [0.0, 0.5, 1.0]

    records = []
    for M in Ms:
        for K in Ks:
            for T in Ts:
                for eps in epses:
                    if args.kernel == "compare":
                        rec_naive = run_case(M, K, T, eps, kernel="naive")
                        rec_tiled = run_case(M, K, T, eps, kernel="tiled")
                        records.extend([rec_naive, rec_tiled])
                        if args.jsonl:
                            with open(args.jsonl, "a", encoding="utf-8") as f:
                                f.write(json.dumps(rec_naive) + "\n")
                                f.write(json.dumps(rec_tiled) + "\n")
                    else:
                        rec = run_case(M, K, T, eps, kernel=args.kernel)
                        records.append(rec)
                        if args.jsonl:
                            with open(args.jsonl, "a", encoding="utf-8") as f:
                                f.write(json.dumps(rec) + "\n")

    if args.csv:
        header = "M,K,T,eps_margin,kernel,avg_T_eff,min_T_eff,max_T_eff,time_ms,GOPS"
        print(header)
        if args.kernel == "compare":
            # group by (M,K,T,eps) and print speedup
            from collections import defaultdict
            groups = defaultdict(list)
            for r in records:
                key = (r['M'], r['K'], r['T'], r['eps_margin'])
                groups[key].append(r)
            for key, recs in groups.items():
                # print both rows
                for r in recs:
                    print(f"{r['M']},{r['K']},{r['T']},{r['eps_margin']},{r['kernel']},{r['avg_T_eff']:.3f},{r['min_T_eff']},{r['max_T_eff']},{r['time_ms']:.3f},{r['GOPS']:.3f}")
                if len(recs) == 2:
                    a = next((x for x in recs if x['kernel']=="naive"), None)
                    b = next((x for x in recs if x['kernel']=="tiled"), None)
                    if a and b:
                        speedup = a['time_ms'] / b['time_ms'] if b['time_ms']>0 else float('nan')
                        print(f"# speedup {key}: {speedup:.3f}x (naive/tiled)")
        else:
            for r in records:
                print(f"{r['M']},{r['K']},{r['T']},{r['eps_margin']},{r['kernel']},{r['avg_T_eff']:.3f},{r['min_T_eff']},{r['max_T_eff']},{r['time_ms']:.3f},{r['GOPS']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

