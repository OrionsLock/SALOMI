"""CLI tool to dump Stage-A probe results in JSONL format.

Usage:
    python -m onebit.cli.dump_stageA --K 1024 --L 256 --kA 16 --seed 12345 --out stageA.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
import numpy as np

from onebit.ops.attention_probe import stageA_probe_topT
from onebit.core.packbits import pack_input_signs


def _rand_pm1(K, seed):
    """Generate random ±1 vector."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=K, dtype=np.int8)
    return (x * 2 - 1).astype(np.int8)


def _pack_pm1(x):
    """Pack ±1 vector to bits."""
    return pack_input_signs(x.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Dump Stage-A probe results")
    parser.add_argument("--K", type=int, default=1024, help="Dimension (bits)")
    parser.add_argument("--L", type=int, default=256, help="Number of keys")
    parser.add_argument("--kA", type=int, default=16, help="Stage-A probe ticks")
    parser.add_argument("--seed", type=int, default=12345, help="PRF seed")
    parser.add_argument("--out", type=str, default="stageA.jsonl", help="Output JSONL file")
    parser.add_argument("--opencl", action="store_true", help="Use OpenCL backend")
    
    args = parser.parse_args()
    
    # Generate random data
    rng = np.random.default_rng(args.seed)
    Q_vec = _rand_pm1(args.K, int(rng.integers(0, 1<<31)))
    K_mat = np.array([_rand_pm1(args.K, int(rng.integers(0, 1<<31))) for _ in range(args.L)])
    
    Q_bits = _pack_pm1(Q_vec)
    K_bits = np.array([_pack_pm1(K_mat[i]) for i in range(args.L)])
    
    # Run CPU
    print(f"Running CPU Stage-A probe: K={args.K}, L={args.L}, kA={args.kA}, seed={args.seed}")
    result_cpu = stageA_probe_topT(
        Q_bits, K_bits,
        kA=args.kA, T_set=(8, 12, 16),
        prf_seed=args.seed,
        walsh_N=2, antithetic=True,
        order=2, beta=0.30, lambd=1.0/256.0,
    )
    
    # Prepare CPU record
    record_cpu = {
        "backend": "cpu",
        "K": args.K,
        "L": args.L,
        "kA": args.kA,
        "prf_seed": args.seed,
        "T_sel": result_cpu["T_sel"],
        "idx_top": result_cpu["idx_top"].tolist(),
        "gap12": float(result_cpu["stats"]["gap12"]),
        "elbow_T_raw": float(result_cpu["stats"]["elbow_T_raw"]),
        "teff": result_cpu["stats"]["teff"],
        "mu_top10": result_cpu["stats"]["mu"][result_cpu["idx_top"][:10]].tolist(),
    }
    
    records = [record_cpu]
    
    # Run OpenCL if requested
    if args.opencl:
        try:
            from onebit.backends.opencl.host_opencl import OpenCLBinGemm
            gemm = OpenCLBinGemm()
            
            print(f"Running OpenCL Stage-A probe...")
            result_cl = gemm.stageA_probe_topT_opencl(
                Q_bits, K_bits,
                kA=args.kA, T_set=(8, 12, 16),
                prf_seed=args.seed,
                walsh_N=2, antithetic=True,
                order=2, beta=0.30, lambd=1.0/256.0,
                local_size=256,
            )
            
            # Prepare OpenCL record
            record_cl = {
                "backend": "opencl",
                "K": args.K,
                "L": args.L,
                "kA": args.kA,
                "prf_seed": args.seed,
                "T_sel": result_cl["T_sel"],
                "idx_top": result_cl["idx_top"].tolist(),
                "gap12": float(result_cl["stats"]["gap12"]),
                "elbow_T_raw": float(result_cl["stats"]["elbow_T_raw"]),
                "teff": result_cl["stats"]["teff"],
                "mu_top10": result_cl["stats"]["mu"][result_cl["idx_top"][:10]].tolist(),
            }
            
            records.append(record_cl)
            
            # Verify parity
            if result_cpu["T_sel"] != result_cl["T_sel"]:
                print(f"WARNING: T_sel mismatch: CPU={result_cpu['T_sel']}, OpenCL={result_cl['T_sel']}")
            elif not np.array_equal(result_cpu["idx_top"], result_cl["idx_top"]):
                print(f"WARNING: idx_top mismatch")
            else:
                print(f"✓ CPU↔OpenCL parity confirmed: T_sel={result_cpu['T_sel']}")
                
        except Exception as e:
            print(f"OpenCL not available: {e}")
    
    # Write JSONL
    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    
    print(f"Wrote {len(records)} records to {args.out}")
    print(f"CPU: T_sel={result_cpu['T_sel']}, gap12={result_cpu['stats']['gap12']:.6f}")


if __name__ == "__main__":
    main()

