"""
Stress test CLI harness.

Usage:
    python -m onebit.cli.stress \
      --preset {smoke,nightly,max} \
      --backend {cpu,opencl} --kernel {auto,naive,tiled} \
      --ctg {0,1} --early-exit {0,1} \
      --cases N --seed-base S \
      --out out/stress_summary.json --log out/stress.jsonl --triage-dir out/triage
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 and return first 16 hex chars."""
    import hashlib
    return hashlib.sha256(data).hexdigest()[:16]


def generate_test_cases(preset: str, seed_base: int, cases: int) -> List[Dict[str, Any]]:
    """Generate test case configurations based on preset."""
    
    if preset == "smoke":
        shapes = [
            (1, 32, 16),
            (2, 64, 16),
            (33, 1024, 16),
        ]
        seeds = list(range(seed_base, seed_base + min(cases, 10)))
        orders = [2]
        ctg_vals = [0]
        early_exit_vals = [0]
    elif preset == "nightly":
        shapes = [
            (1, 32, 16),
            (2, 64, 16),
            (33, 1024, 16),
            (128, 2048, 16),
            (256, 4096, 32),
        ]
        seeds = list(range(seed_base, seed_base + min(cases, 100)))
        orders = [1, 2]
        ctg_vals = [0, 1]
        early_exit_vals = [0, 1]
    else:  # max
        shapes = [
            (1, 32, 16),
            (2, 64, 16),
            (33, 1024, 16),
            (128, 2048, 16),
            (256, 4096, 32),
            (512, 8192, 32),
        ]
        seeds = list(range(seed_base, seed_base + cases))
        orders = [1, 2]
        ctg_vals = [0, 1]
        early_exit_vals = [0, 1]
    
    test_cases = []
    case_id = 0
    
    for seed in seeds:
        for M, Kw, k in shapes:
            for order in orders:
                for ctg in ctg_vals:
                    for early_exit in early_exit_vals:
                        test_cases.append({
                            "case_id": f"case_{case_id:06d}",
                            "seed": seed,
                            "M": M,
                            "Kw": Kw,
                            "k": k,
                            "order": order,
                            "ctg": ctg,
                            "early_exit": early_exit,
                            "eps": 0.05,
                            "delta": 1e-3,
                        })
                        case_id += 1
                        if len(test_cases) >= cases:
                            return test_cases
    
    return test_cases


def run_case_cpu(case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case on CPU."""
    from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig
    from onebit.core.packbits import pack_input_signs
    
    seed = case["seed"]
    M, Kw, k = case["M"], case["Kw"], case["k"]
    d = Kw * 32
    
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    cfg = SDConfig(
        order=case["order"],
        beta=0.30,
        lambd=1.0/256.0,
        walsh_N=2,
        antithetic=True
    )
    
    results = []
    for row_idx in range(M):
        est, diags = bsdm_w_dot(
            W[row_idx], X, k=k, cfg=cfg, seed=seed,
            want_pc32=True, want_y_pack=True, instr_on=False
        )
        results.append({
            "y_mean": est,
            "y_main": diags.get("y_bits_main"),
            "y_twin": diags.get("y_bits_twin"),
            "pc32_main": diags.get("pc32_main"),
            "pc32_twin": diags.get("pc32_twin"),
        })
    
    # Compute digests
    y_mean_arr = np.array([r["y_mean"] for r in results], dtype=np.float32)
    y_main_all = np.concatenate([r["y_main"] for r in results])
    y_twin_all = np.concatenate([r["y_twin"] for r in results])
    pc32_main_all = np.concatenate([r["pc32_main"] for r in results])
    pc32_twin_all = np.concatenate([r["pc32_twin"] for r in results])
    
    return {
        "digests": {
            "y_mean": sha256_hex(y_mean_arr.tobytes()),
            "y_main": sha256_hex(y_main_all.tobytes()),
            "y_twin": sha256_hex(y_twin_all.tobytes()),
            "pc32_main": sha256_hex(pc32_main_all.tobytes()),
            "pc32_twin": sha256_hex(pc32_twin_all.tobytes()),
        },
        "status": "PASS",
        "runtime_ms": 0.0,
    }


def run_case_opencl(case: Dict[str, Any], kernel: str) -> Dict[str, Any]:
    """Run a single test case on OpenCL."""
    from onebit.backends.opencl.host_opencl import OpenCLBinGemm
    from onebit.core.packbits import pack_input_signs
    
    seed = case["seed"]
    M, Kw, k = case["M"], case["Kw"], case["k"]
    d = Kw * 32
    
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, d), dtype=np.float32)
    X = rng.standard_normal(d, dtype=np.float32)
    
    # Pack to bits
    W_bits = np.array([pack_input_signs(row) for row in W], dtype=np.uint32)
    X_bits = pack_input_signs(X).reshape(1, -1)
    X_bits_tiled = np.tile(X_bits, (k, 1))
    
    backend = OpenCLBinGemm()
    
    t0 = time.perf_counter()
    result = backend.run_bsdm_w_naive_norm(
        W_bits, X_bits_tiled, T=k,
        eps=case["eps"], delta=case["delta"],
        order=case["order"], beta=0.30, lambd=1.0/256.0,
        walsh_N=2, antithetic=True,
        use_ctg=bool(case["ctg"]),
        prf_seed=seed,
        early_exit_enable=bool(case["early_exit"]),
        want_y_pack=True, want_pc32=True,
        kernel=kernel,
        instr_on=False
    )
    t1 = time.perf_counter()
    runtime_ms = (t1 - t0) * 1000.0
    
    # Compute digests
    return {
        "digests": {
            "y_mean": sha256_hex(result["Y"].tobytes()),
            "y_main": sha256_hex(result["y_bits_main"].tobytes()),
            "y_twin": sha256_hex(result["y_bits_twin"].tobytes()),
            "pc32_main": sha256_hex(result["pc32_main"].tobytes()),
            "pc32_twin": sha256_hex(result["pc32_twin"].tobytes()),
        },
        "status": "PASS",
        "runtime_ms": runtime_ms,
        "T_eff": result["T_eff"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Stress test harness")
    parser.add_argument("--preset", choices=["smoke", "nightly", "max"], default="smoke")
    parser.add_argument("--backend", choices=["cpu", "opencl"], default="opencl")
    parser.add_argument("--kernel", choices=["auto", "naive", "tiled"], default="auto")
    parser.add_argument("--ctg", type=int, choices=[0, 1], default=None)
    parser.add_argument("--early-exit", type=int, choices=[0, 1], default=None)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=1000000)
    parser.add_argument("--out", type=str, default="out/stress_summary.json")
    parser.add_argument("--log", type=str, default="out/stress.jsonl")
    parser.add_argument("--triage-dir", type=str, default="out/triage")
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.triage_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate test cases
    test_cases = generate_test_cases(args.preset, args.seed_base, args.cases)
    
    print(f"Running {len(test_cases)} test cases...")
    print(f"Preset: {args.preset}, Backend: {args.backend}, Kernel: {args.kernel}")
    
    # Run cases
    results = []
    fails = 0
    runtimes = []
    
    with open(args.log, "w") as log_file:
        for i, case in enumerate(test_cases):
            try:
                if args.backend == "cpu":
                    result = run_case_cpu(case)
                else:
                    result = run_case_opencl(case, args.kernel)
                
                # Log result
                log_entry = {**case, **result, "backend": args.backend, "kernel": args.kernel}
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
                
                results.append(result)
                if result.get("runtime_ms"):
                    runtimes.append(result["runtime_ms"])
                
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(test_cases)}")
            
            except Exception as e:
                print(f"  FAIL: case {case['case_id']}: {e}")
                fails += 1
                log_entry = {**case, "status": "FAIL", "error": str(e), "backend": args.backend, "kernel": args.kernel}
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
    
    # Compute summary
    summary = {
        "runs": len(test_cases),
        "fails": fails,
        "flakes": 0,  # Computed by determinism tests
        "p50_ms": float(np.median(runtimes)) if runtimes else 0.0,
        "p95_ms": float(np.percentile(runtimes, 95)) if runtimes else 0.0,
        "empirical_bound_viol": 0,  # Computed by bounds tests
    }
    
    # Save summary
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary:")
    print(f"  Runs: {summary['runs']}")
    print(f"  Fails: {summary['fails']}")
    print(f"  p50 runtime: {summary['p50_ms']:.2f} ms")
    print(f"  p95 runtime: {summary['p95_ms']:.2f} ms")
    print(f"\nResults written to: {args.out}")
    print(f"Log written to: {args.log}")
    
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()

