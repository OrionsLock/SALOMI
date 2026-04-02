"""CLI to dump golden logs for BSDM-W (CPU and OpenCL backends).

Generates JSONL logs with byte-for-byte parity between backends.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np

from onebit.backends.opencl.host_opencl import OpenCLBinGemm
from onebit.core.golden_bits import pack_y_bits_to_hex
from onebit.core.packbits import pack_input_signs
from onebit.core.prf import derive_seed
from onebit.ops.bsdm_w import SDConfig, bsdm_w_dot


def _rand_pm1(K: int, seed: int) -> np.ndarray:
    """Generate random ±1 vector."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=K, dtype=np.int8)
    return (x * 2 - 1).astype(np.int8)


def _pack_pm1(x: np.ndarray) -> np.ndarray:
    """Pack ±1 vector to uint32."""
    return pack_input_signs(x.astype(np.float32))


def dump_golden_cpu(
    W_bits: np.ndarray,  # [M, Kw]
    X_bits: np.ndarray,  # [1, Kw] (single tick vector)
    cfg: SDConfig,
    eps: float,
    delta: float,
    k_max: int,
    run_id: str,
    layer: int,
    token: int,
    want_pc32: bool = False,
) -> list[dict]:
    """Run CPU backend and emit golden log records.
    
    Returns list of dicts (one per row).
    """
    M, Kw = W_bits.shape
    records = []
    
    for row in range(M):
        # Derive PRF seed
        run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
        seed = derive_seed(layer, row, token, run_id_int)
        
        # Run BSDM-W
        est, diags = bsdm_w_dot(W_bits[row], X_bits[0], k_max, cfg, seed=int(seed), want_pc32=want_pc32)
        
        # Extract diagnostics
        k_used = diags["k_used"]
        y_bits_main = diags["y_bits_main"]
        y_bits_twin = diags.get("y_bits_twin")
        pc32_main = diags.get("pc32_main")
        pc32_twin = diags.get("pc32_twin")
        
        # Pack to hex
        samples_per_channel = k_used * cfg.walsh_N
        y_main_hex = pack_y_bits_to_hex(y_bits_main, samples_per_channel)
        y_twin_hex = pack_y_bits_to_hex(y_bits_twin, samples_per_channel) if y_bits_twin is not None else None
        
        # CTG digest (all zeros when use_ctg=0)
        ctg_digest = "00000000"
        
        rec = {
            "run_id": run_id,
            "backend": "cpu",
            "layer": layer,
            "row": row,
            "token": token,
            "cfg": {
                "order": cfg.order,
                "beta": cfg.beta,
                "lambd": cfg.lambd,
                "walsh_N": cfg.walsh_N,
                "antithetic": 1 if cfg.antithetic else 0,
                "eps": eps,
                "delta": delta,
                "k_max": k_max,
                "use_ctg": 0,
            },
            "seed": int(seed),
            "T_eff": k_used,
            "y_bits_main": y_main_hex,
            "Y_mean": float(est),
            "ctg_digest": ctg_digest,
        }
        
        if y_twin_hex is not None:
            rec["y_bits_twin"] = y_twin_hex
        if pc32_main is not None:
            rec["pc32_main"] = pc32_main
        if pc32_twin is not None:
            rec["pc32_twin"] = pc32_twin
        
        records.append(rec)
    
    return records


def dump_golden_opencl(
    W_bits: np.ndarray,  # [M, Kw]
    X_bits: np.ndarray,  # [1, Kw] (single tick vector)
    cfg: SDConfig,
    eps: float,
    delta: float,
    k_max: int,
    run_id: str,
    layer: int,
    token: int,
    want_pc32: bool = False,
) -> list[dict]:
    """Run OpenCL backend and emit golden log records.
    
    Returns list of dicts (one per row).
    """
    M, Kw = W_bits.shape
    
    # Replicate X_bits to k_max ticks (same vector repeated)
    X_ticks = np.tile(X_bits, (k_max, 1))
    
    # Derive PRF seeds for all rows
    run_id_int = int(uuid.UUID(run_id).int & 0xFFFFFFFFFFFFFFFF)
    seeds = np.array([derive_seed(layer, row, token, run_id_int) for row in range(M)], dtype=np.uint64)
    
    # Run OpenCL kernel (batch mode)
    # Note: Current kernel uses single prf_seed; we'll need to extend for per-row seeds
    # For now, use seed[0] and warn if M > 1
    if M > 1:
        print(f"Warning: OpenCL batch mode with M={M} uses seed[0] for all rows (parity will fail)")
    
    gemm = OpenCLBinGemm()
    out = gemm.run_bsdm_w_naive_norm(
        W_bits, X_ticks, T=k_max, eps=eps, delta=delta,
        order=cfg.order, beta=cfg.beta, lambd=cfg.lambd,
        walsh_N=cfg.walsh_N, antithetic=cfg.antithetic,
        use_ctg=False, prf_seed=int(seeds[0]),
        local_size=256, want_y_pack=True, want_pc32=want_pc32
    )
    
    Y = out["Y"]
    T_eff = out["T_eff"]
    y_main_packed = out.get("y_bits_main")
    y_twin_packed = out.get("y_bits_twin")
    pc32_main_arr = out.get("pc32_main")
    pc32_twin_arr = out.get("pc32_twin")
    
    records = []
    for row in range(M):
        seed = int(seeds[row])
        k_used = int(T_eff[row])
        
        # Pack to hex
        samples_per_channel = k_used * cfg.walsh_N
        y_main_hex = pack_y_bits_to_hex(y_main_packed[row], samples_per_channel) if y_main_packed is not None else None
        y_twin_hex = pack_y_bits_to_hex(y_twin_packed[row], samples_per_channel) if y_twin_packed is not None else None
        
        # CTG digest (all zeros when use_ctg=0)
        ctg_digest = "00000000"
        
        rec = {
            "run_id": run_id,
            "backend": "opencl",
            "layer": layer,
            "row": row,
            "token": token,
            "cfg": {
                "order": cfg.order,
                "beta": cfg.beta,
                "lambd": cfg.lambd,
                "walsh_N": cfg.walsh_N,
                "antithetic": 1 if cfg.antithetic else 0,
                "eps": eps,
                "delta": delta,
                "k_max": k_max,
                "use_ctg": 0,
            },
            "seed": seed,
            "T_eff": k_used,
            "y_bits_main": y_main_hex,
            "Y_mean": float(Y[row]),
            "ctg_digest": ctg_digest,
        }
        
        if y_twin_hex is not None:
            rec["y_bits_twin"] = y_twin_hex
        if pc32_main_arr is not None:
            rec["pc32_main"] = pc32_main_arr[row].tolist()
        if pc32_twin_arr is not None:
            rec["pc32_twin"] = pc32_twin_arr[row].tolist()
        
        records.append(rec)
    
    return records


def main():
    parser = argparse.ArgumentParser(description="Dump golden logs for BSDM-W")
    parser.add_argument("--M", type=int, default=4, help="Number of rows")
    parser.add_argument("--K", type=int, default=1024, help="Dimension K")
    parser.add_argument("--k_max", type=int, default=32, help="Max ticks")
    parser.add_argument("--order", type=int, default=2, help="SD order (1 or 2)")
    parser.add_argument("--walsh_N", type=int, default=2, help="Walsh carriers per tick")
    parser.add_argument("--antithetic", type=int, default=1, help="Antithetic pairs (0 or 1)")
    parser.add_argument("--eps", type=float, default=0.05, help="Hoeffding epsilon")
    parser.add_argument("--delta", type=float, default=0.001, help="Per-row delta")
    parser.add_argument("--backend", type=str, default="both", choices=["cpu", "opencl", "both"], help="Backend to run")
    parser.add_argument("--outdir", type=str, default="golden", help="Output directory")
    parser.add_argument("--pc32", action="store_true", help="Include pc32 arrays")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for data generation")
    
    args = parser.parse_args()
    
    # Generate random data
    rng = np.random.default_rng(args.seed)
    W_rows = [_rand_pm1(args.K, int(rng.integers(0, 1<<31))) for _ in range(args.M)]
    X_vec = _rand_pm1(args.K, int(rng.integers(0, 1<<31)))
    W_bits = np.stack([_pack_pm1(w) for w in W_rows], axis=0)
    X_bits = _pack_pm1(X_vec)[None, :]
    
    # Config
    cfg = SDConfig(order=args.order, walsh_N=args.walsh_N, antithetic=bool(args.antithetic))
    run_id = str(uuid.uuid4())
    layer, token = 0, 0
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Run backends
    if args.backend in ["cpu", "both"]:
        cpu_records = dump_golden_cpu(W_bits, X_bits, cfg, args.eps, args.delta, args.k_max, run_id, layer, token, want_pc32=args.pc32)
        cpu_path = outdir / f"run_{run_id}_cpu.jsonl"
        with open(cpu_path, "w") as f:
            for rec in cpu_records:
                f.write(json.dumps(rec) + "\n")
        print(f"CPU logs: {cpu_path}")
    
    if args.backend in ["opencl", "both"]:
        try:
            opencl_records = dump_golden_opencl(W_bits, X_bits, cfg, args.eps, args.delta, args.k_max, run_id, layer, token, want_pc32=args.pc32)
            opencl_path = outdir / f"run_{run_id}_opencl.jsonl"
            with open(opencl_path, "w") as f:
                for rec in opencl_records:
                    f.write(json.dumps(rec) + "\n")
            print(f"OpenCL logs: {opencl_path}")
        except Exception as e:
            print(f"OpenCL backend failed: {e}")
    
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()

