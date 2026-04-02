"""CLI demo for controller-based inference."""
from __future__ import annotations

import argparse
import time
import numpy as np

from ..runtime.controller import infer_one_token, CtrlConfig
from ..logs.records import controller_record, csv_header, record_to_csv_row
from ..core.packbits import pack_input_signs


def main():
    """Run controller demo with synthetic data."""
    parser = argparse.ArgumentParser(description="Controller demo for SALOMI inference")
    parser.add_argument("--tokens", type=int, default=4, help="Number of tokens to process")
    parser.add_argument("--backend", choices=["cpu", "opencl"], default="opencl", help="Backend to use")
    parser.add_argument("--seed", type=int, default=12345, help="PRF seed")
    parser.add_argument("--keys", type=int, default=64, help="Number of keys")
    parser.add_argument("--dim", type=int, default=1024, help="Dimension")
    parser.add_argument("--kA", type=int, default=16, help="Stage-A ticks")
    parser.add_argument("--k-max", type=int, default=64, help="Max SPRT ticks")
    parser.add_argument("--delta", type=float, default=0.01, help="Total risk budget")
    parser.add_argument("--eps", type=float, default=0.05, help="SPRT effect size")
    parser.add_argument("--output", type=str, default="controller_demo.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Create synthetic data
    np.random.seed(args.seed)
    
    K = args.keys
    d = args.dim
    
    # Generate keys
    K_mat = np.random.randn(K, d)
    K_bits = np.array([pack_input_signs(K_mat[i]) for i in range(K)])
    
    # Controller config
    cfg = CtrlConfig(
        kA=args.kA,
        k_max=args.k_max,
        delta_total=args.delta,
        eps=args.eps,
        backend=args.backend,
    )
    
    # Open output file
    with open(args.output, "w") as f:
        # Write header
        f.write(csv_header() + "\n")
        
        # Process tokens
        for token_idx in range(args.tokens):
            # Generate query for this token
            Q = np.random.randn(d)
            Q_bits = pack_input_signs(Q)
            
            # Derive seed for this token
            token_seed = args.seed + token_idx * 1000
            
            # Run inference
            t0 = time.perf_counter()
            cert = infer_one_token(
                Q_bits, K_bits,
                cfg=cfg,
                prf_seed=token_seed,
            )
            t1 = time.perf_counter()
            time_ms = (t1 - t0) * 1000.0
            
            # Log record
            record = controller_record(cert, token_idx=token_idx, time_ms=time_ms)
            csv_row = record_to_csv_row(record)
            f.write(csv_row + "\n")
            
            # Print to console
            print(f"Token {token_idx}: status={record['status']}, top1={record['top1']}, "
                  f"k_attn={record['k_attn_used']}, pairs={record['pairs_evaluated']}, "
                  f"time={time_ms:.2f}ms")
    
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()

