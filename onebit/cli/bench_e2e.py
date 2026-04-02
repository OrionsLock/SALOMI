"""End-to-end benchmark: measure per-token latency and k-stats."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import numpy as np

from onebit.runtime.controller_e2e import infer_one_token_e2e, E2EConfig
from onebit.runtime.cpg_policy import CpgPolicy
from onebit.runtime.ctg_grammar import CTG, CTGRule, CTGState, make_default_programs
from onebit.runtime.pulse_scheduler import PulseScheduler
from onebit.runtime.shortlist import ShortlistCache, CarryCfg
from onebit.core.packbits import pack_input_signs
from onebit.metrics.summarize import summarize_tokens, save_summary_json


def main():
    parser = argparse.ArgumentParser(description="End-to-end benchmark")
    
    # Shape parameters
    parser.add_argument("--tokens", type=int, default=256, help="Number of tokens to process")
    parser.add_argument("--keys", type=int, default=1024, help="Number of keys in context")
    parser.add_argument("--d", type=int, default=768, help="Model dimension")
    
    # Controller parameters
    parser.add_argument("--kA", type=int, default=16, help="Stage-A probe ticks")
    parser.add_argument("--k-max", type=int, default=64, help="Max ticks for SPRT")
    
    # Backend and kernel
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "opencl"], 
                       help="Backend to use")
    parser.add_argument("--kernel", type=str, default="auto", choices=["auto", "naive", "tiled"],
                       help="Kernel selection (for OpenCL)")
    
    # Flags
    parser.add_argument("--ctg", type=int, default=0, choices=[0, 1], help="Enable CTG")
    parser.add_argument("--ctg-prog", type=int, default=0, choices=[0, 1], help="Enable CTG-PROG (multi-program)")
    parser.add_argument("--ctg-K", type=int, default=4, help="Number of CTG programs (for CTG-PROG)")
    parser.add_argument("--early-exit", type=int, default=1, choices=[0, 1],
                       help="Enable early exit in SPRT")

    # Pulse scheduler
    parser.add_argument("--pulse-enable", type=int, default=0, choices=[0, 1], help="Enable pulse scheduler")
    parser.add_argument("--pulse-period", type=int, default=64, help="Pulse period (tokens between repairs)")
    parser.add_argument("--pulse-duty", type=int, default=1, help="Pulse duty (groups per repair)")

    # CTG shadow
    parser.add_argument("--ctg-shadow", type=int, default=0, choices=[0, 1], help="Enable CTG shadow sampling")
    parser.add_argument("--ctg-shadow-rate", type=float, default=0.01, help="CTG shadow sampling rate")

    # Shortlist carry
    parser.add_argument("--carry-enable", type=int, default=1, choices=[0, 1], help="Enable shortlist carry")
    parser.add_argument("--carry-frac", type=float, default=0.5, help="Fraction of shortlist to carry")
    parser.add_argument("--carry-cap", type=int, default=256, help="Max carried IDs")
    parser.add_argument("--carry-ttl", type=int, default=8, help="TTL for carried IDs")
    parser.add_argument("--carry-explore", type=int, default=128, help="Min fresh candidates")

    # Seed and runs
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (for averaging)")
    
    # Output
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--summary", type=str, default=None, help="Output summary JSON path")
    
    args = parser.parse_args()
    
    # Print config
    print("=" * 60)
    print("End-to-End Benchmark")
    print("=" * 60)
    print(f"Tokens: {args.tokens}")
    print(f"Keys: {args.keys}")
    print(f"Dimension: {args.d}")
    print(f"kA: {args.kA}, k_max: {args.k_max}")
    print(f"Backend: {args.backend}, Kernel: {args.kernel}")
    print(f"CTG: {args.ctg}, Early-exit: {args.early_exit}")
    print(f"Pulse: enable={args.pulse_enable}, period={args.pulse_period}, duty={args.pulse_duty}")
    print(f"CTG Shadow: enable={args.ctg_shadow}, rate={args.ctg_shadow_rate}")
    print(f"Carry: enable={args.carry_enable}, frac={args.carry_frac}, cap={args.carry_cap}, ttl={args.carry_ttl}")
    print(f"Seed: {args.seed}, Runs: {args.runs}")
    print()
    
    # Generate synthetic data
    np.random.seed(args.seed)
    
    n_ctx = args.keys
    d_attn = args.d
    d_model = args.d
    d_kv = args.d // 2  # Typical KV dimension
    vocab_size = 32000  # Typical vocab size
    
    # Attention matrices (shared across tokens for simplicity)
    K_attn = np.random.randn(n_ctx, d_attn).astype(np.float32)
    K_attn_bits = np.array([pack_input_signs(K_attn[i]) for i in range(n_ctx)])
    
    # KV cache
    K_kv = np.random.randn(n_ctx, d_kv).astype(np.float32)
    V_kv = np.random.randn(n_ctx, d_kv).astype(np.float32)
    K_kv_bits = np.array([pack_input_signs(K_kv[i]) for i in range(n_ctx)])
    V_kv_bits = np.array([pack_input_signs(V_kv[i]) for i in range(n_ctx)])
    
    # Vocab IDs
    vocab_ids = np.arange(vocab_size, dtype=np.int32)
    
    # Create CTG grammar (if enabled)
    ctg = None
    ctg_state = None
    if args.ctg:
        if args.ctg_prog:
            # CTG-PROG mode: multiple programs
            programs = make_default_programs(vocab_size, K=args.ctg_K)
            ctg = CTG(programs=programs, vocab_size=vocab_size)
            ctg_state = CTGState()
        else:
            # CTG-FIXED mode: single PASS rule
            rules = [CTGRule(op="PASS", ids=None)]
            ctg = CTG(rules=rules, vocab_size=vocab_size)
            ctg_state = CTGState()

    # Create config
    cfg = E2EConfig(
        kA=args.kA,
        k_max_attn=args.k_max,
        d_kv=d_kv,
        backend=args.backend,
        ctg=ctg,
        ctg_state=ctg_state,
    )

    # Create CTG policy (if shadow enabled)
    cpg_policy = None
    if args.ctg_shadow:
        cpg_policy = CpgPolicy(shadow_rate=args.ctg_shadow_rate, seed=args.seed)

    # Create pulse scheduler (if enabled)
    pulse_scheduler = None
    if args.pulse_enable:
        n_layers = 1  # Single layer for now
        n_groups = (n_ctx + 63) // 64  # 64 positions per group
        pulse_scheduler = PulseScheduler(
            n_layers=n_layers,
            n_groups=n_groups,
            group_size=64,
            base_interval=args.pulse_period,
        )

    # Create shortlist cache (if enabled)
    shortlist_cache = None
    carry_cfg = None
    if args.carry_enable:
        carry_cfg = CarryCfg(
            enable=True,
            frac=args.carry_frac,
            cap=args.carry_cap,
            ttl=args.carry_ttl,
            explore=args.carry_explore,
            seed=args.seed,
        )
        shortlist_cache = ShortlistCache(
            cap=carry_cfg.cap,
            ttl=carry_cfg.ttl,
            ema=0.30,
            seed=carry_cfg.seed,
        )

    # Warmup (1 token to trigger JIT compilation)
    print("Warming up...")
    Q_warmup = np.random.randn(d_attn).astype(np.float32)
    Q_warmup_attn_bits = pack_input_signs(Q_warmup)
    Q_warmup_logits = np.random.randn(d_model).astype(np.float32)
    Q_warmup_logits_bits = pack_input_signs(Q_warmup_logits)

    ctg_state_current = ctg_state

    warmup_result = infer_one_token_e2e(
        Q_warmup_attn_bits, K_attn_bits,
        K_kv_bits, V_kv_bits,
        Q_warmup_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=args.seed - 1,
        d_attn=d_attn,
        d_model=d_model,
        token_idx=0,
        cpg_policy=cpg_policy,
        pulse_scheduler=pulse_scheduler,
        shortlist_cache=shortlist_cache,
        carry_cfg=carry_cfg,
        ctg=ctg,
        ctg_state=ctg_state_current,
    )
    if ctg is not None:
        ctg_state_current = warmup_result.get("ctg_state", ctg_state_current)

    print("Warmup complete.\n")

    # Prepare CSV output
    csv_rows = []

    # Process tokens
    print(f"Processing {args.tokens} tokens...")
    for token_idx in range(args.tokens):
        # Generate query for this token
        Q_attn = np.random.randn(d_attn).astype(np.float32)
        Q_attn_bits = pack_input_signs(Q_attn)
        
        Q_logits = np.random.randn(d_model).astype(np.float32)
        Q_logits_bits = pack_input_signs(Q_logits)
        
        # Measure wall time
        t0 = time.perf_counter()

        result = infer_one_token_e2e(
            Q_attn_bits, K_attn_bits,
            K_kv_bits, V_kv_bits,
            Q_logits_bits, vocab_ids,
            cfg=cfg,
            prf_seed=args.seed + token_idx,
            d_attn=d_attn,
            d_model=d_model,
            token_idx=token_idx,
            cpg_policy=cpg_policy,
            pulse_scheduler=pulse_scheduler,
            shortlist_cache=shortlist_cache,
            carry_cfg=carry_cfg,
            ctg=ctg,
            ctg_state=ctg_state_current,
        )

        if ctg is not None:
            ctg_state_current = result.get("ctg_state", ctg_state_current)

        t1 = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0
        
        # Extract metrics
        k_attn_used = result["k_attn_used"]
        k_logits_used = result["k_logits_used"]
        status = result["status"]
        unsure = result["unsure"]

        # Teff_qk: use k_attn_used as proxy (actual Teff would need y_bits analysis)
        Teff_qk = k_attn_used

        # Append row with telemetry
        csv_rows.append({
            "token_idx": token_idx,
            "time_ms": f"{time_ms:.3f}",
            "k_attn_used": k_attn_used,
            "k_logits_used": k_logits_used,
            "Teff_qk": Teff_qk,
            "status": status,
            "unsure": unsure,
            "carry_count": result.get("carry_count", 0),
            "fresh_count": result.get("fresh_count", 0),
            "shortlist_total": result.get("shortlist_total", 0),
            "pairs_total": result.get("pairs_total", 0),
            "pairs_reduced_pct": f"{result.get('pairs_reduced_pct', 0.0):.2f}",
            "hcl_chunks": result.get("hcl_chunks", 0),
            "hcl_chunks_reduced_pct": f"{result.get('hcl_chunks_reduced_pct', 0.0):.2f}",
            "ctg_shadow_calls": result.get("ctg_shadow_calls", 0),
            "ctg_match_rate": f"{result.get('ctg_match_rate', 1.0):.4f}",
            "pulse_repairs": result.get("pulse_repairs", 0),
            "kv_bytes_write": result.get("kv_bytes_write", 0),
            "ctg_active": result.get("ctg_active", 0),
            "ctg_phase": result.get("ctg_phase", 0),
            "ctg_masked_count": result.get("ctg_masked_count", 0),
            "ctg_mask_digest": result.get("ctg_mask_digest", 0),
            "ctg_inv_flag": result.get("ctg_inv_flag", 0),
            "ctg_prog_id": result.get("ctg_prog_id", 0),
        })

        # Progress
        if (token_idx + 1) % 50 == 0:
            print(f"  Processed {token_idx + 1}/{args.tokens} tokens...")
    
    print(f"Completed {args.tokens} tokens.\n")
    
    # Write CSV
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                "token_idx", "time_ms", "k_attn_used", "k_logits_used",
                "Teff_qk", "status", "unsure",
                "carry_count", "fresh_count", "shortlist_total",
                "pairs_total", "pairs_reduced_pct",
                "hcl_chunks", "hcl_chunks_reduced_pct",
                "ctg_shadow_calls", "ctg_match_rate",
                "pulse_repairs", "kv_bytes_write",
                "ctg_active", "ctg_phase", "ctg_masked_count",
                "ctg_mask_digest", "ctg_inv_flag", "ctg_prog_id",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"CSV written to: {csv_path}")
        
        # Generate summary
        summary = summarize_tokens(csv_path)
        
        print("\nSummary:")
        print(f"  P50 latency: {summary['P50_ms']:.3f} ms")
        print(f"  P95 latency: {summary['P95_ms']:.3f} ms")
        print(f"  Mean k: {summary['mean_k']:.1f}")
        print(f"  Median k: {summary['median_k']:.1f}")
        print(f"  P95 k: {summary['P95_k']}")
        print(f"  Unsure rate: {summary['unsure_rate']:.2%}")
        print(f"  Total tokens: {summary['total_tokens']}")
        
        # Write summary JSON
        if args.summary:
            save_summary_json(summary, args.summary)
            print(f"\nSummary JSON written to: {args.summary}")
    else:
        print("No CSV output specified (use --csv)")
    
    print("\nDone.")


if __name__ == "__main__":
    main()

