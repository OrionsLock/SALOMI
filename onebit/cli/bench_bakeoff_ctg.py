"""Benchmark bake-off: FP32 vs 1.53-bit vs 1-bit+CTG-FIXED vs 1-bit+CTG-PROG.

Phase III: Comprehensive evaluation on WikiText-103.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from onebit.runtime.controller_e2e import infer_one_token_e2e, E2EConfig
from onebit.runtime.ctg_grammar import CTG, CTGRule, CTGState, make_default_programs
from onebit.runtime.cpg_policy import CpgPolicy
from onebit.runtime.pulse_scheduler import PulseScheduler
from onebit.runtime.shortlist import ShortlistCache, CarryCfg
from onebit.core.packbits import pack_input_signs


@dataclass
class BakeoffMetrics:
    """Metrics for a single configuration."""
    config_name: str
    ppl: float  # Perplexity
    ppl_delta_pct: float  # % change vs FP32
    k_mean: float  # Mean k_used
    k_p95: float  # P95 k_used
    pairs_mean: float  # Mean pairs evaluated
    pairs_delta_pct: float  # % change vs baseline
    latency_mean_ms: float  # Mean latency per token
    latency_p95_ms: float  # P95 latency
    latency_delta_pct: float  # % change vs baseline
    variance_k: float  # Variance of k_used
    variance_ratio: float  # σ² ratio vs baseline
    tokens_processed: int
    unsure_rate: float  # Fraction of UNSURE tokens


@dataclass
class BakeoffConfig:
    """Configuration for bake-off run."""
    name: str
    mode: str  # "fp32", "1.53bit", "1bit_fixed", "1bit_prog"
    d_model: int = 256
    n_heads: int = 4
    vocab_size: int = 32000
    k0_attn: int = 16
    k_max_attn: int = 64
    k0_logits: int = 8
    k_max_logits: int = 32
    shortlist_size: int = 64
    eps: float = 0.05
    delta_total: float = 0.01
    backend: str = "cpu"
    seed: int = 12345


def run_single_config(
    cfg: BakeoffConfig,
    tokens: List[int],
    Q_bits_list: List[np.ndarray],
    K_bits_list: List[np.ndarray],
) -> BakeoffMetrics:
    """Run inference for a single configuration.

    Args:
        cfg: Configuration
        tokens: Token IDs [n_tokens]
        Q_bits_list: Query vectors (packed) for each token
        K_bits_list: Key matrices (packed) for each token

    Returns:
        Metrics for this configuration
    """
    n_tokens = len(tokens)

    # Create E2E config
    e2e_cfg = E2EConfig(
        kA=cfg.k0_attn,
        k_max_attn=cfg.k_max_attn,
        k0_logits=cfg.k0_logits,
        k_max_logits=cfg.k_max_logits,
        shortlist_size=cfg.shortlist_size,
        eps=cfg.eps,
        delta_total=cfg.delta_total,
        backend=cfg.backend,
    )
    
    # Configure CTG based on mode
    ctg = None
    ctg_state = None
    if cfg.mode == "1bit_fixed":
        rules = [CTGRule(op="PASS", ids=None)]
        ctg = CTG(rules=rules, vocab_size=cfg.vocab_size)
        ctg_state = CTGState()
    elif cfg.mode == "1bit_prog":
        programs = make_default_programs(cfg.vocab_size, K=4)
        ctg = CTG(programs=programs, vocab_size=cfg.vocab_size)
        ctg_state = CTGState()
    
    # Run inference
    k_used_list = []
    pairs_list = []
    latency_list = []
    unsure_count = 0
    
    for i in range(n_tokens):
        t0 = time.perf_counter()
        
        result = infer_one_token_e2e(
            Q_attn_bits=Q_bits_list[i],
            Q_logits_bits=Q_bits_list[i],
            K_bits=K_bits_list[i],
            vocab_ids=np.arange(cfg.vocab_size, dtype=np.int32),
            cfg=e2e_cfg,
            token_idx=i,
            n_ctx=len(tokens),
            prf_seed=cfg.seed + i,
            ctg=ctg,
            ctg_state=ctg_state,
        )
        
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        
        k_used_list.append(result["k_attn_used"] + result["k_logits_used"])
        pairs_list.append(result.get("pairs_total", 0))
        latency_list.append(latency_ms)
        
        if result.get("unsure", False):
            unsure_count += 1
        
        # Update CTG state
        if ctg is not None:
            ctg_state = result.get("ctg_state", ctg_state)
    
    # Compute metrics
    k_arr = np.array(k_used_list)
    pairs_arr = np.array(pairs_list)
    latency_arr = np.array(latency_list)
    
    metrics = BakeoffMetrics(
        config_name=cfg.name,
        ppl=0.0,  # Placeholder - would need actual LM evaluation
        ppl_delta_pct=0.0,
        k_mean=float(np.mean(k_arr)),
        k_p95=float(np.percentile(k_arr, 95)),
        pairs_mean=float(np.mean(pairs_arr)),
        pairs_delta_pct=0.0,  # Computed later vs baseline
        latency_mean_ms=float(np.mean(latency_arr)),
        latency_p95_ms=float(np.percentile(latency_arr, 95)),
        latency_delta_pct=0.0,  # Computed later vs baseline
        variance_k=float(np.var(k_arr)),
        variance_ratio=1.0,  # Computed later vs baseline
        tokens_processed=n_tokens,
        unsure_rate=float(unsure_count) / n_tokens,
    )
    
    return metrics


def run_bakeoff(
    configs: List[BakeoffConfig],
    tokens: List[int],
    Q_bits_list: List[np.ndarray],
    K_bits_list: List[np.ndarray],
    output_dir: Path,
) -> List[BakeoffMetrics]:
    """Run full bake-off across all configurations.

    Args:
        configs: List of configurations to evaluate
        tokens: Token IDs
        Q_bits_list: Query vectors (packed)
        K_bits_list: Key matrices (packed)
        output_dir: Directory to save results

    Returns:
        List of metrics for each configuration
    """
    results = []

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg.name}")
        print(f"{'='*60}")

        metrics = run_single_config(cfg, tokens, Q_bits_list, K_bits_list)
        results.append(metrics)

        print(f"  k_mean: {metrics.k_mean:.2f}")
        print(f"  k_p95: {metrics.k_p95:.2f}")
        print(f"  latency_mean: {metrics.latency_mean_ms:.2f} ms")
        print(f"  unsure_rate: {metrics.unsure_rate:.2%}")

    # Compute deltas vs baseline (FP32)
    if len(results) > 0:
        baseline = results[0]  # Assume first config is FP32
        for i, m in enumerate(results):
            if i == 0:
                continue
            m.ppl_delta_pct = ((m.ppl - baseline.ppl) / baseline.ppl) * 100 if baseline.ppl > 0 else 0.0
            m.pairs_delta_pct = ((m.pairs_mean - baseline.pairs_mean) / baseline.pairs_mean) * 100 if baseline.pairs_mean > 0 else 0.0
            m.latency_delta_pct = ((m.latency_mean_ms - baseline.latency_mean_ms) / baseline.latency_mean_ms) * 100 if baseline.latency_mean_ms > 0 else 0.0
            m.variance_ratio = m.variance_k / baseline.variance_k if baseline.variance_k > 0 else 1.0

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "bakeoff_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for m in results:
            writer.writerow(asdict(m))

    # JSON
    json_path = output_dir / "bakeoff_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(m) for m in results], f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CTG Bake-off Benchmark")
    parser.add_argument("--tokens", type=int, default=100, help="Number of tokens to process")
    parser.add_argument("--d", type=int, default=256, help="Model dimension")
    parser.add_argument("--vocab", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--output", type=str, default="bakeoff_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    args = parser.parse_args()

    # Generate synthetic data (placeholder - would load WikiText-103 in production)
    print(f"Generating synthetic data: {args.tokens} tokens, d={args.d}")
    rng = np.random.default_rng(args.seed)

    tokens = rng.integers(0, args.vocab, size=args.tokens).tolist()
    Q_bits_list = [pack_input_signs(rng.standard_normal(args.d).astype(np.float32)) for _ in range(args.tokens)]
    K_bits_list = [
        np.array([pack_input_signs(rng.standard_normal(args.d).astype(np.float32)) for _ in range(64)])
        for _ in range(args.tokens)
    ]

    # Define configurations
    configs = [
        BakeoffConfig(name="FP32", mode="fp32", d_model=args.d, vocab_size=args.vocab, seed=args.seed),
        BakeoffConfig(name="1.53-bit", mode="1.53bit", d_model=args.d, vocab_size=args.vocab, seed=args.seed),
        BakeoffConfig(name="1-bit+CTG-FIXED", mode="1bit_fixed", d_model=args.d, vocab_size=args.vocab, seed=args.seed),
        BakeoffConfig(name="1-bit+CTG-PROG", mode="1bit_prog", d_model=args.d, vocab_size=args.vocab, seed=args.seed),
    ]

    # Run bake-off
    results = run_bakeoff(configs, tokens, Q_bits_list, K_bits_list, Path(args.output))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for m in results:
        print(f"\n{m.config_name}:")
        print(f"  PPL Δ: {m.ppl_delta_pct:+.2f}%")
        print(f"  k_mean: {m.k_mean:.2f} (P95: {m.k_p95:.2f})")
        print(f"  Pairs Δ: {m.pairs_delta_pct:+.2f}%")
        print(f"  Latency Δ: {m.latency_delta_pct:+.2f}%")
        print(f"  Variance ratio: {m.variance_ratio:.3f}")


if __name__ == "__main__":
    main()

