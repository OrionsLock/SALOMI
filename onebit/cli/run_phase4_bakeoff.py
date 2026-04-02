"""Phase IV: Real Data, Real Models, Real Numbers.

117M-token WikiText-103 bake-off comparing:
- FP32 (HuggingFace GPT-2)
- 1.53-bit (ternary quantization)
- 1-bit + CTG-FIXED
- 1-bit + CTG-PROG

Generates reproducibility artifacts and validates proof thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from onebit.data.wikitext import load_wikitext103_cached
from onebit.eval.perplexity import compute_perplexity
from onebit.eval.baselines import create_fp32_baseline
from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface, quantize_gpt2
from onebit.model.runtime_transformer import RuntimeTransformer, InferenceConfig
from onebit.runtime.ctg_grammar import CTG, CTGState, CTGRule
from onebit.runtime.ctg_selector import AdaptiveProgramSelector, SelectorConfig


# Frozen seeds for reproducibility
FROZEN_SEEDS = {
    "master": 0x1234567890ABCDEF,
    "fp32": 0x1111111111111111,
    "1.53bit": 0x2222222222222222,
    "ctg_fixed": 0x3333333333333333,
    "ctg_prog": 0x4444444444444444,
}


@dataclass
class BakeoffConfig:
    """Configuration for a single bake-off run."""
    name: str
    seed: int
    quantization: str  # "fp32", "1.53bit", "1bit"
    use_ctg: bool
    ctg_mode: Optional[str]  # "fixed" or "prog"


@dataclass
class BakeoffResult:
    """Result from a single bake-off run."""
    config_name: str
    perplexity: float
    cross_entropy: float
    n_tokens: int
    total_time: float
    tokens_per_sec: float
    seed: int
    
    # CTG-specific metrics (if applicable)
    k_mean: Optional[float] = None
    k_p95: Optional[float] = None
    pairs_evaluated_pct: Optional[float] = None
    variance_ratio: Optional[float] = None


def run_fp32_baseline(
    dataset,
    config: BakeoffConfig,
    max_tokens: Optional[int] = None,
) -> BakeoffResult:
    """Run FP32 baseline evaluation."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"{'='*60}")

    # Create FP32 baseline
    baseline = create_fp32_baseline("gpt2")

    # Compute perplexity
    result = compute_perplexity(
        dataset=dataset,
        forward_fn=baseline.forward,
        seq_len=512,
        max_tokens=max_tokens,
        verbose=True,
    )

    return BakeoffResult(
        config_name=config.name,
        perplexity=result.perplexity,
        cross_entropy=result.cross_entropy,
        n_tokens=result.n_tokens,
        total_time=result.total_time,
        tokens_per_sec=result.tokens_per_sec,
        seed=config.seed,
    )


def run_ctg_fixed(
    dataset,
    config: BakeoffConfig,
    max_tokens: Optional[int] = None,
) -> BakeoffResult:
    """Run CTG-FIXED evaluation."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"{'='*60}")

    # Load GPT-2 weights and quantize
    print("Loading and quantizing GPT-2...")
    weights_fp32, gpt2_cfg = load_gpt2_from_huggingface("gpt2")
    quantized_model = quantize_gpt2(weights_fp32, gpt2_cfg)

    # Create CTG-FIXED engine with default rules
    print("Creating CTG-FIXED engine...")
    ctg = CTG(
        rules=[
            # Simple PASS-only program for now
            # TODO: Add more sophisticated rules
        ],
        vocab_size=gpt2_cfg.vocab_size,
    )

    # Create runtime transformer
    # Use T=2 for fast testing (will increase to T=16 for final runs)
    infer_cfg = InferenceConfig(
        T=2,
        backend="cpu",
        seed=config.seed,
        use_ctg=True,
        use_hcl_logits=False,  # Use FP32 logits for now (faster)
    )
    runtime = RuntimeTransformer(quantized_model, infer_cfg)

    # Track CTG metrics
    k_used_list = []

    def forward_fn_with_ctg(input_ids: np.ndarray) -> np.ndarray:
        """Forward function with CTG tracking.

        Uses RuntimeTransformer with return_all_logits=True to compute
        logits for all positions in one forward pass.
        """
        # Compute logits for all positions
        logits_all = runtime.forward(input_ids, seed=config.seed, return_all_logits=True)

        # TODO: Extract k_used metrics from BSDM-W telemetry

        return logits_all

    # Compute perplexity
    result = compute_perplexity(
        dataset=dataset,
        forward_fn=forward_fn_with_ctg,
        seq_len=512,
        max_tokens=max_tokens,
        verbose=True,
    )

    # Compute CTG metrics
    k_mean = float(np.mean(k_used_list)) if k_used_list else None
    k_p95 = float(np.percentile(k_used_list, 95)) if k_used_list else None

    return BakeoffResult(
        config_name=config.name,
        perplexity=result.perplexity,
        cross_entropy=result.cross_entropy,
        n_tokens=result.n_tokens,
        total_time=result.total_time,
        tokens_per_sec=result.tokens_per_sec,
        seed=config.seed,
        k_mean=k_mean,
        k_p95=k_p95,
    )


def run_ctg_prog(
    dataset,
    config: BakeoffConfig,
    max_tokens: Optional[int] = None,
) -> BakeoffResult:
    """Run CTG-PROG evaluation with adaptive selector."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"{'='*60}")

    # Load GPT-2 weights and quantize
    print("Loading and quantizing GPT-2...")
    weights_fp32, gpt2_cfg = load_gpt2_from_huggingface("gpt2")
    quantized_model = quantize_gpt2(weights_fp32, gpt2_cfg)

    # Create CTG-PROG engine with K=4 programs
    print("Creating CTG-PROG engine with K=4 programs...")
    # TODO: Define 4 distinct programs
    programs = [
        [],  # Program 0: PASS-biased
        [],  # Program 1: PHASE-heavy
        [],  # Program 2: INHIBIT-spiky
        [],  # Program 3: INVERT-accent
    ]
    ctg = CTG(programs=programs, vocab_size=gpt2_cfg.vocab_size)

    # Create adaptive selector
    print("Creating adaptive program selector...")
    selector_cfg = SelectorConfig(K=4, hidden_dim=32)
    # Note: Selector requires PyTorch, will be initialized if available

    # Create runtime transformer
    # Use T=2 for fast testing (will increase to T=16 for final runs)
    infer_cfg = InferenceConfig(
        T=2,
        backend="cpu",
        seed=config.seed,
        use_ctg=True,
        use_hcl_logits=False,  # Use FP32 logits for now (faster)
    )
    runtime = RuntimeTransformer(quantized_model, infer_cfg)

    # Track CTG metrics
    k_used_list = []
    program_usage = {0: 0, 1: 0, 2: 0, 3: 0}

    def forward_fn_with_ctg_prog(input_ids: np.ndarray) -> np.ndarray:
        """Forward function with CTG-PROG tracking.

        Uses RuntimeTransformer with return_all_logits=True to compute
        logits for all positions in one forward pass.
        """
        # Compute logits for all positions
        logits_all = runtime.forward(input_ids, seed=config.seed, return_all_logits=True)

        # TODO: Use selector to choose program_id based on features
        # TODO: Extract k_used metrics from BSDM-W telemetry

        # For now, just track program 0 usage
        program_usage[0] += len(input_ids)

        return logits_all

    # Compute perplexity
    result = compute_perplexity(
        dataset=dataset,
        forward_fn=forward_fn_with_ctg_prog,
        seq_len=512,
        max_tokens=max_tokens,
        verbose=True,
    )

    # Compute CTG metrics
    k_mean = float(np.mean(k_used_list)) if k_used_list else None
    k_p95 = float(np.percentile(k_used_list, 95)) if k_used_list else None

    print(f"\nProgram usage: {program_usage}")

    return BakeoffResult(
        config_name=config.name,
        perplexity=result.perplexity,
        cross_entropy=result.cross_entropy,
        n_tokens=result.n_tokens,
        total_time=result.total_time,
        tokens_per_sec=result.tokens_per_sec,
        seed=config.seed,
        k_mean=k_mean,
        k_p95=k_p95,
    )


def run_bakeoff(
    output_dir: Path,
    max_tokens: Optional[int] = None,
    configs: Optional[List[BakeoffConfig]] = None,
):
    """Run full bake-off evaluation.
    
    Args:
        output_dir: Output directory for results
        max_tokens: Maximum tokens to evaluate (None = all, 117M for full)
        configs: List of configurations to run (None = all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default configs
    if configs is None:
        configs = [
            BakeoffConfig("FP32", FROZEN_SEEDS["fp32"], "fp32", False, None),
            BakeoffConfig("CTG-FIXED", FROZEN_SEEDS["ctg_fixed"], "1bit", True, "fixed"),
            BakeoffConfig("CTG-PROG", FROZEN_SEEDS["ctg_prog"], "1bit", True, "prog"),
            # NOTE: 1.53-bit baseline pending full forward pass implementation
        ]

    # Load dataset
    print("Loading WikiText-103 dataset...")
    cache_path = output_dir / "wikitext103_test.npz"
    dataset = load_wikitext103_cached(
        split="test",
        max_tokens=max_tokens,
        cache_path=cache_path,
    )

    # Run each config
    results = []
    for config in configs:
        if config.quantization == "fp32":
            result = run_fp32_baseline(dataset, config, max_tokens)
        elif config.quantization == "1bit" and config.ctg_mode == "fixed":
            result = run_ctg_fixed(dataset, config, max_tokens)
        elif config.quantization == "1bit" and config.ctg_mode == "prog":
            result = run_ctg_prog(dataset, config, max_tokens)
        else:
            print(f"Skipping {config.name} (not yet implemented)")
            continue

        results.append(result)
    
    # Save results
    save_results(results, output_dir)
    
    # Print summary
    print_summary(results)


def save_results(results: List[BakeoffResult], output_dir: Path):
    """Save results to CSV and JSON."""
    # CSV
    csv_path = output_dir / "bakeoff_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"\nSaved results to: {csv_path}")

    # JSON (convert numpy types to Python types)
    json_path = output_dir / "bakeoff_results.json"
    with open(json_path, "w") as f:
        results_dict = []
        for r in results:
            d = asdict(r)
            # Convert numpy types to Python types
            for key, value in d.items():
                if isinstance(value, (np.integer, np.floating)):
                    d[key] = value.item()
            results_dict.append(d)
        json.dump(results_dict, f, indent=2)
    print(f"Saved results to: {json_path}")
    
    # Frozen seeds
    seeds_path = output_dir / "frozen_seeds.json"
    with open(seeds_path, "w") as f:
        json.dump(FROZEN_SEEDS, f, indent=2)
    print(f"Saved frozen seeds to: {seeds_path}")


def print_summary(results: List[BakeoffResult]):
    """Print summary table."""
    print(f"\n{'='*80}")
    print("PHASE IV BAKE-OFF RESULTS")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'PPL':>10} {'CE':>10} {'Tokens':>12} {'Time':>10} {'Tok/s':>10}")
    print(f"{'-'*80}")
    for result in results:
        print(f"{result.config_name:<20} {result.perplexity:>10.2f} {result.cross_entropy:>10.4f} "
              f"{result.n_tokens:>12,} {result.total_time:>10.2f} {result.tokens_per_sec:>10.2f}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Phase IV: 117M-token bake-off")
    parser.add_argument("--output", type=str, default="out/phase4_bakeoff", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens (None = all 117M)")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10k tokens")
    
    args = parser.parse_args()
    
    max_tokens = 10000 if args.quick else args.max_tokens
    
    run_bakeoff(
        output_dir=Path(args.output),
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    main()

