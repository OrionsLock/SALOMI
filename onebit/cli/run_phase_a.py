"""Phase A: FP32 backbone + 1-bit CTG logits head.

Prove CTG works for precision recovery in the most critical layer.

Comparison:
1. FP32 logits (baseline)
2. 1-bit logits (no CTG)
3. 1-bit logits + CTG-FIXED
4. 1-bit logits + CTG-PROG
5. 1.53-bit ternary logits
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from transformers import GPT2LMHeadModel
import torch

from onebit.data.wikitext import load_wikitext103
from onebit.eval.perplexity import compute_perplexity, PerplexityResult
from onebit.model.hybrid_logits import OneBitLogitsHead, TernaryLogitsHead, LogitsConfig


def run_fp32_logits(model: GPT2LMHeadModel, dataset, seq_len: int, max_tokens: int) -> PerplexityResult:
    """Run with FP32 logits (baseline)."""
    print("\n" + "=" * 60)
    print("Running: FP32 Logits (Baseline)")
    print("=" * 60)
    
    def forward_fn(input_ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids]))
            logits = outputs.logits[0].numpy()  # [seq_len, vocab_size]
        return logits
    
    result = compute_perplexity(dataset, forward_fn, seq_len=seq_len, max_tokens=max_tokens)
    return result


def run_1bit_logits(
    model: GPT2LMHeadModel,
    dataset,
    seq_len: int,
    max_tokens: int,
    use_ctg: bool = False,
    T: int = 16,
    seed: int = 0,
) -> PerplexityResult:
    """Run with 1-bit logits."""
    ctg_str = "+CTG" if use_ctg else ""
    print("\n" + "=" * 60)
    print(f"Running: 1-bit Logits{ctg_str} (T={T})")
    print("=" * 60)
    
    # Get wte matrix from model
    wte_fp32 = model.transformer.wte.weight.detach().cpu().numpy()  # [vocab_size, d_model]
    
    # Create 1-bit logits head
    logits_head = OneBitLogitsHead(
        wte_fp32,
        LogitsConfig(T=T, use_ctg=use_ctg, seed=seed),
    )
    
    def forward_fn(input_ids: np.ndarray) -> np.ndarray:
        # Get hidden states from FP32 backbone
        with torch.no_grad():
            outputs = model.transformer(torch.tensor([input_ids]))
            hidden_states = outputs.last_hidden_state[0].numpy()  # [seq_len, d_model]
        
        # Compute logits using 1-bit head
        logits = logits_head.forward(hidden_states)  # [seq_len, vocab_size]
        return logits
    
    result = compute_perplexity(dataset, forward_fn, seq_len=seq_len, max_tokens=max_tokens)
    return result


def run_ternary_logits(model: GPT2LMHeadModel, dataset, seq_len: int, max_tokens: int) -> PerplexityResult:
    """Run with 1.53-bit ternary logits."""
    print("\n" + "=" * 60)
    print("Running: 1.53-bit Ternary Logits")
    print("=" * 60)
    
    # Get wte matrix from model
    wte_fp32 = model.transformer.wte.weight.detach().cpu().numpy()
    
    # Create ternary logits head
    logits_head = TernaryLogitsHead(wte_fp32)
    
    def forward_fn(input_ids: np.ndarray) -> np.ndarray:
        # Get hidden states from FP32 backbone
        with torch.no_grad():
            outputs = model.transformer(torch.tensor([input_ids]))
            hidden_states = outputs.last_hidden_state[0].numpy()  # [seq_len, d_model]
        
        # Compute logits using ternary head
        logits = logits_head.forward(hidden_states)  # [seq_len, vocab_size]
        return logits
    
    result = compute_perplexity(dataset, forward_fn, seq_len=seq_len, max_tokens=max_tokens)
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase A: FP32 backbone + 1-bit CTG logits")
    parser.add_argument("--output", type=str, default="out/phase_a", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max tokens to evaluate")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--T", type=int, default=16, help="BSDM-W samples")
    parser.add_argument("--quick", action="store_true", help="Quick test with 100 tokens")
    args = parser.parse_args()
    
    if args.quick:
        args.max_tokens = 100
        args.T = 4
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading WikiText-103 dataset...")
    dataset = load_wikitext103(split="test", max_tokens=args.max_tokens, cache_dir=str(output_dir))
    
    # Load FP32 model
    print("\nLoading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    # Run experiments
    results = {}
    
    # 1. FP32 baseline
    results["fp32"] = run_fp32_logits(model, dataset, args.seq_len, args.max_tokens)
    
    # 2. 1-bit logits (no CTG)
    results["1bit"] = run_1bit_logits(model, dataset, args.seq_len, args.max_tokens, use_ctg=False, T=args.T, seed=0x1111)
    
    # 3. 1-bit logits + CTG
    results["1bit_ctg"] = run_1bit_logits(model, dataset, args.seq_len, args.max_tokens, use_ctg=True, T=args.T, seed=0x2222)
    
    # 4. 1.53-bit ternary
    results["ternary"] = run_ternary_logits(model, dataset, args.seq_len, args.max_tokens)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PHASE A RESULTS: FP32 Backbone + Quantized Logits Head")
    print("=" * 80)
    print(f"{'Config':<25} {'PPL':>12} {'CE':>10} {'Tokens':>10} {'Time':>10} {'Tok/s':>10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<25} {result.perplexity:>12.2f} {result.cross_entropy:>10.4f} "
              f"{result.n_tokens:>10} {result.total_time:>10.2f} {result.tokens_per_sec:>10.2f}")
    
    print("=" * 80)
    
    # Save results
    results_json = {
        name: {
            "perplexity": float(r.perplexity),
            "cross_entropy": float(r.cross_entropy),
            "n_tokens": int(r.n_tokens),
            "total_time": float(r.total_time),
            "tokens_per_sec": float(r.tokens_per_sec),
        }
        for name, r in results.items()
    }
    
    with open(output_dir / "phase_a_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'phase_a_results.json'}")


if __name__ == "__main__":
    main()

