"""CLI demo for end-to-end inference: Attention → KV → Logits."""
from __future__ import annotations

import argparse
import time
import numpy as np

from ..runtime.controller_e2e import infer_one_token_e2e, E2EConfig
from ..core.packbits import pack_input_signs


def main():
    parser = argparse.ArgumentParser(description="End-to-end inference demo")
    parser.add_argument("--n-ctx", type=int, default=128, help="Context length")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--d-attn", type=int, default=512, help="Attention dimension")
    parser.add_argument("--d-kv", type=int, default=512, help="KV dimension")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "opencl"], help="Backend")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kA", type=int, default=16, help="Stage-A ticks")
    parser.add_argument("--k-max-attn", type=int, default=64, help="Max ticks for attention")
    parser.add_argument("--top-k-kv", type=int, default=8, help="Top-k KV positions")
    parser.add_argument("--shortlist-size", type=int, default=32, help="Logits shortlist size")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("End-to-End Inference Demo: Attention → KV → Logits")
    print("=" * 80)
    print(f"Context length: {args.n_ctx}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Dimensions: d_attn={args.d_attn}, d_kv={args.d_kv}, d_model={args.d_model}")
    print(f"Backend: {args.backend}")
    print(f"Seed: {args.seed}")
    print()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create synthetic data
    print("Creating synthetic data...")
    
    # Attention query and keys
    Q_attn = np.random.randn(args.d_attn)
    K_attn = np.random.randn(args.n_ctx, args.d_attn)
    
    # Make position 10 highly correlated with query
    K_attn[10] = Q_attn + 0.1 * np.random.randn(args.d_attn)
    
    Q_attn_bits = pack_input_signs(Q_attn)
    K_attn_bits = np.array([pack_input_signs(K_attn[i]) for i in range(args.n_ctx)])
    
    # KV cache
    K_kv = np.random.randn(args.n_ctx, args.d_kv)
    V_kv = np.random.randn(args.n_ctx, args.d_kv)
    
    K_kv_bits = np.array([pack_input_signs(K_kv[i]) for i in range(args.n_ctx)])
    V_kv_bits = np.array([pack_input_signs(V_kv[i]) for i in range(args.n_ctx)])
    
    # Logits query and vocabulary
    Q_logits = np.random.randn(args.d_model)
    Q_logits_bits = pack_input_signs(Q_logits)
    
    vocab_ids = np.arange(args.vocab_size, dtype=np.int32)
    
    print(f"  Q_attn: {Q_attn_bits.shape}")
    print(f"  K_attn: {K_attn_bits.shape}")
    print(f"  K_kv: {K_kv_bits.shape}")
    print(f"  V_kv: {V_kv_bits.shape}")
    print(f"  Q_logits: {Q_logits_bits.shape}")
    print(f"  vocab_ids: {vocab_ids.shape}")
    print()
    
    # Create config
    cfg = E2EConfig(
        kA=args.kA,
        k_max_attn=args.k_max_attn,
        d_kv=args.d_kv,
        top_k_kv=args.top_k_kv,
        shortlist_size=args.shortlist_size,
        backend=args.backend,
    )
    
    # Run inference
    print("Running end-to-end inference...")
    t0 = time.perf_counter()
    
    result = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits,
        K_kv_bits, V_kv_bits,
        Q_logits_bits, vocab_ids,
        cfg=cfg,
        prf_seed=args.seed,
        d_attn=args.d_attn,
        d_model=args.d_model,
    )
    
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Unsure: {result['unsure']}")
    print()
    print(f"Attention:")
    print(f"  Top-1 position: {result['attn_top1']}")
    print(f"  Ticks used: {result['k_attn_used']}")
    print()
    print(f"KV Retrieval:")
    print(f"  Retrieved positions: {result['kv_positions']}")
    print(f"  Ticks used: {result['k_kv_used']}")
    print()
    print(f"Logits:")
    print(f"  Top-1 token ID: {result['logits_top1']}")
    print(f"  Ticks used: {result['k_logits_used']}")
    print()
    print(f"Total time: {elapsed_ms:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()

