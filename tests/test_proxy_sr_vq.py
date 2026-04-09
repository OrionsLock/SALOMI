#!/usr/bin/env python3
"""Proxy-SR-VQ end-to-end validation script.

Runs the "weekend validation" path from the proposal:
  1. Load GPT-2 124M and at least one larger variant.
  2. Compute Redun Scores for each, same calibration budget.
  3. Run dynamic allocation + block-wise calibration.
  4. Compare: does higher Redun Score correlate with better compression?
  5. Fit scaling law from available models.
  6. Report: PPL, BPP, Redun Score distributions, scaling curve.

Usage:
    python tests/test_proxy_sr_vq.py [--sizes 124M,355M] [--device cpu]
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from typing import Dict, List, Tuple
from functools import partial

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

_orig_print = print
print = partial(_orig_print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onebit.quantization.redun_score import RedunScoreComputer
from onebit.quantization.dynamic_allocator import DynamicAllocator
from onebit.quantization.ternary_sparse import TernarySparse
from onebit.quantization.hessian_vq import HessianVQ
from onebit.proxy.model_factory import load_proxy_family, get_model_info, ModelAdapter
from onebit.proxy.scaling_law import ScalingLawFitter
from onebit.proxy.qat_loop import BlockCalibrator, QATConfig, QATLoop
from onebit.proxy.policy_export import export_policy, PolicyMetadata

CALIB_TEXTS = [
    "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
    "Quantization reduces model size by representing weights with fewer bits, trading precision for efficiency.",
    "Language models learn statistical patterns in text data to predict the next token in a sequence.",
    "Gradient descent optimizes neural network parameters by following the direction of steepest loss decrease.",
    "The attention mechanism allows models to focus on relevant parts of the input when generating each output.",
    "Matrix multiplication is the core computational operation in deep neural networks and transformers.",
    "Perplexity measures how well a probability model predicts a sample and is commonly used to evaluate language models.",
    "Binary quantization represents each weight with a single bit, achieving extreme compression at the cost of accuracy.",
    "The softmax function converts logits into a probability distribution over the vocabulary.",
    "Hessian information captures the curvature of the loss landscape and guides quantization sensitivity analysis.",
    "Modern GPUs accelerate matrix operations through massive parallelism and specialized tensor cores.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task using a smaller dataset.",
    "The residual connection in transformers helps gradient flow and enables training of very deep networks.",
    "Entropy coding compresses data by assigning shorter codes to more frequent symbols.",
    "Weight pruning removes less important connections to create sparse, efficient neural networks.",
    "The learning rate controls the step size during gradient descent optimization of neural networks.",
    "Batch normalization stabilizes training by normalizing layer inputs to have zero mean and unit variance.",
    "Knowledge distillation trains a smaller student model to mimic the behavior of a larger teacher model.",
    "The embedding layer maps discrete tokens to continuous vector representations in a high-dimensional space.",
    "Calibration data provides representative inputs for post-training quantization parameter optimization.",
    "Vector quantization groups weights into clusters and represents each group with a shared centroid.",
    "The cross-entropy loss function measures the difference between predicted and true probability distributions.",
    "Low-rank factorization approximates weight matrices as products of smaller matrices to reduce parameters.",
    "Tokenization breaks input text into subword units that form the vocabulary of a language model.",
    "Activation functions introduce nonlinearity into neural networks, enabling them to learn complex patterns.",
    "Mixed-precision training uses different numerical formats for different operations to balance speed and accuracy.",
    "The Hessian diagonal approximation provides per-parameter sensitivity estimates for quantization.",
    "Dropout randomly zeroes activations during training to prevent overfitting and improve generalization.",
    "Scale-aware quantization adjusts bit allocation based on the sensitivity of each layer or block.",
    "The feed-forward network in each transformer block applies two linear transformations with a nonlinearity.",
    "Straight-through estimators approximate gradients through non-differentiable quantization operations.",
    "Model compression enables deployment of large language models on resource-constrained devices.",
]


def compute_perplexity_internal(model, tokenizer, texts: List[str], max_len: int = 128) -> float:
    """Lightweight perplexity computation using internal text."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            ids = {k: v.to(device) for k, v in ids.items()}
            outputs = model(**ids, labels=ids["input_ids"])
            n_tokens = ids["input_ids"].numel()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / (total_tokens + 1e-10)
    return math.exp(min(avg_loss, 20.0))


def run_proxy_sweep(
    sizes: List[str],
    device: str = "cpu",
    dtype: str = None,
    skip_calib: bool = False,
) -> Dict[str, Dict]:
    """Run the full proxy sweep and return structured results."""

    print("=" * 70)
    print("PROXY-SR-VQ VALIDATION SWEEP")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load models
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading proxy models ---")
    family = load_proxy_family(sizes=sizes, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Step 2: Compute Redun Scores
    # ------------------------------------------------------------------
    print("\n--- Step 2: Computing Redun Scores ---")
    redun = RedunScoreComputer(alpha=1.0, beta=1.0, gamma=1.0)
    all_redun: Dict[str, Dict] = {}
    model_sizes: Dict[str, int] = {}

    for label, (model, tokenizer, info, adapter) in family.items():
        print(f"\n  Probing {label} (arch={adapter.arch})...")
        n_probe = min(16, 8 if info.n_params > 1_000_000_000 else 16)
        scores = redun.compute_model_redun(
            model, tokenizer, CALIB_TEXTS, n_samples=n_probe, max_len=64,
            adapter=adapter,
        )
        all_redun[label] = scores
        model_sizes[label] = info.n_params

        attn_scores = []
        mlp_scores = []
        for lk, comps in scores.items():
            for comp, r in comps.items():
                if comp.startswith("attn"):
                    attn_scores.append(r.redun_score)
                else:
                    mlp_scores.append(r.redun_score)
        print(f"    Mean Redun -- attn: {np.mean(attn_scores):.4f}, mlp: {np.mean(mlp_scores):.4f}")

    # ------------------------------------------------------------------
    # Step 3: Dynamic allocation
    # ------------------------------------------------------------------
    print("\n--- Step 3: Dynamic Allocation ---")
    allocator = DynamicAllocator(redun_threshold=1.0)
    configs: Dict[str, any] = {}

    for label in sizes:
        model, tokenizer, info, adapter = family[label]
        config = allocator.allocate(
            all_redun[label], n_layers=info.n_layers, target_bpp=1.2,
        )
        configs[label] = config
        hvq_count = sum(
            1 for b in config.layer_budgets
            if b.attn_method == "hessian_vq" or b.mlp_method == "hessian_vq"
        )
        ts_count = sum(
            1 for b in config.layer_budgets
            if b.attn_method == "ternary_sparse" or b.mlp_method == "ternary_sparse"
        )
        print(f"  {label}: {hvq_count} HessianVQ, {ts_count} TernarySparse, "
              f"{len(config.layer_budgets)} layers")

    # ------------------------------------------------------------------
    # Step 4: Block-wise calibration (Phase 1) on smallest model
    # ------------------------------------------------------------------
    smallest = sizes[0]
    model_sm, tok_sm, info_sm, adapter_sm = family[smallest]
    calib_results = []

    if not skip_calib:
        print(f"\n--- Step 4: Block-wise Calibration (Phase 1 on {smallest}) ---")
        model_calib = copy.deepcopy(model_sm).float()
        adapter_calib = ModelAdapter(model_calib)

        ppl_before = compute_perplexity_internal(model_calib, tok_sm, CALIB_TEXTS[:8])
        print(f"  PPL before calibration: {ppl_before:.2f}")

        calibrator = BlockCalibrator(n_calib_samples=32, max_len=64)
        calib_results = calibrator.calibrate_model(
            model_calib, tok_sm, configs[smallest], CALIB_TEXTS, device,
            adapter=adapter_calib,
        )

        ppl_after = compute_perplexity_internal(model_calib, tok_sm, CALIB_TEXTS[:8])
        print(f"  PPL after calibration: {ppl_after:.2f}")
        del model_calib
    else:
        print("\n--- Step 4: Calibration skipped (--skip-calib) ---")

    # ------------------------------------------------------------------
    # Step 5: Scaling law fit
    # ------------------------------------------------------------------
    print("\n--- Step 5: Scaling Law Fit ---")
    proxy_flat: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, scores in all_redun.items():
        proxy_flat[label] = {}
        for lk, comps in scores.items():
            proxy_flat[label][lk] = {c: r.redun_score for c, r in comps.items()}

    fitter = ScalingLawFitter()
    fitted = fitter.fit_redun_scaling(proxy_flat, model_sizes)

    for ctype, fp in fitted.items():
        print(f"  {ctype}: R = {fp.a:.4f} * ln(N) + {fp.b:.4f}  (R²={fp.r_squared:.3f}, n={fp.n_points})")

    if len(sizes) >= 2:
        pred_70b = fitter.predict_redun(70_000_000_000)
        print(f"  Extrapolation to 70B: {pred_70b}")

    try:
        fig = fitter.plot_scaling_curve(
            output_path="scaling_curve.png",
            target_sizes=[7_000_000_000, 70_000_000_000],
        )
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    # ------------------------------------------------------------------
    # Step 6: Compression quality (Redun-only, no deep copy for big models)
    # ------------------------------------------------------------------
    print("\n--- Step 6: Redun Score vs Compression Quality ---")
    quality_data: Dict[str, Dict] = {}

    for label in sizes:
        model, tok, info, adapter = family[label]
        ppl_fp = compute_perplexity_internal(model, tok, CALIB_TEXTS[:8])

        scores = all_redun[label]
        mean_redun = np.mean([r.redun_score for lk in scores for r in scores[lk].values()])

        quality_data[label] = {
            "n_params": info.n_params,
            "mean_redun": float(mean_redun),
            "ppl_fp": ppl_fp,
        }
        print(f"  {label}: Redun={mean_redun:.4f}, PPL_fp={ppl_fp:.2f}")

    if len(quality_data) >= 2:
        redun_vals = [v["mean_redun"] for v in quality_data.values()]
        ppl_vals = [v["ppl_fp"] for v in quality_data.values()]
        if np.std(redun_vals) > 1e-10 and np.std(ppl_vals) > 1e-10:
            corr = np.corrcoef(redun_vals, ppl_vals)[0, 1]
            print(f"\n  Correlation(Redun, PPL_fp): {corr:.4f}")

    # ------------------------------------------------------------------
    # Step 7: Policy export
    # ------------------------------------------------------------------
    print("\n--- Step 7: Exporting Policy ---")
    meta = PolicyMetadata(
        model_name="Proxy sweep",
        n_params=model_sizes[smallest],
        n_layers=info_sm.n_layers,
        target_bpp=1.2,
        proxy_sizes_used=sizes,
        redun_threshold=1.0,
    )
    policy_path = export_policy(
        configs[smallest],
        all_redun[smallest],
        scaling_params=fitter.to_dict(),
        metadata=meta,
        output_path="proxy_sr_vq_policy.json",
    )
    print(f"  Policy written to {policy_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label in sizes:
        q = quality_data.get(label, {})
        print(f"  {label:>12s}  params={model_sizes[label]:>14,}  "
              f"redun={q.get('mean_redun', 0):.4f}  "
              f"ppl_fp={q.get('ppl_fp', 0):.2f}")

    for ctype, fp in fitted.items():
        print(f"  Scaling [{ctype}]: R = {fp.a:.4f}*ln(N) + {fp.b:.4f}  R²={fp.r_squared:.3f}")

    return {
        "quality": quality_data,
        "scaling": fitter.to_dict(),
        "calibration": [
            {"layer": r.layer_idx, "mse_before": r.mse_before, "mse_after": r.mse_after, "bpp": r.bpp}
            for r in calib_results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Proxy-SR-VQ validation sweep")
    parser.add_argument("--sizes", default="124M,355M",
                        help="Comma-separated model labels (see MODEL_CATALOG)")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--dtype", default=None, choices=["float16", "bfloat16"],
                        help="Load large models in reduced precision")
    parser.add_argument("--skip-calib", action="store_true",
                        help="Skip Phase 1 block calibration (faster for probing)")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]

    t0 = time.time()
    results = run_proxy_sweep(
        sizes=sizes, device=args.device, dtype=args.dtype,
        skip_calib=args.skip_calib,
    )
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.1f}s")

    import json
    with open("proxy_sr_vq_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Results written to proxy_sr_vq_results.json")


if __name__ == "__main__":
    main()
