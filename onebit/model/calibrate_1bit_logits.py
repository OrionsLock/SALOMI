"""Calibration harness for 1-bit logits head.

Collects (h_t, y_t, ẑ_t) on calibration data and fits global scaling factors.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from transformers import GPT2LMHeadModel
from tqdm import tqdm

from onebit.data.wikitext import load_wikitext103
from onebit.core.packbits import pack_input_signs, pack_signs_rowmajor
from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig


def collect_calibration_data(
    model: GPT2LMHeadModel,
    wte_fp32: np.ndarray,
    wte_bits: np.ndarray,
    dataset,
    max_tokens: int = 5000,
    T: int = 64,
    vocab_sample_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect calibration data: hidden states, FP32 logits, 1-bit estimates.
    
    Args:
        model: FP32 GPT-2 model
        wte_fp32: FP32 token embeddings [vocab_size, d_model]
        wte_bits: 1-bit token embeddings [vocab_size, ceil(d_model/32)]
        dataset: WikiText dataset
        max_tokens: Number of tokens to collect
        T: Number of BSDM-W samples
        vocab_sample_size: Sample this many vocab tokens per hidden state (for speed)
        
    Returns:
        hidden_states: [n_samples, d_model]
        logits_fp32: [n_samples, vocab_sample_size]
        logits_1bit_raw: [n_samples, vocab_sample_size] (raw BSDM-W estimates in [-1, 1])
    """
    vocab_size, d_model = wte_fp32.shape
    
    # Sample vocab indices to evaluate (for speed)
    np.random.seed(42)
    vocab_indices = np.random.choice(vocab_size, size=vocab_sample_size, replace=False)
    
    hidden_states_list = []
    logits_fp32_list = []
    logits_1bit_raw_list = []
    
    # ΣΔ configuration
    sd_cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0, walsh_N=2, antithetic=False)
    
    n_collected = 0

    print(f"Collecting calibration data: {max_tokens} tokens, {vocab_sample_size} vocab samples, T={T}")

    # Iterate over batches
    with tqdm(total=max_tokens, desc="Calibration") as pbar:
        for input_ids in dataset.iter_batches(seq_len=512):
            if n_collected >= max_tokens:
                break

            # Get hidden states from FP32 backbone
            with torch.no_grad():
                outputs = model.transformer(torch.tensor([input_ids]))
                h_all = outputs.last_hidden_state[0].numpy()  # [seq_len, d_model]

            # Process each position
            for i in range(len(input_ids)):
                if n_collected >= max_tokens:
                    break

                h = h_all[i]  # [d_model]

                # FP32 logits (sampled vocab)
                logits_fp32 = wte_fp32[vocab_indices] @ h  # [vocab_sample_size]

                # Sign-quantize hidden state
                h_signs = np.sign(h)
                h_signs[h_signs == 0] = 1

                # 1-bit logits (sampled vocab)
                h_bits = pack_input_signs(h_signs)
                logits_1bit_raw = np.zeros(vocab_sample_size, dtype=np.float32)

                for j, v in enumerate(vocab_indices):
                    row_v_bits = wte_bits[v, :]
                    est, _ = bsdm_w_dot(
                        row_v_bits,
                        h_bits,
                        k=T,
                        cfg=sd_cfg,
                        seed=v,
                        eps=0.05,
                        delta=0.001 / vocab_size,
                        early_exit_enable=False,
                        use_ctg=False,
                    )
                    logits_1bit_raw[j] = est  # Keep raw estimate in [-1, 1]

                hidden_states_list.append(h)
                logits_fp32_list.append(logits_fp32)
                logits_1bit_raw_list.append(logits_1bit_raw)

                n_collected += 1
                pbar.update(1)
    
    hidden_states = np.array(hidden_states_list)  # [n_samples, d_model]
    logits_fp32 = np.array(logits_fp32_list)  # [n_samples, vocab_sample_size]
    logits_1bit_raw = np.array(logits_1bit_raw_list)  # [n_samples, vocab_sample_size]
    
    return hidden_states, logits_fp32, logits_1bit_raw


def fit_scaling_factors(
    hidden_states: np.ndarray,
    logits_fp32: np.ndarray,
    logits_1bit_raw: np.ndarray,
) -> Tuple[float, float]:
    """Fit global scaling factors a, b for 1-bit logits.
    
    Model: ŷ = a * ||h|| * sqrt(d) * ẑ + b
    
    Args:
        hidden_states: [n_samples, d_model]
        logits_fp32: [n_samples, vocab_sample_size]
        logits_1bit_raw: [n_samples, vocab_sample_size] (raw estimates in [-1, 1])
        
    Returns:
        a: Global scale factor
        b: Global bias
    """
    n_samples, d_model = hidden_states.shape
    
    # Compute ||h|| for each sample
    h_norms = np.linalg.norm(hidden_states, axis=1)  # [n_samples]
    
    # Compute structural factor: ||h|| * sqrt(d)
    structural_factor = h_norms[:, None] * np.sqrt(d_model)  # [n_samples, 1]
    
    # Flatten for linear regression
    y_true = logits_fp32.flatten()  # [n_samples * vocab_sample_size]
    z_raw = logits_1bit_raw.flatten()  # [n_samples * vocab_sample_size]
    structural = np.repeat(structural_factor, logits_fp32.shape[1])  # [n_samples * vocab_sample_size]
    
    # Linear regression: y = a * (structural * z) + b
    # X = [structural * z, 1]
    X = np.column_stack([structural * z_raw, np.ones_like(z_raw)])
    
    # Solve: (X^T X)^{-1} X^T y
    params, residuals, rank, s = np.linalg.lstsq(X, y_true, rcond=None)
    a, b = params
    
    # Compute metrics
    y_pred = a * structural * z_raw + b
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    print(f"\nCalibration Results:")
    print(f"  a (scale): {a:.4f}")
    print(f"  b (bias): {b:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Correlation: {corr:.4f}")
    
    return a, b


def main():
    parser = argparse.ArgumentParser(description="Calibrate 1-bit logits scaling")
    parser.add_argument("--output", type=str, default="out/calibration", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Calibration tokens")
    parser.add_argument("--T", type=int, default=64, help="BSDM-W samples for calibration")
    parser.add_argument("--vocab-sample", type=int, default=1000, help="Vocab tokens to sample")
    args = parser.parse_args()

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

    # Get wte matrix
    wte_fp32 = model.transformer.wte.weight.detach().cpu().numpy()

    # Quantize to 1-bit
    print("\nQuantizing wte to 1-bit...")
    wte_bits = pack_signs_rowmajor(wte_fp32)

    # Collect calibration data
    hidden_states, logits_fp32, logits_1bit_raw = collect_calibration_data(
        model,
        wte_fp32,
        wte_bits,
        dataset,
        max_tokens=args.max_tokens,
        T=args.T,
        vocab_sample_size=args.vocab_sample,
    )

    print(f"\nCollected {len(hidden_states)} samples")
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"FP32 logits shape: {logits_fp32.shape}")
    print(f"1-bit raw logits shape: {logits_1bit_raw.shape}")

    # Fit scaling factors
    a, b = fit_scaling_factors(hidden_states, logits_fp32, logits_1bit_raw)

    # Save results
    results = {
        "a": float(a),
        "b": float(b),
        "n_samples": int(len(hidden_states)),
        "T": int(args.T),
        "vocab_sample_size": int(args.vocab_sample),
    }

    with open(output_dir / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'calibration_results.json'}")


if __name__ == "__main__":
    main()

