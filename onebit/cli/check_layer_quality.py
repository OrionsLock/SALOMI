import numpy as np
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface, pack_signs_rowmajor
from onebit.core.packbits import pack_input_signs, pack_float_to_stream
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute signal quality metrics."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Correlation
    if np.std(y_true_flat) < 1e-9 or np.std(y_pred_flat) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
        
    # Error metrics
    diff = y_true_flat - y_pred_flat
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    
    # Relative Error
    norm_true = np.linalg.norm(y_true_flat)
    rel_err = float(np.linalg.norm(diff) / (norm_true + 1e-9))
    
    # Magnitude Ratio (Energy preservation)
    mag_ratio = float(np.linalg.norm(y_pred_flat) / (norm_true + 1e-9))
    
    return {
        "corr": corr,
        "mae": mae,
        "mse": mse,
        "rel_err": rel_err,
        "mag_ratio": mag_ratio
    }

def print_metrics(metrics: Dict[str, float], name: str):
    print(f"[{name}]")
    print(f"  Corr:      {metrics['corr']:.4f}")
    print(f"  MAE:       {metrics['mae']:.4f}")
    print(f"  MSE:       {metrics['mse']:.4f}")
    print(f"  Rel Err:   {metrics['rel_err']:.4f}")
    print(f"  Mag Ratio: {metrics['mag_ratio']:.4f}")

def test_layer(W_fp32: np.ndarray, layer_name: str, T: int, bias: float = 0.0, 
               results_dict: Dict[str, Any] = None) -> None:
    print(f"\n=== Testing Layer: {layer_name} ===")
    d_out, d_in = W_fp32.shape
    print(f"Shape: [{d_out}, {d_in}]")
    
    # 1. Generate input x (Random Normal + Bias)
    # Use fixed seed for reproducibility of input data generation within this run
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(d_in, dtype=np.float32) * 0.1) + bias
    
    # 2. FP32 Ground Truth
    y_fp32 = W_fp32 @ x
    
    # 3. Weight Quantization (Standard Sign)
    # Center per row
    mean_w = np.mean(W_fp32, axis=1, keepdims=True)
    W_centered = W_fp32 - mean_w
    scale_w = np.mean(np.abs(W_centered), axis=1) # [d_out]
    
    # Avoid div by zero
    scale_w[scale_w < 1e-9] = 1e-9
    
    W_bits = pack_signs_rowmajor(W_centered)
    Kw = W_bits.shape[1]
    Kbits = Kw * 32
    
    # Config for BSDM (Order 2, consistent with runtime)
    sd_cfg = SDConfig(order=2, beta=0.3, lambd=0.0, walsh_N=2, antithetic=False)
    
    # --- Method: Mag-Aware + Mean Correction (Best Method) ---
    # Normalize input
    max_x = np.max(np.abs(x))
    if max_x < 1e-9: max_x = 1e-9
    x_norm = x / max_x
    
    # Pad if necessary
    padded_d_in = Kw * 32
    if d_in < padded_d_in:
        x_norm_padded = np.zeros(padded_d_in, dtype=x_norm.dtype)
        x_norm_padded[:d_in] = x_norm
        x_norm = x_norm_padded
        
    x_stream = pack_float_to_stream(x_norm, k=T)
    
    # Run BSDM
    y_raw = bsdm_w_matmul(
        W_bits, x_stream, k=T, cfg=sd_cfg, seed=42, scale=1.0
    )
    
    # Reconstruct scaling
    y_est = y_raw * scale_w * max_x * Kbits
    
    # Apply Mean Correction (DC Gain)
    # mean_w is [d_out, 1]
    correction = mean_w.flatten() * np.sum(x)
    y_final = y_est + correction
    
    # Compute Metrics
    metrics = compute_metrics(y_fp32, y_final)
    print_metrics(metrics, f"Mag-Aware + Mean Correction (T={T})")
    
    if results_dict is not None:
        results_dict[layer_name] = metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--save-golden", type=str, help="Path to save golden metrics JSON")
    parser.add_argument("--verify-golden", type=str, help="Path to load golden metrics JSON for verification")
    args = parser.parse_args()
    
    print("Loading weights...")
    weights, cfg = load_gpt2_from_huggingface(args.model)
    
    results = {
        "config": {
            "T": args.T,
            "model": args.model,
            "method": "Mag-Aware + Mean Correction"
        },
        "layers": {}
    }
    
    # 1. MLP Layer (fc)
    # GPT-2 weights are [d_in, d_out], need transpose
    W_mlp = weights["h.0.mlp.c_fc.w"].T # [3072, 768]
    
    # Test with Bias (Shift=5.0) as this is the most challenging/realistic scenario
    test_layer(W_mlp, "h.0.mlp.c_fc_biased", T=args.T, bias=5.0, results_dict=results["layers"])
    
    # 2. Attention QKV
    W_attn = weights["h.0.attn.c_attn.w"].T # [2304, 768]
    test_layer(W_attn, "h.0.attn.c_attn", T=args.T, bias=0.0, results_dict=results["layers"])
    
    # 3. Logits (wte transposed) - usually zero mean roughly?
    W_logits = weights["wte"] # [vocab, d_model]
    test_layer(W_logits, "wte_logits", T=args.T, bias=0.0, results_dict=results["layers"])

    # Save Golden
    if args.save_golden:
        with open(args.save_golden, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nGolden metrics saved to {args.save_golden}")

    # Verify Golden
    if args.verify_golden:
        print(f"\nVerifying against {args.verify_golden}...")
        with open(args.verify_golden, 'r') as f:
            golden = json.load(f)
        
        failed = False
        for layer, metrics in results["layers"].items():
            if layer not in golden["layers"]:
                print(f"WARNING: Layer {layer} not in golden.")
                continue
            
            g_metrics = golden["layers"][layer]
            
            # Check correlation (tolerance 0.05)
            if metrics["corr"] < g_metrics["corr"] - 0.05:
                print(f"FAIL {layer}: Corr {metrics['corr']:.4f} < Golden {g_metrics['corr']:.4f}")
                failed = True
            elif metrics["corr"] < g_metrics["corr"]:
                print(f"WARN {layer}: Corr {metrics['corr']:.4f} < Golden {g_metrics['corr']:.4f}")
            else:
                print(f"PASS {layer}: Corr {metrics['corr']:.4f} >= Golden {g_metrics['corr']:.4f}")
                
        if failed:
            print("\nVERIFICATION FAILED")
            sys.exit(1)
        else:
            print("\nVERIFICATION PASSED")

if __name__ == "__main__":
    main()
