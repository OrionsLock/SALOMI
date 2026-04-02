import numpy as np
import torch
from onebit.ops.bsdm_w import bsdm_w_matmul
from onebit.core.packbits import pack_float_to_stream, pack_signs_rowmajor
from onebit.ops.bsdm_w import SDConfig

def check_scaling():
    # 1. Setup meaningful data
    d_in = 768
    d_out = 768
    
    # Create a random matrix with non-zero mean to test centering
    rng = np.random.default_rng(42)
    W_fp32 = rng.normal(loc=0.05, scale=0.1, size=(d_out, d_in)).astype(np.float32)
    
    # Create input with non-zero mean
    x_fp32 = rng.normal(loc=0.5, scale=1.0, size=(d_in,)).astype(np.float32)
    
    # 2. FP32 Ground Truth
    y_fp32 = W_fp32 @ x_fp32
    print(f"FP32: Mean={y_fp32.mean():.4f}, Std={y_fp32.std():.4f}, Norm={np.linalg.norm(y_fp32):.4f}")
    
    # 3. Quantization (mirroring quantize_gpt2.py)
    mean_w = np.mean(W_fp32, axis=1, keepdims=True)
    W_centered = W_fp32 - mean_w
    scale_w = np.mean(np.abs(W_centered), axis=1)
    
    W_bits = pack_signs_rowmajor(W_centered)
    
    # 4. Runtime Simulation (mirroring RuntimeTransformer._matmul_1bit)
    T = 64 # Test with small T first
    input_scale = np.max(np.abs(x_fp32))
    x_norm = x_fp32 / input_scale
    
    x_stream = pack_float_to_stream(x_norm, k=T)
    
    Kbits = W_bits.shape[1] * 32
    
    # This is the scaling formula used in runtime
    # scale = scale * input_scale * Kbits
    runtime_scale = scale_w * input_scale * Kbits
    
    cfg = SDConfig(order=2, beta=0.3, lambd=0.0, walsh_N=2, antithetic=True)
    
    y_raw = bsdm_w_matmul(
        W_bits, x_stream, k=T, cfg=cfg, seed=123,
        scale=runtime_scale
    )
    
    # Mean Correction
    x_sum = float(np.sum(x_fp32))
    correction = mean_w.flatten() * x_sum
    y_1bit = y_raw + correction
    
    print(f"1Bit: Mean={y_1bit.mean():.4f}, Std={y_1bit.std():.4f}, Norm={np.linalg.norm(y_1bit):.4f}")
    
    # 5. Comparison
    corr = np.corrcoef(y_fp32, y_1bit)[0, 1]
    mag_ratio = np.linalg.norm(y_1bit) / np.linalg.norm(y_fp32)
    
    print(f"Correlation: {corr:.4f}")
    print(f"Mag Ratio:   {mag_ratio:.4f}")
    
    # 6. Component Analysis
    print("\n--- Component Analysis ---")
    y_centered_fp32 = W_centered @ x_fp32
    print(f"FP32 Centered Norm: {np.linalg.norm(y_centered_fp32):.4f}")
    print(f"1Bit Raw Norm:      {np.linalg.norm(y_raw):.4f}")
    print(f"Raw Ratio:          {np.linalg.norm(y_raw)/np.linalg.norm(y_centered_fp32):.4f}")
    
    correction_fp32 = mean_w.flatten() * np.sum(x_fp32)
    print(f"Correction Norm:    {np.linalg.norm(correction):.4f}")

    # Sweep boost factors to find optimal scaling for AC component
    print("\n--- Optimization ---")
    best_corr = -1
    best_boost = 1.0
    
    y_centered_target = y_centered_fp32
    
    for boost in [1.0, 1.5, 2.0, 2.3, 2.5, 3.0]:
        # Apply boost only to the BSDM part
        y_boosted = y_raw * boost + correction
        
        corr = np.corrcoef(y_fp32, y_boosted)[0, 1]
        ac_norm = np.linalg.norm(y_raw * boost)
        ratio = ac_norm / np.linalg.norm(y_centered_target)
        
        print(f"Boost {boost:.1f}x -> Corr: {corr:.4f}, AC Ratio: {ratio:.4f}")
        
        if corr > best_corr:
            best_corr = corr
            best_boost = boost
            
    print(f"Best Boost: {best_boost}")

if __name__ == "__main__":
    check_scaling()

