import numpy as np
import torch
from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface, pack_signs_rowmajor
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig
from onebit.core.packbits import pack_float_to_stream

def main():
    print("Loading GPT-2 weights...")
    weights, cfg = load_gpt2_from_huggingface("gpt2")
    
    # Pick a weight matrix (e.g. h.0.attn.c_attn.w)
    name = "h.0.attn.c_attn.w"
    W_fp32 = weights[name] # [2304, 768]
    print(f"Testing {name}, shape {W_fp32.shape}")
    
    # Center and scale
    mean = np.mean(W_fp32)
    W_centered = W_fp32 - mean
    scale = np.mean(np.abs(W_centered))
    print(f"Mean: {mean:.6f}, Scale: {scale:.6f}")
    
    # Quantize
    W_bits = pack_signs_rowmajor(W_centered)
    
    # Create random input (scaled)
    d_in = W_fp32.shape[1]
    x = np.random.randn(d_in).astype(np.float32) * 0.1
    
    # Normalize input
    x_scale = np.max(np.abs(x))
    x_norm = x / x_scale
    
    # FP32 Ground Truth
    y_fp32 = W_fp32 @ x
    
    sd_cfg = SDConfig(order=2, beta=0.30, lambd=1.0/256.0)
    
    # Test various T with stochastic input stream
    for T in [16, 32, 64, 128, 256, 1024]:
        # Generate stochastic bitstream for input
        x_stream = pack_float_to_stream(x_norm, k=T)
        
        # Estimate using BSDM-W with stochastic input
        # Need to call bsdm_w_matmul with stream support (not yet implemented in matmul, need to loop)
        
        # For check, we use bsdm_w_matmul in a loop over chunks if we modify it, 
        # or here we just implement the logic inline or call a new function.
        # Actually, let's modify bsdm_w_matmul to accept x_stream.
        
        # Assume bsdm_w_matmul handles x_stream if provided.
        y_1bit = bsdm_w_matmul(
            W_bits, 
            x_stream, # Pass stream! 
            k=T, 
            cfg=sd_cfg, 
            seed=42, 
            scale=scale * x_scale
        )
        
        # Compute correlation
        if T == 16:
            print(f"y_1bit stats: min={y_1bit.min():.4f}, max={y_1bit.max():.4f}, mean={y_1bit.mean():.4f}, std={y_1bit.std():.4f}")
        
        corr = np.corrcoef(y_fp32, y_1bit)[0, 1]
        mse = np.mean((y_fp32 - y_1bit)**2)
        rel_err = np.linalg.norm(y_fp32 - y_1bit) / np.linalg.norm(y_fp32)
        
        print(f"T={T:4d}: Corr={corr:.4f}, MSE={mse:.4f}, RelErr={rel_err:.4f}")

if __name__ == "__main__":
    main()

