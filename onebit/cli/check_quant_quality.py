import numpy as np
from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface, pack_signs_rowmajor
from onebit.core.packbits import pack_input_signs, pack_float_to_stream
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig

def main():
    print("Loading GPT-2 weights...")
    weights, cfg = load_gpt2_from_huggingface("gpt2")
    
    # Pick a weight matrix (e.g. h.0.attn.c_attn.w)
    name = "h.0.attn.c_attn.w"
    W_hf = weights[name] # [768, 2304]
    print(f"Testing {name}, shape {W_hf.shape}")
    
    # Transpose to get [2304, 768]
    W_target = W_hf.T
    
    # Center and scale PER ROW
    mean = np.mean(W_target, axis=1, keepdims=True)
    W_centered = W_target - mean
    scale = np.mean(np.abs(W_centered), axis=1) # [2304]
    print(f"Mean scale: {np.mean(scale):.6f}")
    
    # Quantize
    W_bits = pack_signs_rowmajor(W_centered)
    
    # Create input x
    # Use a real embedding vector instead of random noise
    print("Using real embedding for input x...")
    wte = weights["wte"] # [vocab_size, d_model]
    token_id = 1234
    x = wte[token_id].astype(np.float32)
    
    # Normalize input x to [-1, 1] for magnitude-aware stream
    max_abs_x = np.max(np.abs(x))
    x_norm = x / max_abs_x
    print(f"Max abs x: {max_abs_x:.6f}")
    
    # FP32 Ground Truth
    y_fp32 = W_target @ x
    
    # Config
    sd_cfg = SDConfig(order=2, beta=0.30, lambd=0.0, walsh_N=2, antithetic=True)
    
    # Test Stream Mode
    print("\n--- Magnitude-Aware Stream Test ---")
    
    for T in [256, 1024, 4096]:
        # Pack x into stochastic stream
        x_stream = pack_float_to_stream(x_norm, T)
        
        # Compute using BSDM-W
        # scale = scale_w * scale_x
        # We pass scale=1.0 to get raw normalized output, then rescale manually to debug
        y_raw = bsdm_w_matmul(
            W_bits, x_stream, k=T, cfg=sd_cfg, seed=42, scale=1.0
        )
        
        # Reconstruct
        # y_est = y_raw * scale_w * max_abs_x * C?
        # Let's find optimal scalar C again to see if it's consistent
        
        y_est_base = y_raw * scale * max_abs_x
        
        # Find optimal scalar C
        C = np.dot(y_fp32, y_est_base) / np.dot(y_est_base, y_est_base)
        
        y_final = y_est_base * C
        
        corr = np.corrcoef(y_fp32, y_final)[0, 1]
        rel_err = np.linalg.norm(y_fp32 - y_final) / np.linalg.norm(y_fp32)
        
        print(f"T={T:4d}: Corr={corr:.4f}, Optimal C={C:.4f}, RelErr={rel_err:.4f}")

if __name__ == "__main__":
    main()
