import numpy as np
from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface, pack_signs_rowmajor
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig

def main():
    print("Loading GPT-2 weights...")
    weights, cfg = load_gpt2_from_huggingface("gpt2")
    
    # Pick a weight matrix (e.g. h.0.attn.c_attn.w)
    name = "h.0.attn.c_attn.w"
    W_fp32 = weights[name] # [768, 2304]
    print(f"Testing {name}, shape {W_fp32.shape}")
    
    # Transpose to get [2304, 768] (output, input)
    W_target = W_fp32.T
    
    # Center and scale PER ROW
    mean = np.mean(W_target, axis=1, keepdims=True)
    W_centered = W_target - mean
    scale = np.mean(np.abs(W_centered), axis=1) # [2304]
    print(f"Mean scale: {np.mean(scale):.6f}")
    
    # Quantize
    W_bits = pack_signs_rowmajor(W_centered)
    
    # Create random input
    d_in = W_target.shape[1]
    x = np.random.randn(d_in).astype(np.float32)
    
    # FP32 Ground Truth
    y_fp32 = W_target @ x
    
    # 1-Bit Estimate (BSDM-W)
    from onebit.core.packbits import pack_input_signs
    input_scale = np.mean(np.abs(x))
    x_bits = pack_input_signs(x)
    
    # Config
    sd_cfg = SDConfig(order=2, beta=0.30, lambd=0.0, walsh_N=2, antithetic=True)
    
    # Test scaling factor
    # Analytical: E[sign(w)*sign(x)] * scale_w * scale_x * C
    # What is C?
    
    for T in [256, 1024]:
        # Try raw output without extra scalar
        y_raw = bsdm_w_matmul(
            W_bits, x_bits, k=T, cfg=sd_cfg, seed=42, scale=scale * input_scale
        )
        
        # Find optimal scalar C
        # y_fp32 = C * y_raw
        C = np.dot(y_fp32, y_raw) / np.dot(y_raw, y_raw)
        
        y_est = y_raw * C
        
        corr = np.corrcoef(y_fp32, y_raw)[0, 1]
        rel_err = np.linalg.norm(y_fp32 - y_est) / np.linalg.norm(y_fp32)
        
        print(f"T={T:4d}: Corr={corr:.4f}, Optimal C={C:.4f}, RelErr={rel_err:.4f}")

if __name__ == "__main__":
    main()

