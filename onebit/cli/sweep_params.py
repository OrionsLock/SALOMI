import numpy as np
from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface, pack_signs_rowmajor
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig
from onebit.core.packbits import pack_input_signs

def main():
    print("Loading GPT-2 weights...")
    weights, cfg = load_gpt2_from_huggingface("gpt2")
    
    name = "h.0.attn.c_attn.w"
    W_fp32 = weights[name] # [2304, 768]
    print(f"Testing {name}, shape {W_fp32.shape}")
    
    # Center and scale
    mean = np.mean(W_fp32)
    W_centered = W_fp32 - mean
    scale = np.mean(np.abs(W_centered))
    
    # Transpose for correct packing
    W_target = W_fp32.T # [768, 2304] - wait, quantization code does W_fp32.T
    # In load_gpt2_from_huggingface, we return W.T. 
    # Let's trust quantize_gpt2 logic: it packs W_fp32.T
    
    # Let's replicate quantize_gpt2 logic exactly
    # In quantize_gpt2: 
    # W_target = W_fp32.T  (where W_fp32 is from load_gpt2...)
    # W_centered = W_target - mean
    # W_bits = pack_signs_rowmajor(W_centered)
    
    W_target = W_fp32.T # [768, 2304]
    mean = np.mean(W_target)
    W_centered = W_target - mean
    scale = float(np.mean(np.abs(W_centered)))
    W_bits = pack_signs_rowmajor(W_centered)
    
    # Create input
    d_in = W_target.shape[1] # 2304
    x = np.random.randn(d_in).astype(np.float32)
    input_scale = np.mean(np.abs(x))
    x_bits = pack_input_signs(x)
    
    # FP32 Truth
    y_fp32 = W_target @ x
    
    print(f"Weight scale: {scale:.6f}, Input scale: {input_scale:.6f}")
    
    # Sweep
    betas = [0.1, 0.3, 0.5, 0.8]
    lambdas = [0.0, 1.0/256.0, 0.1, 0.5]
    orders = [1, 2]
    T = 256
    
    best_corr = -1.0
    best_cfg = None
    
    print(f"\nSweeping parameters with T={T}...")
    print(f"{'Order':<5} {'Beta':<6} {'Lambda':<8} | {'Corr':<8} {'RelErr':<8}")
    print("-" * 45)
    
    for order in orders:
        for beta in betas:
            if order == 1 and beta != betas[0]: continue # Beta irrelevant for order 1
            
            for lambd in lambdas:
                sd_cfg = SDConfig(order=order, beta=beta, lambd=lambd, walsh_N=2, antithetic=True)
                
                # Run matmul
                # Total scale = weight_scale * input_scale
                y_1bit = bsdm_w_matmul(
                    W_bits, x_bits, k=T, cfg=sd_cfg, seed=42, scale=scale * input_scale
                )
                
                corr = np.corrcoef(y_fp32, y_1bit)[0, 1]
                rel_err = np.linalg.norm(y_fp32 - y_1bit) / np.linalg.norm(y_fp32)
                
                print(f"{order:<5} {beta:<6.2f} {lambd:<8.4f} | {corr:<8.4f} {rel_err:<8.4f}")
                
                if corr > best_corr:
                    best_corr = corr
                    best_cfg = (order, beta, lambd)

    print("-" * 45)
    print(f"Best Correlation: {best_corr:.4f}")
    print(f"Best Config: Order={best_cfg[0]}, Beta={best_cfg[1]}, Lambda={best_cfg[2]}")

if __name__ == "__main__":
    main()

