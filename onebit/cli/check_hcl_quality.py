import numpy as np
from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface
from onebit.model.hcl_logits_head import HCLLogitsHead
from onebit.core.hadamard import fwht, inverse_fwht
from onebit.ops.bsdm_w import SDConfig
from onebit.core.packbits import pack_float_to_stream

def check_fwht():
    print("\n=== Checking FWHT ===")
    n = 8
    x = np.random.randn(n).astype(np.float32)
    y = fwht(x)
    z = inverse_fwht(y)
    
    err = np.linalg.norm(x - z)
    print(f"FWHT -> IFWHT error: {err:.6e}")
    if err > 1e-5:
        print("FAIL")
    else:
        print("PASS")

def check_hcl_reconstruction(wte):
    print("\n=== Checking HCL Reconstruction (FP32 Ideal) ===")
    vocab_size, d_model = wte.shape
    print(f"Vocab: {vocab_size}, d_model: {d_model}")
    
    # Create Head
    head = HCLLogitsHead.from_wte(wte)
    
    # Create random hidden state
    h = np.random.randn(d_model).astype(np.float32)
    
    # 1. True logits
    logits_true = wte @ h
    
    # 2. HCL logits (Ideal reconstruction)
    # Skip explicit reconstruction of W_code_fp32, just test forward
    # This avoids unpacking complexity in the test script
    
    # Let's just run forward with high T to simulate ideal 1-bit
    T = 4096
    sd_cfg = SDConfig(order=2, beta=0.3, lambd=0.0, walsh_N=2, antithetic=False)
    
    max_h = np.max(np.abs(h))
    h_norm = h / max_h
    h_stream = pack_float_to_stream(h_norm, T)
    h_sum = np.sum(h)
    
    logits_hcl = head.forward(h_stream, h_sum, max_h, sd_cfg, 42, T)
    
    # Metrics
    corr = np.corrcoef(logits_true, logits_hcl)[0, 1]
    mae = np.mean(np.abs(logits_true - logits_hcl))
    print(f"Correlation: {corr:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Check top-k overlap
    k = 10
    top_true = np.argsort(logits_true)[-k:]
    top_hcl = np.argsort(logits_hcl)[-k:]
    overlap = len(set(top_true) & set(top_hcl))
    print(f"Top-{k} overlap: {overlap}/{k}")

def main():
    check_fwht()
    
    print("\nLoading weights...")
    weights, cfg = load_gpt2_from_huggingface("gpt2")
    wte = weights["wte"]
    
    check_hcl_reconstruction(wte)

if __name__ == "__main__":
    main()

