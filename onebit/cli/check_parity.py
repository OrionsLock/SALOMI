import numpy as np
from onebit.model.quantize_gpt2 import load_gpt2_from_huggingface, pack_signs_rowmajor
from onebit.core.packbits import pack_input_signs
from onebit.ops.bsdm_w import _xnor_popcount_dot, _ensure_bits

def main():
    print("Loading GPT-2 weights...")
    weights, cfg = load_gpt2_from_huggingface("gpt2")
    
    # Pick a weight matrix
    name = "h.0.attn.c_attn.w"
    W_hf = weights[name] # [768, 2304]
    W_target = W_hf.T    # [2304, 768]
    print(f"Testing {name}, shape {W_target.shape}")
    
    # Center
    mean = np.mean(W_target, axis=1, keepdims=True)
    W_centered = W_target - mean
    
    # Create input x
    print("Using real embedding for input x...")
    wte = weights["wte"]
    x = wte[1234].astype(np.float32)
    
    # 1. Reference: FP32 Sign Agreement
    print("\n--- Parity Test ---")
    w_sign = np.sign(W_centered)
    # Replace 0 with 1 to match packing logic (val >= 0 -> 1)
    w_sign[w_sign == 0] = 1
    
    x_sign = np.sign(x)
    x_sign[x_sign == 0] = 1
    
    # Compute mean sign agreement per row (normalized dot in [-1, 1])
    # dot = sum(sign(w)*sign(x)). normalized = dot / K
    b_ref = np.mean(w_sign * x_sign, axis=1)
    
    # 2. Bitwise: XNOR Popcount
    # Pack W
    W_bits = pack_signs_rowmajor(W_centered)
    # Pack x
    x_bits = pack_input_signs(x)
    
    # Compute bitwise dot for each row
    d_out = W_target.shape[0]
    K = W_target.shape[1]
    b_bits = np.zeros(d_out, dtype=np.float32)
    
    for i in range(d_out):
        # _xnor_popcount_dot returns integer dot in [-K, K]
        dot_int = _xnor_popcount_dot(W_bits[i], x_bits)
        # Normalize to [-1, 1]. Note: _xnor_popcount_dot uses Kbits (multiple of 32)
        # We need to be careful if K is not multiple of 32.
        # pack_signs pads with 0s (which are -1s).
        # But pack_input_signs also pads with 0s.
        # So padding bits are (-1)*(-1) = +1.
        # The normalization in bsdm_w divides by Kbits.
        
        Kw = W_bits.shape[1]
        Kbits = Kw * 32
        b_bits[i] = float(dot_int) / float(Kbits)

    # Compare
    print(f"b_ref  stats: mean={b_ref.mean():.4f}, std={b_ref.std():.4f}")
    print(f"b_bits stats: mean={b_bits.mean():.4f}, std={b_bits.std():.4f}")
    
    diff = np.abs(b_ref - b_bits)
    print(f"Max abs diff: {np.max(diff):.4f}")
    print(f"Mean abs diff: {np.mean(diff):.4f}")
    
    corr = np.corrcoef(b_ref, b_bits)[0, 1]
    print(f"Parity Correlation: {corr:.4f}")
    
    if corr > 0.98:
        print("PASS: Bit packing and XNOR logic match FP32 signs.")
    else:
        print("FAIL: Bit packing or XNOR logic is broken.")

if __name__ == "__main__":
    main()

