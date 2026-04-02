import numpy as np
import torch
from onebit.ops.bsdm_w import bsdm_w_matmul
from onebit.core.packbits import pack_float_to_stream, pack_signs_rowmajor
from onebit.ops.bsdm_w import SDConfig

def check_ac_quality():
    print("Checking AC Component Quality...")
    d_in = 768
    d_out = 768
    rng = np.random.default_rng(42)
    
    # Test Case 1: Gaussian
    # W = Gaussian
    # x = Gaussian
    W_centered = rng.normal(0, 0.1, (d_out, d_in)).astype(np.float32)
    x_centered = rng.normal(0, 1.0, (d_in,)).astype(np.float32)
    
    # Quantize
    W_bits = pack_signs_rowmajor(W_centered)
    scale_w = np.mean(np.abs(W_centered), axis=1) # Per row
    
    input_scale = np.max(np.abs(x_centered))
    x_norm = x_centered / input_scale
    
    for T in [64, 256, 768]:
        x_stream = pack_float_to_stream(x_norm, k=T)
        
        cfg = SDConfig(order=2, beta=0.3, lambd=0.0, walsh_N=2, antithetic=True)
        Kbits = W_bits.shape[1] * 32
        
        # Raw BSDM output (normalized mean)
        # We want to recover sum(w * x).
        # BSDM returns approx mean(w * x) * scale.
        # My formula: scale = scale_w * input_scale * Kbits
        
        scale_factor = scale_w * input_scale * Kbits
        
        y_true = W_centered @ x_centered

        y_est = bsdm_w_matmul(
            W_bits, x_stream, k=T, cfg=cfg, seed=123,
            scale=scale_factor
        )
        
        # DEBUG: Manually compute W_bits @ mean(x_stream)
        # Decode stream average
        x_decoded = np.zeros(d_in)
        for t in range(T):
            # Unpack row
            row = x_stream[t]
            bits = []
            for w in row:
                for b in range(32):
                    bits.append(1.0 if (w & (1<<b)) else -1.0)
            x_decoded += np.array(bits[:d_in])
        x_decoded /= T
        
        # Unpack W_bits
        W_decoded = np.zeros((d_out, d_in))
        for r in range(d_out):
            row_bits = []
            for w in range(W_bits.shape[1]):
                word = W_bits[r, w]
                for b in range(32):
                    row_bits.append(1.0 if (word & (1<<b)) else -1.0)
            W_decoded[r] = np.array(row_bits[:d_in])
            
        y_manual = W_decoded @ x_decoded
        
        # Compare
        corr_manual = np.corrcoef(y_true, y_manual)[0, 1]
        corr_bsdm = np.corrcoef(y_manual, y_est)[0, 1]
        
        print(f"T={T}: Corr(True, Manual)={corr_manual:.4f}, Corr(Manual, BSDM)={corr_bsdm:.4f}")
        
        # y_true = W_centered @ x_centered (removed duplicate)
        
        # corr = np.corrcoef(y_true, y_est)[0, 1] (old code)
        # ratio = np.linalg.norm(y_est) / np.linalg.norm(y_true)
        # print(f"T={T}: Corr={corr:.4f}, MagRatio={ratio:.4f}")
        
        corr = np.corrcoef(y_true, y_est)[0, 1]
        ratio = np.linalg.norm(y_est) / np.linalg.norm(y_true)
        
        print(f"T={T}: Corr={corr:.4f}, MagRatio={ratio:.4f}")

if __name__ == "__main__":
    check_ac_quality()

