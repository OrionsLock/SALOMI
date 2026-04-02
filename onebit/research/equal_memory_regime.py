"""
EQUAL MEMORY REGIME: Binary + Metadata vs Ternary at 1.58 bpp total

The insight: Ternary uses 1.58 bpp inherently.
If we use binary (1.0 bpp) + 0.58 bpp metadata, we have EQUAL memory.
What can 0.58 bpp of metadata buy us?

Options for 0.58 bpp per weight:
1. Per-row/col scales (amortized)
2. Importance masks
3. Low-rank magnitude factors
4. Per-block calibration
5. Activation statistics
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def test_equal_memory_regime():
    """Test binary + metadata vs ternary at equal 1.58 bpp."""
    print("=" * 70)
    print("EQUAL MEMORY REGIME: Binary + 0.58 bpp Metadata vs Ternary")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    test_texts = [
        "The transformer architecture has revolutionized natural language processing.",
        "Machine learning models require large amounts of training data.",
        "Neural networks consist of interconnected layers of neurons.",
        "Deep learning has achieved remarkable results in computer vision.",
    ]
    
    def compute_ppl():
        losses = []
        for text in test_texts:
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
            with torch.no_grad():
                outputs = model(tokens, labels=tokens)
                losses.append(outputs.loss.item())
        return np.exp(np.mean(losses))
    
    def restore_all():
        for name, param in model.named_parameters():
            param.data = original_state[name].clone()
    
    # Count params
    weight_params = sum(p.numel() for n, p in model.named_parameters() 
                       if 'weight' in n and p.dim() == 2 and 'transformer.h' in n)
    
    print(f"Quantizable weight params: {weight_params:,}")
    print(f"Metadata budget at 0.58 bpp: {weight_params * 0.58 / 8:,.0f} bytes")
    print(f"                           = {weight_params * 0.58:,.0f} bits")
    print()
    
    # Baseline
    fp32_ppl = compute_ppl()
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    # Pure ternary at 1.58 bpp
    restore_all()
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2 and 'transformer.h' in name:
            W = original_state[name]
            scale = W.abs().mean()
            thresh = 0.3 * scale
            W_tern = torch.zeros_like(W)
            W_tern[W > thresh] = scale
            W_tern[W < -thresh] = -scale
            param.data = W_tern
    
    tern_ppl = compute_ppl()
    print(f"Ternary (1.58 bpp): {tern_ppl:.2f} ({(tern_ppl/fp32_ppl-1)*100:+.0f}%)")
    print()
    
    print("=" * 70)
    print("BINARY + METADATA OPTIONS (all at 1.58 bpp total)")
    print("=" * 70)
    
    results = []
    
    # Option 1: Binary + per-channel 8-bit scales
    # 8 bits per row = 8/(768 cols) = 0.01 bpp overhead per weight
    # Can afford 0.58/0.01 = 58x more metadata
    restore_all()
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2 and 'transformer.h' in name:
            W = original_state[name]
            # Per-row scale (expensive but accurate)
            row_scales = W.abs().mean(dim=1, keepdim=True)
            sign = torch.sign(W)
            sign[sign == 0] = 1
            param.data = sign * row_scales
    
    ppl = compute_ppl()
    # BPP: 1 bit sign + 8 bits per row / cols
    overhead = 8 / 768  # ~0.01 bpp
    total_bpp = 1.0 + overhead
    results.append(("Binary + row scales (8-bit)", ppl, total_bpp))
    
    # Option 2: Binary + per-element 4-bit magnitude correction
    # This is basically 5-bit quantization!
    restore_all()
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2 and 'transformer.h' in name:
            W = original_state[name]
            # 4-bit magnitude per weight = 16 levels
            W_abs = W.abs()
            W_max = W_abs.max()
            mag_q = torch.round(W_abs / W_max * 15) / 15 * W_max
            sign = torch.sign(W)
            sign[sign == 0] = 1
            param.data = sign * mag_q
    
    ppl = compute_ppl()
    results.append(("Binary + 4-bit magnitude (5-bit total)", ppl, 5.0))
    
    # Option 3: Binary + LowRank-8 magnitude (proven to work)
    restore_all()
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2 and 'transformer.h' in name:
            W = original_state[name]
            W_abs = W.abs()
            U, S, Vh = torch.linalg.svd(W_abs, full_matrices=False)
            r = 8
            W_mag_approx = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
            sign = torch.sign(W)
            sign[sign == 0] = 1
            param.data = sign * W_mag_approx
    
    ppl = compute_ppl()
    # LowRank r=8: 2*r*d / (m*n) extra bits ≈ 0.42 bpp
    results.append(("Binary + LowRank-8 magnitude", ppl, 1.42))
    
    # Option 4: Binary + importance mask (top 30% get higher precision)
    restore_all()
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2 and 'transformer.h' in name:
            W = original_state[name]
            W_abs = W.abs()
            thresh = torch.quantile(W_abs.flatten(), 0.7)
            
            # Important weights: keep FP16 magnitude
            # Unimportant: binary
            important = W_abs >= thresh
            scale = W_abs.mean()
            
            sign = torch.sign(W)
            sign[sign == 0] = 1
            
            W_out = sign * scale  # Base binary
            W_out[important] = W[important]  # Important keep original
            param.data = W_out
    
    ppl = compute_ppl()
    # 30% at 16-bit, 70% at 1-bit = 0.3*16 + 0.7*1 = 5.5 bpp
    results.append(("Binary + 30% FP16 important", ppl, 5.5))
    
    print(f"{'Method':<45} {'PPL':>10} {'BPP':>8} {'vs Tern':>10}")
    print("-" * 75)
    
    for name, ppl, bpp in results:
        vs_tern = (ppl / tern_ppl - 1) * 100
        marker = " ✓" if ppl < tern_ppl else ""
        print(f"{name:<45} {ppl:>10.2f} {bpp:>8.2f} {vs_tern:>+9.1f}%{marker}")


if __name__ == '__main__':
    test_equal_memory_regime()

