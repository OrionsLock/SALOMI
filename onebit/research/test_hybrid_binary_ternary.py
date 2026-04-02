"""Test hybrid quantization: Binary attention + Ternary/higher-precision MLP."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def quantize_binary(W):
    """Quantize to binary {-1, +1}."""
    scale = W.abs().mean()
    sign = torch.sign(W)
    sign[sign == 0] = 1
    return sign * scale

def quantize_ternary(W, threshold_ratio=0.3):
    """Quantize to ternary {-1, 0, +1}."""
    scale = W.abs().mean()
    threshold = threshold_ratio * scale
    W_tern = torch.zeros_like(W)
    W_tern[W > threshold] = scale
    W_tern[W < -threshold] = -scale
    return W_tern

def quantize_4bit(W):
    """Quantize to 4-bit (16 levels)."""
    W_min, W_max = W.min(), W.max()
    scale = (W_max - W_min) / 15
    W_q = torch.round((W - W_min) / scale)
    W_q = torch.clamp(W_q, 0, 15)
    return W_q * scale + W_min


def test_hybrid_quantization():
    """Test various hybrid quantization schemes."""
    print("=" * 70)
    print("HYBRID QUANTIZATION: Binary Attention + Higher-Precision MLP")
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
    
    def apply_hybrid(attn_quant, mlp_quant):
        """Apply different quantization to attention vs MLP."""
        for name, param in model.named_parameters():
            if 'weight' not in name or param.dim() != 2:
                continue
            W = original_state[name]
            
            if 'attn' in name:
                param.data = attn_quant(W)
            elif 'mlp' in name:
                param.data = mlp_quant(W)
    
    def count_params(pattern):
        return sum(p.numel() for n, p in model.named_parameters() 
                  if pattern in n and 'weight' in n and p.dim() == 2)
    
    attn_params = count_params('attn')
    mlp_params = count_params('mlp')
    total_params = attn_params + mlp_params
    
    print(f"Attention params: {attn_params:,} ({attn_params/total_params*100:.1f}%)")
    print(f"MLP params: {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
    print()
    
    # Baseline
    fp32_ppl = compute_ppl()
    print(f"FP32 baseline PPL: {fp32_ppl:.2f}")
    print()
    
    # Test configurations
    configs = [
        ("FP32 Attn + FP32 MLP", lambda w: w.clone(), lambda w: w.clone(), 32, 32),
        ("Binary Attn + FP32 MLP", quantize_binary, lambda w: w.clone(), 1, 32),
        ("Binary Attn + Ternary MLP", quantize_binary, quantize_ternary, 1, 1.58),
        ("Binary Attn + 4-bit MLP", quantize_binary, quantize_4bit, 1, 4),
        ("Ternary Attn + Ternary MLP", quantize_ternary, quantize_ternary, 1.58, 1.58),
        ("Binary Attn + Binary MLP", quantize_binary, quantize_binary, 1, 1),
    ]
    
    print(f"{'Config':<35} {'PPL':>10} {'vs FP32':>12} {'Eff BPP':>10}")
    print("-" * 70)
    
    for name, attn_q, mlp_q, attn_bpp, mlp_bpp in configs:
        restore_all()
        apply_hybrid(attn_q, mlp_q)
        ppl = compute_ppl()
        gap = (ppl / fp32_ppl - 1) * 100
        
        # Calculate effective BPP
        eff_bpp = (attn_params * attn_bpp + mlp_params * mlp_bpp) / total_params
        
        print(f"{name:<35} {ppl:>10.2f} {gap:>+11.1f}% {eff_bpp:>10.2f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)


if __name__ == '__main__':
    test_hybrid_quantization()

