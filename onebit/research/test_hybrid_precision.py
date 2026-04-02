"""Test hybrid precision: keep critical layers at FP32, rest binary."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def test_hybrid_precision():
    """Test keeping first/last layers FP32, middle binary."""
    print("=" * 70)
    print("HYBRID PRECISION TEST")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Save original state
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    test_texts = [
        "The transformer architecture has revolutionized natural language processing.",
        "Machine learning models require large amounts of training data.",
        "Neural networks consist of interconnected layers of neurons.",
    ]
    
    def compute_ppl():
        losses = []
        for text in test_texts:
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
            with torch.no_grad():
                outputs = model(tokens, labels=tokens)
                losses.append(outputs.loss.item())
        return np.exp(np.mean(losses))
    
    def binarize_layer(layer_idx):
        """Binarize weights in a specific layer."""
        prefix = f"transformer.h.{layer_idx}."
        for name, param in model.named_parameters():
            if name.startswith(prefix) and 'weight' in name and param.dim() == 2:
                W = original_state[name]
                scale = W.abs().mean()
                sign = torch.sign(W)
                sign[sign == 0] = 1
                param.data = sign * scale
    
    def restore_all():
        for name, param in model.named_parameters():
            param.data = original_state[name].clone()
    
    # Baseline
    fp32_ppl = compute_ppl()
    print(f"FP32 baseline PPL: {fp32_ppl:.2f}")
    
    # Test different hybrid configurations
    configs = [
        ("All binary (layers 0-11)", list(range(12))),
        ("Layer 0 FP32, rest binary (1-11)", list(range(1, 12))),
        ("Layers 0-1 FP32, rest binary (2-11)", list(range(2, 12))),
        ("First+Last FP32, middle binary (1-10)", list(range(1, 11))),
        ("Only middle binary (2-9)", list(range(2, 10))),
        ("Attn binary, MLP FP32", "attn_only"),
        ("MLP binary, Attn FP32", "mlp_only"),
    ]
    
    print("\nHybrid configurations:")
    print(f"{'Config':<45} {'PPL':>10} {'vs FP32':>12}")
    print("-" * 70)
    
    for config_name, layers in configs:
        restore_all()
        
        if layers == "attn_only":
            # Only binarize attention
            for name, param in model.named_parameters():
                if 'attn' in name and 'weight' in name and param.dim() == 2:
                    W = original_state[name]
                    scale = W.abs().mean()
                    sign = torch.sign(W)
                    sign[sign == 0] = 1
                    param.data = sign * scale
        elif layers == "mlp_only":
            # Only binarize MLP
            for name, param in model.named_parameters():
                if 'mlp' in name and 'weight' in name and param.dim() == 2:
                    W = original_state[name]
                    scale = W.abs().mean()
                    sign = torch.sign(W)
                    sign[sign == 0] = 1
                    param.data = sign * scale
        else:
            for layer_idx in layers:
                binarize_layer(layer_idx)
        
        ppl = compute_ppl()
        gap = (ppl / fp32_ppl - 1) * 100
        print(f"{config_name:<45} {ppl:>10.2f} {gap:>+11.1f}%")
    
    # Calculate effective BPP for best config
    print("\n" + "=" * 70)
    print("BEST HYBRID: Layers 0-1 FP32, Layers 2-11 Binary")
    print("=" * 70)
    
    # Count params
    layer0_params = sum(p.numel() for n, p in model.named_parameters() 
                       if n.startswith('transformer.h.0.') and 'weight' in n and p.dim() == 2)
    layer1_params = sum(p.numel() for n, p in model.named_parameters() 
                       if n.startswith('transformer.h.1.') and 'weight' in n and p.dim() == 2)
    other_params = sum(p.numel() for n, p in model.named_parameters() 
                      if 'transformer.h.' in n and 'weight' in n and p.dim() == 2 
                      and not n.startswith('transformer.h.0.') and not n.startswith('transformer.h.1.'))
    
    # BPP calculation
    fp32_bits = (layer0_params + layer1_params) * 32
    binary_bits = other_params * 1
    total_bits = fp32_bits + binary_bits
    total_params = layer0_params + layer1_params + other_params
    effective_bpp = total_bits / total_params
    
    print(f"Layer 0-1 params: {layer0_params + layer1_params:,}")
    print(f"Layer 2-11 params: {other_params:,}")
    print(f"Effective BPP: {effective_bpp:.2f}")


if __name__ == '__main__':
    test_hybrid_precision()

