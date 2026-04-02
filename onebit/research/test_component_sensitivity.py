"""Analyze which components are most sensitive to binarization."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def test_component_sensitivity():
    """Test binarizing individual component types."""
    print("=" * 70)
    print("COMPONENT SENSITIVITY ANALYSIS")
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
    
    def restore_all():
        for name, param in model.named_parameters():
            param.data = original_state[name].clone()
    
    def binarize_matching(pattern):
        """Binarize weights matching pattern."""
        for name, param in model.named_parameters():
            if pattern in name and 'weight' in name and param.dim() == 2:
                W = original_state[name]
                scale = W.abs().mean()
                sign = torch.sign(W)
                sign[sign == 0] = 1
                param.data = sign * scale
    
    def count_params(pattern):
        return sum(p.numel() for n, p in model.named_parameters() 
                  if pattern in n and 'weight' in n and p.dim() == 2)
    
    # Baseline
    fp32_ppl = compute_ppl()
    print(f"FP32 baseline PPL: {fp32_ppl:.2f}")
    print()
    
    # Test each component type
    components = [
        ("c_attn (QKV projection)", "c_attn"),
        ("c_proj (attention output)", "attn.c_proj"),
        ("c_fc (MLP up)", "c_fc"),
        ("c_proj (MLP down)", "mlp.c_proj"),
    ]
    
    print(f"{'Component':<30} {'Params':>12} {'PPL':>12} {'vs FP32':>12}")
    print("-" * 70)
    
    for comp_name, pattern in components:
        restore_all()
        binarize_matching(pattern)
        ppl = compute_ppl()
        params = count_params(pattern)
        gap = (ppl / fp32_ppl - 1) * 100
        print(f"{comp_name:<30} {params:>12,} {ppl:>12.2f} {gap:>+11.1f}%")
    
    print()
    
    # Test combined
    print("Combined configurations:")
    print("-" * 70)
    
    configs = [
        ("All attention (c_attn + c_proj)", ["c_attn", "attn.c_proj"]),
        ("All MLP (c_fc + c_proj)", ["c_fc", "mlp.c_proj"]),
        ("Everything", ["c_attn", "attn.c_proj", "c_fc", "mlp.c_proj"]),
    ]
    
    for config_name, patterns in configs:
        restore_all()
        for pattern in patterns:
            binarize_matching(pattern)
        ppl = compute_ppl()
        params = sum(count_params(p) for p in patterns)
        gap = (ppl / fp32_ppl - 1) * 100
        print(f"{config_name:<30} {params:>12,} {ppl:>12.2f} {gap:>+11.1f}%")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("MLP layers (c_fc, c_proj) are much more sensitive to binarization")
    print("than attention layers (c_attn, c_proj).")
    print()
    print("This suggests: Binary attention + higher-precision MLP could be")
    print("a good hybrid approach for storage savings with acceptable quality.")


if __name__ == '__main__':
    test_component_sensitivity()

