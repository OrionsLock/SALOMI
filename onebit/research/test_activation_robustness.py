"""Test if different activations are more robust to binary quantization."""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def test_activation_robustness():
    """Compare GELU vs ReLU robustness to binarization."""
    print("=" * 70)
    print("ACTIVATION ROBUSTNESS TO BINARIZATION")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Get layer 0 MLP
    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(text, return_tensors='pt')
    
    with torch.no_grad():
        embeddings = model.transformer.wte(tokens) + model.transformer.wpe(torch.arange(tokens.size(1)))
        x = model.transformer.h[0].ln_1(embeddings)
    
    c_fc = model.transformer.h[0].mlp.c_fc.weight.data
    c_proj = model.transformer.h[0].mlp.c_proj.weight.data
    
    # Binary c_fc
    scale_fc = c_fc.abs().mean()
    c_fc_bin = torch.sign(c_fc) * scale_fc
    c_fc_bin[c_fc_bin == 0] = scale_fc
    
    # FP32 forward
    with torch.no_grad():
        hidden_fp32 = x @ c_fc
        
    # Binary forward  
    with torch.no_grad():
        hidden_bin = x @ c_fc_bin
    
    print(f"Pre-activation error: {(hidden_bin - hidden_fp32).abs().mean() / hidden_fp32.abs().mean() * 100:.1f}%")
    print()
    
    # Test different activations
    activations = [
        ("GELU", F.gelu),
        ("ReLU", F.relu),
        ("LeakyReLU", lambda x: F.leaky_relu(x, 0.1)),
        ("SiLU/Swish", F.silu),
        ("Tanh", torch.tanh),
        ("Sigmoid", torch.sigmoid),
        ("Softplus", F.softplus),
    ]
    
    print(f"{'Activation':<15} {'FP32 out':>12} {'Binary out':>12} {'Error':>10}")
    print("-" * 55)
    
    for name, act_fn in activations:
        with torch.no_grad():
            out_fp32 = act_fn(hidden_fp32) @ c_proj
            out_bin = act_fn(hidden_bin) @ c_proj
            
            error = (out_bin - out_fp32).abs().mean() / out_fp32.abs().mean() * 100
            
            print(f"{name:<15} {out_fp32.abs().mean():>12.4f} {out_bin.abs().mean():>12.4f} {error:>9.1f}%")
    
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("GELU, SiLU, and similar smooth activations are very sensitive")
    print("to sign errors because they have different behavior for +x vs -x.")
    print()
    print("ReLU is more robust because ReLU(-x) = 0, so sign errors just")
    print("zero out activations rather than flipping their sign.")
    print()
    print("This suggests: Models trained with ReLU might be more amenable")
    print("to post-hoc binary quantization.")
    
    # Test actual perplexity impact
    print()
    print("=" * 70)
    print("PERPLEXITY TEST: Replace GELU with ReLU + Binarize MLP")
    print("=" * 70)
    
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    test_texts = [
        "The transformer architecture has revolutionized natural language processing.",
        "Machine learning models require large amounts of training data.",
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
    
    # Baseline with GELU
    fp32_ppl = compute_ppl()
    print(f"FP32 (GELU): {fp32_ppl:.2f}")
    
    # Binary MLP with GELU
    for name, param in model.named_parameters():
        if 'mlp' in name and 'weight' in name and param.dim() == 2:
            W = original_state[name]
            scale = W.abs().mean()
            sign = torch.sign(W)
            sign[sign == 0] = 1
            param.data = sign * scale
    
    binary_gelu_ppl = compute_ppl()
    print(f"Binary MLP (GELU): {binary_gelu_ppl:.2f} ({(binary_gelu_ppl/fp32_ppl-1)*100:+.0f}%)")
    
    # Replace GELU with ReLU and test
    restore_all()
    for block in model.transformer.h:
        block.mlp.act = F.relu
    
    fp32_relu_ppl = compute_ppl()
    print(f"FP32 (ReLU): {fp32_relu_ppl:.2f} ({(fp32_relu_ppl/fp32_ppl-1)*100:+.0f}%)")
    
    # Binary MLP with ReLU
    for name, param in model.named_parameters():
        if 'mlp' in name and 'weight' in name and param.dim() == 2:
            W = original_state[name]
            scale = W.abs().mean()
            sign = torch.sign(W)
            sign[sign == 0] = 1
            param.data = sign * scale
    
    binary_relu_ppl = compute_ppl()
    print(f"Binary MLP (ReLU): {binary_relu_ppl:.2f} ({(binary_relu_ppl/fp32_ppl-1)*100:+.0f}%)")


if __name__ == '__main__':
    test_activation_robustness()

