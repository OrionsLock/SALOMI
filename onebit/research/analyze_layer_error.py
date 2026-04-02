"""Analyze how binary quantization errors compound through layers."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def analyze_error_propagation():
    """See how errors grow through transformer layers."""
    print("=" * 70)
    print("ERROR PROPAGATION ANALYSIS")
    print("=" * 70)
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Test input
    text = "The quick brown fox"
    tokens = tokenizer.encode(text, return_tensors='pt')
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model.transformer.wte(tokens) + model.transformer.wpe(torch.arange(tokens.size(1)))
        x_fp32 = embeddings.clone()
        x_binary = embeddings.clone()
    
    print(f"Input: '{text}'")
    print(f"Embedding norm: {x_fp32.norm():.2f}")
    print()
    
    # Process through each layer
    print("Layer-by-layer error accumulation:")
    print(f"{'Layer':<10} {'FP32 norm':>12} {'Binary norm':>12} {'Rel Error':>12}")
    print("-" * 50)
    
    for layer_idx, block in enumerate(model.transformer.h):
        # FP32 forward
        with torch.no_grad():
            x_fp32 = block(x_fp32)[0]
        
        # Create binary version of this layer
        binary_block = block
        for name, param in binary_block.named_parameters():
            if 'weight' in name and param.dim() == 2:
                W = param.data
                scale = W.abs().mean()
                sign = torch.sign(W)
                sign[sign == 0] = 1
                param.data = sign * scale
        
        # Binary forward
        with torch.no_grad():
            x_binary = binary_block(x_binary)[0]
        
        # Error
        rel_error = (x_binary - x_fp32).norm() / x_fp32.norm()
        
        print(f"Layer {layer_idx:<3} {x_fp32.norm():>12.2f} {x_binary.norm():>12.2f} {rel_error.item():>12.2%}")
        
        if layer_idx >= 5:
            print("... (stopping early, error already huge)")
            break


def test_single_layer_binary():
    """Test binary on ONLY ONE layer, rest FP32."""
    print("\n" + "=" * 70)
    print("SINGLE LAYER BINARY TEST")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Save original state
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
    
    # Baseline
    fp32_ppl = compute_ppl()
    print(f"FP32 baseline PPL: {fp32_ppl:.2f}")
    
    # Test binarizing each layer individually
    print("\nBinarizing one layer at a time:")
    print(f"{'Layer':<20} {'PPL':>10} {'vs FP32':>12}")
    print("-" * 45)
    
    for layer_idx in range(12):
        # Restore all weights
        for name, param in model.named_parameters():
            param.data = original_state[name].clone()
        
        # Binarize only this layer
        prefix = f"transformer.h.{layer_idx}."
        for name, param in model.named_parameters():
            if name.startswith(prefix) and 'weight' in name and param.dim() == 2:
                W = original_state[name]
                scale = W.abs().mean()
                sign = torch.sign(W)
                sign[sign == 0] = 1
                param.data = sign * scale
        
        ppl = compute_ppl()
        gap = (ppl / fp32_ppl - 1) * 100
        print(f"Layer {layer_idx:<15} {ppl:>10.2f} {gap:>+11.1f}%")
    
    # Test binarizing ALL layers
    print("\nBinarizing ALL layers:")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            W = original_state[name]
            scale = W.abs().mean()
            sign = torch.sign(W)
            sign[sign == 0] = 1
            param.data = sign * scale
    
    all_binary_ppl = compute_ppl()
    print(f"All layers binary: {all_binary_ppl:.2f} ({(all_binary_ppl/fp32_ppl-1)*100:+.1f}%)")


if __name__ == '__main__':
    test_single_layer_binary()

