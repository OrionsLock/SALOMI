"""
SIMPLE CALIBRATION: Learn just per-layer scales for binary weights.
Much more stable than low-rank calibration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class ScaledBinaryConv1D(nn.Module):
    """Binary signs with learnable scalar scale per layer."""
    def __init__(self, orig_weight):
        super().__init__()
        sign = torch.sign(orig_weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        self.scale = nn.Parameter(torch.tensor(orig_weight.abs().mean().item()))
        
    def forward(self, x):
        return x @ (self.sign * F.softplus(self.scale))  # softplus ensures positive


def compute_ppl(model, tokenizer, texts, max_samples=50):
    losses = []
    for i, text in enumerate(texts):
        if i >= max_samples:
            break
        if len(text.strip()) < 20:
            continue
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True)
        if tokens.shape[1] < 5:
            continue
        with torch.no_grad():
            out = model(tokens, labels=tokens)
            if not torch.isnan(out.loss):
                losses.append(out.loss.item())
    return np.exp(np.mean(losses)) if losses else float('inf')


def main():
    print("=" * 70)
    print("SIMPLE CALIBRATION: Per-Layer Scales Only")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Simple diverse texts for calibration
    train_texts = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Machine learning algorithms can process vast amounts of data efficiently.",
        "Scientists recently discovered a new species in the Amazon rainforest.",
        "The stock market showed significant gains in the technology sector.",
        "Climate change poses serious risks to global food production systems.",
        "Modern architecture combines functionality with innovative design elements.",
        "The history of mathematics spans thousands of years of human discovery.",
        "Quantum computers may revolutionize cryptography and drug discovery.",
    ]
    
    val_texts = [
        "Artificial neural networks are inspired by biological brain structures.",
        "The Renaissance marked a period of cultural rebirth in Europe.",
        "Electric vehicles are becoming increasingly popular worldwide.",
        "Deep ocean exploration has revealed many unknown species.",
    ]
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    fp32_ppl = compute_ppl(model, tokenizer, val_texts)
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    # Get teacher outputs
    teacher_outputs = []
    for text in train_texts:
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True)
        with torch.no_grad():
            teacher_outputs.append((tokens, model(tokens).logits.detach()))
    
    # Replace with scaled binary
    scale_params = []
    for layer_idx in range(12):
        block = model.transformer.h[layer_idx]
        for name in ['c_fc', 'c_proj']:
            orig = getattr(block.mlp, name).weight.data.clone()
            scaled = ScaledBinaryConv1D(orig)
            scale_params.append(scaled.scale)
            setattr(block.mlp, name, scaled)
        for name in ['c_attn', 'c_proj']:
            orig = getattr(block.attn, name).weight.data.clone()
            scaled = ScaledBinaryConv1D(orig)
            scale_params.append(scaled.scale)
            setattr(block.attn, name, scaled)
    
    before_ppl = compute_ppl(model, tokenizer, val_texts)
    print(f"Before calibration: {before_ppl:.2f}")
    
    # Simple scale calibration
    optimizer = torch.optim.Adam(scale_params, lr=0.1)
    print("\nCalibrating scales...")
    
    for epoch in range(200):
        total_loss = 0
        for tokens, teacher_logits in teacher_outputs:
            optimizer.zero_grad()
            student_logits = model(tokens).logits
            loss = F.mse_loss(student_logits, teacher_logits)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            ppl = compute_ppl(model, tokenizer, val_texts)
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, val_ppl={ppl:.2f}")
    
    after_ppl = compute_ppl(model, tokenizer, val_texts)
    
    # BPP: 1 bit per weight + 1 FP32 scale per layer (48 total for 85M weights)
    n_weights = 84_934_656
    n_scales = 48
    bpp = 1.0 + (n_scales * 32) / n_weights  # Essentially 1.00
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Method':<30} {'BPP':>8} {'PPL':>10} {'vs FP32':>12}")
    print("-" * 62)
    print(f"{'FP32':<30} {'32.00':>8} {fp32_ppl:>10.2f} {'baseline':>12}")
    print(f"{'Before calib':<30} {'1.00':>8} {before_ppl:>10.2f} {(before_ppl/fp32_ppl-1)*100:>+11.0f}%")
    print(f"{'After calib (scales only)':<30} {bpp:>8.4f} {after_ppl:>10.2f} {(after_ppl/fp32_ppl-1)*100:>+11.1f}%")


if __name__ == '__main__':
    main()

