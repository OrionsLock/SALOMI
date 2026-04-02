"""
CRITICAL TEST: The tests that actually matter

KEY FINDINGS SO FAR:
1. Per-layer correlation: VQ wins
2. But Ternary FULL MODEL PPL = 768,107 (CATASTROPHIC!)
3. This means BOTH methods fail at full model level!

Let's focus on:
- Why does ternary fail so badly at full model PPL?
- Is our ternary implementation correct?
- Is there something fundamentally broken?
"""

import numpy as np
import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CRITICAL ANALYSIS: WHY DOES EVERYTHING FAIL?")
print("=" * 80)


def quantize_ternary(W):
    """Standard ternary quantization."""
    thr = np.percentile(np.abs(W), 30)
    W_q = np.zeros_like(W)
    W_q[W > thr] = 1.0
    W_q[W < -thr] = -1.0
    scale = np.mean(np.abs(W[np.abs(W) > thr])) if (np.abs(W) > thr).any() else 1.0
    return W_q * scale


def quantize_rtn(W, bits):
    """RTN uniform quantization."""
    levels = 2 ** bits
    w_min, w_max = W.min(), W.max()
    scale = (w_max - w_min) / (levels - 1) + 1e-10
    return np.round((W - w_min) / scale) * scale + w_min


# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Test text
test_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(test_text, return_tensors="pt")

# Baseline
with torch.no_grad():
    out = model(inputs.input_ids, labels=inputs.input_ids)
    ppl_base = math.exp(out.loss.item())
print(f"FP16 Baseline PPL: {ppl_base:.2f}")


print("\n" + "=" * 80)
print("TEST A: QUANTIZE ONLY FIRST LAYER")
print("=" * 80)

import copy

# Test ternary on just first layer
model_t1 = copy.deepcopy(model)
layer = model_t1.transformer.h[0].mlp.c_fc
W = layer.weight.data.numpy()
W_q = quantize_ternary(W.T).T
layer.weight.data = torch.tensor(W_q, dtype=torch.float32)

with torch.no_grad():
    out = model_t1(inputs.input_ids, labels=inputs.input_ids)
    ppl_t1 = math.exp(out.loss.item())
print(f"Ternary 1 layer PPL: {ppl_t1:.2f} ({ppl_t1/ppl_base:.1f}x)")

# Test RTN 4-bit on just first layer
model_r4 = copy.deepcopy(model)
layer = model_r4.transformer.h[0].mlp.c_fc
W = layer.weight.data.numpy()
W_q = quantize_rtn(W.T, 4).T
layer.weight.data = torch.tensor(W_q, dtype=torch.float32)

with torch.no_grad():
    out = model_r4(inputs.input_ids, labels=inputs.input_ids)
    ppl_r4 = math.exp(out.loss.item())
print(f"RTN 4-bit 1 layer PPL: {ppl_r4:.2f} ({ppl_r4/ppl_base:.1f}x)")


print("\n" + "=" * 80)
print("TEST B: WEIGHT STATISTICS BEFORE/AFTER")
print("=" * 80)

W_orig = model.transformer.h[0].mlp.c_fc.weight.data.numpy()
W_tern = quantize_ternary(W_orig.T).T

print(f"Original W: mean={W_orig.mean():.4f}, std={W_orig.std():.4f}, range=[{W_orig.min():.4f}, {W_orig.max():.4f}]")
print(f"Ternary W:  mean={W_tern.mean():.4f}, std={W_tern.std():.4f}, range=[{W_tern.min():.4f}, {W_tern.max():.4f}]")
print(f"Unique values in ternary: {np.unique(W_tern)}")
print(f"Sparsity: {100 * np.mean(W_tern == 0):.1f}%")


print("\n" + "=" * 80)
print("TEST C: BITNET-STYLE TERNARY (with proper scaling)")
print("=" * 80)

def bitnet_ternary(W):
    """BitNet b1.58 style: {-1, 0, +1} with per-channel scaling."""
    # Absmean quantization
    gamma = np.mean(np.abs(W), axis=1, keepdims=True)
    W_scaled = W / (gamma + 1e-5)
    W_q = np.round(np.clip(W_scaled, -1, 1))
    return W_q * gamma  # Dequantize

model_bn = copy.deepcopy(model)
layer = model_bn.transformer.h[0].mlp.c_fc
W = layer.weight.data.numpy()
W_q = bitnet_ternary(W.T).T
layer.weight.data = torch.tensor(W_q, dtype=torch.float32)

with torch.no_grad():
    out = model_bn(inputs.input_ids, labels=inputs.input_ids)
    ppl_bn = math.exp(out.loss.item())
print(f"BitNet-style 1 layer PPL: {ppl_bn:.2f} ({ppl_bn/ppl_base:.1f}x)")


print("\n" + "=" * 80)
print("TEST D: PROGRESSIVE LAYER QUANTIZATION")
print("=" * 80)

print(f"{'Layers':>10} {'Tern PPL':>12} {'RTN4 PPL':>12}")
print("-" * 40)

for n_layers in [1, 3, 6, 12]:
    # Ternary
    model_t = copy.deepcopy(model)
    for i in range(n_layers):
        for name in ['c_attn', 'c_proj']:
            layer = getattr(model_t.transformer.h[i].attn, name)
            W = layer.weight.data.numpy()
            layer.weight.data = torch.tensor(bitnet_ternary(W.T).T, dtype=torch.float32)
        for name in ['c_fc', 'c_proj']:
            layer = getattr(model_t.transformer.h[i].mlp, name)
            W = layer.weight.data.numpy()
            layer.weight.data = torch.tensor(bitnet_ternary(W.T).T, dtype=torch.float32)
    
    with torch.no_grad():
        out = model_t(inputs.input_ids, labels=inputs.input_ids)
        ppl_t = math.exp(out.loss.item())
    
    # RTN 4-bit
    model_r = copy.deepcopy(model)
    for i in range(n_layers):
        for name in ['c_attn', 'c_proj']:
            layer = getattr(model_r.transformer.h[i].attn, name)
            W = layer.weight.data.numpy()
            layer.weight.data = torch.tensor(quantize_rtn(W.T, 4).T, dtype=torch.float32)
        for name in ['c_fc', 'c_proj']:
            layer = getattr(model_r.transformer.h[i].mlp, name)
            W = layer.weight.data.numpy()
            layer.weight.data = torch.tensor(quantize_rtn(W.T, 4).T, dtype=torch.float32)
    
    with torch.no_grad():
        out = model_r(inputs.input_ids, labels=inputs.input_ids)
        ppl_r = math.exp(out.loss.item())
    
    print(f"{n_layers:>10} {ppl_t:>12.2f} {ppl_r:>12.2f}")


print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("""
If PPL explodes as we add more quantized layers, then:
1. The problem is ERROR ACCUMULATION across layers
2. Per-layer correlation is MISLEADING
3. We need GPTQ-style layer-wise optimization

If PPL stays reasonable:
1. Our ternary implementation was wrong
2. The correlation metric is valid
""")

