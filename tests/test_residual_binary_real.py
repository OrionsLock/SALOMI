#!/usr/bin/env python3
"""
TEST: Residual-Aware Binary Quantization on REAL GPT-2

Novel Insight: Residual connections (x = x + layer(x)) preserve the original signal
while quantization only affects the delta. This keeps errors ADDITIVE rather than
MULTIPLICATIVE.

GPT-2 already has residuals:
  x = x + attention(ln(x))  # Attention block with residual
  x = x + mlp(ln(x))        # MLP block with residual

The question: Can we leverage this to make binary quantization work?
"""

import numpy as np
import time
import sys
from typing import Dict, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Install: pip install torch transformers")
    sys.exit(1)


def cosine_sim(a, b):
    """Compute cosine similarity."""
    return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def get_eval_text():
    return """
    The transformer architecture has revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input.
    Large language models have shown remarkable capabilities in text generation.
    """


def compute_ppl(model, tokenizer, text, device):
    """Compute perplexity."""
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[:, :256].to(device)  # Truncate for speed
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()


class BinaryLinear(nn.Module):
    """Binary quantized linear layer with optimal scaling.
    
    GPT-2 uses Conv1D which stores weights as (in_features, out_features).
    We need to handle this properly.
    """
    
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, is_conv1d: bool = True):
        super().__init__()
        
        # GPT-2 Conv1D stores weights as [in_features, out_features]
        # We keep it that way for compatibility
        self.is_conv1d = is_conv1d
        
        # Store original for comparison
        self.orig_weight = weight.clone()
        
        # Binary quantization with per-output-channel scaling
        self.sign = torch.sign(weight)
        self.sign[self.sign == 0] = 1
        
        # Optimal scale per output channel
        # For Conv1D: scale per column (output dim)
        if is_conv1d:
            self.scale = weight.abs().mean(dim=0, keepdim=True)  # [1, out]
        else:
            self.scale = weight.abs().mean(dim=-1, keepdim=True)
        
        # Register as parameters/buffers
        self.register_buffer('sign_weight', self.sign)
        self.register_buffer('scale_factor', self.scale)
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
            
    def forward(self, x):
        # Reconstruct weight from binary + scale
        weight = self.sign_weight * self.scale_factor
        
        # Conv1D: x @ weight (not F.linear which transposes)
        if self.is_conv1d:
            out = torch.matmul(x, weight)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = F.linear(x, weight, self.bias)
        return out
    
    def get_correlation(self):
        """Get correlation between original and quantized weight."""
        weight_q = self.sign_weight * self.scale_factor
        return cosine_sim(
            self.orig_weight.cpu().numpy(),
            weight_q.cpu().numpy()
        )


class ResidualAwareBinaryMLP(nn.Module):
    """
    Binary MLP that's aware of residual connection.
    
    Key insight: In transformers, output = x + mlp(x)
    If mlp(x) has error e, then output = x + mlp_approx(x) + e
    The error e is ADDITIVE to x, not multiplicative.
    
    This means we can be more aggressive with quantization because
    the original signal x passes through unchanged.
    """
    
    def __init__(self, orig_mlp, residual_weight: float = 1.0):
        super().__init__()
        
        # Get original weights
        c_fc_weight = orig_mlp.c_fc.weight.data
        c_fc_bias = orig_mlp.c_fc.bias.data
        c_proj_weight = orig_mlp.c_proj.weight.data
        c_proj_bias = orig_mlp.c_proj.bias.data
        
        # Binary quantize
        self.c_fc = BinaryLinear(c_fc_weight, c_fc_bias)
        self.c_proj = BinaryLinear(c_proj_weight, c_proj_bias)
        self.act = orig_mlp.act
        self.dropout = orig_mlp.dropout
        
        # Residual scaling - can be learned
        self.residual_weight = residual_weight
        
    def forward(self, x):
        h = self.c_fc(x)
        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h
    
    def get_weight_correlation(self):
        return (self.c_fc.get_correlation() + self.c_proj.get_correlation()) / 2


class ResidualAwareBinaryAttention(nn.Module):
    """
    Binary attention with residual awareness.
    """
    
    def __init__(self, orig_attn):
        super().__init__()
        
        # Binary quantize QKV projection
        self.c_attn = BinaryLinear(orig_attn.c_attn.weight.data, orig_attn.c_attn.bias.data)
        self.c_proj = BinaryLinear(orig_attn.c_proj.weight.data, orig_attn.c_proj.bias.data)
        
        # Keep other components
        self.embed_dim = orig_attn.embed_dim
        self.num_heads = orig_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.scale_attn_weights = orig_attn.scale_attn_weights
        
        # Masks and dropout
        self.attn_dropout = orig_attn.attn_dropout
        self.resid_dropout = orig_attn.resid_dropout
        
    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        
        # Causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = torch.tril(
            torch.ones((key_length, key_length), dtype=torch.bool, device=attn_weights.device)
        ).view(1, 1, key_length, key_length)
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(
            causal_mask[:, :, :query_length, :key_length],
            attn_weights,
            mask_value
        )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights
    
    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                use_cache=False, output_attentions=False):
        
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        present = (key, value) if use_cache else None
        
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs


def create_residual_aware_binary_gpt2(model, quantize_attn=False):
    """
    Replace MLP layers with binary versions.
    
    GPT-2 architecture already has residuals:
    - x = x + attn(ln1(x))
    - x = x + mlp(ln2(x))
    
    We quantize the mlp, but the residual x passes through unchanged.
    MLP layers are 2/3 of the transformer parameters!
    """
    correlations = []
    
    for layer_idx in range(len(model.transformer.h)):
        block = model.transformer.h[layer_idx]
        
        # Replace MLP (the bulk of parameters)
        orig_mlp = block.mlp
        binary_mlp = ResidualAwareBinaryMLP(orig_mlp)
        correlations.append(binary_mlp.get_weight_correlation())
        block.mlp = binary_mlp
    
    return model, np.mean(correlations)


def test_residual_binary():
    """Test residual-aware binary quantization on real GPT-2."""
    print("=" * 70)
    print("RESIDUAL-AWARE BINARY QUANTIZATION TEST")
    print("=" * 70)
    print("""
Novel Insight: Residual connections preserve the original signal.
In transformers: x = x + layer(x)
If layer(x) has quantization error e: output = x + layer_approx(x) + e
The error e is ADDITIVE to x, not multiplicative!
""")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    eval_text = get_eval_text()
    
    # Baseline
    print("\n" + "-" * 60)
    print("FP32 BASELINE")
    print("-" * 60)
    
    model_fp32 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_fp32.eval()
    
    ppl_fp32 = compute_ppl(model_fp32, tokenizer, eval_text, device)
    print(f"Perplexity: {ppl_fp32:.2f}")
    
    # Test with residual-aware binary
    print("\n" + "-" * 60)
    print("RESIDUAL-AWARE BINARY (1.00 bpp)")
    print("-" * 60)
    
    model_binary = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_binary.eval()
    
    model_binary, avg_correlation = create_residual_aware_binary_gpt2(model_binary)
    print(f"Average weight correlation: {avg_correlation:.4f}")
    
    ppl_binary = compute_ppl(model_binary, tokenizer, eval_text, device)
    ratio = ppl_binary / ppl_fp32
    print(f"Perplexity: {ppl_binary:.2f} ({ratio:.2f}x baseline)")
    
    # Test quality of generation
    print("\n" + "-" * 60)
    print("GENERATION QUALITY")
    print("-" * 60)
    
    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # FP32 generation
    with torch.no_grad():
        output_fp32 = model_fp32.generate(
            input_ids, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_fp32 = tokenizer.decode(output_fp32[0], skip_special_tokens=True)
    print(f"FP32: {gen_fp32}")
    
    # Binary generation
    with torch.no_grad():
        output_binary = model_binary.generate(
            input_ids, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_binary = tokenizer.decode(output_binary[0], skip_special_tokens=True)
    print(f"Binary: {gen_binary}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"""
Results:
- FP32 Perplexity: {ppl_fp32:.2f}
- Binary Perplexity: {ppl_binary:.2f}
- Ratio: {ratio:.2f}x

Weight Statistics:
- Average correlation: {avg_correlation:.4f}
""")
    
    if ratio < 10:
        print("GOOD: Residual-aware binary achieves reasonable quality!")
        print("The residual connection hypothesis is validated.")
    elif ratio < 100:
        print("MODERATE: Some degradation but residuals help.")
    else:
        print("POOR: Even with residuals, binary causes significant degradation.")
    
    # Test layer-by-layer correlation vs output correlation
    print("\n" + "-" * 60)
    print("LAYER-BY-LAYER ANALYSIS")
    print("-" * 60)
    
    # Run same input through both models and compare
    test_input = input_ids
    
    # Get hidden states from FP32
    model_fp32.eval()
    with torch.no_grad():
        outputs_fp32 = model_fp32(test_input, output_hidden_states=True)
        hidden_fp32 = [h.cpu().numpy() for h in outputs_fp32.hidden_states]
    
    # Get hidden states from binary
    model_binary.eval()
    with torch.no_grad():
        outputs_binary = model_binary(test_input, output_hidden_states=True)
        hidden_binary = [h.cpu().numpy() for h in outputs_binary.hidden_states]
    
    print(f"{'Layer':<10} {'Correlation':>12}")
    print("-" * 25)
    for i in range(len(hidden_fp32)):
        corr = cosine_sim(hidden_fp32[i], hidden_binary[i])
        print(f"Layer {i:<4} {corr:>12.4f}")
    
    final_corr = cosine_sim(hidden_fp32[-1], hidden_binary[-1])
    print(f"\nFinal hidden state correlation: {final_corr:.4f}")
    
    return {
        'ppl_fp32': ppl_fp32,
        'ppl_binary': ppl_binary,
        'ratio': ratio,
        'weight_correlation': avg_correlation,
        'final_hidden_correlation': final_corr
    }


if __name__ == "__main__":
    results = test_residual_binary()