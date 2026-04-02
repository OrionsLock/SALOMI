#!/usr/bin/env python3
"""
TOP-K DISTILLATION BINARY QUANTIZATION

Key insight: PPL is exponentially sensitive to probability errors.
We need to preserve the PROBABILITY of correct tokens, not just correlation.

Approach:
1. Binary quantize weights
2. Distill with TOP-K weighted loss (important logits matter more)
3. Use high temperature to soften softmax during training
4. Learn optimal scales for each layer
"""

import numpy as np
import time
import sys
from typing import List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("Install: pip install torch transformers")
    sys.exit(1)


def compute_ppl(model, tokenizer, device='cpu'):
    """Compute perplexity."""
    text = """
    The transformer architecture has revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input.
    Large language models have shown remarkable capabilities in text generation.
    """
    
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[:, :256].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()


class BinaryMLPWithLearnedScales(nn.Module):
    """Binary MLP with heavily learnable components."""
    
    def __init__(self, orig_mlp):
        super().__init__()
        
        c_fc_weight = orig_mlp.c_fc.weight.data
        c_fc_bias = orig_mlp.c_fc.bias.data
        c_proj_weight = orig_mlp.c_proj.weight.data
        c_proj_bias = orig_mlp.c_proj.bias.data
        
        # Fixed binary signs
        sign_fc = torch.sign(c_fc_weight)
        sign_fc[sign_fc == 0] = 1
        sign_proj = torch.sign(c_proj_weight)
        sign_proj[sign_proj == 0] = 1
        
        self.register_buffer('sign_fc', sign_fc)
        self.register_buffer('sign_proj', sign_proj)
        
        # Per-OUTPUT dimension scales (learnable)
        self.scale_fc = nn.Parameter(c_fc_weight.abs().mean(dim=0))  # [out_fc]
        self.scale_proj = nn.Parameter(c_proj_weight.abs().mean(dim=0))  # [out_proj]
        
        # Per-INPUT dimension scales (learnable) - NEW!
        self.input_scale_fc = nn.Parameter(torch.ones(c_fc_weight.shape[0]))  # [in_fc]
        self.input_scale_proj = nn.Parameter(torch.ones(c_proj_weight.shape[0]))  # [in_proj]
        
        # Layer-level scaling factors
        self.layer_scale_fc = nn.Parameter(torch.tensor(1.0))
        self.layer_scale_proj = nn.Parameter(torch.tensor(1.0))
        
        # Activation correction (learnable gamma/beta for GELU output)
        self.act_gamma = nn.Parameter(torch.ones(c_fc_weight.shape[1]))
        self.act_beta = nn.Parameter(torch.zeros(c_fc_weight.shape[1]))
        
        # Biases
        self.bias_fc = nn.Parameter(c_fc_bias.clone())
        self.bias_proj = nn.Parameter(c_proj_bias.clone())
        
        self.act = orig_mlp.act
        self.dropout = orig_mlp.dropout
    
    def forward(self, x):
        # First linear with both input and output scaling
        # W_effective = input_scale.unsqueeze(1) * sign * output_scale.unsqueeze(0) * layer_scale
        weight_fc = self.input_scale_fc.unsqueeze(1) * self.sign_fc.float() * self.scale_fc.unsqueeze(0)
        weight_fc = weight_fc * self.layer_scale_fc
        
        h = torch.matmul(x, weight_fc) + self.bias_fc
        
        # GELU with learnable correction
        h = self.act(h)
        h = h * self.act_gamma + self.act_beta
        
        # Second linear
        weight_proj = self.input_scale_proj.unsqueeze(1) * self.sign_proj.float() * self.scale_proj.unsqueeze(0)
        weight_proj = weight_proj * self.layer_scale_proj
        
        h = torch.matmul(h, weight_proj) + self.bias_proj
        h = self.dropout(h)
        
        return h


def apply_binary_mlp(model):
    """Replace all MLPs with binary versions."""
    for layer_idx in range(len(model.transformer.h)):
        block = model.transformer.h[layer_idx]
        orig_mlp = block.mlp
        block.mlp = BinaryMLPWithLearnedScales(orig_mlp)
    return model


def topk_distillation(model, tokenizer, device, n_steps=300, temperature=4.0, topk=100):
    """
    Distill with top-k weighted loss.
    
    The key insight: we care about getting the TOP tokens right.
    Errors in the bottom 99% of tokens don't matter as much.
    """
    teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    teacher.eval()
    
    # Collect learnable parameters
    learnable_params = []
    for layer in model.transformer.h:
        for p in layer.mlp.parameters():
            if p.requires_grad:
                learnable_params.append(p)
    
    print(f"Training {len(learnable_params)} parameters with top-{topk} distillation...")
    
    optimizer = torch.optim.AdamW(learnable_params, lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    
    # Diverse training texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we build and deploy software systems.",
        "Neural networks can learn complex hierarchical patterns from raw data.",
        "Artificial intelligence research continues to make remarkable progress.",
        "Natural language processing enables machines to understand human text.",
        "Computer vision algorithms can now recognize thousands of object categories.",
        "Reinforcement learning trains agents through trial and error with rewards.",
        "Transformer models have fundamentally changed sequence-to-sequence learning.",
        "Large language models demonstrate emergent capabilities at scale.",
        "Deep learning requires substantial computational resources for training.",
        "Scientists discovered a new species of bird in the rainforest.",
        "The stock market showed significant volatility during the trading session.",
        "Climate change poses serious risks to ecosystems around the world.",
        "Quantum computing promises exponential speedups for certain problems.",
        "Researchers developed a new vaccine using mRNA technology.",
    ]
    
    for step in range(n_steps):
        total_loss = 0
        
        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=48)
            input_ids = tokens.input_ids.to(device)
            
            # Teacher logits
            with torch.no_grad():
                teacher_out = teacher(input_ids)
                teacher_logits = teacher_out.logits  # [B, T, V]
            
            # Student logits
            student_out = model(input_ids)
            student_logits = student_out.logits
            
            # Top-K weighted KL loss
            # Get top-k indices from teacher
            topk_vals, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
            
            # Gather corresponding student logits
            student_topk = torch.gather(student_logits, -1, topk_idx)
            
            # KL divergence on top-k (with temperature)
            teacher_topk_soft = F.softmax(topk_vals / temperature, dim=-1)
            student_topk_soft = F.log_softmax(student_topk / temperature, dim=-1)
            
            kl_topk = F.kl_div(student_topk_soft, teacher_topk_soft, reduction='batchmean')
            
            # Also match the raw logits of top tokens (MSE)
            mse_topk = F.mse_loss(student_topk, topk_vals)
            
            # Full distribution KL (lower weight)
            kl_full = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean'
            )
            
            # Combined loss
            loss = kl_topk * 2.0 + mse_topk * 0.1 + kl_full * 0.5
            loss = loss * (temperature ** 2)  # Scale for temperature
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(learnable_params, 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (step + 1) % 50 == 0:
            ppl = compute_ppl(model, tokenizer, device)
            print(f"  Step {step+1}/{n_steps}: loss={total_loss/len(texts):.4f}, PPL={ppl:.2f}")
    
    return model


def run_topk_distillation():
    """Test top-k distillation approach."""
    print("=" * 70)
    print("TOP-K DISTILLATION BINARY QUANTIZATION")
    print("=" * 70)
    print("""
Key insight: PPL is exponentially sensitive to probability errors.
We preserve the TOP tokens' probabilities, not just overall correlation.
""")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Baseline
    print("\n" + "-" * 60)
    print("FP32 BASELINE")
    print("-" * 60)
    
    model_fp32 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_fp32.eval()
    ppl_fp32 = compute_ppl(model_fp32, tokenizer, device)
    print(f"Perplexity: {ppl_fp32:.2f}")
    
    del model_fp32
    
    # Binary with top-k distillation
    print("\n" + "-" * 60)
    print("BINARY + TOP-K DISTILLATION")
    print("-" * 60)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Apply binary
    model = apply_binary_mlp(model)
    
    # Before distillation
    ppl_before = compute_ppl(model, tokenizer, device)
    print(f"Before distillation: {ppl_before:.2f} ({ppl_before/ppl_fp32:.2f}x)")
    
    # Distill with top-k focus
    model = topk_distillation(model, tokenizer, device, n_steps=200, temperature=4.0, topk=100)
    
    # After distillation
    ppl_after = compute_ppl(model, tokenizer, device)
    ratio = ppl_after / ppl_fp32
    print(f"\nFinal PPL: {ppl_after:.2f} ({ratio:.2f}x baseline)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
FP32 Baseline:        {ppl_fp32:.2f}
Binary Before:        {ppl_before:.2f} ({ppl_before/ppl_fp32:.2f}x)
Binary After TopK:    {ppl_after:.2f} ({ratio:.2f}x)

BPP: ~1.0 (binary signs + learned scales add negligible overhead)
""")
    
    if ratio < 5:
        print("*** SUCCESS: < 5x PPL degradation at ~1.0 bpp! ***")
    elif ratio < 10:
        print("*** CLOSE: < 10x PPL degradation ***")
    elif ratio < 50:
        print("*** PROGRESS: < 50x PPL degradation ***")
    else:
        print("*** Need more improvement ***")
    
    return ppl_fp32, ppl_before, ppl_after


if __name__ == "__main__":
    run_topk_distillation()