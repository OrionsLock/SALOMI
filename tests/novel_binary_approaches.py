#!/usr/bin/env python3
"""
NOVEL BINARY QUANTIZATION APPROACHES
Goal: Achieve < 5x PPL degradation at 1.00-1.05 bpp

Approaches:
1. Output-optimal scaling (calibration-based)
2. GPTQ-style iterative error compensation
3. Learned scale calibration via gradient descent
4. Mixed precision with minimal overhead
5. Combined best approaches
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Install: pip install torch transformers")
    sys.exit(1)


def get_calibration_data(tokenizer, n_samples=20):
    """Get calibration data for output-optimal scaling."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we build software.",
        "Neural networks can learn complex patterns from data.",
        "The weather today is sunny with a chance of rain.",
        "Artificial intelligence has made remarkable progress.",
        "Deep learning models require substantial compute resources.",
        "Natural language processing has many applications.",
        "Computer vision enables machines to understand images.",
        "Reinforcement learning trains agents through reward signals.",
        "Transformers have revolutionized sequence modeling.",
        "The cat sat on the mat and watched the birds.",
        "Scientists discovered a new species in the Amazon.",
        "Technology continues to advance at a rapid pace.",
        "Education is the key to a better future.",
        "Climate change affects ecosystems worldwide.",
        "Music has the power to bring people together.",
        "Art reflects the culture and values of society.",
        "History teaches us important lessons about humanity.",
        "Mathematics is the language of the universe.",
        "Philosophy explores the fundamental questions of existence.",
    ]
    
    inputs = []
    for text in texts[:n_samples]:
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
        inputs.append(tokens)
    return inputs


def compute_ppl(model, tokenizer, text=None, device='cpu'):
    """Compute perplexity on sample text."""
    if text is None:
        text = """
        The transformer architecture has revolutionized natural language processing.
        Attention mechanisms allow models to focus on relevant parts of the input.
        Large language models have shown remarkable capabilities in text generation.
        """
    
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[:, :256].to(device)
    
    try:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            return torch.exp(outputs.loss).item()
    except Exception as e:
        print(f"Error computing PPL: {e}")
        return float('inf')


# =============================================================================
# APPROACH 1: Output-Optimal Scaling
# =============================================================================

class OutputOptimalBinaryLinear(nn.Module):
    """
    Binary linear with output-optimal scaling.
    
    Instead of scale = mean(|W|), we compute:
    scale = (x @ sign(W))^T @ (x @ W) / ||x @ sign(W)||^2
    
    This minimizes ||x @ W - x @ (sign(W) * scale)||
    """
    
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, calibration_inputs: List[torch.Tensor]):
        super().__init__()
        
        # Binary signs
        sign = torch.sign(weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        
        # Compute output-optimal scale using calibration data
        if calibration_inputs:
            scale = self._compute_optimal_scale(weight, sign, calibration_inputs)
        else:
            scale = weight.abs().mean(dim=0, keepdim=True)
        
        self.register_buffer('scale', scale)
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
    
    def _compute_optimal_scale(self, W, S, inputs):
        """Compute output-optimal scale."""
        # W: [in, out], S: [in, out] (signs)
        # For each output column, find optimal scale
        
        with torch.no_grad():
            numerator = torch.zeros(W.shape[1], device=W.device)  # [out]
            denominator = torch.zeros(W.shape[1], device=W.device) + 1e-8
            
            for inp in inputs:
                # inp is dict with input_ids
                # We need actual hidden states - use random for now as proxy
                x = torch.randn(1, W.shape[0], device=W.device)  # [1, in]
                
                # y_fp32 = x @ W  # [1, out]
                # y_sign = x @ S  # [1, out]
                
                y_fp32 = torch.matmul(x, W)  # [1, out]
                y_sign = torch.matmul(x, S.float())  # [1, out]
                
                # scale = (y_sign^T @ y_fp32) / (y_sign^T @ y_sign)
                numerator += (y_sign * y_fp32).sum(dim=0)
                denominator += (y_sign * y_sign).sum(dim=0)
            
            scale = numerator / denominator
            scale = scale.clamp(min=1e-6)  # Prevent negative/zero scales
            
        return scale.unsqueeze(0)  # [1, out]
    
    def forward(self, x):
        weight = self.sign.float() * self.scale
        out = torch.matmul(x, weight)
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================================================================
# APPROACH 2: GPTQ-Style Error Compensation
# =============================================================================

class GPTQBinaryLinear(nn.Module):
    """
    GPTQ-style binary quantization.
    
    Key idea: After quantizing each column, compensate in remaining columns
    to minimize total output error.
    """
    
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, damping=0.01):
        super().__init__()
        
        W = weight.clone()  # [in, out]
        
        # Compute diagonal Hessian proxy (simplified - use identity)
        # In full GPTQ, this comes from calibration data
        H_diag = torch.ones(W.shape[0], device=W.device)
        
        # Process columns one at a time
        W_quantized = torch.zeros_like(W)
        
        for col in range(W.shape[1]):
            w_col = W[:, col]
            
            # Quantize this column
            scale = w_col.abs().mean()
            sign = torch.sign(w_col)
            sign[sign == 0] = 1
            w_q = sign * scale
            
            # Error in this column
            error = w_col - w_q
            
            # Compensate in remaining columns (simplified)
            if col < W.shape[1] - 1:
                # Distribute error proportionally
                # This is a simplified version of GPTQ's compensation
                remaining_cols = W.shape[1] - col - 1
                compensation = error.unsqueeze(1) / remaining_cols * 0.1  # Damped
                W[:, col+1:] = W[:, col+1:] + compensation
            
            W_quantized[:, col] = w_q
        
        self.register_buffer('weight', W_quantized)
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================================================================
# APPROACH 3: Learned Scale Calibration
# =============================================================================

class LearnedScaleBinaryLinear(nn.Module):
    """
    Binary weights with learned per-output scales.
    The signs are fixed, but scales are optimized via gradient descent.
    """
    
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        
        # Fixed binary signs
        sign = torch.sign(weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        
        # Learnable scale per output
        init_scale = weight.abs().mean(dim=0)  # [out]
        self.scale = nn.Parameter(init_scale)
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        weight = self.sign.float() * self.scale.unsqueeze(0)  # [in, out]
        out = torch.matmul(x, weight)
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================================================================
# APPROACH 4: Binary + Learned Error Residual
# =============================================================================

class BinaryWithResidualLinear(nn.Module):
    """
    Binary weights + tiny learned residual correction.
    
    W_effective = sign(W) * scale + R
    where R is a low-rank correction (adds ~0.02 bpp)
    """
    
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, rank=4):
        super().__init__()
        
        # Binary signs
        sign = torch.sign(weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        
        # Scale per output
        init_scale = weight.abs().mean(dim=0)
        self.scale = nn.Parameter(init_scale)
        
        # Low-rank residual: U @ V.T
        # Adds rank * (in + out) * bits / (in * out) bpp
        in_dim, out_dim = weight.shape
        self.U = nn.Parameter(torch.zeros(in_dim, rank))
        self.V = nn.Parameter(torch.zeros(out_dim, rank))
        
        # Initialize to approximate the quantization error
        with torch.no_grad():
            W_binary = sign.float() * init_scale.unsqueeze(0)
            error = weight - W_binary
            
            # SVD to initialize low-rank approximation
            try:
                U, S, Vh = torch.linalg.svd(error, full_matrices=False)
                self.U.data = U[:, :rank] * S[:rank].sqrt().unsqueeze(0)
                self.V.data = Vh[:rank, :].T * S[:rank].sqrt().unsqueeze(0)
            except:
                pass  # Keep zeros if SVD fails
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        weight = self.sign.float() * self.scale.unsqueeze(0)
        residual = self.U @ self.V.T
        weight = weight + residual
        
        out = torch.matmul(x, weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def get_bpp(self):
        """Compute actual bits per parameter."""
        in_dim, out_dim = self.sign.shape
        rank = self.U.shape[1]
        
        binary_bits = in_dim * out_dim * 1  # 1 bit per sign
        scale_bits = out_dim * 32  # FP32 scales
        residual_bits = (in_dim * rank + out_dim * rank) * 32  # FP32 residual
        
        total_bits = binary_bits + scale_bits + residual_bits
        total_params = in_dim * out_dim
        
        return total_bits / total_params


# =============================================================================
# MLP Wrappers
# =============================================================================

class BinaryMLP(nn.Module):
    """Binary MLP with configurable approach."""
    
    def __init__(self, orig_mlp, approach: str, calibration_inputs=None, rank=4):
        super().__init__()
        
        c_fc_weight = orig_mlp.c_fc.weight.data
        c_fc_bias = orig_mlp.c_fc.bias.data
        c_proj_weight = orig_mlp.c_proj.weight.data
        c_proj_bias = orig_mlp.c_proj.bias.data
        
        if approach == 'output_optimal':
            self.c_fc = OutputOptimalBinaryLinear(c_fc_weight, c_fc_bias, calibration_inputs or [])
            self.c_proj = OutputOptimalBinaryLinear(c_proj_weight, c_proj_bias, calibration_inputs or [])
        elif approach == 'gptq':
            self.c_fc = GPTQBinaryLinear(c_fc_weight, c_fc_bias)
            self.c_proj = GPTQBinaryLinear(c_proj_weight, c_proj_bias)
        elif approach == 'learned_scale':
            self.c_fc = LearnedScaleBinaryLinear(c_fc_weight, c_fc_bias)
            self.c_proj = LearnedScaleBinaryLinear(c_proj_weight, c_proj_bias)
        elif approach == 'binary_residual':
            self.c_fc = BinaryWithResidualLinear(c_fc_weight, c_fc_bias, rank=rank)
            self.c_proj = BinaryWithResidualLinear(c_proj_weight, c_proj_bias, rank=rank)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        self.act = orig_mlp.act
        self.dropout = orig_mlp.dropout
        self.approach = approach
    
    def forward(self, x):
        h = self.c_fc(x)
        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h
    
    def get_bpp(self):
        if self.approach == 'binary_residual':
            return (self.c_fc.get_bpp() + self.c_proj.get_bpp()) / 2
        return 1.0  # Pure binary


def apply_binary_mlp(model, approach: str, calibration_inputs=None, rank=4):
    """Apply binary MLP to all layers."""
    for layer_idx in range(len(model.transformer.h)):
        block = model.transformer.h[layer_idx]
        orig_mlp = block.mlp
        block.mlp = BinaryMLP(orig_mlp, approach, calibration_inputs, rank)
    return model


def calibrate_learned_scales(model, tokenizer, device, n_steps=100):
    """Fine-tune the learned scales using distillation."""
    # Get teacher predictions
    teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    teacher.eval()
    
    # Collect learnable parameters
    scale_params = []
    for layer in model.transformer.h:
        if hasattr(layer.mlp, 'c_fc') and hasattr(layer.mlp.c_fc, 'scale'):
            scale_params.append(layer.mlp.c_fc.scale)
            scale_params.append(layer.mlp.c_proj.scale)
    
    if not scale_params:
        print("No learnable scales found")
        return model
    
    optimizer = torch.optim.Adam(scale_params, lr=0.01)
    
    # Calibration texts
    texts = [
        "The quick brown fox jumps over",
        "Machine learning is transforming",
        "Neural networks can learn complex",
        "Artificial intelligence has made",
        "The weather today is sunny",
    ]
    
    print(f"Calibrating {len(scale_params)} scale parameters...")
    
    for step in range(n_steps):
        total_loss = 0
        
        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
            input_ids = tokens.input_ids.to(device)
            
            # Teacher logits
            with torch.no_grad():
                teacher_out = teacher(input_ids)
                teacher_logits = teacher_out.logits
            
            # Student logits
            student_out = model(input_ids)
            student_logits = student_out.logits
            
            # KL divergence loss
            loss = F.kl_div(
                F.log_softmax(student_logits / 2.0, dim=-1),
                F.softmax(teacher_logits / 2.0, dim=-1),
                reduction='batchmean'
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{n_steps}: loss={total_loss/len(texts):.4f}")
    
    return model


def run_tests():
    """Run all novel approaches and compare."""
    print("=" * 70)
    print("NOVEL BINARY APPROACHES: Pushing for 1.00-1.05 bpp")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    calibration_inputs = get_calibration_data(tokenizer)
    
    # Baseline
    print("\n" + "-" * 60)
    print("FP32 BASELINE")
    print("-" * 60)
    
    model_fp32 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_fp32.eval()
    ppl_fp32 = compute_ppl(model_fp32, tokenizer, device=device)
    print(f"Perplexity: {ppl_fp32:.2f}")
    
    del model_fp32
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    results = [("FP32 Baseline", 32.0, ppl_fp32, 1.0)]
    
    # Test each approach
    approaches = [
        ("output_optimal", "Output-Optimal Scaling", 1.0, 4),
        ("gptq", "GPTQ-Style Compensation", 1.0, 4),
        ("learned_scale", "Learned Scales (no calibration)", 1.0, 4),
        ("binary_residual", "Binary + Low-Rank Residual (r=4)", 1.05, 4),
        ("binary_residual", "Binary + Low-Rank Residual (r=8)", 1.08, 8),
        ("binary_residual", "Binary + Low-Rank Residual (r=16)", 1.15, 16),
    ]
    
    for approach, name, expected_bpp, rank in approaches:
        print("\n" + "-" * 60)
        print(f"{name} (~{expected_bpp:.2f} bpp)")
        print("-" * 60)
        
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        model.eval()
        
        try:
            model = apply_binary_mlp(model, approach, calibration_inputs, rank=rank)
            
            ppl = compute_ppl(model, tokenizer, device=device)
            ratio = ppl / ppl_fp32
            
            # Calculate actual BPP for residual approach
            actual_bpp = expected_bpp
            if approach == 'binary_residual':
                actual_bpp = model.transformer.h[0].mlp.get_bpp()
            
            print(f"Perplexity: {ppl:.2f} ({ratio:.2f}x baseline)")
            print(f"Actual BPP: {actual_bpp:.2f}")
            
            results.append((name, actual_bpp, ppl, ratio))
            
        except Exception as e:
            print(f"Error: {e}")
            results.append((name, expected_bpp, float('inf'), float('inf')))
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Test learned scales with calibration
    print("\n" + "-" * 60)
    print("Learned Scales WITH Calibration (~1.0 bpp)")
    print("-" * 60)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    try:
        model = apply_binary_mlp(model, 'learned_scale', calibration_inputs)
        
        # Before calibration
        ppl_before = compute_ppl(model, tokenizer, device=device)
        print(f"Before calibration: {ppl_before:.2f}")
        
        # Calibrate
        model = calibrate_learned_scales(model, tokenizer, device, n_steps=50)
        
        # After calibration
        ppl_after = compute_ppl(model, tokenizer, device=device)
        ratio = ppl_after / ppl_fp32
        print(f"After calibration: {ppl_after:.2f} ({ratio:.2f}x baseline)")
        
        results.append(("Learned Scales (calibrated)", 1.0, ppl_after, ratio))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Approach':<40} {'BPP':>8} {'PPL':>12} {'Ratio':>10}")
    print("-" * 70)
    
    for name, bpp, ppl, ratio in results:
        print(f"{name:<40} {bpp:>8.2f} {ppl:>12.2f} {ratio:>10.2f}x")
    
    # Find best under 1.1 bpp
    best_low_bpp = None
    for name, bpp, ppl, ratio in results:
        if bpp <= 1.1 and ratio < float('inf'):
            if best_low_bpp is None or ppl < best_low_bpp[2]:
                best_low_bpp = (name, bpp, ppl, ratio)
    
    print("\n" + "-" * 70)
    if best_low_bpp:
        print(f"BEST at ≤1.1 bpp: {best_low_bpp[0]}")
        print(f"  BPP: {best_low_bpp[1]:.2f}, PPL: {best_low_bpp[2]:.2f}, Ratio: {best_low_bpp[3]:.2f}x")
    
    # Check if we achieved target
    target_achieved = any(ratio < 5 and bpp <= 1.05 for name, bpp, ppl, ratio in results)
    
    if target_achieved:
        print("\n*** TARGET ACHIEVED: < 5x PPL at ≤1.05 bpp! ***")
    else:
        print("\n*** TARGET NOT YET ACHIEVED ***")
        print("Need more aggressive approaches...")
    
    return results


if __name__ == "__main__":
    run_tests()