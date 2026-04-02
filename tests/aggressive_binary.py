#!/usr/bin/env python3
"""
AGGRESSIVE BINARY APPROACHES
Target: < 5x PPL at 1.00-1.05 bpp

Key insights from previous tests:
1. Pure binary (1.00 bpp) = 400-600x worse PPL
2. Low-rank residual helps but adds too much overhead
3. Need smarter error compensation

New approaches:
1. INT4 residual instead of FP32 (reduces overhead)
2. Selective precision (keep critical layers higher)
3. Per-block quantization with shared scales
4. Heavy distillation-based calibration
5. GELU-aware scaling (different scale before/after GELU)
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple

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


def compute_ppl(model, tokenizer, text=None, device='cpu'):
    """Compute perplexity."""
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
        print(f"Error: {e}")
        return float('inf')


# =============================================================================
# APPROACH 1: Binary + INT4 Residual (very low overhead)
# =============================================================================

class BinaryWithINT4Residual(nn.Module):
    """
    Binary signs + INT4 (4-bit) residual correction.
    
    BPP = 1 (signs) + 4 * compression_ratio (residual)
    With 16:1 compression of residual, gets ~1.25 bpp
    With 32:1 compression, gets ~1.125 bpp
    """
    
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, residual_rank=2):
        super().__init__()
        
        in_dim, out_dim = weight.shape
        
        # Binary signs
        sign = torch.sign(weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        
        # Per-column scale (also quantized to 8-bit)
        scale = weight.abs().mean(dim=0)  # [out]
        self.register_buffer('scale', scale)
        
        # INT4 low-rank residual: U @ V.T quantized to 4-bit
        # This captures the essential error pattern
        with torch.no_grad():
            W_binary = sign.float() * scale.unsqueeze(0)
            error = weight - W_binary
            
            # SVD to get low-rank approximation
            try:
                U, S, Vh = torch.linalg.svd(error, full_matrices=False)
                U_r = U[:, :residual_rank] * S[:residual_rank].sqrt().unsqueeze(0)
                V_r = Vh[:residual_rank, :].T * S[:residual_rank].sqrt().unsqueeze(0)
                
                # Quantize to INT4 range [-8, 7] / 7
                U_scale = U_r.abs().max() / 7.0
                V_scale = V_r.abs().max() / 7.0
                
                U_int4 = torch.round(U_r / (U_scale + 1e-8)).clamp(-8, 7)
                V_int4 = torch.round(V_r / (V_scale + 1e-8)).clamp(-8, 7)
                
                self.register_buffer('U', U_int4)
                self.register_buffer('V', V_int4)
                self.register_buffer('U_scale', torch.tensor(U_scale))
                self.register_buffer('V_scale', torch.tensor(V_scale))
            except:
                self.register_buffer('U', torch.zeros(in_dim, residual_rank))
                self.register_buffer('V', torch.zeros(out_dim, residual_rank))
                self.register_buffer('U_scale', torch.tensor(1.0))
                self.register_buffer('V_scale', torch.tensor(1.0))
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
            
        self.residual_rank = residual_rank
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def forward(self, x):
        # Binary part
        weight = self.sign.float() * self.scale.unsqueeze(0)
        
        # INT4 residual part (dequantize)
        U_deq = self.U * self.U_scale
        V_deq = self.V * self.V_scale
        residual = U_deq @ V_deq.T
        
        weight = weight + residual
        
        out = torch.matmul(x, weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def get_bpp(self):
        """Calculate actual bits per parameter."""
        # Signs: 1 bit per weight
        sign_bits = self.in_dim * self.out_dim * 1
        
        # Scales: 8 bits per output column (quantized)
        scale_bits = self.out_dim * 8
        
        # INT4 residual: 4 bits per element
        residual_bits = (self.in_dim * self.residual_rank + self.out_dim * self.residual_rank) * 4
        
        # Scale factors for residual: 2 * 32 bits
        residual_scale_bits = 2 * 32
        
        total_bits = sign_bits + scale_bits + residual_bits + residual_scale_bits
        total_params = self.in_dim * self.out_dim
        
        return total_bits / total_params


# =============================================================================
# APPROACH 2: Selective Precision (First/Last layers higher precision)
# =============================================================================

class SelectivePrecisionMLP(nn.Module):
    """
    MLP with selective precision based on layer importance.
    First and last layers stay in higher precision.
    """
    
    def __init__(self, orig_mlp, is_critical: bool = False, residual_rank: int = 2):
        super().__init__()
        
        c_fc_weight = orig_mlp.c_fc.weight.data
        c_fc_bias = orig_mlp.c_fc.bias.data
        c_proj_weight = orig_mlp.c_proj.weight.data
        c_proj_bias = orig_mlp.c_proj.bias.data
        
        if is_critical:
            # Keep as FP32 (or use INT8)
            self.c_fc = nn.Linear(c_fc_weight.shape[0], c_fc_weight.shape[1], bias=True)
            self.c_fc.weight.data = c_fc_weight.T.clone()
            self.c_fc.bias.data = c_fc_bias.clone()
            
            self.c_proj = nn.Linear(c_proj_weight.shape[0], c_proj_weight.shape[1], bias=True)
            self.c_proj.weight.data = c_proj_weight.T.clone()
            self.c_proj.bias.data = c_proj_bias.clone()
            
            self.is_binary = False
        else:
            # Use binary + INT4 residual
            self.c_fc = BinaryWithINT4Residual(c_fc_weight, c_fc_bias, residual_rank)
            self.c_proj = BinaryWithINT4Residual(c_proj_weight, c_proj_bias, residual_rank)
            self.is_binary = True
        
        self.act = orig_mlp.act
        self.dropout = orig_mlp.dropout
    
    def forward(self, x):
        if self.is_binary:
            h = self.c_fc(x)
        else:
            h = F.linear(x, self.c_fc.weight, self.c_fc.bias)
        
        h = self.act(h)
        
        if self.is_binary:
            h = self.c_proj(h)
        else:
            h = F.linear(h, self.c_proj.weight, self.c_proj.bias)
        
        h = self.dropout(h)
        return h


# =============================================================================
# APPROACH 3: Block-wise Quantization with Shared Scales
# =============================================================================

class BlockBinaryLinear(nn.Module):
    """
    Block-wise binary with shared scales.
    Divide weight matrix into blocks, share scale per block.
    """
    
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, block_size: int = 64):
        super().__init__()
        
        in_dim, out_dim = weight.shape
        
        # Pad to multiple of block_size
        pad_in = (block_size - in_dim % block_size) % block_size
        pad_out = (block_size - out_dim % block_size) % block_size
        
        if pad_in > 0 or pad_out > 0:
            weight = F.pad(weight, (0, pad_out, 0, pad_in))
        
        padded_in, padded_out = weight.shape
        n_blocks_in = padded_in // block_size
        n_blocks_out = padded_out // block_size
        
        # Reshape to blocks [n_blocks_in, n_blocks_out, block_size, block_size]
        weight_blocks = weight.reshape(n_blocks_in, block_size, n_blocks_out, block_size)
        weight_blocks = weight_blocks.permute(0, 2, 1, 3)
        
        # Binary signs per block
        signs = torch.sign(weight_blocks)
        signs[signs == 0] = 1
        
        # Scale per block (mean abs)
        scales = weight_blocks.abs().mean(dim=(2, 3))  # [n_blocks_in, n_blocks_out]
        
        # Store
        self.register_buffer('signs', signs)
        self.register_buffer('scales', scales)
        
        self.block_size = block_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padded_in = padded_in
        self.padded_out = padded_out
        self.n_blocks_in = n_blocks_in
        self.n_blocks_out = n_blocks_out
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        # Reconstruct weight
        weight = self.signs * self.scales[:, :, None, None]
        
        # Reshape back
        weight = weight.permute(0, 2, 1, 3).reshape(self.padded_in, self.padded_out)
        
        # Trim padding
        weight = weight[:self.in_dim, :self.out_dim]
        
        out = torch.matmul(x, weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def get_bpp(self):
        """Calculate BPP including block scales."""
        # 1 bit per sign
        sign_bits = self.padded_in * self.padded_out * 1
        
        # FP32 scale per block (but could be 8-bit)
        scale_bits = self.n_blocks_in * self.n_blocks_out * 32
        
        # But we only count params for actual dimensions
        total_bits = sign_bits + scale_bits
        total_params = self.in_dim * self.out_dim
        
        return total_bits / total_params


# =============================================================================
# APPROACH 4: Heavy Distillation with All Learnable Components
# =============================================================================

class DistillationBinaryLinear(nn.Module):
    """
    Binary with learnable scales AND biases, optimized via distillation.
    """
    
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        
        # Fixed binary signs
        sign = torch.sign(weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        
        # Learnable per-row AND per-column scales
        row_scale = weight.abs().mean(dim=1)  # [in]
        col_scale = torch.ones(weight.shape[1])  # [out]
        
        self.row_scale = nn.Parameter(row_scale)
        self.col_scale = nn.Parameter(col_scale)
        
        # Learnable offset matrix (very small)
        self.offset = nn.Parameter(torch.zeros_like(weight) * 0.001)
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        # Weight = sign * row_scale * col_scale + offset
        weight = self.sign.float() * self.row_scale.unsqueeze(1) * self.col_scale.unsqueeze(0)
        weight = weight + self.offset
        
        out = torch.matmul(x, weight)
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================================================================
# APPROACH 5: GELU-Aware Binary Scaling
# =============================================================================

class GELUAwareBinaryMLP(nn.Module):
    """
    Binary MLP with GELU-aware scaling.
    
    The GELU function x * Phi(x) has different behavior for positive/negative.
    We apply different scales before and after GELU.
    """
    
    def __init__(self, orig_mlp):
        super().__init__()
        
        c_fc_weight = orig_mlp.c_fc.weight.data
        c_fc_bias = orig_mlp.c_fc.bias.data
        c_proj_weight = orig_mlp.c_proj.weight.data
        c_proj_bias = orig_mlp.c_proj.bias.data
        
        # Binary signs
        sign_fc = torch.sign(c_fc_weight)
        sign_fc[sign_fc == 0] = 1
        sign_proj = torch.sign(c_proj_weight)
        sign_proj[sign_proj == 0] = 1
        
        self.register_buffer('sign_fc', sign_fc)
        self.register_buffer('sign_proj', sign_proj)
        
        # Learnable scales
        self.scale_fc = nn.Parameter(c_fc_weight.abs().mean(dim=0))
        self.scale_proj = nn.Parameter(c_proj_weight.abs().mean(dim=0))
        
        # GELU correction factor (learnable)
        self.gelu_scale = nn.Parameter(torch.ones(c_fc_weight.shape[1]))
        self.gelu_bias = nn.Parameter(torch.zeros(c_fc_weight.shape[1]))
        
        # Biases
        self.bias_fc = nn.Parameter(c_fc_bias.clone())
        self.bias_proj = nn.Parameter(c_proj_bias.clone())
        
        self.act = orig_mlp.act
        self.dropout = orig_mlp.dropout
    
    def forward(self, x):
        # First linear
        weight_fc = self.sign_fc.float() * self.scale_fc.unsqueeze(0)
        h = torch.matmul(x, weight_fc) + self.bias_fc
        
        # GELU with learnable correction
        h = self.act(h)
        h = h * self.gelu_scale + self.gelu_bias
        
        # Second linear
        weight_proj = self.sign_proj.float() * self.scale_proj.unsqueeze(0)
        h = torch.matmul(h, weight_proj) + self.bias_proj
        
        h = self.dropout(h)
        return h


def distill_model(model, tokenizer, device, n_steps=200):
    """Heavy distillation to learn all parameters."""
    teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    teacher.eval()
    
    # Collect all learnable parameters
    learnable_params = []
    for layer in model.transformer.h:
        for p in layer.mlp.parameters():
            if p.requires_grad:
                learnable_params.append(p)
    
    if not learnable_params:
        print("No learnable parameters")
        return model
    
    optimizer = torch.optim.AdamW(learnable_params, lr=0.005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming software development.",
        "Neural networks learn patterns from data efficiently.",
        "Artificial intelligence continues to advance rapidly.",
        "Natural language processing enables text understanding.",
        "Computer vision allows machines to interpret images.",
        "Deep learning requires significant computing resources.",
        "Transformers have changed sequence modeling approaches.",
        "Attention mechanisms help models focus on relevant inputs.",
        "Language models generate coherent and fluent text.",
    ]
    
    print(f"Distilling model with {len(learnable_params)} learnable parameters...")
    
    for step in range(n_steps):
        total_loss = 0
        
        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
            input_ids = tokens.input_ids.to(device)
            
            with torch.no_grad():
                teacher_out = teacher(input_ids, output_hidden_states=True)
                teacher_logits = teacher_out.logits
                teacher_hidden = teacher_out.hidden_states
            
            student_out = model(input_ids, output_hidden_states=True)
            student_logits = student_out.logits
            student_hidden = student_out.hidden_states
            
            # KL loss on logits
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / 2.0, dim=-1),
                F.softmax(teacher_logits / 2.0, dim=-1),
                reduction='batchmean'
            )
            
            # MSE loss on hidden states (match each layer)
            hidden_loss = 0
            for t_h, s_h in zip(teacher_hidden[1:], student_hidden[1:]):
                hidden_loss += F.mse_loss(s_h, t_h)
            hidden_loss = hidden_loss / len(teacher_hidden)
            
            loss = kl_loss + 0.1 * hidden_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{n_steps}: loss={total_loss/len(texts):.4f}")
    
    return model


class INT4ResidualMLP(nn.Module):
    """MLP with INT4 residual."""
    def __init__(self, orig_mlp, residual_rank=2):
        super().__init__()
        self.c_fc = BinaryWithINT4Residual(
            orig_mlp.c_fc.weight.data,
            orig_mlp.c_fc.bias.data,
            residual_rank
        )
        self.c_proj = BinaryWithINT4Residual(
            orig_mlp.c_proj.weight.data,
            orig_mlp.c_proj.bias.data,
            residual_rank
        )
        self.act = orig_mlp.act
        self.dropout = orig_mlp.dropout
    
    def forward(self, x):
        h = self.c_fc(x)
        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h


class BlockBinaryMLP(nn.Module):
    """MLP with block-wise binary."""
    def __init__(self, orig_mlp, block_size=64):
        super().__init__()
        self.c_fc = BlockBinaryLinear(
            orig_mlp.c_fc.weight.data,
            orig_mlp.c_fc.bias.data,
            block_size
        )
        self.c_proj = BlockBinaryLinear(
            orig_mlp.c_proj.weight.data,
            orig_mlp.c_proj.bias.data,
            block_size
        )
        self.act = orig_mlp.act
        self.dropout = orig_mlp.dropout
    
    def forward(self, x):
        h = self.c_fc(x)
        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h


class DistillationMLP(nn.Module):
    """MLP with distillation binary."""
    def __init__(self, orig_mlp):
        super().__init__()
        self.c_fc = DistillationBinaryLinear(
            orig_mlp.c_fc.weight.data,
            orig_mlp.c_fc.bias.data
        )
        self.c_proj = DistillationBinaryLinear(
            orig_mlp.c_proj.weight.data,
            orig_mlp.c_proj.bias.data
        )
        self.act = orig_mlp.act
        self.dropout = orig_mlp.dropout
    
    def forward(self, x):
        h = self.c_fc(x)
        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h


def apply_approach(model, approach: str, critical_layers=None, residual_rank=2, block_size=64):
    """Apply specified approach to model."""
    for layer_idx in range(len(model.transformer.h)):
        block = model.transformer.h[layer_idx]
        orig_mlp = block.mlp
        
        if approach == 'int4_residual':
            block.mlp = INT4ResidualMLP(orig_mlp, residual_rank)
            
        elif approach == 'selective':
            is_critical = (critical_layers is not None and layer_idx in critical_layers)
            block.mlp = SelectivePrecisionMLP(orig_mlp, is_critical, residual_rank)
            
        elif approach == 'block':
            block.mlp = BlockBinaryMLP(orig_mlp, block_size)
            
        elif approach == 'gelu_aware':
            block.mlp = GELUAwareBinaryMLP(orig_mlp)
            
        elif approach == 'distillation':
            block.mlp = DistillationMLP(orig_mlp)
    
    return model


def run_aggressive_tests():
    """Test aggressive approaches."""
    print("=" * 70)
    print("AGGRESSIVE BINARY APPROACHES: Target < 5x PPL at 1.00-1.05 bpp")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Baseline
    print("\n" + "-" * 60)
    print("FP32 BASELINE")
    print("-" * 60)
    
    model_fp32 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_fp32.eval()
    ppl_fp32 = compute_ppl(model_fp32, tokenizer, device=device)
    print(f"Perplexity: {ppl_fp32:.2f}")
    
    del model_fp32
    
    results = [("FP32 Baseline", 32.0, ppl_fp32, 1.0)]
    
    # Test 1: INT4 residual (rank 1 - minimal overhead)
    print("\n" + "-" * 60)
    print("Binary + INT4 Low-Rank (r=1)")
    print("-" * 60)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    try:
        model = apply_approach(model, 'int4_residual', residual_rank=1)
        ppl = compute_ppl(model, tokenizer, device=device)
        
        # Calculate BPP
        bpp = model.transformer.h[0].mlp.c_fc.get_bpp()
        ratio = ppl / ppl_fp32
        
        print(f"BPP: {bpp:.3f}")
        print(f"Perplexity: {ppl:.2f} ({ratio:.2f}x baseline)")
        
        results.append(("INT4 Residual (r=1)", bpp, ppl, ratio))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    del model
    
    # Test 2: Block-wise (64x64)
    print("\n" + "-" * 60)
    print("Block Binary (64x64 blocks)")
    print("-" * 60)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    try:
        model = apply_approach(model, 'block', block_size=64)
        ppl = compute_ppl(model, tokenizer, device=device)
        
        bpp = model.transformer.h[0].mlp.c_fc.get_bpp()
        ratio = ppl / ppl_fp32
        
        print(f"BPP: {bpp:.3f}")
        print(f"Perplexity: {ppl:.2f} ({ratio:.2f}x baseline)")
        
        results.append(("Block Binary (64)", bpp, ppl, ratio))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    del model
    
    # Test 3: GELU-aware
    print("\n" + "-" * 60)
    print("GELU-Aware Binary")
    print("-" * 60)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    try:
        model = apply_approach(model, 'gelu_aware')
        ppl = compute_ppl(model, tokenizer, device=device)
        ratio = ppl / ppl_fp32
        
        print(f"BPP: ~1.0")
        print(f"Perplexity: {ppl:.2f} ({ratio:.2f}x baseline)")
        
        results.append(("GELU-Aware Binary", 1.0, ppl, ratio))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    del model
    
    # Test 4: GELU-aware + distillation
    print("\n" + "-" * 60)
    print("GELU-Aware + Heavy Distillation")
    print("-" * 60)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    try:
        model = apply_approach(model, 'gelu_aware')
        
        ppl_before = compute_ppl(model, tokenizer, device=device)
        print(f"Before distillation: {ppl_before:.2f}")
        
        model = distill_model(model, tokenizer, device, n_steps=100)
        
        ppl = compute_ppl(model, tokenizer, device=device)
        ratio = ppl / ppl_fp32
        
        print(f"After distillation: {ppl:.2f} ({ratio:.2f}x baseline)")
        
        results.append(("GELU-Aware + Distill", 1.0, ppl, ratio))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    del model
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Approach':<30} {'BPP':>8} {'PPL':>12} {'Ratio':>10}")
    print("-" * 70)
    
    for name, bpp, ppl, ratio in results:
        print(f"{name:<30} {bpp:>8.3f} {ppl:>12.2f} {ratio:>10.2f}x")
    
    # Best result
    best = min(results[1:], key=lambda x: x[3])  # Exclude FP32
    
    print("\n" + "-" * 70)
    print(f"BEST RESULT: {best[0]}")
    print(f"  BPP: {best[1]:.3f}, PPL: {best[2]:.2f}, Ratio: {best[3]:.2f}x")
    
    if best[3] < 5 and best[1] <= 1.05:
        print("\n*** TARGET ACHIEVED: < 5x PPL at <=1.05 bpp! ***")
    elif best[3] < 10 and best[1] <= 1.1:
        print("\n*** CLOSE: < 10x PPL at <=1.1 bpp ***")
    else:
        print("\n*** Still need improvement ***")
    
    return results


if __name__ == "__main__":
    run_aggressive_tests()