"""
PROGRESSIVE CALIBRATION: Calibrate layer-by-layer using actual model outputs.
After calibrating layer i, use the calibrated model to get activations for layer i+1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class LowRankBinaryConv1D(nn.Module):
    def __init__(self, orig_weight, rank=8):
        super().__init__()
        n_in, n_out = orig_weight.shape
        sign = torch.sign(orig_weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        W_abs = orig_weight.abs()
        U, S, Vh = torch.linalg.svd(W_abs, full_matrices=False)
        self.U = nn.Parameter(U[:, :rank] * S[:rank].sqrt().unsqueeze(0))
        self.V = nn.Parameter(Vh[:rank, :].T * S[:rank].sqrt().unsqueeze(0))
        
    def forward(self, x):
        mag = F.relu(self.U @ self.V.T) + 1e-8
        return x @ (self.sign * mag)


def compute_ppl(model, tokenizer, texts):
    losses = []
    for text in texts:
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
    print("PROGRESSIVE CALIBRATION")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    train_texts = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Machine learning algorithms process vast amounts of data efficiently.",
        "Scientists recently discovered a new species in the Amazon rainforest.",
        "The stock market showed significant gains in the technology sector.",
        "Climate change poses serious risks to global food production systems.",
        "Modern architecture combines functionality with innovative design.",
        "The history of mathematics spans thousands of years of discovery.",
        "Quantum computers may revolutionize cryptography and drug discovery.",
    ]
    
    val_texts = [
        "Neural networks can learn complex patterns from training data.",
        "Deep ocean exploration has revealed many unknown species.",
        "Electric vehicles are becoming popular as technology improves.",
        "Space exploration continues to push the boundaries of science.",
    ]
    
    # Teacher model (FP32)
    teacher = GPT2LMHeadModel.from_pretrained('gpt2')
    teacher.eval()
    fp32_ppl = compute_ppl(teacher, tokenizer, val_texts)
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    # Student model (will be progressively binarized)
    student = GPT2LMHeadModel.from_pretrained('gpt2')
    student.eval()
    
    # Get teacher outputs for distillation
    teacher_outputs = []
    for text in train_texts:
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True)
        with torch.no_grad():
            teacher_outputs.append((tokens, teacher(tokens).logits.clone()))
    
    RANK = 8
    
    # Progressive layer-by-layer calibration
    for layer_idx in range(12):
        block = student.transformer.h[layer_idx]
        
        # Replace this layer's weights with calibrated binary
        calibrated_params = []
        for name in ['c_fc', 'c_proj']:
            orig = getattr(block.mlp, name).weight.data.clone()
            calib = LowRankBinaryConv1D(orig, rank=RANK)
            calibrated_params.extend([calib.U, calib.V])
            setattr(block.mlp, name, calib)
        for name in ['c_attn', 'c_proj']:
            orig = getattr(block.attn, name).weight.data.clone()
            calib = LowRankBinaryConv1D(orig, rank=RANK)
            calibrated_params.extend([calib.U, calib.V])
            setattr(block.attn, name, calib)
        
        # Calibrate this layer using end-to-end distillation
        optimizer = torch.optim.Adam(calibrated_params, lr=1e-3)
        
        for epoch in range(50):
            for tokens, teacher_logits in teacher_outputs:
                optimizer.zero_grad()
                student_logits = student(tokens).logits
                loss = F.mse_loss(student_logits, teacher_logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(calibrated_params, 1.0)
                optimizer.step()
        
        # Freeze this layer
        for p in calibrated_params:
            p.requires_grad = False
        
        ppl = compute_ppl(student, tokenizer, val_texts)
        print(f"  Layer {layer_idx}: val_ppl={ppl:.2f}")
    
    final_ppl = compute_ppl(student, tokenizer, val_texts)
    print(f"\nFinal: {final_ppl:.2f} ({(final_ppl/fp32_ppl-1)*100:+.1f}% vs FP32)")


if __name__ == '__main__':
    main()

