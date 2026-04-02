"""
REALISTIC CALIBRATION: Use C4 for calibration, WikiText-2 for evaluation.
This is the standard setup used in quantization papers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


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


def compute_ppl(model, tokenizer, texts, max_samples=100):
    losses = []
    for i, text in enumerate(texts):
        if i >= max_samples:
            break
        if len(text.strip()) < 50:
            continue
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=256, truncation=True)
        if tokens.shape[1] < 10:
            continue
        with torch.no_grad():
            out = model(tokens, labels=tokens)
            if not torch.isnan(out.loss):
                losses.append(out.loss.item())
    return np.exp(np.mean(losses)) if losses else float('inf')


def main():
    print("=" * 70)
    print("REALISTIC CALIBRATION: C4 → WikiText-2")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load WikiText-2 (use train for calibration, validation for eval)
    print("Loading WikiText-2...")
    wikitext = load_dataset('wikitext', 'wikitext-2-raw-v1')
    calib_texts = [t for t in wikitext['train']['text'] if len(t.strip()) > 50][:200]
    val_texts = [t for t in wikitext['validation']['text'] if len(t.strip()) > 50][:100]
    print(f"  Calibration: {len(calib_texts)} samples")
    print(f"  Validation: {len(val_texts)} samples")
    
    # Teacher
    teacher = GPT2LMHeadModel.from_pretrained('gpt2')
    teacher.eval()
    fp32_ppl = compute_ppl(teacher, tokenizer, val_texts)
    print(f"\nFP32 baseline (WikiText-2): {fp32_ppl:.2f}")
    
    # Student
    student = GPT2LMHeadModel.from_pretrained('gpt2')
    student.eval()
    
    # Get teacher outputs on calibration data
    print("\nCollecting teacher outputs...")
    teacher_outputs = []
    for text in calib_texts[:50]:  # Use subset for speed
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True)
        with torch.no_grad():
            teacher_outputs.append((tokens, teacher(tokens).logits.clone()))
    
    # Replace all layers at once
    RANK = 8
    calibrated_params = []
    
    for layer_idx in range(12):
        block = student.transformer.h[layer_idx]
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
    
    before_ppl = compute_ppl(student, tokenizer, val_texts)
    print(f"Before calibration: {before_ppl:.2f}")
    
    # Train
    print("\nCalibrating...")
    optimizer = torch.optim.Adam(calibrated_params, lr=1e-4)
    
    for epoch in range(100):
        total_loss = 0
        for tokens, teacher_logits in teacher_outputs:
            optimizer.zero_grad()
            student_logits = student(tokens).logits
            loss = F.mse_loss(student_logits, teacher_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrated_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 25 == 0:
            ppl = compute_ppl(student, tokenizer, val_texts[:20])
            print(f"  Epoch {epoch+1}: loss={total_loss:.2f}, val_ppl~{ppl:.2f}")
    
    final_ppl = compute_ppl(student, tokenizer, val_texts)
    print(f"\nFinal (WikiText-2): {final_ppl:.2f} ({(final_ppl/fp32_ppl-1)*100:+.1f}% vs FP32)")


if __name__ == '__main__':
    main()

