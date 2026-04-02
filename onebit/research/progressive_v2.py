"""
PROGRESSIVE CALIBRATION V2: More epochs, higher rank, cosine schedule.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class LowRankBinaryConv1D(nn.Module):
    def __init__(self, orig_weight, rank=16):
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
    print("PROGRESSIVE CALIBRATION V2: Higher rank, more epochs")
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
        "Artificial intelligence is transforming industries worldwide.",
        "The Renaissance marked a golden age of art and culture.",
    ]
    
    val_texts = [
        "Neural networks can learn complex patterns from training data.",
        "Deep ocean exploration has revealed many unknown species.",
        "Electric vehicles are becoming popular as technology improves.",
        "Space exploration continues to push the boundaries of science.",
    ]
    
    teacher = GPT2LMHeadModel.from_pretrained('gpt2')
    teacher.eval()
    fp32_ppl = compute_ppl(teacher, tokenizer, val_texts)
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    student = GPT2LMHeadModel.from_pretrained('gpt2')
    student.eval()
    
    teacher_outputs = []
    for text in train_texts:
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True)
        with torch.no_grad():
            teacher_outputs.append((tokens, teacher(tokens).logits.clone()))
    
    RANK = 16
    EPOCHS_PER_LAYER = 100
    
    for layer_idx in range(12):
        block = student.transformer.h[layer_idx]
        
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
        
        optimizer = torch.optim.AdamW(calibrated_params, lr=5e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_PER_LAYER)
        
        for epoch in range(EPOCHS_PER_LAYER):
            for tokens, teacher_logits in teacher_outputs:
                optimizer.zero_grad()
                student_logits = student(tokens).logits
                loss = F.mse_loss(student_logits, teacher_logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(calibrated_params, 1.0)
                optimizer.step()
            scheduler.step()
        
        for p in calibrated_params:
            p.requires_grad = False
        
        ppl = compute_ppl(student, tokenizer, val_texts)
        print(f"  Layer {layer_idx}: val_ppl={ppl:.2f}")
    
    final_ppl = compute_ppl(student, tokenizer, val_texts)
    
    # Calculate BPP
    n_weights = 84_934_656
    n_calib = 48 * (768 * RANK + 3072 * RANK + 768 * RANK + 2304 * RANK)  # Rough estimate
    bpp = 1.0 + (n_calib * 8) / n_weights
    
    print(f"\nFinal: {final_ppl:.2f} ({(final_ppl/fp32_ppl-1)*100:+.1f}% vs FP32)")
    print(f"BPP: ~{bpp:.2f}")


if __name__ == '__main__':
    main()

