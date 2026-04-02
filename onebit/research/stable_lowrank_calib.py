"""
STABLE LOW-RANK CALIBRATION: Careful init and training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class StableLowRankBinaryConv1D(nn.Module):
    """Binary + low-rank magnitude with careful initialization."""
    def __init__(self, orig_weight, rank=4):
        super().__init__()
        n_in, n_out = orig_weight.shape
        
        # Binary signs
        sign = torch.sign(orig_weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        
        # Initialize from SVD of magnitude matrix
        W_abs = orig_weight.abs()
        U, S, Vh = torch.linalg.svd(W_abs, full_matrices=False)
        
        # Low-rank factors: magnitude = U @ diag(S) @ V.T
        self.U = nn.Parameter(U[:, :rank] * S[:rank].sqrt().unsqueeze(0))
        self.V = nn.Parameter(Vh[:rank, :].T * S[:rank].sqrt().unsqueeze(0))
        
    def forward(self, x):
        mag = self.U @ self.V.T
        mag = F.relu(mag) + 1e-8
        return x @ (self.sign * mag)


def compute_ppl(model, tokenizer, texts):
    losses = []
    for text in texts:
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
    print("STABLE LOW-RANK CALIBRATION")
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
        "Artificial intelligence is transforming how we build software.",
        "The Renaissance marked a golden age of art and culture in Europe.",
    ]
    
    val_texts = [
        "Neural networks can learn complex patterns from training data.",
        "Deep ocean exploration has revealed many unknown species.",
        "Electric vehicles are becoming popular as technology improves.",
        "Space exploration continues to push the boundaries of science.",
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
            teacher_outputs.append((tokens, model(tokens).logits.clone()))
    
    # Replace with low-rank binary
    RANK = 8
    calibrated_params = []
    total_weights = 0
    total_calib = 0
    
    for layer_idx in range(12):
        block = model.transformer.h[layer_idx]
        for name in ['c_fc', 'c_proj']:
            orig = getattr(block.mlp, name).weight.data.clone()
            calib = StableLowRankBinaryConv1D(orig, rank=RANK)
            total_weights += orig.numel()
            total_calib += calib.U.numel() + calib.V.numel()
            calibrated_params.extend([calib.U, calib.V])
            setattr(block.mlp, name, calib)
        for name in ['c_attn', 'c_proj']:
            orig = getattr(block.attn, name).weight.data.clone()
            calib = StableLowRankBinaryConv1D(orig, rank=RANK)
            total_weights += orig.numel()
            total_calib += calib.U.numel() + calib.V.numel()
            calibrated_params.extend([calib.U, calib.V])
            setattr(block.attn, name, calib)
    
    before_ppl = compute_ppl(model, tokenizer, val_texts)
    bpp = 1.0 + (total_calib * 8) / total_weights
    print(f"Before calibration: {before_ppl:.2f} (BPP: {bpp:.3f})")
    
    # Careful training
    optimizer = torch.optim.AdamW(calibrated_params, lr=1e-4, weight_decay=0.01)
    
    for epoch in range(300):
        for tokens, teacher_logits in teacher_outputs:
            optimizer.zero_grad()
            student_logits = model(tokens).logits
            loss = F.mse_loss(student_logits, teacher_logits)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(calibrated_params, 0.5)
                optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            ppl = compute_ppl(model, tokenizer, val_texts)
            print(f"  Epoch {epoch+1}: val_ppl={ppl:.2f}")
    
    after_ppl = compute_ppl(model, tokenizer, val_texts)
    print(f"\nFinal: {after_ppl:.2f} ({(after_ppl/fp32_ppl-1)*100:+.1f}% vs FP32)")
    print(f"BPP: {bpp:.3f}")


if __name__ == '__main__':
    main()

