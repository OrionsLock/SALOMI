"""
CALIBRATED BINARY V2: All weights (MLP + Attention), more training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class CalibratedBinaryConv1D(nn.Module):
    """Binary weights + learnable low-rank calibration."""
    def __init__(self, orig_weight, rank=4):
        super().__init__()
        n_in, n_out = orig_weight.shape
        sign = torch.sign(orig_weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        self.a = nn.Parameter(torch.tensor(orig_weight.abs().mean().item()))
        self.U = nn.Parameter(torch.randn(n_in, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n_out, rank) * 0.01)
        
    def forward(self, x):
        mag = F.relu(self.a + self.U @ self.V.T) + 1e-6
        return x @ (self.sign * mag)


def main():
    print("=" * 70)
    print("CALIBRATED BINARY V2: Full Model (MLP + Attention)")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # More training texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we build software.",
        "Neural networks can learn complex patterns from data.",
        "The weather today is sunny with a chance of rain.",
        "Artificial intelligence has made remarkable progress.",
        "Deep learning models require substantial compute resources.",
        "The transformer architecture revolutionized NLP.",
        "Attention mechanisms allow models to focus on relevant parts.",
    ]
    
    def compute_ppl():
        losses = []
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
            with torch.no_grad():
                out = model(tokens, labels=tokens)
                losses.append(out.loss.item())
        return np.exp(np.mean(losses))
    
    fp32_ppl = compute_ppl()
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    # Get teacher outputs
    teacher_outputs = []
    for text in texts:
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            teacher_outputs.append((tokens, model(tokens).logits))
    
    # Replace ALL Conv1D layers (MLP + Attention)
    calibrated_params = []
    total_weights = 0
    total_calib = 0
    RANK = 8
    
    for layer_idx in range(12):
        block = model.transformer.h[layer_idx]
        
        # MLP
        for name in ['c_fc', 'c_proj']:
            orig = getattr(block.mlp, name).weight.data.clone()
            calib = CalibratedBinaryConv1D(orig, rank=RANK)
            total_weights += orig.numel()
            total_calib += sum(p.numel() for p in calib.parameters())
            calibrated_params.extend(calib.parameters())
            setattr(block.mlp, name, calib)
        
        # Attention
        for name in ['c_attn', 'c_proj']:
            orig = getattr(block.attn, name).weight.data.clone()
            calib = CalibratedBinaryConv1D(orig, rank=RANK)
            total_weights += orig.numel()
            total_calib += sum(p.numel() for p in calib.parameters())
            calibrated_params.extend(calib.parameters())
            setattr(block.attn, name, calib)
    
    before_ppl = compute_ppl()
    print(f"Before calibration: {before_ppl:.2f}")
    
    bpp_8bit = 1.0 + (total_calib * 8) / total_weights
    print(f"Weights: {total_weights:,}, Calib: {total_calib:,}, BPP: {bpp_8bit:.2f}")
    
    # Calibrate with longer training
    optimizer = torch.optim.Adam(calibrated_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    print("\nCalibrating...")
    for epoch in range(500):
        total_loss = 0
        for tokens, teacher_logits in teacher_outputs:
            optimizer.zero_grad()
            student_logits = model(tokens).logits
            loss = F.kl_div(
                F.log_softmax(student_logits / 2.0, dim=-1),
                F.softmax(teacher_logits / 2.0, dim=-1),
                reduction='batchmean'
            ) * 4.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrated_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            ppl = compute_ppl()
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, PPL={ppl:.2f}, lr={lr:.2e}")
    
    after_ppl = compute_ppl()
    print(f"\nFinal: {after_ppl:.2f} (FP32: {fp32_ppl:.2f})")
    print(f"Gap: {(after_ppl/fp32_ppl - 1)*100:+.1f}%")
    print(f"BPP: {bpp_8bit:.2f} (ternary=1.58)")


if __name__ == '__main__':
    main()

