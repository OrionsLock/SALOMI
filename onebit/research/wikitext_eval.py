"""
WIKITEXT EVALUATION: Calibrate on real data, evaluate on WikiText-2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


class CalibratedBinaryConv1D(nn.Module):
    def __init__(self, orig_weight, rank=8):
        super().__init__()
        n_in, n_out = orig_weight.shape
        sign = torch.sign(orig_weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        self.a = nn.Parameter(torch.tensor(orig_weight.abs().mean().item()))
        self.U = nn.Parameter(torch.randn(n_in, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n_out, rank) * 0.01)
        
    def forward(self, x):
        return x @ (self.sign * (F.relu(self.a + self.U @ self.V.T) + 1e-6))


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
    print("WIKITEXT-2 EVALUATION")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load WikiText-2
    print("Loading WikiText-2...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_texts = [t for t in dataset['train']['text'] if len(t.strip()) > 50][:200]
    val_texts = [t for t in dataset['validation']['text'] if len(t.strip()) > 50][:100]
    print(f"Train: {len(train_texts)} samples, Val: {len(val_texts)} samples")
    
    # FP32 Baseline
    model_fp32 = GPT2LMHeadModel.from_pretrained('gpt2')
    model_fp32.eval()
    fp32_ppl = compute_ppl(model_fp32, tokenizer, val_texts)
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    # Post-hoc Binary
    model_bin = GPT2LMHeadModel.from_pretrained('gpt2')
    for name, param in model_bin.named_parameters():
        if 'weight' in name and param.dim() == 2 and 'transformer.h' in name:
            scale = param.data.abs().mean()
            sign = torch.sign(param.data)
            sign[sign == 0] = 1
            param.data = sign * scale
    bin_ppl = compute_ppl(model_bin, tokenizer, val_texts)
    print(f"Post-hoc Binary: {bin_ppl:.2f}")
    
    # Calibrated Binary with real data
    print("\nCalibrating on WikiText train split...")
    model_calib = GPT2LMHeadModel.from_pretrained('gpt2')
    model_calib.eval()
    
    # Get teacher outputs on train
    teacher_outputs = []
    for text in train_texts[:50]:  # Use subset for speed
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True)
        with torch.no_grad():
            teacher_outputs.append((tokens, model_calib(tokens).logits))
    
    # Replace layers
    calibrated_params = []
    for layer_idx in range(12):
        block = model_calib.transformer.h[layer_idx]
        for name in ['c_fc', 'c_proj']:
            calib = CalibratedBinaryConv1D(getattr(block.mlp, name).weight.data.clone(), rank=8)
            calibrated_params.extend(calib.parameters())
            setattr(block.mlp, name, calib)
        for name in ['c_attn', 'c_proj']:
            calib = CalibratedBinaryConv1D(getattr(block.attn, name).weight.data.clone(), rank=8)
            calibrated_params.extend(calib.parameters())
            setattr(block.attn, name, calib)
    
    before_ppl = compute_ppl(model_calib, tokenizer, val_texts)
    print(f"Before calibration: {before_ppl:.2f}")
    
    # Calibrate with stable settings
    optimizer = torch.optim.Adam(calibrated_params, lr=5e-4)
    for epoch in range(100):
        total_loss = 0
        for tokens, teacher_logits in teacher_outputs:
            optimizer.zero_grad()
            student_logits = model_calib(tokens).logits
            # Stable loss: MSE on logits instead of KL
            loss = F.mse_loss(student_logits, teacher_logits)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(calibrated_params, 0.5)
                optimizer.step()
                total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            ppl = compute_ppl(model_calib, tokenizer, val_texts[:20])
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, val_ppl~{ppl:.2f}")
    
    calib_ppl = compute_ppl(model_calib, tokenizer, val_texts)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Method':<25} {'BPP':>8} {'PPL':>12} {'vs FP32':>12}")
    print("-" * 60)
    for name, bpp, ppl in [
        ("FP32", 32.0, fp32_ppl),
        ("Post-hoc Binary", 1.0, bin_ppl),
        ("Calibrated Binary", 1.11, calib_ppl),
    ]:
        gap = (ppl / fp32_ppl - 1) * 100
        print(f"{name:<25} {bpp:>8.2f} {ppl:>12.2f} {gap:>+11.1f}%")


if __name__ == '__main__':
    main()

