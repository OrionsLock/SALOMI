"""
CALIBRATED BINARY - FULL MODEL: Learn calibration for all layers.
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
        mag = self.a + self.U @ self.V.T
        mag = F.relu(mag) + 1e-6
        weight = self.sign * mag
        return x @ weight


def calibrate_full_model():
    """Calibrate ALL layers with binary + learned calibration."""
    print("=" * 70)
    print("CALIBRATED BINARY: Full Model (All MLP Layers)")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we build software.",
        "Neural networks can learn complex patterns from data.",
        "The weather today is sunny with a chance of rain.",
        "Artificial intelligence has made remarkable progress.",
        "Deep learning models require substantial compute resources.",
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
            out = model(tokens)
            teacher_outputs.append((tokens, out.logits))
    
    # Replace ALL MLP layers
    calibrated_params = []
    total_weights = 0
    total_calib = 0
    
    RANK = 8  # Low-rank calibration rank
    
    for layer_idx in range(12):
        mlp = model.transformer.h[layer_idx].mlp
        
        orig_c_fc = mlp.c_fc.weight.data.clone()
        orig_c_proj = mlp.c_proj.weight.data.clone()
        
        calib_c_fc = CalibratedBinaryConv1D(orig_c_fc, rank=RANK)
        calib_c_proj = CalibratedBinaryConv1D(orig_c_proj, rank=RANK)
        
        # Track params
        total_weights += orig_c_fc.numel() + orig_c_proj.numel()
        total_calib += sum(p.numel() for p in calib_c_fc.parameters())
        total_calib += sum(p.numel() for p in calib_c_proj.parameters())
        
        calibrated_params.extend(calib_c_fc.parameters())
        calibrated_params.extend(calib_c_proj.parameters())
        
        # Replace in model
        class CalibratedMLP(nn.Module):
            def __init__(self, orig_mlp, c_fc, c_proj):
                super().__init__()
                self.c_fc = c_fc
                self.c_proj = c_proj
                self.act = orig_mlp.act
                self.dropout = orig_mlp.dropout
            def forward(self, x):
                return self.dropout(self.c_proj(self.act(self.c_fc(x))))
        
        model.transformer.h[layer_idx].mlp = CalibratedMLP(mlp, calib_c_fc, calib_c_proj)
    
    before_ppl = compute_ppl()
    print(f"Before calibration (all MLP binary): {before_ppl:.2f}")
    
    # Calculate BPP
    # Binary: 1 bit per weight
    # Calibration: FP32 (but could be quantized to 8-bit)
    bpp_binary = 1.0
    bpp_calib_fp32 = (total_calib * 32) / total_weights
    bpp_calib_8bit = (total_calib * 8) / total_weights
    print(f"\nWeights: {total_weights:,}, Calibration params: {total_calib:,}")
    print(f"BPP with FP32 calib: {bpp_binary + bpp_calib_fp32:.2f}")
    print(f"BPP with 8-bit calib: {bpp_binary + bpp_calib_8bit:.2f}")
    
    # Calibrate
    optimizer = torch.optim.Adam(calibrated_params, lr=5e-4)
    
    print("\nCalibrating...")
    for epoch in range(200):
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
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            ppl = compute_ppl()
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, PPL={ppl:.2f}")
    
    after_ppl = compute_ppl()
    print(f"\nAfter calibration: {after_ppl:.2f}")
    print(f"Recovery: {(before_ppl - after_ppl) / (before_ppl - fp32_ppl) * 100:.1f}%")


if __name__ == '__main__':
    calibrate_full_model()

