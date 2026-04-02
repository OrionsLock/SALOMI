"""
CALIBRATED BINARY: Learn calibration parameters after quantization.

The idea: Freeze binary weights, learn small calibration tensors.
- Binary weights: 1.0 bpp (frozen)
- Calibration: 0.58 bpp budget 
- Total: 1.58 bpp = equal to ternary
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class CalibratedBinaryConv1D(nn.Module):
    """Binary weights + learnable calibration for GPT-2 Conv1D."""
    def __init__(self, orig_weight, rank=4):
        super().__init__()
        # GPT-2 Conv1D: weight is (out_features, in_features) but used as x @ W
        n_in, n_out = orig_weight.shape

        # Frozen binary signs
        sign = torch.sign(orig_weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)

        # Learnable: low-rank calibration of magnitude
        # W_calibrated = sign * (a + U @ V^T)
        self.a = nn.Parameter(torch.tensor(orig_weight.abs().mean().item()))
        self.U = nn.Parameter(torch.randn(n_in, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n_out, rank) * 0.01)

    def forward(self, x):
        # Magnitude = a + low-rank correction
        mag = self.a + self.U @ self.V.T
        mag = F.relu(mag) + 1e-6  # Ensure positive
        weight = self.sign * mag
        # GPT-2 Conv1D: output = x @ weight
        return x @ weight


def calibrate_binary_model():
    """Calibrate binary weights with learnable parameters."""
    print("=" * 70)
    print("CALIBRATED BINARY: Learn calibration after binarization")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we build software.",
        "Neural networks can learn complex patterns from data.",
        "The weather today is sunny with a chance of rain.",
    ]
    
    def compute_ppl(model):
        losses = []
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
            with torch.no_grad():
                out = model(tokens, labels=tokens)
                losses.append(out.loss.item())
        return np.exp(np.mean(losses))
    
    fp32_ppl = compute_ppl(model)
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    # Get teacher outputs for distillation
    teacher_outputs = []
    for text in texts:
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            out = model(tokens)
            teacher_outputs.append((tokens, out.logits))
    
    # Replace one MLP layer with calibrated binary
    layer_idx = 6  # Middle layer
    orig_c_fc = model.transformer.h[layer_idx].mlp.c_fc.weight.data.clone()
    orig_c_proj = model.transformer.h[layer_idx].mlp.c_proj.weight.data.clone()
    
    # Create calibrated binary layers
    calib_c_fc = CalibratedBinaryConv1D(orig_c_fc, rank=8)
    calib_c_proj = CalibratedBinaryConv1D(orig_c_proj, rank=8)
    
    # Replace in model
    class CalibratedMLP(nn.Module):
        def __init__(self, orig_mlp, calib_fc, calib_proj):
            super().__init__()
            self.c_fc = calib_fc
            self.c_proj = calib_proj
            self.act = orig_mlp.act
            self.dropout = orig_mlp.dropout
            
        def forward(self, x):
            h = self.act(self.c_fc(x))
            h = self.c_proj(h)
            h = self.dropout(h)
            return h
    
    orig_mlp = model.transformer.h[layer_idx].mlp
    model.transformer.h[layer_idx].mlp = CalibratedMLP(orig_mlp, calib_c_fc, calib_c_proj)
    
    # Before calibration
    before_ppl = compute_ppl(model)
    print(f"Before calibration (layer {layer_idx} binary): {before_ppl:.2f}")
    
    # Calibration: minimize KL divergence to teacher
    params = list(calib_c_fc.parameters()) + list(calib_c_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    print("\nCalibrating...")
    for epoch in range(100):
        total_loss = 0
        for tokens, teacher_logits in teacher_outputs:
            optimizer.zero_grad()
            student_logits = model(tokens).logits
            
            # KL divergence loss
            loss = F.kl_div(
                F.log_softmax(student_logits / 2.0, dim=-1),
                F.softmax(teacher_logits / 2.0, dim=-1),
                reduction='batchmean'
            ) * 4.0  # temperature^2
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            ppl = compute_ppl(model)
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, PPL={ppl:.2f}")
    
    after_ppl = compute_ppl(model)
    print(f"\nAfter calibration: {after_ppl:.2f}")
    print(f"Recovery: {(before_ppl - after_ppl) / (before_ppl - fp32_ppl) * 100:.1f}%")
    
    # Calculate BPP
    n_weights = orig_c_fc.numel() + orig_c_proj.numel()
    n_calib = sum(p.numel() for p in params)
    bpp = (n_weights * 1 + n_calib * 32) / n_weights  # binary + FP32 calibration
    print(f"\nBPP for layer {layer_idx}: {bpp:.2f}")


if __name__ == '__main__':
    calibrate_binary_model()

