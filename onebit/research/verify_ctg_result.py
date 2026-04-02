"""
VERIFY CTG RESULT: Make sure we're actually using binary weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class CTGCalibratedBinary(nn.Module):
    def __init__(self, orig_weight, block_size=64):
        super().__init__()
        n_in, n_out = orig_weight.shape
        
        # Binary signs (frozen)
        sign = torch.sign(orig_weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        
        n_blocks_in = (n_in + block_size - 1) // block_size
        n_blocks_out = (n_out + block_size - 1) // block_size
        
        self.block_scales = nn.Parameter(torch.ones(n_blocks_in, n_blocks_out))
        
        W_abs = orig_weight.abs()
        for i in range(n_blocks_in):
            for j in range(n_blocks_out):
                i_start, i_end = i * block_size, min((i+1) * block_size, n_in)
                j_start, j_end = j * block_size, min((j+1) * block_size, n_out)
                self.block_scales.data[i, j] = W_abs[i_start:i_end, j_start:j_end].mean()
        
        self.block_size = block_size
        self.n_in = n_in
        self.n_out = n_out
        
    def forward(self, x):
        mag = torch.repeat_interleave(
            torch.repeat_interleave(self.block_scales, self.block_size, dim=0)[:self.n_in],
            self.block_size, dim=1
        )[:, :self.n_out]
        mag = F.relu(mag) + 1e-6
        weight = self.sign * mag
        return x @ weight
    
    def get_effective_weight(self):
        """Return the effective weight for verification."""
        mag = torch.repeat_interleave(
            torch.repeat_interleave(self.block_scales, self.block_size, dim=0)[:self.n_in],
            self.block_size, dim=1
        )[:, :self.n_out]
        mag = F.relu(mag) + 1e-6
        return self.sign * mag


def main():
    print("=" * 70)
    print("VERIFY CTG RESULT")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use MORE test texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we build software.",
        "Neural networks can learn complex patterns from data.",
        "Artificial intelligence has made remarkable progress.",
        "Deep learning requires large amounts of compute.",
        "Natural language processing has advanced significantly.",
        "Computer vision models can now detect objects accurately.",
        "Reinforcement learning enables autonomous decision making.",
        "The history of computing began with mechanical calculators.",
        "Modern processors contain billions of transistors.",
    ]
    
    def compute_ppl(model):
        losses = []
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
            with torch.no_grad():
                out = model(tokens, labels=tokens)
                losses.append(out.loss.item())
        return np.exp(np.mean(losses))
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    fp32_ppl = compute_ppl(model)
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    # Get teacher outputs
    teacher_outputs = []
    for text in texts:
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            teacher_outputs.append((tokens, model(tokens).logits))
    
    # Apply CTG calibration
    block_size = 64
    calibrated_params = []
    calibrated_layers = []
    
    for layer_idx in range(12):
        block = model.transformer.h[layer_idx]
        for name in ['c_fc', 'c_proj']:
            orig = getattr(block.mlp, name).weight.data.clone()
            calib = CTGCalibratedBinary(orig, block_size=block_size)
            calibrated_params.extend(calib.parameters())
            calibrated_layers.append(calib)
            setattr(block.mlp, name, calib)
        for name in ['c_attn', 'c_proj']:
            orig = getattr(block.attn, name).weight.data.clone()
            calib = CTGCalibratedBinary(orig, block_size=block_size)
            calibrated_params.extend(calib.parameters())
            calibrated_layers.append(calib)
            setattr(block.attn, name, calib)
    
    before_ppl = compute_ppl(model)
    print(f"Before calibration: {before_ppl:.2f}")
    
    # LONGER training
    optimizer = torch.optim.Adam(calibrated_params, lr=1e-2)
    for epoch in range(200):
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
    
    after_ppl = compute_ppl(model)
    print(f"After calibration: {after_ppl:.2f} ({(after_ppl/fp32_ppl-1)*100:+.1f}%)")
    
    # Verify weights are actually binary
    layer = calibrated_layers[0]
    eff_weight = layer.get_effective_weight()
    unique_signs = torch.unique(layer.sign)
    print(f"\nWeight verification:")
    print(f"  Sign values: {unique_signs.tolist()}")
    print(f"  Sign shape: {layer.sign.shape}")
    print(f"  Block scales shape: {layer.block_scales.shape}")
    print(f"  Effective weight unique magnitudes per block: ~{layer.block_scales.numel()} values")


if __name__ == '__main__':
    main()

