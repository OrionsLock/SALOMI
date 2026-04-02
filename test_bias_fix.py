"""
Test bias correction fix
"""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from onebit.quantization import HessianVQ

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Get calibration
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n\n".join(dataset["text"][:10])
calib_ids = tokenizer(text, return_tensors="pt").input_ids[:, :512]

# Test Layer 0 MLP c_fc
layer0 = model.transformer.h[0]
mlp_fc = layer0.mlp.c_fc

# Capture input
activations = []
def hook(m, i, o):
    activations.append(i[0].detach())
    
handle = mlp_fc.register_forward_hook(hook)
with torch.no_grad():
    model(calib_ids)
handle.remove()

X = activations[0]
X_flat = X.reshape(-1, 768).numpy()
H_diag = (X_flat**2).mean(axis=0)

# Quantize (with NEW bias correction!)
hvq = HessianVQ(n_codes=32)
W_orig = mlp_fc.weight.detach().numpy()
W_in = W_orig.T
W_recon = hvq.quantize(W_in, H_diag)

# Check mean preservation
print("BIAS CORRECTION TEST:")
print(f"  Original Mean: {W_in.mean():.8f}")
print(f"  Reconstructed Mean: {W_recon.mean():.8f}")
print(f"  Difference: {abs(W_in.mean() - W_recon.mean()):.8f}")

if abs(W_in.mean() - W_recon.mean()) < 1e-6:
    print("  STATUS: PERFECT - Bias corrected!")
elif abs(W_in.mean() - W_recon.mean()) < 0.001:
    print("  STATUS: GOOD - Bias mostly corrected")
else:
    print("  STATUS: FAILED - Bias still present")
