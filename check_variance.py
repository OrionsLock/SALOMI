"""
Check variance/scale preservation
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

# Quantize
hvq = HessianVQ(n_codes=32)
W_orig = mlp_fc.weight.detach().numpy()
W_in = W_orig.T
W_recon = hvq.quantize(W_in, H_diag)

# Check ALL statistics
print("STATISTICS CHECK:")
print(f"  Mean   - Original: {W_in.mean():.8f}, Quantized: {W_recon.mean():.8f}, Diff: {abs(W_in.mean() - W_recon.mean()):.8f}")
print(f"  Std    - Original: {W_in.std():.8f}, Quantized: {W_recon.std():.8f}, Ratio: {W_recon.std()/W_in.std():.4f}")
print(f"  Min    - Original: {W_in.min():.8f}, Quantized: {W_recon.min():.8f}")
print(f"  Max    - Original: {W_in.max():.8f}, Quantized: {W_recon.max():.8f}")

# Correlation
corr = np.corrcoef(W_in.flatten(), W_recon.flatten())[0, 1]
print(f"  Correlation: {corr:.6f}")

# Check output difference
mlp_fc.weight.data = torch.from_numpy(W_recon.T)
with torch.no_grad():
    Y_q = mlp_fc(X)

# Restore original
mlp_fc.weight.data = torch.from_numpy(W_orig)
with torch.no_grad():
    Y_fp16 = mlp_fc(X)

print(f"\nOUTPUT CHECK:")
print(f"  Mean - FP16: {Y_fp16.mean():.6f}, Quantized: {Y_q.mean():.6f}, Diff: {abs(Y_fp16.mean() - Y_q.mean()).item():.6f}")
print(f"  Std  - FP16: {Y_fp16.std():.6f}, Quantized: {Y_q.std():.6f}, Ratio: {(Y_q.std()/Y_fp16.std()).item():.4f}")

mse = ((Y_fp16 - Y_q)**2).mean().item()
print(f"  MSE: {mse:.6f}")
