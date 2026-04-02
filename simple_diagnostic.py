"""
Simple diagnostic: Check if quantization preserves mean and std
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

# Get FP16 output
with torch.no_grad():
    Y_fp16 = mlp_fc(X)

# Quantize
hvq = HessianVQ(n_codes=32)
W_orig = mlp_fc.weight.detach().numpy()
W_in = W_orig.T
W_recon = hvq.quantize(W_in, H_diag)

# Check weight stats
print("WEIGHT STATISTICS:")
print(f"  FP16 Mean: {W_orig.mean():.6f}, Std: {W_orig.std():.6f}")
print(f"  Reconstructed Mean: {W_recon.T.mean():.6f}, Std: {W_recon.T.std():.6f}")
print(f"  Difference in Mean: {(W_recon.T.mean() - W_orig.mean()):.6f}")

# Update and get output
mlp_fc.weight.data = torch.from_numpy(W_recon.T)
with torch.no_grad():
    Y_q = mlp_fc(X)

# Check output stats
print("\nOUTPUT STATISTICS:")
print(f"  FP16 Mean: {Y_fp16.mean():.6f}, Std: {Y_fp16.std():.6f}")
print(f"  Quantized Mean: {Y_q.mean():.6f}, Std: {Y_q.std():.6f}")
print(f"  Bias (Mean Diff): {(Y_q.mean() - Y_fp16.mean()).item():.6f}")

# Correlation
corr = np.corrcoef(Y_fp16.flatten().numpy(), Y_q.flatten().numpy())[0, 1]
print(f"  Correlation: {corr:.4f}")

if abs((Y_q.mean() - Y_fp16.mean()).item()) > 0.01:
    print("\n❌ BIAS DETECTED: Quantization shifts the mean significantly!")
else:
    print("\n✅ No significant bias")
