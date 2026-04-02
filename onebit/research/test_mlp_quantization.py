"""Deep dive: Why is MLP so sensitive to quantization?"""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F


def analyze_mlp_sensitivity():
    """Analyze why MLP layers are so sensitive."""
    print("=" * 70)
    print("WHY IS MLP SO SENSITIVE TO BINARIZATION?")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Get a sample input
    text = "The quick brown fox"
    tokens = tokenizer.encode(text, return_tensors='pt')
    
    with torch.no_grad():
        embeddings = model.transformer.wte(tokens) + model.transformer.wpe(torch.arange(tokens.size(1)))
        x = model.transformer.h[0].ln_1(embeddings)
    
    # Get layer 0 MLP weights
    c_fc = model.transformer.h[0].mlp.c_fc.weight.data  # (768, 3072)
    c_proj = model.transformer.h[0].mlp.c_proj.weight.data  # (3072, 768)
    
    print(f"MLP c_fc shape: {c_fc.shape}")
    print(f"MLP c_proj shape: {c_proj.shape}")
    print()
    
    # Analyze weight distributions
    print("Weight Statistics:")
    print(f"  c_fc  - mean: {c_fc.mean():.4f}, std: {c_fc.std():.4f}, |mean|: {c_fc.abs().mean():.4f}")
    print(f"  c_proj - mean: {c_proj.mean():.4f}, std: {c_proj.std():.4f}, |mean|: {c_proj.abs().mean():.4f}")
    print()
    
    # Compute MLP output with FP32
    with torch.no_grad():
        hidden = x @ c_fc  # (1, 4, 3072)
        hidden_gelu = F.gelu(hidden)
        out_fp32 = hidden_gelu @ c_proj  # (1, 4, 768)
    
    print(f"FP32 MLP output stats:")
    print(f"  Pre-GELU: mean={hidden.mean():.4f}, std={hidden.std():.4f}")
    print(f"  Post-GELU: mean={hidden_gelu.mean():.4f}, std={hidden_gelu.std():.4f}")
    print(f"  Output: mean={out_fp32.mean():.4f}, std={out_fp32.std():.4f}")
    print()
    
    # Binary c_fc
    scale_fc = c_fc.abs().mean()
    c_fc_bin = torch.sign(c_fc) * scale_fc
    c_fc_bin[c_fc_bin == 0] = scale_fc
    
    with torch.no_grad():
        hidden_bin = x @ c_fc_bin
        hidden_bin_gelu = F.gelu(hidden_bin)
        out_bin_fc = hidden_bin_gelu @ c_proj  # c_proj still FP32
    
    print(f"Binary c_fc (c_proj FP32):")
    print(f"  Pre-GELU: mean={hidden_bin.mean():.4f}, std={hidden_bin.std():.4f}")
    print(f"  Post-GELU: mean={hidden_bin_gelu.mean():.4f}, std={hidden_bin_gelu.std():.4f}")
    print(f"  Output error vs FP32: {(out_bin_fc - out_fp32).abs().mean() / out_fp32.abs().mean() * 100:.1f}%")
    print()
    
    # The key insight: GELU amplifies errors
    print("=" * 70)
    print("KEY INSIGHT: GELU AMPLIFICATION")
    print("=" * 70)
    
    # Compare pre-GELU distributions
    fp32_pre = hidden.flatten()
    bin_pre = hidden_bin.flatten()
    
    # What fraction is in the "sensitive" GELU region?
    sensitive_region = (fp32_pre.abs() < 1.0).float().mean() * 100
    print(f"Fraction of activations in GELU sensitive region (|x| < 1): {sensitive_region:.1f}%")
    
    # GELU derivative analysis
    def gelu_derivative(x):
        """Approximate GELU derivative."""
        return 0.5 * (1 + torch.erf(x / np.sqrt(2))) + x * torch.exp(-x**2/2) / np.sqrt(2*np.pi)
    
    # At which values does GELU have high derivative (amplifies errors)?
    print("\nGELU behavior:")
    test_vals = torch.tensor([-2., -1., -0.5, 0., 0.5, 1., 2.])
    for v in test_vals:
        g = F.gelu(v.unsqueeze(0)).item()
        dg = gelu_derivative(v).item()
        print(f"  x={v:+.1f}: GELU(x)={g:+.3f}, GELU'(x)={dg:.3f}")
    
    print("\nThe problem: When pre-GELU values are near 0, small errors in c_fc")
    print("can flip the sign of the activation, causing the output to be 0 instead")
    print("of a positive value (or vice versa). This is a discontinuous error!")
    
    # Demonstrate this
    print("\n" + "=" * 70)
    print("DEMONSTRATING GELU ERROR AMPLIFICATION")
    print("=" * 70)
    
    # Take a value near 0 in FP32
    near_zero_fp32 = torch.tensor([0.1])
    near_zero_bin = torch.tensor([-0.1])  # Binary flipped the sign
    
    gelu_fp32 = F.gelu(near_zero_fp32).item()
    gelu_bin = F.gelu(near_zero_bin).item()
    
    print(f"FP32 pre-GELU: {near_zero_fp32.item():.2f} → post-GELU: {gelu_fp32:.3f}")
    print(f"Binary pre-GELU: {near_zero_bin.item():.2f} → post-GELU: {gelu_bin:.3f}")
    print(f"Relative error: {abs(gelu_bin - gelu_fp32) / abs(gelu_fp32) * 100:.1f}%")
    print()
    print("A small sign error in c_fc can cause ~100-200% error after GELU!")


if __name__ == '__main__':
    analyze_mlp_sensitivity()

