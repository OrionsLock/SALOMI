"""Test if ternary's zeros help specifically in the GELU-sensitive region."""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def test_zeros_help_gelu():
    """Test if putting zeros in GELU-sensitive weights helps."""
    print("=" * 70)
    print("DO TERNARY'S ZEROS HELP IN GELU-SENSITIVE REGIONS?")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Get layer 0 MLP
    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(text, return_tensors='pt')
    
    with torch.no_grad():
        embeddings = model.transformer.wte(tokens) + model.transformer.wpe(torch.arange(tokens.size(1)))
        x = model.transformer.h[0].ln_1(embeddings)
    
    c_fc = model.transformer.h[0].mlp.c_fc.weight.data.clone()
    c_proj = model.transformer.h[0].mlp.c_proj.weight.data.clone()
    
    scale = c_fc.abs().mean()
    
    # FP32 reference
    with torch.no_grad():
        hidden_fp32 = x @ c_fc
        gelu_fp32 = F.gelu(hidden_fp32)
        out_fp32 = gelu_fp32 @ c_proj
    
    print(f"FP32 reference: output mean = {out_fp32.abs().mean():.4f}")
    print()
    
    # Binary: sign * scale
    c_fc_bin = torch.sign(c_fc) * scale
    c_fc_bin[c_fc_bin == 0] = scale
    
    with torch.no_grad():
        hidden_bin = x @ c_fc_bin
        gelu_bin = F.gelu(hidden_bin)
        out_bin = gelu_bin @ c_proj
    
    bin_error = (out_bin - out_fp32).abs().mean() / out_fp32.abs().mean() * 100
    print(f"Binary: error = {bin_error:.1f}%")
    
    # Ternary: zeros for small weights
    thresh = 0.3 * scale
    c_fc_tern = torch.zeros_like(c_fc)
    c_fc_tern[c_fc > thresh] = scale
    c_fc_tern[c_fc < -thresh] = -scale
    
    zero_pct = (c_fc_tern == 0).float().mean() * 100
    print(f"Ternary zeros: {zero_pct:.1f}%")
    
    with torch.no_grad():
        hidden_tern = x @ c_fc_tern
        gelu_tern = F.gelu(hidden_tern)
        out_tern = gelu_tern @ c_proj
    
    tern_error = (out_tern - out_fp32).abs().mean() / out_fp32.abs().mean() * 100
    print(f"Ternary: error = {tern_error:.1f}%")
    
    print()
    print("=" * 70)
    print("WHY ZEROS HELP")
    print("=" * 70)
    
    # Analyze where zeros are placed
    small_weights = (c_fc.abs() <= thresh)
    small_weight_pct = small_weights.float().mean() * 100
    print(f"Small weights (|w| <= 0.3*scale): {small_weight_pct:.1f}%")
    
    # What happens to activations from small weights?
    # If weight is small, its contribution to pre-GELU is small
    # Binary assigns it full scale → potentially flips sign of sum
    # Ternary assigns it 0 → removes its contribution entirely
    
    print()
    print("The insight:")
    print("- Small weights contribute small amounts to pre-GELU activations")
    print("- Binary gives them full scale → they can flip signs incorrectly")
    print("- Ternary zeros them → they don't contribute at all")
    print("- For GELU-sensitive regions, 'no contribution' is better than 'wrong contribution'")
    print()
    
    # Test: what if we use zeros for weights where BINARY would flip the sign?
    print("=" * 70)
    print("SMART ZEROS: Zero out weights where binary would flip sign")
    print("=" * 70)
    
    # A weight "flips" if sign(w) != sign of w's "true" contribution
    # We can't know true contribution, but small |w| are more likely to flip
    
    for zero_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        thresh_abs = zero_thresh * scale
        c_fc_smart = torch.sign(c_fc) * scale
        c_fc_smart[c_fc.abs() <= thresh_abs] = 0
        
        zero_pct = (c_fc_smart == 0).float().mean() * 100
        
        with torch.no_grad():
            out_smart = F.gelu(x @ c_fc_smart) @ c_proj
        
        error = (out_smart - out_fp32).abs().mean() / out_fp32.abs().mean() * 100
        
        # Calculate effective BPP
        n_zeros = (c_fc_smart == 0).sum().item()
        n_nonzero = c_fc.numel() - n_zeros
        # Ternary encoding: log2(3) ≈ 1.58 bits for {-1, 0, +1}
        bpp = 1.58  # All positions need to encode 3 states
        
        print(f"Thresh={zero_thresh:.1f}: zeros={zero_pct:>5.1f}%, error={error:>5.1f}%, bpp={bpp:.2f}")


if __name__ == '__main__':
    test_zeros_help_gelu()

