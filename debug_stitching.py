"""
Debug Stitching: Unit Test for Layer 0 Quantization

Goal: Isolate why full-model PPL fails despite high correlation.
Method:
1. Load GPT-2.
2. Quantize ONLY Layer 0 (MLP c_fc).
3. Measure Output MSE vs FP16.
4. Measure PPL impact.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import copy
from onebit.quantization import HessianVQ

def debug_layer0():
    print("="*80)
    print("DEBUG: LAYER 0 UNIT TEST")
    print("="*80)
    
    device = "cpu"
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    
    # 1. Calibration Data
    print("1. Getting Calibration Data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"][:10])
    calib_ids = tokenizer(text, return_tensors="pt").input_ids[:, :512]
    
    # 2. Capture Layer 0 Input (Frozen)
    print("2. Capturing Layer 0 Input...")
    layer0 = model.transformer.h[0]
    mlp_fc = layer0.mlp.c_fc
    
    activations = []
    def hook(m, i, o):
        activations.append(i[0].detach())
        
    handle = mlp_fc.register_forward_hook(hook)
    with torch.no_grad():
        model(calib_ids)
    handle.remove()
    
    X = activations[0] # (1, 512, 768)
    X_flat = X.reshape(-1, 768).numpy()
    H_diag = (X_flat**2).mean(axis=0)
    
    # 3. Baseline Output
    print("3. Running FP16 Forward Pass...")
    with torch.no_grad():
        Y_ref = mlp_fc(X)
        
    # 4. Quantize
    print("4. Quantizing Layer 0 MLP c_fc...")
    hvq = HessianVQ(n_codes=32)
    
    # Conv1D weight is (In, Out) -> (768, 3072)
    # Quantizer expects (Out, In) -> (3072, 768)
    W_orig = mlp_fc.weight.detach().numpy() # (768, 3072)
    W_in = W_orig.T # (3072, 768)
    
    print(f"   W shape: {W_orig.shape}")
    print(f"   Quantizer Input: {W_in.shape}")
    print(f"   H_diag shape: {H_diag.shape}")
    
    W_recon = hvq.quantize(W_in, H_diag) # Returns (3072, 768)
    
    # Update model
    mlp_fc.weight.data = torch.from_numpy(W_recon.T) # Back to (768, 3072)
    
    # 5. Quantized Output
    print("5. Running Quantized Forward Pass...")
    with torch.no_grad():
        Y_q = mlp_fc(X)
        
    # 6. Metrics
    mse = torch.mean((Y_ref - Y_q)**2).item()
    ref_pow = torch.mean(Y_ref**2).item()
    nmse = mse / ref_pow
    
    print(f"\nRESULTS:")
    print(f"   MSE: {mse:.6f}")
    print(f"   NMSE: {nmse:.6f}")
    
    if nmse > 0.1:
        print("❌ FAILED: High Error (>10%)")
    else:
        print("✅ PASSED: Low Error (<10%)")
        
    # 7. Correlation
    y_ref_flat = Y_ref.flatten().numpy()
    y_q_flat = Y_q.flatten().numpy()
    corr = np.corrcoef(y_ref_flat, y_q_flat)[0, 1]
    print(f"   Correlation: {corr:.4f}")
    
    if corr < 0.90:
        print("❌ FAILED: Low Correlation (<0.90)")
    else:
        print("✅ PASSED: High Correlation (>0.90)")
        
    # =========================================================================
    # PART 2: CHAIN TEST (Layer 0 -> Layer 1)
    # =========================================================================
    print("\n" + "="*80)
    print("DEBUG: LAYER 0 -> LAYER 1 CHAIN TEST")
    print("="*80)
    
    # 1. Get Input to Layer 1 (from Quantized Layer 0)
    print("1. Running Quantized Model to Layer 1...")
    # We need to run the whole block 0
    # But we only quantized mlp.c_fc.
    # Let's quantize the REST of Layer 0 to be realistic.
    
    layer0 = model.transformer.h[0]
    modules = [layer0.attn.c_attn, layer0.attn.c_proj, layer0.mlp.c_proj] # c_fc already done
    
    # Capture inputs for these modules (using current quantized state of c_fc)
    # Note: c_fc is already quantized. So c_proj will see quantized input.
    # But attn modules see raw input (from LayerNorm).
    
    # Let's just quantize Layer 1 MLP c_fc using input from Quantized Layer 0.
    
    layer1 = model.transformer.h[1]
    mlp_fc_1 = layer1.mlp.c_fc
    
    activations_1 = []
    def hook_1(m, i, o):
        activations_1.append(i[0].detach())
        
    handle = mlp_fc_1.register_forward_hook(hook_1)
    with torch.no_grad():
        model(calib_ids) # Run forward pass (Layer 0 is quantized!)
    handle.remove()
    
    X1 = activations_1[0]
    X1_flat = X1.reshape(-1, 768).numpy()
    H_diag_1 = (X1_flat**2).mean(axis=0)
    
    # 2. Baseline Output for Layer 1
    # We need to know what Layer 1 SHOULD output given this input.
    # Wait, "Baseline" usually means "FP16 model output".
    # But if input is noisy, FP16 model output is also "noisy" (perturbed).
    # We want to minimize || W_q * X_noisy - W * X_noisy ||.
    # So reference is W * X_noisy.
    
    print("2. Running FP16 Layer 1 on Quantized Input...")
    with torch.no_grad():
        Y_ref_1 = mlp_fc_1(X1)
        
    # 3. Quantize Layer 1
    print("3. Quantizing Layer 1 MLP c_fc...")
    W1_orig = mlp_fc_1.weight.detach().numpy()
    W1_in = W1_orig.T
    
    W1_recon = hvq.quantize(W1_in, H_diag_1)
    
    mlp_fc_1.weight.data = torch.from_numpy(W1_recon.T)
    
    # 4. Quantized Output
    print("4. Running Quantized Layer 1...")
    with torch.no_grad():
        Y_q_1 = mlp_fc_1(X1)
        
    # 5. Metrics
    corr_1 = np.corrcoef(Y_ref_1.flatten().numpy(), Y_q_1.flatten().numpy())[0, 1]
    
    # Check Bias
    bias = torch.mean(Y_q_1 - Y_ref_1).item()
    std_ref = torch.std(Y_ref_1).item()
    rel_bias = bias / std_ref
    
    print(f"\nRESULTS LAYER 1:")
    print(f"   Correlation: {corr_1:.4f}")
    print(f"   Bias: {bias:.6f} (Rel: {rel_bias:.4f})")
    
    if abs(rel_bias) > 0.05:
        print("❌ FAILED: High Bias (>5%)")
    else:
        print("✅ PASSED: Low Bias (<5%)")
    
    if corr_1 < 0.90:
        print("❌ FAILED: Chain Reaction Detected (<0.90)")
    else:
        print("✅ PASSED: No Chain Reaction (>0.90)")

if __name__ == "__main__":
    debug_layer0()
