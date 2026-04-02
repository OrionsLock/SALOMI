"""
Full-Model PPL Test with Bias-Corrected HessianVQ
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import copy
from onebit.quantization import HessianVQ

def evaluate_ppl_fast(model, tokenizer):
    """Fast PPL evaluation on WikiText-2 subset"""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])[:20000] 
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)
    except:
        input_ids = torch.randint(0, 50257, (1, 1024))
        
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        if input_ids.size(1) > 1024:
            input_ids = input_ids[:, :1024]
            
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()

def quantize_model_biascorrected():
    print("="*80)
    print("FULL-MODEL PPL TEST (BIAS-CORRECTED HessianVQ)")
    print("="*80)
    
    device = "cpu"
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    
    # 1. Baseline
    print("\n1. FP16 Baseline...")
    ppl_fp16 = evaluate_ppl_fast(model, tokenizer)
    print(f"   PPL: {ppl_fp16:.2f}")
    
    # 2. Calibration
    print("\n2. Preparing Calibration Data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"][:10])
    calib_ids = tokenizer(text, return_tensors="pt").input_ids[:, :512]
    
    # Keep frozen model for calibration
    model_fp16 = copy.deepcopy(model)
    model_fp16.eval()
    
    # 3. Quantize
    print("\n3. Quantizing with Bias Correction...")
    hvq = HessianVQ(n_codes=32)
    
    layers = model.transformer.h
    layers_fp16 = model_fp16.transformer.h
    
    for i in tqdm(range(len(layers)), desc="   Layers"):
        layer = layers[i]
        layer_fp16 = layers_fp16[i]
        
        modules = [layer.attn.c_attn, layer.attn.c_proj, layer.mlp.c_fc, layer.mlp.c_proj]
        modules_fp16 = [layer_fp16.attn.c_attn, layer_fp16.attn.c_proj, layer_fp16.mlp.c_fc, layer_fp16.mlp.c_proj]
        
        # Capture activations from frozen model
        activations = {}
        def hook(m, i, o):
            activations[m] = i[0].detach()
            
        handles = []
        for m in modules_fp16:
            handles.append(m.register_forward_hook(hook))
            
        with torch.no_grad():
            model_fp16(calib_ids)
            
        for h in handles: h.remove()
        
        # Quantize each module
        for idx, m in enumerate(modules):
            m_fp16 = modules_fp16[idx]
            
            X = activations[m_fp16].reshape(-1, activations[m_fp16].shape[-1]).numpy()
            H_diag = (X**2).mean(axis=0)
            
            W = m.weight.detach().numpy().T
            W_recon = hvq.quantize(W, H_diag)
            
            m.weight.data = torch.from_numpy(W_recon.T)
    
    # 4. Evaluate
    print("\n4. Evaluating Quantized Model...")
    ppl_hvq = evaluate_ppl_fast(model, tokenizer)
    print(f"   PPL: {ppl_hvq:.2f}")
    
    print("\n" + "="*80)
    print("RESULTS:")
    print(f"  FP16:        {ppl_fp16:.2f}")
    print(f"  HessianVQ-32: {ppl_hvq:.2f}")
    
    if ppl_hvq < 35:
        print("  STATUS: SUCCESS!")
    elif ppl_hvq < 100:
        print("  STATUS: Partial success (better but not great)")
    else:
        print("  STATUS: Still broken")
    
    with open("ppl_biascorrected.txt", "w") as f:
        f.write(f"FP16: {ppl_fp16:.2f}\n")
        f.write(f"HessianVQ-32: {ppl_hvq:.2f}\n")

if __name__ == "__main__":
    quantize_model_biascorrected()
