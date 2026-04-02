"""
Test larger codebook (K=256) to see if better per-layer quality helps PPL
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import copy
from onebit.quantization import HessianVQ

def evaluate_ppl_fast(model, tokenizer):
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

def test_large_codebook():
    print("="*80)
    print("TESTING K=256 (vs K=32)")
    print("="*80)
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    
    # FP16 baseline
    print("\n1. FP16 Baseline...")
    ppl_fp16 = evaluate_ppl_fast(model, tokenizer)
    print(f"   PPL: {ppl_fp16:.2f}")
    
    # Calibration
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"][:10])
    calib_ids = tokenizer(text, return_tensors="pt").input_ids[:, :512]
    
    model_fp16 = copy.deepcopy(model)
    
    # Test K=256 (much larger codebook, ~8 bits per block vs ~5 bits)
    print("\n2. Quantizing with K=256...")
    hvq = HessianVQ(n_codes=256)  # Larger codebook!
    
    layers = model.transformer.h
    layers_fp16 = model_fp16.transformer.h
    
    for i in tqdm(range(len(layers)), desc="   Layers"):
        layer = layers[i]
        layer_fp16 = layers_fp16[i]
        
        modules = [layer.attn.c_attn, layer.attn.c_proj, layer.mlp.c_fc, layer.mlp.c_proj]
        modules_fp16 = [layer_fp16.attn.c_attn, layer_fp16.attn.c_proj, layer_fp16.mlp.c_fc, layer_fp16.mlp.c_proj]
        
        activations = {}
        handles = [m.register_forward_hook(lambda m,i,o,key=m: activations.__setitem__(key, i[0].detach())) for m in modules_fp16]
        
        with torch.no_grad():
            model_fp16(calib_ids)
            
        for h in handles: h.remove()
        
        for idx, m in enumerate(modules):
            m_fp16 = modules_fp16[idx]
            X = activations[m_fp16].reshape(-1, activations[m_fp16].shape[-1]).numpy()
            H_diag = (X**2).mean(axis=0)
            
            W = m.weight.detach().numpy().T
            W_recon = hvq.quantize(W, H_diag)
            m.weight.data = torch.from_numpy(W_recon.T)
    
    print("\n3. Evaluating K=256...")
    ppl_256 = evaluate_ppl_fast(model, tokenizer)
    print(f"   PPL: {ppl_256:.2f}")
    
    print("\n" + "="*80)
    print(f"FP16:    {ppl_fp16:.2f}")
    print(f"K=256:   {ppl_256:.2f}")
    
    if ppl_256 < 35:
        print("SUCCESS!")
    elif ppl_256 < ppl_fp16 * 2:
        print("Promising - within 2x of FP16")
    else:
        print("Still broken")

if __name__ == "__main__":
    test_large_codebook()
