"""
Clean PPL Test for High-Dimensional VQ
Test the method that achieved 0.9434 correlation on full-model PPL.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import copy

def evaluate_ppl(model, tokenizer, max_length=512):
    """Evaluate perplexity on WikiText-2 test set"""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in dataset["text"] if len(t) > 0][:100])
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids[:, :max_length]
    except:
        print("  Warning: Using random data (dataset failed)")
        input_ids = torch.randint(0, 50257, (1, max_length))
    
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        ppl = torch.exp(outputs.loss).item()
    
    return ppl

class SimpleVQ:
    """Clean implementation of High-Dim VQ (D=16, K=65536)"""
    def __init__(self, n_codes=65536, block_size=16):
        self.n_codes = n_codes
        self.block_size = block_size
        
    def quantize(self, W):
        """Quantize a weight matrix"""
        d_out, d_in = W.shape
        bs = self.block_size
        
        # Pad to block size
        pad = (bs - (W.size % bs)) % bs
        W_flat = W.flatten()
        W_pad = np.pad(W_flat, (0, pad))
        
        # Reshape to blocks
        blocks = W_pad.reshape(-1, bs)
        
        # Train K-Means on sample
        n_train = min(len(blocks), 50000)
        train_idx = np.random.choice(len(blocks), n_train, replace=False)
        train_blocks = blocks[train_idx]
        
        # Initialize centroids
        if self.n_codes > n_train:
            cent_idx = np.random.choice(n_train, self.n_codes, replace=True)
        else:
            cent_idx = np.random.choice(n_train, self.n_codes, replace=False)
        centroids = train_blocks[cent_idx].copy()
        
        # K-Means (3 iters)
        for _ in range(3):
            c_norms = (centroids**2).sum(axis=1)
            new_cents = np.zeros_like(centroids)
            counts = np.zeros(self.n_codes)
            
            # Train on sample
            batch_size = 4096
            for i in range(0, len(train_blocks), batch_size):
                batch = train_blocks[i:i+batch_size]
                dots = np.dot(batch, centroids.T)
                dists = -2 * dots + c_norms[None, :]
                assigns = np.argmin(dists, axis=1)
                
                for j, a in enumerate(assigns):
                    new_cents[a] += batch[j]
                    counts[a] += 1
            
            mask = counts > 0
            centroids[mask] = new_cents[mask] / counts[mask][:, None]
        
        # Quantize all blocks
        recon_blocks = np.zeros_like(blocks)
        c_norms = (centroids**2).sum(axis=1)
        
        for i in range(0, len(blocks), batch_size):
            batch = blocks[i:i+batch_size]
            dots = np.dot(batch, centroids.T)
            dists = -2 * dots + c_norms[None, :]
            assigns = np.argmin(dists, axis=1)
            recon_blocks[i:i+batch_size] = centroids[assigns]
        
        # Reshape back
        W_recon = recon_blocks.flatten()[:W.size].reshape(d_out, d_in)
        return W_recon

def test_ppl():
    print("="*80)
    print("FULL-MODEL PPL TEST: High-Dim VQ (D=16, K=65536)")
    print("="*80)
    
    # Load model
    print("\n1. Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    
    # Baseline PPL
    print("\n2. FP16 Baseline PPL...")
    ppl_fp16 = evaluate_ppl(model, tokenizer)
    print(f"   FP16 PPL: {ppl_fp16:.2f}")
    
    # Quantize model
    print("\n3. Quantizing model...")
    vq = SimpleVQ(n_codes=65536, block_size=16)
    
    layers = model.transformer.h
    for i in tqdm(range(len(layers)), desc="   Layers"):
        layer = layers[i]
        
        # Quantize 4 weight matrices per layer
        for module_name in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
            module = layer
            for part in module_name.split('.'):
                module = getattr(module, part)
            
            # Get weight (transpose for Conv1D)
            W = module.weight.detach().cpu().numpy().T  # (out, in)
            
            # Quantize
            W_recon = vq.quantize(W)
            
            # Put back (transpose back)
            module.weight.data = torch.from_numpy(W_recon.T).to(module.weight.dtype)
    
    # Quantized PPL
    print("\n4. Quantized Model PPL...")
    ppl_quant = evaluate_ppl(model, tokenizer)
    print(f"   Quantized PPL: {ppl_quant:.2f}")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS:")
    print(f"  FP16:      {ppl_fp16:.2f}")
    print(f"  Quantized: {ppl_quant:.2f}")
    
    if ppl_quant < 30:
        print("  STATUS: ✅ SUCCESS!")
    elif ppl_quant < 100:
        print("  STATUS: ⚠️ Degraded but reasonable")
    else:
        print("  STATUS: ❌ BROKEN - needs debugging")
    
    print("="*80)
    
    # Save results
    with open("ppl_highdim_vq.txt", "w") as f:
        f.write(f"FP16: {ppl_fp16:.2f}\n")
        f.write(f"HighDimVQ (D=16, K=65536): {ppl_quant:.2f}\n")

if __name__ == "__main__":
    test_ppl()
