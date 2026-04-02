"""
LAYER-WISE CALIBRATION: Calibrate one layer at a time.
This is more stable than end-to-end training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class LowRankBinaryConv1D(nn.Module):
    def __init__(self, orig_weight, rank=8):
        super().__init__()
        n_in, n_out = orig_weight.shape
        sign = torch.sign(orig_weight)
        sign[sign == 0] = 1
        self.register_buffer('sign', sign)
        
        # SVD init
        W_abs = orig_weight.abs()
        U, S, Vh = torch.linalg.svd(W_abs, full_matrices=False)
        self.U = nn.Parameter(U[:, :rank] * S[:rank].sqrt().unsqueeze(0))
        self.V = nn.Parameter(Vh[:rank, :].T * S[:rank].sqrt().unsqueeze(0))
        
    def forward(self, x):
        mag = F.relu(self.U @ self.V.T) + 1e-8
        return x @ (self.sign * mag)


def compute_ppl(model, tokenizer, texts):
    losses = []
    for text in texts:
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True)
        if tokens.shape[1] < 5:
            continue
        with torch.no_grad():
            out = model(tokens, labels=tokens)
            if not torch.isnan(out.loss):
                losses.append(out.loss.item())
    return np.exp(np.mean(losses)) if losses else float('inf')


def main():
    print("=" * 70)
    print("LAYER-WISE CALIBRATION")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    train_texts = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Machine learning algorithms process vast amounts of data efficiently.",
        "Scientists recently discovered a new species in the Amazon rainforest.",
        "The stock market showed significant gains in the technology sector.",
        "Climate change poses serious risks to global food production systems.",
        "Modern architecture combines functionality with innovative design.",
        "The history of mathematics spans thousands of years of discovery.",
        "Quantum computers may revolutionize cryptography and drug discovery.",
    ]
    
    val_texts = [
        "Neural networks can learn complex patterns from training data.",
        "Deep ocean exploration has revealed many unknown species.",
        "Electric vehicles are becoming popular as technology improves.",
        "Space exploration continues to push the boundaries of science.",
    ]
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    fp32_ppl = compute_ppl(model, tokenizer, val_texts)
    print(f"FP32 baseline: {fp32_ppl:.2f}")
    
    # Collect activations for layer-wise calibration
    def get_activations(model, texts):
        acts = {i: {'mlp_in': [], 'attn_in': []} for i in range(12)}
        hooks = []
        
        def make_hook(layer_idx, name):
            def hook(module, inp, out):
                acts[layer_idx][name].append(inp[0].detach())
            return hook
        
        for i in range(12):
            hooks.append(model.transformer.h[i].mlp.c_fc.register_forward_hook(make_hook(i, 'mlp_in')))
            hooks.append(model.transformer.h[i].attn.c_attn.register_forward_hook(make_hook(i, 'attn_in')))
        
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True)
            with torch.no_grad():
                model(tokens)
        
        for h in hooks:
            h.remove()
        return acts
    
    print("Collecting activations...")
    acts = get_activations(model, train_texts)
    
    # Layer-by-layer calibration
    RANK = 8
    for layer_idx in range(12):
        block = model.transformer.h[layer_idx]
        
        # Calibrate MLP
        for name in ['c_fc', 'c_proj']:
            orig = getattr(block.mlp, name).weight.data.clone()
            calib = LowRankBinaryConv1D(orig, rank=RANK)
            
            # Get input activations - flatten each separately then cat
            if name == 'c_fc':
                X_list = [a.reshape(-1, orig.shape[0]) for a in acts[layer_idx]['mlp_in']]
                X = torch.cat(X_list, dim=0)
            else:
                X = torch.randn(100, orig.shape[0]) * 0.1
            
            # Target: original output
            with torch.no_grad():
                Y_target = X @ orig
            
            # Optimize
            optimizer = torch.optim.Adam([calib.U, calib.V], lr=1e-3)
            for _ in range(100):
                optimizer.zero_grad()
                Y_pred = calib(X)
                loss = F.mse_loss(Y_pred, Y_target)
                loss.backward()
                optimizer.step()
            
            setattr(block.mlp, name, calib)
        
        # Calibrate Attention
        for name in ['c_attn', 'c_proj']:
            orig = getattr(block.attn, name).weight.data.clone()
            calib = LowRankBinaryConv1D(orig, rank=RANK)
            
            if name == 'c_attn':
                X_list = [a.reshape(-1, orig.shape[0]) for a in acts[layer_idx]['attn_in']]
                X = torch.cat(X_list, dim=0)
            else:
                X = torch.randn(100, orig.shape[0]) * 0.1
            
            with torch.no_grad():
                Y_target = X @ orig
            
            optimizer = torch.optim.Adam([calib.U, calib.V], lr=1e-3)
            for _ in range(100):
                optimizer.zero_grad()
                loss = F.mse_loss(calib(X), Y_target)
                loss.backward()
                optimizer.step()
            
            setattr(block.attn, name, calib)
        
        ppl = compute_ppl(model, tokenizer, val_texts)
        print(f"  Layer {layer_idx}: val_ppl={ppl:.2f}")
    
    final_ppl = compute_ppl(model, tokenizer, val_texts)
    print(f"\nFinal: {final_ppl:.2f} ({(final_ppl/fp32_ppl-1)*100:+.1f}% vs FP32)")


if __name__ == '__main__':
    main()

