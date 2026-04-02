"""
Test: Compare bias-only vs bias+variance correction
"""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import copy

# Create two versions of HessianVQ

class HessianVQ_BiasOnly:
    """Version with ONLY bias correction"""
    def __init__(self, n_codes=32, block_size=4):
        self.n_codes = n_codes
        self.block_size = block_size
        
    def quantize(self, W, H_diag):
        d_out, d_in = W.shape
        bs = self.block_size
        
        # Signs and magnitude
        S = np.sign(W)
        S[S == 0] = 1.0
        M = np.abs(W)
        
        # Pad and reshape
        pad_h = (bs - d_out % bs) % bs
        pad_w = (bs - d_in % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)))
        H_mat = np.tile(H_diag, (d_out, 1))
        H_pad = np.pad(H_mat, ((0, pad_h), (0, pad_w)), constant_values=1e-6)
        
        h_p, w_p = M_pad.shape
        n_blocks_h = h_p // bs
        n_blocks_w = w_p // bs
        
        blocks = M_pad.reshape(n_blocks_h, bs, n_blocks_w, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        weights = H_pad.reshape(n_blocks_h, bs, n_blocks_w, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        
        # K-means
        if len(blocks) < self.n_codes:
            indices = np.arange(len(blocks))
            centroids = blocks.copy()
        else:
            indices = np.random.choice(len(blocks), self.n_codes, replace=False)
            centroids = blocks[indices].copy()
        
        for _ in range(5):
            dists = np.linalg.norm(blocks[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_codes):
                mask = (assignments == k)
                if np.sum(mask) > 0:
                    w_sum = np.sum(weights[mask], axis=0) + 1e-8
                    new_centroids[k] = np.sum(blocks[mask] * weights[mask], axis=0) / w_sum
                else:
                    new_centroids[k] = centroids[k]
            centroids = new_centroids
        
        # Reconstruct
        recon_blocks = centroids[assignments]
        M_recon_pad = recon_blocks.reshape(n_blocks_h, n_blocks_w, bs, bs).transpose(0, 2, 1, 3).reshape(h_p, w_p)
        M_recon = M_recon_pad[:d_out, :d_in]
        W_recon = S * M_recon
        
        # ONLY bias correction
        mean_orig = W.mean()
        mean_recon = W_recon.mean()
        W_recon = W_recon - mean_recon + mean_orig
        
        return W_recon

def test_bias_only():
    print("Testing BIAS-ONLY correction...")
    device = "cpu"
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Calibration
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"][:10])
    calib_ids = tokenizer(text, return_tensors="pt").input_ids[:, :512]
    
    # Layer 0 test
    layer0 = model.transformer.h[0]
    mlp_fc = layer0.mlp.c_fc
    
    activations = []
    handle = mlp_fc.register_forward_hook(lambda m,i,o: activations.append(i[0].detach()))
    with torch.no_grad():
        model(calib_ids)
    handle.remove()
    
    X = activations[0]
    H_diag = (X.reshape(-1, 768).numpy()**2).mean(axis=0)
    
    W_orig = mlp_fc.weight.detach().numpy()
    
    # Test bias-only
    hvq = HessianVQ_BiasOnly(n_codes=32)
    W_recon = hvq.quantize(W_orig.T, H_diag)
    
    print(f"Mean - Original: {W_orig.T.mean():.8f}, Recon: {W_recon.mean():.8f}")
    print(f"Std  - Original: {W_orig.T.std():.8f}, Recon: {W_recon.std():.8f}, Ratio: {W_recon.std()/W_orig.T.std():.4f}")
    
    # Output test
    with torch.no_grad():
        Y_fp16 = mlp_fc(X)
    
    mlp_fc.weight.data = torch.from_numpy(W_recon.T)
    with torch.no_grad():
        Y_q = mlp_fc(X)
    
    mse = ((Y_fp16 - Y_q)**2).mean().item()
    print(f"Output MSE: {mse:.6f}")
    print(f"Correlation: {np.corrcoef(Y_fp16.flatten().numpy(), Y_q.flatten().numpy())[0,1]:.4f}")

if __name__ == "__main__":
    test_bias_only()
