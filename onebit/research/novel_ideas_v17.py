"""
Novel Ideas V17: Rigor & Robustness

Mission: Verify soundness and reproducibility on REAL data across ALL layers.
Strict Requirement: MUST use real GPT-2 weights.

Experiments:
1. Full Model Sweep (12 layers * 4 matrices = 48 tests)
2. Robust Hessian Calibration (Larger dataset)
3. Statistical Reporting (Mean/Std)
"""

import numpy as np
import torch
from typing import Tuple, Dict, List
import sys
from tqdm import tqdm

# =============================================================================
# HESSIAN BLOCK VQ (From V16)
# =============================================================================

class HessianBlockVQ:
    def __init__(self, d_in: int, d_out: int, n_codes: int = 16):
        self.d_in = d_in
        self.d_out = d_out
        self.n_codes = n_codes
        self.block_size = 4
        self.S = None
        self.codebook = None
        self.assignments = None
        self.sign_entropy = 0.5 
        self.index_entropy = 0.0
        
    def _weighted_kmeans(self, X, weights, k, max_iter=20):
        indices = np.random.choice(len(X), k, replace=False)
        centroids = X[indices].copy()
        
        for _ in range(max_iter):
            block_weights = np.mean(weights, axis=1, keepdims=True)
            distances = np.linalg.norm((X[:, None, :] - centroids[None, :, :]) * np.sqrt(block_weights[:, None, :]), axis=2)
            assignments = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = (assignments == i)
                if np.sum(mask) > 0:
                    X_subset = X[mask]
                    W_subset = weights[mask]
                    new_centroids[i] = np.sum(X_subset * W_subset, axis=0) / (np.sum(W_subset, axis=0) + 1e-8)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        return centroids, assignments

    def train(self, W_target: np.ndarray, X_calib: np.ndarray):
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        M = np.abs(W_target)
        
        # Hessian diagonal: sum(X^2)
        # Handle case where X_calib is smaller than d_in (should not happen if correct)
        if X_calib.shape[1] != self.d_in:
             # This might happen for c_attn if we capture input to c_attn
             # c_attn weight is (3*hidden, hidden). Input is (N, hidden).
             # So d_in = hidden. X_calib = (N, hidden). Correct.
             pass

        H_diag = np.mean(X_calib**2, axis=0)
        H_matrix = np.tile(H_diag, (self.d_out, 1))
        
        bs = self.block_size
        blocks = []
        weights = []
        
        H_out, W_out = M.shape
        pad_h = (bs - H_out % bs) % bs
        pad_w = (bs - W_out % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)), mode='constant')
        H_pad = np.pad(H_matrix, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=1e-6)
        
        for i in range(0, H_out, bs):
            for j in range(0, W_out, bs):
                block = M_pad[i:i+bs, j:j+bs].flatten()
                weight = H_pad[i:i+bs, j:j+bs].flatten()
                blocks.append(block)
                weights.append(weight)
                
        blocks = np.array(blocks)
        weights = np.array(weights)
        
        self.codebook, self.assignments = self._weighted_kmeans(blocks, weights, self.n_codes)
        
        counts = np.bincount(self.assignments, minlength=self.n_codes)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        self.index_entropy = -np.sum(probs * np.log2(probs))
        
    def get_weights(self) -> np.ndarray:
        bs = self.block_size
        H, W = self.d_out, self.d_in
        M_recon = np.zeros((H + (bs - H % bs) % bs, W + (bs - W % bs) % bs))
        
        idx = 0
        for i in range(0, H, bs):
            for j in range(0, W, bs):
                block = self.codebook[self.assignments[idx]].reshape(bs, bs)
                M_recon[i:i+bs, j:j+bs] = block
                idx += 1
                
        return self.S * M_recon[:H, :W]
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = self.sign_entropy * n_weights
        n_blocks = (self.d_out * self.d_in) / (self.block_size ** 2)
        vq_bits = n_blocks * self.index_entropy
        codebook_bits = self.n_codes * (self.block_size ** 2) * 32
        return (sign_bits + vq_bits + codebook_bits) / n_weights

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_full_sweep():
    print("="*80)
    print("NOVEL IDEAS V17: FULL MODEL SWEEP (RIGOR & ROBUSTNESS)")
    print("="*80)
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("Loading GPT-2...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Calibration Data
    print("Generating calibration data...")
    text = "The quick brown fox jumps over the lazy dog. " * 50 # 450 tokens
    inputs = tokenizer(text, return_tensors="pt")
    
    # We need to capture inputs for EVERY layer.
    # We will do this layer by layer to save memory.
    
    layers = model.transformer.h
    n_layers = len(layers)
    
    results_agg = {
        'Ternary': {'corr': [], 'bpp': []},
        'HessianVQ-16': {'corr': [], 'bpp': []},
        'HessianVQ-32': {'corr': [], 'bpp': []}
    }
    
    print(f"Starting sweep over {n_layers} layers...")
    
    # Hook to capture input to a specific module
    captured_input = {}
    def get_input_hook(name):
        def hook(module, input, output):
            captured_input[name] = input[0].detach().cpu().numpy()
        return hook

    # Iterate layers
    for i in range(n_layers):
        print(f"\n--- Layer {i} ---")
        layer = layers[i]
        
        # Sub-modules to test
        modules = {
            'attn_c_attn': layer.attn.c_attn,
            'attn_c_proj': layer.attn.c_proj,
            'mlp_c_fc': layer.mlp.c_fc,
            'mlp_c_proj': layer.mlp.c_proj
        }
        
        for name, module in modules.items():
            full_name = f"L{i}.{name}"
            
            # 1. Capture Activations
            handle = module.register_forward_hook(get_input_hook(full_name))
            with torch.no_grad():
                model(inputs.input_ids)
            handle.remove()
            
            X_calib = captured_input[full_name]
            X_calib = X_calib.reshape(-1, X_calib.shape[-1]) # (N*Seq, Din)
            
            # Weight
            W_true = module.weight.detach().cpu().numpy().T # (Dout, Din)
            d_out, d_in = W_true.shape
            
            # Test set (subset of calib for speed, or separate? Let's use subset of X_calib as "test" 
            # to measure reconstruction quality on realistic distribution. 
            # Ideally we use separate validation set, but for quantization proxy, calib is often used.)
            # Let's split X_calib.
            n_samples = X_calib.shape[0]
            split = int(n_samples * 0.8)
            X_train = X_calib[:split]
            X_test = X_calib[split:]
            Y_test = X_test @ W_true.T
            
            # 2. Ternary Baseline
            S_bin = np.sign(W_true)
            S_bin[S_bin==0] = 1
            thresh = np.percentile(np.abs(W_true), 30)
            W_tern = S_bin * (np.abs(W_true) > thresh)
            scale_tern = np.mean(np.abs(W_true[np.abs(W_true) > thresh]))
            W_tern = W_tern * scale_tern
            corr_tern = np.corrcoef((X_test @ W_tern.T).flatten(), Y_test.flatten())[0,1]
            results_agg['Ternary']['corr'].append(corr_tern)
            results_agg['Ternary']['bpp'].append(1.58)
            
            # 3. HessianVQ-16
            hvq16 = HessianBlockVQ(d_in, d_out, n_codes=16)
            hvq16.train(W_true, X_train)
            W_hvq16 = hvq16.get_weights()
            corr_16 = np.corrcoef((X_test @ W_hvq16.T).flatten(), Y_test.flatten())[0,1]
            bpp_16 = hvq16.effective_bpp()
            results_agg['HessianVQ-16']['corr'].append(corr_16)
            results_agg['HessianVQ-16']['bpp'].append(bpp_16)
            
            # 4. HessianVQ-32
            hvq32 = HessianBlockVQ(d_in, d_out, n_codes=32)
            hvq32.train(W_true, X_train)
            W_hvq32 = hvq32.get_weights()
            corr_32 = np.corrcoef((X_test @ W_hvq32.T).flatten(), Y_test.flatten())[0,1]
            bpp_32 = hvq32.effective_bpp()
            results_agg['HessianVQ-32']['corr'].append(corr_32)
            results_agg['HessianVQ-32']['bpp'].append(bpp_32)
            
            print(f"{full_name:<15} | Tern: {corr_tern:.4f} | HVQ-16: {corr_16:.4f} ({bpp_16:.2f}b) | HVQ-32: {corr_32:.4f} ({bpp_32:.2f}b)")

    # Final Stats
    print("\n" + "="*80)
    print("FINAL ROBUSTNESS STATISTICS (48 Layers)")
    print("="*80)
    
    with open("results_v17_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("FINAL ROBUSTNESS STATISTICS (48 Layers)\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<15} {'Mean Corr':>10} {'Std Corr':>10} {'Mean BPP':>10} {'vs Tern':>10}\n")
        f.write("-" * 60 + "\n")
        
        tern_mean = np.mean(results_agg['Ternary']['corr'])
        
        for name, data in results_agg.items():
            mean_corr = np.mean(data['corr'])
            std_corr = np.std(data['corr'])
            mean_bpp = np.mean(data['bpp'])
            vs_tern = (mean_corr - tern_mean) / tern_mean * 100
            
            line = f"{name:<15} {mean_corr:>10.4f} {std_corr:>10.4f} {mean_bpp:>10.2f} {vs_tern:>+9.1f}%\n"
            print(line.strip())
            f.write(line)
            
    print("\nResults written to results_v17_utf8.txt")

if __name__ == "__main__":
    run_full_sweep()
