"""
Novel Ideas V20: The 1.00/1.00 Quest

Mission: Achieve 1.000 correlation at exactly 1.00 bpp.
Current Best: ~0.945 @ 1.00 bpp (V16), 0.9817 @ 1.56 bpp (V19)
Target: Bridge the gap!

Experiments:
1. Budget-Constrained Residual VQ (Tune K1/K2 for 1.00 bpp)
2. Triple-Stage Cascaded VQ (3 refinement passes)
3. Pruning + VQ Hybrid (Sparse mask + higher K on important blocks)
"""

import numpy as np
import sys

# Reuse components from V19
from typing import Tuple, Dict

def load_real_gpt2_data():
    """Load real GPT-2 weights and activation data."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        layer = model.transformer.h[0].mlp.c_fc
        W_real = layer.weight.detach().cpu().numpy().T
        
        activations = []
        def hook_fn(module, input, output):
            activations.append(input[0].detach().cpu().numpy())
        
        handle = layer.register_forward_hook(hook_fn)
        text = "The quick brown fox jumps over the lazy dog. " * 10
        inputs = tokenizer(text, return_tensors="pt")
        model(inputs.input_ids)
        handle.remove()
        
        X_real = np.concatenate(activations, axis=0).reshape(-1, activations[0].shape[-1])
        return W_real, X_real
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

# Base HessianBlockVQ (from V19)
class HessianBlockVQ:
    def __init__(self, d_in: int, d_out: int, n_codes: int = 16, block_size: int = 4):
        self.d_in, self.d_out, self.n_codes, self.block_size = d_in, d_out, n_codes, block_size
        self.S, self.codebook, self.assignments, self.index_entropy = None, None, None, 0.0
        
    def _weighted_kmeans(self, X, weights, k, max_iter=10):
        indices = np.random.choice(len(X), min(k, len(X)), replace=False)
        centroids = X[indices].copy()
        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = (assignments == i)
                if np.sum(mask) > 0:
                    new_centroids[i] = np.sum(X[mask] * weights[mask], axis=0) / (np.sum(weights[mask], axis=0) + 1e-8)
                else:
                    new_centroids[i] = centroids[i]
            if np.allclose(centroids, new_centroids, rtol=1e-3):
                break
            centroids = new_centroids
        return centroids, assignments

    def train(self, W_target: np.ndarray, H_diag: np.ndarray):
        self.S = np.sign(W_target)
        self.S[self.S == 0] = 1.0
        M = np.abs(W_target)
        H_mat = np.tile(H_diag, (W_target.shape[0], 1))
        
        bs = self.block_size
        h, w = M.shape
        pad_h, pad_w = (bs - h % bs) % bs, (bs - w % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)), mode='constant')
        H_pad = np.pad(H_mat, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=1e-6)
        
        h_p, w_p = M_pad.shape
        blocks = M_pad.reshape(h_p//bs, bs, w_p//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        weights_blocks = H_pad.reshape(h_p//bs, bs, w_p//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        
        self.codebook, self.assignments = self._weighted_kmeans(blocks, weights_blocks, self.n_codes)
        counts = np.bincount(self.assignments, minlength=self.n_codes)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        self.index_entropy = -np.sum(probs * np.log2(probs))
        
    def get_weights(self) -> np.ndarray:
        bs = self.block_size
        H, W = self.d_out, self.d_in
        h_p = H + (bs - H % bs) % bs
        w_p = W + (bs - W % bs) % bs
        recon_blocks = self.codebook[self.assignments]
        M_recon_pad = recon_blocks.reshape(h_p//bs, w_p//bs, bs, bs).transpose(0, 2, 1, 3).reshape(h_p, w_p)
        return self.S * M_recon_pad[:H, :W]
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = 0.5 * n_weights
        n_blocks = n_weights / (self.block_size ** 2)
        vq_bits = n_blocks * self.index_entropy
        codebook_bits = self.n_codes * (self.block_size ** 2) * 32
        return (sign_bits + vq_bits + codebook_bits) / n_weights

# 1. Budget-Constrained Residual VQ
class BudgetResidualVQ:
    """Tune (K1, K2) to hit exactly 1.00 bpp."""
    def __init__(self, d_in: int, d_out: int, target_bpp: float = 1.00):
        self.d_in, self.d_out, self.target_bpp = d_in, d_out, target_bpp
        self.hvq1, self.hvq2, self.best_k1, self.best_k2 = None, None, None, None
        
    def train(self, W_target: np.ndarray, H_diag: np.ndarray):
        best_corr, best_k1, best_k2 = -1, 16, 4
        X_test = np.random.randn(100, self.d_in) * 0.1
        Y_test = X_test @ W_target.T
        
        # Grid search
        for k1 in [12, 16, 20, 24]:
            for k2 in [4, 6, 8, 10]:
                # Estimate BPP
                estimated_bpp = 0.5 + (np.log2(k1)/16) + (np.log2(k2)/16)  # Simplified
                if abs(estimated_bpp - self.target_bpp) > 0.15:
                    continue
                
                # Train
                hvq1 = HessianBlockVQ(self.d_in, self.d_out, n_codes=k1)
                hvq1.train(W_target, H_diag)
                W_q1 = hvq1.get_weights()
                
                Residual = W_target - W_q1
                hvq2 = HessianBlockVQ(self.d_in, self.d_out, n_codes=k2)
                hvq2.train(Residual, H_diag * 0.5)
                W_q2 = hvq2.get_weights()
                
                W_final = W_q1 + W_q2
                corr = np.corrcoef((X_test @ W_final.T).flatten(), Y_test.flatten())[0,1]
                
                if corr > best_corr:
                    best_corr, best_k1, best_k2 = corr, k1, k2
        
        # Train with best
        self.best_k1, self.best_k2 = best_k1, best_k2
        self.hvq1 = HessianBlockVQ(self.d_in, self.d_out, n_codes=best_k1)
        self.hvq1.train(W_target, H_diag)
        W_q1 = self.hvq1.get_weights()
        
        Residual = W_target - W_q1
        self.hvq2 = HessianBlockVQ(self.d_in, self.d_out, n_codes=best_k2)
        self.hvq2.train(Residual, H_diag * 0.5)
        
    def get_weights(self) -> np.ndarray:
        return self.hvq1.get_weights() + self.hvq2.get_weights()
        
    def effective_bpp(self) -> float:
        return self.hvq1.effective_bpp() + self.hvq2.effective_bpp()

# 2. Triple-Stage Cascaded VQ
class TripleStageVQ:
    """3-pass refinement."""
    def __init__(self, d_in: int, d_out: int, k1: int = 16, k2: int = 8, k3: int = 4):
        self.d_in, self.d_out = d_in, d_out
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.hvq1, self.hvq2, self.hvq3 = None, None, None
        
    def train(self, W_target: np.ndarray, H_diag: np.ndarray):
        # Stage 1
        self.hvq1 = HessianBlockVQ(self.d_in, self.d_out, n_codes=self.k1)
        self.hvq1.train(W_target, H_diag)
        W_q1 = self.hvq1.get_weights()
        
        # Stage 2
        R1 = W_target - W_q1
        self.hvq2 = HessianBlockVQ(self.d_in, self.d_out, n_codes=self.k2)
        self.hvq2.train(R1, H_diag * 0.5)
        W_q2 = self.hvq2.get_weights()
        
        # Stage 3
        R2 = R1 - W_q2
        self.hvq3 = HessianBlockVQ(self.d_in, self.d_out, n_codes=self.k3)
        self.hvq3.train(R2, H_diag * 0.25)
        
    def get_weights(self) -> np.ndarray:
        return self.hvq1.get_weights() + self.hvq2.get_weights() + self.hvq3.get_weights()
        
    def effective_bpp(self) -> float:
        return self.hvq1.effective_bpp() + self.hvq2.effective_bpp() + self.hvq3.effective_bpp()

# 3. Pruning + VQ Hybrid
class PruningHybridVQ:
    """Sparse mask + VQ on important blocks."""
    def __init__(self, d_in: int, d_out: int, prune_ratio: float = 0.15, k_dense: int = 32):
        self.d_in, self.d_out = d_in, d_out
        self.prune_ratio, self.k_dense = prune_ratio, k_dense
        self.sparse_mask, self.hvq = None, None
        
    def train(self, W_target: np.ndarray, H_diag: np.ndarray):
        # Compute block importance
        bs = 4
        h, w = W_target.shape
        M = np.abs(W_target)
        H_mat = np.tile(H_diag, (h, 1))
        
        pad_h, pad_w = (bs - h % bs) % bs, (bs - w % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)))
        H_pad = np.pad(H_mat, ((0, pad_h), (0, pad_w)), constant_values=1e-6)
        
        h_p, w_p = M_pad.shape
        M_blocks = M_pad.reshape(h_p//bs, bs, w_p//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        H_blocks = H_pad.reshape(h_p//bs, bs, w_p//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        
        # Importance = magnitude × Hessian
        importance = np.sum(M_blocks * H_blocks, axis=1)
        threshold = np.percentile(importance, self.prune_ratio * 100)
        
        # Sparse mask
        self.sparse_mask = (importance > threshold)
        
        # VQ on non-pruned blocks (simulate by quantizing full matrix with awareness)
        self.hvq = HessianBlockVQ(self.d_in, self.d_out, n_codes=self.k_dense)
        self.hvq.train(W_target, H_diag)
        
    def get_weights(self) -> np.ndarray:
        W_q = self.hvq.get_weights()
        # Apply sparse mask (zero out pruned blocks)
        bs = 4
        H, W = self.d_out, self.d_in
        h_p = H + (bs - H % bs) % bs
        w_p = W + (bs - W % bs) % bs
        
        W_q_pad = np.pad(W_q, ((0, h_p-H), (0, w_p-W)))
        W_q_blocks = W_q_pad.reshape(h_p//bs, bs, w_p//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        
        # Zero out pruned
        W_q_blocks[~self.sparse_mask] = 0
        
        W_final_pad = W_q_blocks.reshape(h_p//bs, w_p//bs, bs, bs).transpose(0, 2, 1, 3).reshape(h_p, w_p)
        return W_final_pad[:H, :W]
        
    def effective_bpp(self) -> float:
        base_bpp = self.hvq.effective_bpp()
        # Sparse mask cost
        n_blocks = (self.d_out * self.d_in) / 16
        mask_bits = n_blocks  # 1 bit per block
        return base_bpp + (mask_bits / (self.d_out * self.d_in))

# EXPERIMENT RUNNER
def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V20: THE 1.00/1.00 QUEST")
    print("="*80)
    print("Mission: Achieve 1.000 correlation at 1.00 bpp")
    print()
    
    W_real, X_real = load_real_gpt2_data()
    d_out, d_in = W_real.shape
    H_diag = np.mean(X_real**2, axis=0)
    
    X_test = X_real[:500]
    Y_test = X_test @ W_real.T
    
    results = {}
    
    # Baselines
    S_bin = np.sign(W_real)
    S_bin[S_bin==0] = 1
    W_tern = S_bin * (np.abs(W_real) > np.percentile(np.abs(W_real), 30))
    scale_tern = np.mean(np.abs(W_real[np.abs(W_real) > np.percentile(np.abs(W_real), 30)]))
    W_tern = W_tern * scale_tern
    corr_tern = np.corrcoef((X_test @ W_tern.T).flatten(), Y_test.flatten())[0,1]
    results['Ternary'] = {'corr': corr_tern, 'bpp': 1.58}

    
    # 1. Budget-Constrained Residual
    print("\n1. Budget-Constrained Residual VQ...")
    for target in [0.95, 1.00, 1.05]:
        brvq = BudgetResidualVQ(d_in, d_out, target_bpp=target)
        brvq.train(W_real, H_diag)
        W_brvq = brvq.get_weights()
        corr = np.corrcoef((X_test @ W_brvq.T).flatten(), Y_test.flatten())[0,1]
        bpp = brvq.effective_bpp()
        results[f'BudgetResidual (target={target:.2f})'] = {'corr': corr, 'bpp': bpp}
        print(f"Target={target:.2f}, K1={brvq.best_k1}, K2={brvq.best_k2}: {corr:.4f} @ {bpp:.2f} bpp")
    
    # 2. Triple-Stage
    print("\n2. Triple-Stage Cascaded VQ...")
    for k1, k2, k3 in [(16, 8, 4), (20, 10, 5), (12, 8, 6)]:
        tsvq = TripleStageVQ(d_in, d_out, k1=k1, k2=k2, k3=k3)
        tsvq.train(W_real, H_diag)
        W_tsvq = tsvq.get_weights()
        corr = np.corrcoef((X_test @ W_tsvq.T).flatten(), Y_test.flatten())[0,1]
        bpp = tsvq.effective_bpp()
        results[f'TripleStage ({k1}/{k2}/{k3})'] = {'corr': corr, 'bpp': bpp}
        print(f"K1={k1}, K2={k2}, K3={k3}: {corr:.4f} @ {bpp:.2f} bpp")
    
    # 3. Pruning Hybrid
    print("\n3. Pruning + VQ Hybrid...")
    for prune in [0.10, 0.15, 0.20]:
        phvq = PruningHybridVQ(d_in, d_out, prune_ratio=prune)
        phvq.train(W_real, H_diag)
        W_phvq = phvq.get_weights()
        corr = np.corrcoef((X_test @ W_phvq.T).flatten(), Y_test.flatten())[0,1]
        bpp = phvq.effective_bpp()
        results[f'PruningHybrid (p={prune:.2f})'] = {'corr': corr, 'bpp': bpp}
        print(f"Prune={prune:.0%}: {corr:.4f} @ {bpp:.2f} bpp")
    
    # Summary
    print("\n" + "="*80)
    print(f"{'Method':<35} {'Corr':>8} {'BPP':>8} {'To 1.00':>10}")
    print("-" * 70)
    
    with open("results_v20_utf8.txt",   "w", encoding="utf-8") as f:
        for name, res in sorted(results.items(), key=lambda x: -x[1]['corr']):
            to_1 = abs(res['corr'] - 1.0)
            line = f"{name:<35} {res['corr']:>8.4f} {res['bpp']:>8.2f} {to_1:>10.4f}\n"
            print(line.strip())
            f.write(line)
    
    print("\nResults written to results_v20_utf8.txt")

if __name__ == "__main__":
    run_experiments()
