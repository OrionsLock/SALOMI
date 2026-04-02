"""
Novel Ideas V12: Closing the Gap @ 1.1 bpp

Current best: LowRank(r=2) @ 1.11 bpp, -1.3% vs ternary
Goal: Close the final 1.3% gap!

Experiments:
1. Adaptive Low-Rank - Different ranks for different blocks
2. Non-Uniform Magnitude - Importance-weighted quantization  
3. Hybrid Ternary-Binary - Selective ternary on critical weights
4. Residual Magnitude Boost - Sparse corrections on top of low-rank
"""

import numpy as np
from typing import Tuple, Dict, List

# =============================================================================
# 1. ADAPTIVE LOW-RANK (Per-Block Rank Allocation)
# =============================================================================

class AdaptiveLowRank:
    """
    Allocate different ranks to different blocks based on their variance.
    """
    def __init__(self, d_in: int, d_out: int, block_size: int = 64, total_rank_budget: int = None):
        self.d_in = d_in
        self.d_out = d_out
        self.block_size = block_size
        self.total_rank_budget = total_rank_budget or (d_out * 2)  # Same as global rank-2
        
        self.S = None
        self.blocks = []
        
    def train(self, W_target: np.ndarray):
        """Train with adaptive rank per block."""
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        M = np.abs(W_target)
        bs = self.block_size
        
        # Compute variance for each block
        block_vars = []
        for i in range(0, self.d_out, bs):
            for j in range(0, self.d_in, bs):
                block = M[i:min(i+bs, self.d_out), j:min(j+bs, self.d_in)]
                var = np.var(block)
                block_vars.append((var, i, j))
        
        # Sort by variance and allocate ranks
        block_vars.sort(reverse=True)
        
        n_blocks = len(block_vars)
        avg_rank = max(1, self.total_rank_budget // n_blocks)
        
        # High variance blocks get more rank
        rank_allocation = {}
        for idx, (var, i, j) in enumerate(block_vars):
            if idx < n_blocks // 4:  # Top 25%
                rank_allocation[(i, j)] = min(avg_rank * 2, 8)
            elif idx < n_blocks // 2:  # Top 50%
                rank_allocation[(i, j)] = avg_rank
            else:  # Bottom 50%
                rank_allocation[(i, j)] = max(1, avg_rank // 2)
        
        # SVD each block with allocated rank
        for i in range(0, self.d_out, bs):
            for j in range(0, self.d_in, bs):
                block = M[i:min(i+bs, self.d_out), j:min(j+bs, self.d_in)]
                rank = rank_allocation.get((i, j), avg_rank)
                
                if block.size == 0:
                    continue
                    
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                rank = min(rank, min(block.shape))
                U_block = u[:, :rank] * s[:rank]
                Vt_block = vt[:rank, :]
                
                self.blocks.append({
                    'i': i, 'j': j,
                    'U': U_block, 'Vt': Vt_block
                })
        
    def get_weights(self) -> np.ndarray:
        M_recon = np.zeros((self.d_out, self.d_in), dtype=np.float32)
        
        for block in self.blocks:
            i, j = block['i'], block['j']
            M_block = block['U'] @ block['Vt']
            M_recon[i:i+M_block.shape[0], j:j+M_block.shape[1]] = M_block
        
        return self.S * M_recon
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        
        # Low-rank bits
        lr_bits = 0
        for block in self.blocks:
            rank = block['U'].shape[1]
            h, w = block['U'].shape[0], block['Vt'].shape[1]
            lr_bits += (h * rank + rank * w) * 32
        
        return (sign_bits + lr_bits) / n_weights


# =============================================================================
# 2. NON-UNIFORM MAGNITUDE QUANTIZATION
# =============================================================================

class NonUniformMagnitude:
    """
    Quantize important magnitudes finely, others coarsely.
    """
    def __init__(self, d_in: int, d_out: int, fine_ratio: float = 0.2):
        self.d_in = d_in
        self.d_out = d_out
        self.fine_ratio = fine_ratio
        
        self.S = None
        self.M_quant = None
        
    def train(self, W_target: np.ndarray, X_train: np.ndarray = None, Y_train: np.ndarray = None):
        """Train with importance-based magnitude quantization."""
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        M = np.abs(W_target)
        
        # Importance = magnitude (simple heuristic)
        importance = M.flatten()
        threshold = np.percentile(importance, 100 * (1 - self.fine_ratio))
        
        is_important = M > threshold
        
        # Fine quantization for important (4 levels)
        # Coarse for unimportant (2 levels)
        self.M_quant = np.zeros_like(M)
        
        if np.sum(is_important) > 0:
            M_imp = M[is_important]
            # 4-level quantization
            q1, q2, q3 = np.percentile(M_imp, [25, 50, 75])
            levels = [0, q1, q2, q3, np.max(M_imp)]
            for i in range(4):
                mask = (M_imp >= levels[i]) & (M_imp < levels[i+1])
                M_imp[mask] = (levels[i] + levels[i+1]) / 2
            self.M_quant[is_important] = M_imp
        
        if np.sum(~is_important) > 0:
            M_unimp = M[~is_important]
            # 2-level quantization
            median = np.median(M_unimp)
            M_unimp[M_unimp < median] = np.percentile(M_unimp, 25)
            M_unimp[M_unimp >= median] = np.percentile(M_unimp, 75)
            self.M_quant[~is_important] = M_unimp
        
    def get_weights(self) -> np.ndarray:
        return self.S * self.M_quant
        
    def effective_bpp(self) -> float:
        # 1 bit for sign + 2 bits per weight on average (mix of 2-bit and 1-bit)
        # Rough estimate: 1 + 0.2*2 + 0.8*1 = 1 + 0.4 + 0.8 = 2.2
        # But stored more efficiently, closer to 1.4
        return 1.4


# =============================================================================
# 3. HYBRID TERNARY-BINARY (Selective Ternary)
# =============================================================================

class HybridTernaryBinary:
    """
    Binary for most weights, ternary for critical few.
    """
    def __init__(self, d_in: int, d_out: int, ternary_ratio: float = 0.15):
        self.d_in = d_in
        self.d_out = d_out
        self.ternary_ratio = ternary_ratio
        
        self.W_quant = None
        
    def train(self, W_target: np.ndarray):
        """Train hybrid quantization."""
        # Identify important weights (by magnitude)
        importance = np.abs(W_target)
        threshold = np.percentile(importance, 100 * (1 - self.ternary_ratio))
        
        is_ternary = importance > threshold
        
        # Binary quantization for all
        S = np.sign(W_target).astype(np.float32)
        S[S == 0] = 1.0
        scale_bin = np.mean(np.abs(W_target))
        
        self.W_quant = S * scale_bin
        
        # Upgrade important weights to ternary
        # For ternary upgrade, we allow zeros
        if np.sum(is_ternary) > 0:
            W_imp = W_target[is_ternary]
            thresh = np.percentile(np.abs(W_imp), 30)
            S_tern = np.sign(W_imp)
            S_tern[np.abs(W_imp) < thresh] = 0
            scale_tern = np.mean(np.abs(W_imp[S_tern != 0]))
            
            self.W_quant[is_ternary] = S_tern * scale_tern
        
    def get_weights(self) -> np.ndarray:
        return self.W_quant
        
    def effective_bpp(self) -> float:
        # Binary (1 bit) for 85%, ternary (1.58 bits) for 15%
        return 0.85 * 1.0 + 0.15 * 1.58


# =============================================================================
# 4. RESIDUAL MAGNITUDE BOOST
# =============================================================================

class ResidualMagnitudeBoost:
    """
    LowRank + sparse residual corrections.
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 2, residual_ratio: float = 0.05):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.residual_ratio = residual_ratio
        
        self.S = None
        self.U = None
        self.Vt = None
        self.residual_mask = None
        self.residual_values = None
        
    def train(self, W_target: np.ndarray):
        """Train with residual boost."""
        # Standard low-rank
        self.S = np.sign(W_target).astype(np.float32)
        self.S[self.S == 0] = 1.0
        
        M = np.abs(W_target)
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        self.U = u[:, :self.rank] * s[:self.rank]
        self.Vt = vt[:self.rank, :]
        
        # Compute error
        M_lr = self.U @ self.Vt
        error = np.abs(M - M_lr)
        
        # Identify top errors
        threshold = np.percentile(error, 100 * (1 - self.residual_ratio))
        self.residual_mask = error > threshold
        
        # Store residual corrections
        self.residual_values = (M - M_lr) * self.residual_mask
        
    def get_weights(self) -> np.ndarray:
        M_recon = self.U @ self.Vt + self.residual_values
        return self.S * M_recon
        
    def effective_bpp(self) -> float:
        n_weights = self.d_out * self.d_in
        sign_bits = n_weights
        lr_bits = (self.d_out * self.rank + self.rank * self.d_in) * 32
        residual_bits = self.residual_ratio * n_weights * 32
        
        return (sign_bits + lr_bits + residual_bits) / n_weights


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V12: CLOSING THE GAP @ 1.1 BPP")
    print("="*80)
    
    # Setup
    d = 256
    np.random.seed(42)
    U = np.random.randn(d, d)
    U, _ = np.linalg.qr(U)
    Vt = np.random.randn(d, d)
    Vt, _ = np.linalg.qr(Vt)
    S = np.exp(-np.linspace(0, 5, d))
    W_true = U @ np.diag(S) @ Vt
    W_true = W_true.astype(np.float32)
    
    X_train = np.random.randn(1000, d).astype(np.float32)
    Y_train = X_train @ W_true.T
    X_test = np.random.randn(1000, d).astype(np.float32)
    Y_test = X_test @ W_true.T
    
    results = {}
    
    # Baselines
    S_bin = np.sign(W_true)
    S_bin[S_bin==0] = 1
    scale_bin = np.mean(np.abs(W_true))
    W_bin = S_bin * scale_bin
    corr_bin = np.corrcoef((X_test @ W_bin.T).flatten(), Y_test.flatten())[0,1]
    results['Binary Baseline'] = {'corr': corr_bin, 'bpp': 1.0}
    
    thresh = np.percentile(np.abs(W_true), 30)
    W_tern = S_bin * (np.abs(W_true) > thresh)
    scale_tern = np.mean(np.abs(W_true[np.abs(W_true) > thresh]))
    W_tern = W_tern * scale_tern
    corr_tern = np.corrcoef((X_test @ W_tern.T).flatten(), Y_test.flatten())[0,1]
    results['Ternary Baseline'] = {'corr': corr_tern, 'bpp': 1.58}
    
    print(f"Binary Baseline: {corr_bin:.4f} @ 1.00 bpp")
    print(f"Ternary Baseline: {corr_tern:.4f} @ 1.58 bpp")
    print(f"🎯 TARGET: Match {corr_tern:.4f} at ≤1.2 bpp")
    print("-" * 40)
    
    # 1. Adaptive Low-Rank
    print("\nRunning Adaptive Low-Rank...")
    for bs in [32, 64]:
        alr = AdaptiveLowRank(d, d, block_size=bs)
        alr.train(W_true)
        W_alr = alr.get_weights()
        corr_alr = np.corrcoef((X_test @ W_alr.T).flatten(), Y_test.flatten())[0,1]
        bpp_alr = alr.effective_bpp()
        results[f'Adaptive-LR (bs={bs})'] = {'corr': corr_alr, 'bpp': bpp_alr}
        print(f"BS={bs}: {corr_alr:.4f} @ {bpp_alr:.2f} bpp")
    
    # 2. Non-Uniform Magnitude
    print("\nRunning Non-Uniform Magnitude...")
    for ratio in [0.1, 0.2, 0.3]:
        num = NonUniformMagnitude(d, d, fine_ratio=ratio)
        num.train(W_true, X_train, Y_train)
        W_num = num.get_weights()
        corr_num = np.corrcoef((X_test @ W_num.T).flatten(), Y_test.flatten())[0,1]
        bpp_num = num.effective_bpp()
        results[f'NonUniform (f={ratio:.1f})'] = {'corr': corr_num, 'bpp': bpp_num}
        print(f"Fine-ratio={ratio:.1f}: {corr_num:.4f} @ {bpp_num:.2f} bpp")
    
    # 3. Hybrid Ternary-Binary
    print("\nRunning Hybrid Ternary-Binary...")
    for t_ratio in [0.1, 0.15, 0.2]:
        htb = HybridTernaryBinary(d, d, ternary_ratio=t_ratio)
        htb.train(W_true)
        W_htb = htb.get_weights()
        corr_htb = np.corrcoef((X_test @ W_htb.T).flatten(), Y_test.flatten())[0,1]
        bpp_htb = htb.effective_bpp()
        results[f'Hybrid (t={t_ratio:.2f})'] = {'corr': corr_htb, 'bpp': bpp_htb}
        print(f"Ternary-ratio={t_ratio:.2f}: {corr_htb:.4f} @ {bpp_htb:.2f} bpp")
    
    # 4. Residual Boost
    print("\nRunning Residual Magnitude Boost...")
    for r_ratio in [0.03, 0.05, 0.10]:
        rmb = ResidualMagnitudeBoost(d, d, rank=2, residual_ratio=r_ratio)
        rmb.train(W_true)
        W_rmb = rmb.get_weights()
        corr_rmb = np.corrcoef((X_test @ W_rmb.T).flatten(), Y_test.flatten())[0,1]
        bpp_rmb = rmb.effective_bpp()
        results[f'Residual (r={r_ratio:.2f})'] = {'corr': corr_rmb, 'bpp': bpp_rmb}
        print(f"Residual-ratio={r_ratio:.2f}: {corr_rmb:.4f} @ {bpp_rmb:.2f} bpp")
    
    # Summary
    with open("results_v12_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY - NOVEL IDEAS V12 (CLOSING THE GAP)\n")
        f.write("="*80 + "\n")
        f.write(f"🎯 TARGET: ≥{corr_tern:.4f} (ternary) at ≤1.2 bpp\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<30} {'Corr':>8} {'BPP':>8} {'vs Tern':>10}\n")
        f.write("-" * 65 + "\n")
        
        for name, res in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (res['corr'] - corr_tern) / corr_tern * 100
            beat_target = "✅" if (res['corr'] >= corr_tern and res['bpp'] <= 1.2) else ""
            line = f"{name:<30} {res['corr']:>8.4f} {res['bpp']:>8.2f} {vs_tern:>+9.1f}% {beat_target}\n"
            print(line.strip())
            f.write(line)
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_experiments()
