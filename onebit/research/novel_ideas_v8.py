"""
Novel Ideas V8: Strict 1.00 bpp via Density-Based Quantization (Dithering)

Experiments:
1. Floyd-Steinberg Dithering:
   - Error diffusion to neighbors.
   - Hypothesis: Preserves low-frequency magnitude information in the density of signs.

2. Ordered Dithering:
   - Thresholding with a Bayer matrix.
   - Hypothesis: Structured noise helps represent continuous values.

3. Density Optimized Binary:
   - Optimize binary pattern to match local block averages.
   - Hypothesis: Explicitly matching local density captures magnitude.
"""

import numpy as np
from typing import Tuple, Dict, List

# =============================================================================
# 1. FLOYD-STEINBERG DITHERING
# =============================================================================

class FloydSteinbergBinary:
    """
    Error diffusion dithering.
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.W_bin = None
        self.scale = 1.0
        
    def train(self, W_target: np.ndarray):
        # We process the matrix element by element
        # Copy W to avoid modifying original
        W = W_target.copy()
        H, W_dim = W.shape
        
        self.scale = np.mean(np.abs(W_target))
        
        # Output binary matrix
        self.W_bin = np.zeros_like(W)
        
        for r in range(H):
            for c in range(W_dim):
                old_pixel = W[r, c]
                # Quantize to closest level: +scale or -scale
                if old_pixel > 0:
                    new_pixel = self.scale
                    self.W_bin[r, c] = 1.0
                else:
                    new_pixel = -self.scale
                    self.W_bin[r, c] = -1.0
                
                quant_error = old_pixel - new_pixel
                
                # Distribute error to neighbors
                if c + 1 < W_dim:
                    W[r, c + 1] += quant_error * 7 / 16
                if r + 1 < H:
                    if c - 1 >= 0:
                        W[r + 1, c - 1] += quant_error * 3 / 16
                    W[r + 1, c] += quant_error * 5 / 16
                    if c + 1 < W_dim:
                        W[r + 1, c + 1] += quant_error * 1 / 16
                        
    def get_weights(self) -> np.ndarray:
        return self.W_bin * self.scale
        
    def effective_bpp(self) -> float:
        return 1.0


# =============================================================================
# 2. ORDERED DITHERING (Bayer Matrix)
# =============================================================================

class OrderedDitherBinary:
    """
    Thresholding with a Bayer matrix.
    """
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.W_bin = None
        self.scale = 1.0
        
    def _make_bayer_matrix(self, n):
        if n == 2:
            return np.array([[0, 2], [3, 1]]) / 4.0
        else:
            B_prev = self._make_bayer_matrix(n // 2)
            top = np.hstack([4 * B_prev, 4 * B_prev + 2])
            bottom = np.hstack([4 * B_prev + 3, 4 * B_prev + 1])
            return np.vstack([top, bottom]) / 4.0
            
    def train(self, W_target: np.ndarray):
        self.scale = np.mean(np.abs(W_target))
        
        # Create Bayer matrix of size (d_out, d_in)
        # We tile a smaller Bayer matrix
        bayer_size = 4
        B = self._make_bayer_matrix(bayer_size)
        
        # Tile it
        rows = int(np.ceil(self.d_out / bayer_size))
        cols = int(np.ceil(self.d_in / bayer_size))
        B_tiled = np.tile(B, (rows, cols))[:self.d_out, :self.d_in]
        
        # Center Bayer matrix around 0: [-0.5, 0.5]
        B_tiled = B_tiled - 0.5
        
        # Scale Bayer matrix by the local range of weights?
        # Or just use it as a threshold offset.
        # W > Threshold -> +1
        # Threshold = B_tiled * 2 * scale (range of weights)
        
        # Normalize W to [-1, 1] roughly
        W_norm = W_target / (np.max(np.abs(W_target)) + 1e-8)
        
        # Dither
        # If W_norm > B_tiled, then +1, else -1
        # But we want to encode magnitude.
        # A value of 0.5 should be +1 75% of time.
        # A value of 0.0 should be +1 50% of time.
        
        # Standard ordered dithering:
        # Output = 1 if Input > Threshold else 0
        # Here we map to {-1, 1}
        
        # Let's treat W as probability of being +1
        # P(+1) = (W / max_val + 1) / 2  (maps -max..max to 0..1)
        
        W_prob = (W_norm + 1) / 2
        
        # Threshold with Bayer (0..1)
        # Note: B_tiled above was centered, let's use uncentered 0..1
        B_01 = B_tiled + 0.5 
        
        self.W_bin = np.where(W_prob > B_01, 1.0, -1.0).astype(np.float32)
        
    def get_weights(self) -> np.ndarray:
        return self.W_bin * self.scale
        
    def effective_bpp(self) -> float:
        return 1.0


# =============================================================================
# 3. DENSITY OPTIMIZED BINARY
# =============================================================================

class DensityOptimizedBinary:
    """
    Optimize binary pattern to match local block averages.
    """
    def __init__(self, d_in: int, d_out: int, block_size: int = 4):
        self.d_in = d_in
        self.d_out = d_out
        self.block_size = block_size
        self.W_bin = None
        self.scale = 1.0
        
    def train(self, W_target: np.ndarray):
        self.scale = np.mean(np.abs(W_target))
        self.W_bin = np.zeros_like(W_target)
        
        H, W = W_target.shape
        bs = self.block_size
        
        for r in range(0, H, bs):
            for c in range(0, W, bs):
                # Extract block
                r_end = min(r + bs, H)
                c_end = min(c + bs, W)
                block = W_target[r:r_end, c:c_end]
                
                # Target sum for this block
                target_sum = np.sum(block)
                
                # We want sum(bin_block * scale) approx target_sum
                # sum(bin_block) approx target_sum / scale
                target_k = target_sum / self.scale
                
                # Number of elements
                n_elem = block.size
                
                # We need k ones and (n - k) minus ones
                # Sum = k - (n - k) = 2k - n
                # 2k - n = target_k  =>  2k = target_k + n  => k = (target_k + n) / 2
                
                k = int(round((target_k + n_elem) / 2))
                k = max(0, min(n_elem, k)) # Clip to 0..n
                
                # Create block with k ones
                # Which positions? The ones with highest values in original block
                flat_block = block.flatten()
                # Sort indices descending
                idx = np.argsort(flat_block)[::-1]
                
                bin_flat = -np.ones(n_elem, dtype=np.float32)
                bin_flat[idx[:k]] = 1.0
                
                self.W_bin[r:r_end, c:c_end] = bin_flat.reshape(block.shape)
                
    def get_weights(self) -> np.ndarray:
        return self.W_bin * self.scale
        
    def effective_bpp(self) -> float:
        return 1.0


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("="*80)
    print("NOVEL IDEAS V8: Strict 1.00 bpp (Dithering)")
    print("="*80)
    
    # Setup Data
    d = 256
    np.random.seed(42)
    U = np.random.randn(d, d)
    U, _ = np.linalg.qr(U)
    Vt = np.random.randn(d, d)
    Vt, _ = np.linalg.qr(Vt)
    S = np.exp(-np.linspace(0, 5, d))
    W_true = U @ np.diag(S) @ Vt
    W_true = W_true.astype(np.float32)
    
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
    print("-" * 40)
    
    # 1. Floyd-Steinberg
    print("\nRunning Floyd-Steinberg Dithering...")
    fsb = FloydSteinbergBinary(d, d)
    fsb.train(W_true)
    W_fsb = fsb.get_weights()
    corr_fsb = np.corrcoef((X_test @ W_fsb.T).flatten(), Y_test.flatten())[0,1]
    bpp_fsb = fsb.effective_bpp()
    results['Floyd-Steinberg'] = {'corr': corr_fsb, 'bpp': bpp_fsb}
    print(f"Result: {corr_fsb:.4f} @ {bpp_fsb:.2f} bpp")
    
    # 2. Ordered Dithering
    print("\nRunning Ordered Dithering...")
    odb = OrderedDitherBinary(d, d)
    odb.train(W_true)
    W_odb = odb.get_weights()
    corr_odb = np.corrcoef((X_test @ W_odb.T).flatten(), Y_test.flatten())[0,1]
    bpp_odb = odb.effective_bpp()
    results['Ordered Dither'] = {'corr': corr_odb, 'bpp': bpp_odb}
    print(f"Result: {corr_odb:.4f} @ {bpp_odb:.2f} bpp")
    
    # 3. Density Optimized
    print("\nRunning Density Optimized Binary...")
    for bs in [2, 4, 8]:
        dob = DensityOptimizedBinary(d, d, block_size=bs)
        dob.train(W_true)
        W_dob = dob.get_weights()
        corr_dob = np.corrcoef((X_test @ W_dob.T).flatten(), Y_test.flatten())[0,1]
        bpp_dob = dob.effective_bpp()
        results[f'Density-Opt (BS={bs})'] = {'corr': corr_dob, 'bpp': bpp_dob}
        print(f"BS={bs}: {corr_dob:.4f} @ {bpp_dob:.2f} bpp")
    
    # Summary
    with open("results_v8_utf8.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY - NOVEL IDEAS V8\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<25} {'Corr':>8} {'BPP':>8} {'vs Tern':>10}\n")
        f.write("-" * 60 + "\n")
        
        for name, res in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs_tern = (res['corr'] - corr_tern) / corr_tern * 100
            line = f"{name:<25} {res['corr']:>8.4f} {res['bpp']:>8.2f} {vs_tern:>+9.1f}%\n"
            print(line.strip())
            f.write(line)

if __name__ == "__main__":
    run_experiments()
