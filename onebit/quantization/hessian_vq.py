import torch
import numpy as np
from typing import Tuple, Optional

class HessianVQ:
    def __init__(self, n_codes: int = 32, block_size: int = 4):
        self.n_codes = n_codes
        self.block_size = block_size
        self.indices = None
        self.codebook = None
        self.signs = None
        
    def quantize(self, W: np.ndarray, H_diag: np.ndarray) -> np.ndarray:
        """
        Quantize a weight matrix using Hessian-weighted VQ.
        
        Args:
            W: (Out, In) numpy array
            H_diag: (In,) numpy array (Hessian diagonal)
            
        Returns:
            W_recon: (Out, In) numpy array
        """
        d_out, d_in = W.shape
        bs = self.block_size
        
        # 1. Signs
        S = np.sign(W)
        S[S == 0] = 1.0
        M = np.abs(W)
        
        # 2. Prepare Blocks
        # Pad if needed
        pad_h = (bs - d_out % bs) % bs
        pad_w = (bs - d_in % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)))
        
        # H_diag corresponds to columns (In). Tile it across rows (Out).
        # H_mat: (Out, In)
        H_mat = np.tile(H_diag, (d_out, 1))
        H_pad = np.pad(H_mat, ((0, pad_h), (0, pad_w)), constant_values=1e-6)
        
        # Reshape to blocks
        h_p, w_p = M_pad.shape
        n_blocks_h = h_p // bs
        n_blocks_w = w_p // bs
        
        # (N_h, bs, N_w, bs) -> (N_h, N_w, bs, bs) -> (N_blocks, bs*bs)
        # This order groups (row_block, col_block) together
        blocks = M_pad.reshape(n_blocks_h, bs, n_blocks_w, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        weights = H_pad.reshape(n_blocks_h, bs, n_blocks_w, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
        
        # 3. Weighted K-Means
        # Init centroids
        if len(blocks) < self.n_codes:
             indices = np.arange(len(blocks))
             centroids = blocks.copy()
        else:
            indices = np.random.choice(len(blocks), self.n_codes, replace=False)
            centroids = blocks[indices].copy()
        
        # Iterations (K-Means) - using unweighted for now
        for _ in range(5):
            # Distance
            dists = np.linalg.norm(blocks[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            
            # Update (TEST: completely unweighted)
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_codes):
                mask = (assignments == k)
                if np.sum(mask) > 0:
                    # Simple unweighted average
                    new_centroids[k] = np.mean(blocks[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]
            centroids = new_centroids
            
        # 4. Reconstruct
        recon_blocks = centroids[assignments]
        M_recon_pad = recon_blocks.reshape(n_blocks_h, n_blocks_w, bs, bs).transpose(0, 2, 1, 3).reshape(h_p, w_p)
        M_recon = M_recon_pad[:d_out, :d_in]
        
        W_recon = S * M_recon
        
        # 5. Bias Correction ONLY (per user suggestion - variance correction too aggressive)
        # VQ shifts the mean, which accumulates across layers
        # Preserve the original mean to prevent bias accumulation
        mean_original = W.mean()
        mean_recon = W_recon.mean()
        W_recon = W_recon - mean_recon + mean_original
        
        # Store components for later use
        self.indices = assignments
        self.codebook = centroids
        self.signs = S
        
        return W_recon
