import torch
import torch.nn.functional as F
import numpy as np

def decode_vq_fast(indices: torch.Tensor, codebook: torch.Tensor, signs: torch.Tensor, d_out: int, d_in: int, block_size: int = 4) -> torch.Tensor:
    """
    Fast CPU decoding using PyTorch F.embedding.
    
    Args:
        indices: (n_blocks,) LongTensor
        codebook: (n_codes, block_size*block_size) FloatTensor
        signs: (d_out, d_in) FloatTensor (or Int8)
        d_out: Output dimension
        d_in: Input dimension
        block_size: Size of VQ blocks (default 4)
        
    Returns:
        W_recon: (d_out, d_in) FloatTensor
    """
    # 1. Lookup
    # indices: (N_blocks,) -> blocks: (N_blocks, 16)
    # F.embedding is highly optimized (C++ kernel)
    blocks = F.embedding(indices, codebook)
    
    # 2. Reshape
    # (N_blocks, 16) -> (N_h, N_w, 4, 4) -> (N_h, 4, N_w, 4) -> (Out, In)
    n_h = d_out // block_size
    n_w = d_in // block_size
    
    # Note: This reshape order MUST match the quantization order!
    # Quantization: M_pad.reshape(n_h, bs, n_w, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    # Decoding: blocks.view(n_h, n_w, bs, bs).permute(0, 2, 1, 3).reshape(d_out, d_in)
    
    W_recon = blocks.view(n_h, n_w, block_size, block_size).permute(0, 2, 1, 3).reshape(d_out, d_in)
    
    # 3. Apply Signs
    W_recon = W_recon * signs
    
    return W_recon
