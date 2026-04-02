from __future__ import annotations

import numpy as np
from typing import Union

from onebit.core.packbits import pack_signs_rowmajor
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig


class OneBitLogitsHead:
    """Standard 1-bit Linear Head (No HCL/FWHT)."""

    def __init__(
        self,
        W_bits: np.ndarray,  # [vocab, Kw]
        means: np.ndarray,   # [vocab]
        scales: np.ndarray,  # [vocab]
        vocab_size: int,
        d_model: int,
    ):
        self.W_bits = W_bits
        self.means = means
        self.scales = scales
        self.vocab_size = vocab_size
        self.d_model = d_model

    @classmethod
    def from_wte(cls, wte_fp32: np.ndarray) -> OneBitLogitsHead:
        """Create 1-bit head from FP32 embeddings.
        
        Args:
            wte_fp32: [vocab_size, d_model]
        """
        vocab_size, d_model = wte_fp32.shape
        
        # Quantize
        mean_w = np.mean(wte_fp32, axis=1)
        W_centered = wte_fp32 - mean_w[:, None]
        scale_w = np.mean(np.abs(W_centered), axis=1)
        scale_w[scale_w < 1e-9] = 1e-9
        
        W_bits = pack_signs_rowmajor(W_centered)
        
        return cls(
            W_bits=W_bits,
            means=mean_w,
            scales=scale_w,
            vocab_size=vocab_size,
            d_model=d_model,
        )

    def forward(
        self,
        x_stream: np.ndarray,  # [T, Kw] or [Seq, T, Kw]
        x_sum: Union[float, np.ndarray], # scalar or [Seq]
        x_scale: Union[float, np.ndarray], # scalar or [Seq]
        sd_cfg: SDConfig,
        seed: int,
        T: int,
    ) -> np.ndarray:
        """Compute logits using 1-bit stream input."""
        Kw = self.W_bits.shape[1]
        Kbits = Kw * 32
        
        # Detect batch dimension
        is_batched = (x_stream.ndim == 3)
        
        if is_batched:
            x_scale_bc = np.asarray(x_scale)[:, None]
            w_scale_bc = self.scales[None, :]
            effective_scale = w_scale_bc * x_scale_bc * Kbits
        else:
            effective_scale = self.scales * x_scale * Kbits
        
        logits_raw = bsdm_w_matmul(
            self.W_bits,
            x_stream,
            k=T,
            cfg=sd_cfg,
            seed=seed,
            scale=effective_scale,
        )
        
        # Mean correction
        if is_batched:
            logits = logits_raw + self.means[None, :] * np.asarray(x_sum)[:, None]
        else:
            logits = logits_raw + self.means * x_sum
            
        return logits

    def forward_fp32(self, x: np.ndarray) -> np.ndarray:
        """Compute logits using high-precision input (W1A32).
        
        Unpacks 1-bit weights to signed float and performs standard matmul.
        Equivalent to XNOR-Net / BitNet forward pass for this layer.
        
        Args:
            x: [d_model] or [Seq, d_model]
            
        Returns:
            logits: [vocab] or [Seq, vocab]
        """
        # 1. Unpack W_bits to W_sign [vocab, d_model]
        # We unpack on the fly to avoid storing dense matrix
        # But to make it fast, we might need optimized kernel.
        # In numpy, we iterate words.
        
        # Optimization: Unpacking 50k rows in python is slow.
        # We can use a small cache or just pay the cost?
        # Or implement vectorized unpacking?
        # We have check_packer logic.
        
        # Let's do a vectorized unpack
        # W_bits: [vocab, Kw]
        vocab, Kw = self.W_bits.shape
        Kbits = Kw * 32
        
        # Prepare bit masks
        # [1, 32]
        masks = (1 << np.arange(32, dtype=np.uint32)).reshape(1, 32)
        
        # Expand bits: [vocab, Kw, 1] & [1, 1, 32] -> [vocab, Kw, 32]
        # Note: bit order is LSB first.
        # Result is boolean
        w_bool = (self.W_bits[:, :, None] & masks) > 0
        
        # Convert to float {-1, 1}
        w_sign = np.where(w_bool, 1.0, -1.0).astype(np.float32)
        
        # Reshape to [vocab, Kbits]
        w_sign = w_sign.reshape(vocab, Kbits)
        
        # Truncate to d_model (if padding exists)
        if Kbits > self.d_model:
            w_sign = w_sign[:, :self.d_model]
            
        # 2. Matmul: logits = x @ W_sign.T
        # x: [Seq, d_model]
        # W_sign.T: [d_model, vocab]
        logits_raw = x @ w_sign.T # [Seq, vocab]
        
        # 3. Scale: logits = logits_raw * scale + mean * sum(x)
        # scale: [vocab]
        # mean: [vocab]
        
        # Apply scale
        logits = logits_raw * self.scales
        
        # Apply mean correction
        # sum(x): [Seq]
        x_sum = np.sum(x, axis=-1, keepdims=True) # [Seq, 1]
        logits += x_sum * self.means[None, :]
        
        return logits
