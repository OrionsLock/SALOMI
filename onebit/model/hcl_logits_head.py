from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

from onebit.core.hadamard import fwht, inverse_fwht
from onebit.core.packbits import pack_signs_rowmajor
from onebit.ops.bsdm_w import bsdm_w_matmul, SDConfig


class HCLLogitsHead:
    """HCL Logits Head for 1-bit inference."""

    def __init__(
        self,
        W_code_bits: np.ndarray,
        W_code_means: np.ndarray,
        W_code_scales: np.ndarray,
        vocab_size: int,
        d_model: int,
        d_code: int,
    ):
        """Initialize HCL head with pre-computed weights.
        
        Args:
            W_code_bits: Packed 1-bit weights [d_code, Kw]
            W_code_means: Per-row means for DC correction [d_code]
            W_code_scales: Per-row scales [d_code]
            vocab_size: Vocabulary size
            d_model: Model dimension
            d_code: Code dimension (power of 2)
        """
        self.W_code_bits = W_code_bits
        self.W_code_means = W_code_means
        self.W_code_scales = W_code_scales
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_code = d_code

    @classmethod
    def from_wte(cls, wte_fp32: np.ndarray) -> HCLLogitsHead:
        """Create HCL head by projecting existing embeddings.
        
        This analytically solves for W_code such that H @ W_code ~= wte.
        W_code = H^(-1) @ wte = (1/N) * H^T @ wte.
        Since we use FWHT (Sylvester), H is symmetric, so H^T = H.
        
        Args:
            wte_fp32: FP32 token embeddings [vocab_size, d_model]
            
        Returns:
            Initialized HCLLogitsHead
        """
        vocab_size, d_model = wte_fp32.shape
        
        # 1. Determine d_code (next power of 2)
        d_code = 1
        while d_code < vocab_size:
            d_code *= 2
            
        print(f"HCL: vocab_size={vocab_size}, d_code={d_code} (padding {d_code - vocab_size})")
        
        # 2. Pad wte to d_code rows
        wte_padded = np.zeros((d_code, d_model), dtype=np.float32)
        wte_padded[:vocab_size, :] = wte_fp32
        
        # 3. Compute W_code_fp32 = IFWHT(wte_padded)
        # We apply IFWHT to the COLUMNS of wte (i.e., along the vocab dimension).
        # fwht operates on the last axis, so we transpose first.
        # wte_padded.T shape: [d_model, d_code]
        W_code_T = inverse_fwht(wte_padded.T)
        W_code_fp32 = W_code_T.T  # [d_code, d_model]
        
        # 4. Quantize W_code to 1-bit + mean correction
        # Center per row (code dimension)
        mean_w = np.mean(W_code_fp32, axis=1)
        W_centered = W_code_fp32 - mean_w[:, None]
        scale_w = np.mean(np.abs(W_centered), axis=1)
        scale_w[scale_w < 1e-9] = 1e-9
        
        W_code_bits = pack_signs_rowmajor(W_centered)
        
        return cls(
            W_code_bits=W_code_bits,
            W_code_means=mean_w,
            W_code_scales=scale_w,
            vocab_size=vocab_size,
            d_model=d_model,
            d_code=d_code,
        )

    def forward(
        self,
        x_stream: np.ndarray,  # [T, Kw] or [Seq, T, Kw]
        x_sum: Union[float, np.ndarray], # scalar or [Seq]
        x_scale: Union[float, np.ndarray], # scalar or [Seq]
        sd_cfg: SDConfig,
        seed: int,
        T: int,
        logit_scale: float = 0.1, # New scale factor
    ) -> np.ndarray:
        """Compute logits.
        
        Args:
            x_stream: Magnitude-aware input stream
            x_sum: Sum of input vector (for mean correction)
            x_scale: Input scale factor (max_abs)
            sd_cfg: Sigma-Delta config
            seed: Random seed
            T: Number of ticks
            logit_scale: Scaling factor for final logits (calibration).
            
        Returns:
            logits [vocab_size] or [Seq, vocab_size]
        """
        # 1. Project to code space: c = W_code @ x
        # W_code_bits is [d_code, Kw]
        
        Kw = self.W_code_bits.shape[1]
        Kbits = Kw * 32
        
        # Total scale = weight_scale * input_scale * Kbits
        # self.W_code_scales is [d_code]
        
        # Detect batch dimension
        is_batched = (x_stream.ndim == 3)
        
        if is_batched:
            # Batch mode:
            # x_scale is [Seq]
            # W_code_scales is [d_code]
            # effective_scale should be [Seq, d_code]
            # Broadcasting: [Seq, 1] * [1, d_code]
            x_scale_bc = np.asarray(x_scale)[:, None]
            w_scale_bc = self.W_code_scales[None, :]
            effective_scale = w_scale_bc * x_scale_bc * Kbits
        else:
            # Scalar mode
            effective_scale = self.W_code_scales * x_scale * Kbits
        
        c_raw = bsdm_w_matmul(
            self.W_code_bits,
            x_stream,
            k=T,
            cfg=sd_cfg,
            seed=seed,
            scale=effective_scale,
        )
        
        # Apply mean correction: c += mean_w * sum(x)
        if is_batched:
            # c_raw is [Seq, d_code]
            # x_sum is [Seq]
            # W_code_means is [d_code]
            c = c_raw + self.W_code_means[None, :] * np.asarray(x_sum)[:, None]
        else:
            c = c_raw + self.W_code_means * x_sum
        
        # 2. Decode via FWHT: logits = H @ c
        # fwht handles batch dimension automatically (operates on last axis)
        logits_full = fwht(c)
        
        # 3. Slice to vocab size
        if is_batched:
            logits = logits_full[:, :self.vocab_size]
        else:
            logits = logits_full[:self.vocab_size]
            
        # 4. Apply Calibration Scale
        return logits * logit_scale
