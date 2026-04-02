"""Runtime transformer for 1-bit GPT-2 inference.

This module implements the GPT-2 transformer architecture using 1-bit operators:
- BSDM-W for matrix multiplications
- HCL or 1-bit Linear for logits computation
- LDP-KV for KV cache (optional)

Key features:
- Autoregressive generation
- KV caching
- Configurable compute budgets (T values)
- Support for CPU and OpenCL backends
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from onebit.model.quantize_gpt2 import QuantizedGPT2, GPT2Config
from onebit.core.packbits import pack_input_signs, pack_float_to_stream
from onebit.ops.bsdm_w import bsdm_w_dot, bsdm_w_matmul, SDConfig
from onebit.ops.hcl import hcl_energy_cpu
from onebit.model.hcl_logits_head import HCLLogitsHead
from onebit.model.onebit_logits_head import OneBitLogitsHead


@dataclass
class InferenceConfig:
    """Configuration for 1-bit inference.

    Attributes:
        T: Number of BSDM-W ticks (compute budget)
        backend: "cpu" or "opencl"
        seed: Master seed for PRF
        eps: SPRT effect size
        delta: SPRT risk budget
        order: ΣΔ order (1 or 2)
        beta: ΣΔ-2 beta parameter
        lambd: ΣΔ leak parameter
        walsh_N: Walsh carriers per tick
        antithetic: Use antithetic pairs
        use_ctg: Enable CTG
        use_hcl_logits: (Deprecated) Use HCL head
        head_type: "fp32", "1bit", or "hcl"
        logits_input_bits: int = 32   # Precision of input to logits head
    """
    T: int = 64
    backend: str = "cpu"
    seed: int = 42
    eps: float = 0.05
    delta: float = 0.01
    order: int = 2
    beta: float = 0.30
    lambd: float = 1.0 / 256.0
    walsh_N: int = 2
    antithetic: bool = True
    use_ctg: bool = False
    use_hcl_logits: bool = False  # Backward compat
    head_type: str = "fp32"       # Default to FP32 head for stability
    logits_input_bits: int = 32   # Precision of input to logits head


class RuntimeTransformer:
    """1-bit GPT-2 transformer runtime."""
    
    def __init__(
        self,
        model: QuantizedGPT2,
        infer_cfg: InferenceConfig,
    ):
        """Initialize runtime transformer."""
        self.model = model
        self.cfg = model.config
        self.infer_cfg = infer_cfg
        
        # BSDM-W config
        self.sd_cfg = SDConfig(
            order=infer_cfg.order,
            beta=infer_cfg.beta,
            lambd=infer_cfg.lambd,
            walsh_N=infer_cfg.walsh_N,
            antithetic=infer_cfg.antithetic,
        )
        
        # KV cache (optional)
        self.kv_cache: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        
        # Initialize Logits Head
        self.logits_head = None
        
        # Resolve head type
        head_type = infer_cfg.head_type
        if infer_cfg.use_hcl_logits:
            head_type = "hcl"
            
        if head_type == "hcl":
            if "wte" in self.model.weights_fp32:
                print("Initializing HCL Logits Head...")
                self.logits_head = HCLLogitsHead.from_wte(self.model.weights_fp32["wte"])
            else:
                print("WARNING: wte weights not found, cannot initialize HCL head.")
        elif head_type == "1bit":
            if "wte" in self.model.weights_fp32:
                print("Initializing 1-bit Linear Logits Head...")
                self.logits_head = OneBitLogitsHead.from_wte(self.model.weights_fp32["wte"])
            else:
                print("WARNING: wte weights not found, cannot initialize 1-bit head.")
        elif head_type == "fp32":
            pass # Use direct wte
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
    
    def reset_kv_cache(self):
        """Reset KV cache."""
        self.kv_cache = None
    
    def _matmul_1bit(
        self,
        x_fp32: np.ndarray,  # [d_in] or [Seq, d_in]
        W_bits: np.ndarray,  # [d_out, d_in_words] uint32 (row-major packing)
        W_shape: Tuple[int, int],  # (d_out, d_in) - original shape before packing
        seed: int,
        scale: np.ndarray | float = 1.0,
        mean_w: Optional[np.ndarray] = None,  # Per-row mean for DC correction
    ) -> np.ndarray:
        """1-bit matrix multiplication: y = W @ x"""
        d_out, d_in = W_shape
        
        # Normalize input shape
        is_1d = (x_fp32.ndim == 1)
        if is_1d:
            if len(x_fp32) != d_in:
                raise ValueError(f"Input dimension mismatch: x has {len(x_fp32)}, W has {d_in} columns")
            x_batch = x_fp32[None, :] # [1, d_in]
        else:
            if x_fp32.shape[1] != d_in:
                raise ValueError(f"Input dimension mismatch: x has {x_fp32.shape[1]}, W has {d_in} columns")
            x_batch = x_fp32 # [Seq, d_in]

        # Calculate input scale (max absolute value) PER TOKEN
        input_scale = np.max(np.abs(x_batch), axis=1, keepdims=True)
        input_scale = np.maximum(input_scale, 1e-9) # Avoid div/0
            
        # Normalize input to [-1, 1]
        x_norm = x_batch / input_scale

        # Pad normalized input if d_in is not multiple of 32
        Kw = W_bits.shape[1]
        padded_d_in = Kw * 32
        if d_in < padded_d_in:
            x_padded = np.zeros((x_norm.shape[0], padded_d_in), dtype=x_norm.dtype)
            x_padded[:, :d_in] = x_norm
            x_norm = x_padded
            
        x_stream = pack_float_to_stream(x_norm, k=self.infer_cfg.T) # [Seq, T, Kw]

        Kbits = W_bits.shape[1] * 32
        
        # CORRECTION: Global damping to prevent magnitude explosion (observed 1.7x growth per layer)
        damping_factor = 0.6
        effective_scale = (scale * Kbits * damping_factor) * input_scale
        
        y = bsdm_w_matmul(
            W_bits,      # [d_out, Kw]
            x_stream,    # [Seq, T, Kw]
            k=self.infer_cfg.T,
            cfg=self.sd_cfg,
            seed=seed,
            scale=effective_scale,
        )
        
        # Apply DC correction: y += mean_w * sum(x)
        if mean_w is not None:
            x_sum = np.sum(x_batch, axis=1, keepdims=True) # [Seq, 1]
            correction = mean_w * x_sum
            y += correction

        if is_1d:
            return y[0]
        else:
            return y
    
    def _layer_norm(
        self,
        x: np.ndarray,  # [d] or [Seq, d]
        g: np.ndarray,  # [d] gain
        b: np.ndarray,  # [d] bias
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Layer normalization (Vectorized)."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return g * x_norm + b
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation (approximate)."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    
    def _attention_layer(
        self,
        x: np.ndarray,  # [seq_len, d_model]
        layer_idx: int,
        seed: int,
    ) -> np.ndarray:
        """Full attention layer with multi-head self-attention."""
        seq_len, d_model = x.shape
        n_heads = self.cfg.n_heads
        d_head = d_model // n_heads
        prefix = f"h.{layer_idx}.attn"

        W_qkv_bits = self.model.weights_1bit[f"{prefix}.c_attn.w"]
        b_qkv = self.model.weights_fp32[f"{prefix}.c_attn.b"]
        W_proj_bits = self.model.weights_1bit[f"{prefix}.c_proj.w"]
        b_proj = self.model.weights_fp32[f"{prefix}.c_proj.b"]
        
        scale_qkv = self.model.scales.get(f"{prefix}.c_attn.w", 1.0)
        mean_qkv = self.model.means.get(f"{prefix}.c_attn.w", None)
        
        scale_proj = self.model.scales.get(f"{prefix}.c_proj.w", 1.0)
        mean_proj = self.model.means.get(f"{prefix}.c_proj.w", None)

        qkv = self._matmul_1bit(
            x,
            W_qkv_bits,
            W_shape=(3 * d_model, d_model),
            seed=seed,
            scale=scale_qkv,
            mean_w=mean_qkv,
        ) + b_qkv

        q = qkv[:, :d_model]
        k = qkv[:, d_model:2*d_model]
        v = qkv[:, 2*d_model:]

        q_heads = q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        k_heads = k.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        v_heads = v.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

        scores = np.matmul(q_heads, k_heads.transpose(0, 2, 1))
        scores = scores / np.sqrt(d_head)
        
        mask = np.tril(np.ones((seq_len, seq_len)))
        scores = np.where(mask == 1, scores, -1e9)
        
        max_score = np.max(scores, axis=-1, keepdims=True)
        exp_score = np.exp(scores - max_score)
        attn_weights = exp_score / np.sum(exp_score, axis=-1, keepdims=True)
        
        attn_out = np.matmul(attn_weights, v_heads)
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d_model)

        out = self._matmul_1bit(
            attn_out,
            W_proj_bits,
            W_shape=(d_model, d_model),
            seed=seed + 1,
            scale=scale_proj,
            mean_w=mean_proj,
        ) + b_proj

        return x + out
    
    def _ffn_layer(
        self,
        x: np.ndarray,  # [seq_len, d_model]
        layer_idx: int,
        seed: int,
    ) -> np.ndarray:
        """FFN layer."""
        seq_len, d_model = x.shape
        prefix = f"h.{layer_idx}.mlp"
        
        W_fc_bits = self.model.weights_1bit[f"{prefix}.c_fc.w"]
        b_fc = self.model.weights_fp32[f"{prefix}.c_fc.b"]
        W_proj_bits = self.model.weights_1bit[f"{prefix}.c_proj.w"]
        b_proj = self.model.weights_fp32[f"{prefix}.c_proj.b"]
        
        scale_fc = self.model.scales.get(f"{prefix}.c_fc.w", 1.0)
        mean_fc = self.model.means.get(f"{prefix}.c_fc.w", None)
        scale_proj = self.model.scales.get(f"{prefix}.c_proj.w", 1.0)
        mean_proj = self.model.means.get(f"{prefix}.c_proj.w", None)
        
        h = self._matmul_1bit(
            x,
            W_fc_bits,
            W_shape=(self.cfg.d_ff, d_model),
            seed=seed,
            scale=scale_fc,
            mean_w=mean_fc,
        ) + b_fc

        h = self._gelu(h)

        out = self._matmul_1bit(
            h,
            W_proj_bits,
            W_shape=(d_model, self.cfg.d_ff),
            seed=seed + 1,
            scale=scale_proj,
            mean_w=mean_proj,
        ) + b_proj
        
        return x + out

    def _compute_logits(
        self,
        x: np.ndarray,  # [d_model] or [Seq, d_model]
        seed: int,
    ) -> np.ndarray:
        """Compute logits using selected head."""
        # If head is None, we must use FP32 wte
        if self.logits_head is None:
            wte = self.model.weights_fp32["wte"]
            return x @ wte.T
            
        d_model = x.shape[-1]
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x[None, :] # [1, d_model]
            
        if self.infer_cfg.logits_input_bits > 1 and isinstance(self.logits_head, OneBitLogitsHead):
            # High precision input path (W1A32)
            logits = self.logits_head.forward_fp32(x)
            if is_1d:
                return logits[0]
            else:
                return logits
            
        # Normalize and pack input (PER TOKEN)
        # Both HCL and 1-bit linear heads expect packed input
        
        # Check head type to determine padding/width
        if isinstance(self.logits_head, HCLLogitsHead):
            Kw = self.logits_head.W_code_bits.shape[1]
        elif isinstance(self.logits_head, OneBitLogitsHead):
            Kw = self.logits_head.W_bits.shape[1]
        else:
            raise RuntimeError(f"Unknown head type {type(self.logits_head)}")
            
        padded_d_in = Kw * 32
        
        input_scale = np.max(np.abs(x), axis=1, keepdims=True)
        input_scale = np.maximum(input_scale, 1e-9)
        x_norm = x / input_scale
        
        if d_model < padded_d_in:
            x_padded = np.zeros((x.shape[0], padded_d_in), dtype=x_norm.dtype)
            x_padded[:, :d_model] = x_norm
            x_norm = x_padded
            
        x_stream = pack_float_to_stream(x_norm, k=self.infer_cfg.T)
        
        logits = self.logits_head.forward(
            x_stream=x_stream,
            x_sum=np.sum(x, axis=1), # [Seq]
            x_scale=input_scale.flatten(), # [Seq]
            sd_cfg=self.sd_cfg,
            seed=seed,
            T=self.infer_cfg.T
        )
        
        if is_1d:
            return logits[0]
        else:
            return logits

    def forward(
        self,
        input_ids: np.ndarray,  # [seq_len]
        seed: Optional[int] = None,
        return_all_logits: bool = False,
    ) -> np.ndarray:
        """Forward pass."""
        if seed is None:
            seed = self.infer_cfg.seed

        seq_len = len(input_ids)

        # Token + position embeddings
        wte = self.model.weights_fp32["wte"]
        wpe = self.model.weights_fp32["wpe"]

        x = wte[input_ids] + wpe[:seq_len]  # [seq_len, d_model]

        # Transformer layers
        for layer_idx in range(self.cfg.n_layers):
            prefix = f"h.{layer_idx}"

            # Layer norm 1 (Vectorized)
            ln1_g = self.model.weights_fp32[f"{prefix}.ln_1.g"]
            ln1_b = self.model.weights_fp32[f"{prefix}.ln_1.b"]
            x_norm = self._layer_norm(x, ln1_g, ln1_b)

            # Attention (Full Sequence)
            x = self._attention_layer(x_norm, layer_idx, seed + layer_idx * 1000)

            # Layer norm 2 (Vectorized)
            ln2_g = self.model.weights_fp32[f"{prefix}.ln_2.g"]
            ln2_b = self.model.weights_fp32[f"{prefix}.ln_2.b"]
            x_norm = self._layer_norm(x, ln2_g, ln2_b)

            # FFN (Vectorized)
            x = self._ffn_layer(x_norm, layer_idx, seed + layer_idx * 1000 + 100)

        # Final layer norm
        ln_f_g = self.model.weights_fp32["ln_f.g"]
        ln_f_b = self.model.weights_fp32["ln_f.b"]
        x_final = self._layer_norm(x, ln_f_g, ln_f_b)  # [Seq, d_model]

        if return_all_logits:
            return self._compute_logits(x_final, seed + 999999)
        else:
            return self._compute_logits(x_final[-1], seed + 999999)
