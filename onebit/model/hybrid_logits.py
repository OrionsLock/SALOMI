"""Hybrid model: FP32 backbone + 1-bit logits head.

Phase A: Prove CTG works for precision recovery in the most critical layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from onebit.core.packbits import pack_signs_rowmajor
from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig


@dataclass
class LogitsConfig:
    """Configuration for 1-bit logits computation."""
    T: int = 16  # Number of BSDM-W samples
    backend: str = "cpu"
    seed: int = 0x1234567890ABCDEF
    use_ctg: bool = False
    eps: float = 0.05
    delta: float = 0.001
    sd_order: int = 2
    sd_beta: float = 0.30
    sd_lambda: float = 1.0 / 256.0
    # Calibrated scaling parameters (from calibration harness)
    scale_a: float = 0.1689  # Global scale factor
    scale_b: float = -71.9659  # Global bias


class OneBitLogitsHead:
    """1-bit quantized logits head using BSDM-W.
    
    Computes logits = wte @ hidden_states where wte is 1-bit quantized.
    """
    
    def __init__(
        self,
        wte_fp32: np.ndarray,  # [vocab_size, d_model]
        config: LogitsConfig,
        calibration_hidden_states: Optional[np.ndarray] = None,  # [n_samples, d_model]
    ):
        """Initialize 1-bit logits head.

        Args:
            wte_fp32: Token embedding matrix [vocab_size, d_model]
            config: Configuration for 1-bit computation
            calibration_hidden_states: Optional hidden states for calibration
        """
        self.config = config
        self.vocab_size, self.d_model = wte_fp32.shape

        # Quantize to 1-bit (signs only)
        self.wte_bits = pack_signs_rowmajor(wte_fp32)  # [vocab_size, ceil(d_model/32)]

        # Compute scaling factor to match FP32 magnitude
        # For sign-quantized dot products, we need to scale by:
        # 1. The RMS of the weight matrix (to restore magnitude)
        # 2. The expected magnitude of hidden states
        # 3. sqrt(d_model) (dimension scaling)

        wte_rms = np.sqrt(np.mean(wte_fp32 ** 2))
        self.scale = wte_rms * self.d_model  # Scale factor

        # If calibration data provided, compute a better scaling factor
        if calibration_hidden_states is not None:
            # Compute FP32 logits for calibration
            logits_fp32 = wte_fp32 @ calibration_hidden_states.T  # [vocab_size, n_samples]
            fp32_std = np.std(logits_fp32)

            # Compute 1-bit logits for calibration (just a few samples)
            # This is expensive, so we only do it once during initialization
            # For now, use the RMS-based scaling
            pass

        # ΣΔ configuration
        self.sd_cfg = SDConfig(
            order=config.sd_order,
            beta=config.sd_beta,
            lambd=config.sd_lambda,
            walsh_N=2,
            antithetic=False,  # Disable antithetic - it cancels out the signal!
        )
    
    def forward(
        self,
        hidden_states: np.ndarray,  # [seq_len, d_model] or [d_model]
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Compute logits using 1-bit wte matrix.
        
        Args:
            hidden_states: Hidden states from transformer [seq_len, d_model] or [d_model]
            seed: Optional seed for BSDM-W
            
        Returns:
            Logits [seq_len, vocab_size] or [vocab_size]
        """
        if seed is None:
            seed = self.config.seed
        
        # Handle both [d_model] and [seq_len, d_model] inputs
        if hidden_states.ndim == 1:
            return self._compute_logits_single(hidden_states, seed)
        else:
            # Compute logits for each position
            seq_len = hidden_states.shape[0]
            logits_all = np.zeros((seq_len, self.vocab_size), dtype=np.float32)
            for i in range(seq_len):
                logits_all[i] = self._compute_logits_single(hidden_states[i], seed + i)
            return logits_all
    
    def _compute_logits_single(
        self,
        hidden_state: np.ndarray,  # [d_model]
        seed: int,
    ) -> np.ndarray:
        """Compute logits for a single hidden state.

        Args:
            hidden_state: Hidden state [d_model]
            seed: Seed for BSDM-W

        Returns:
            Logits [vocab_size]
        """
        from onebit.core.packbits import pack_input_signs

        # Quantize hidden state to {-1, +1} (sign quantization)
        # This is what BSDM-W expects: binary vectors in {-1, +1}
        h_signs = np.sign(hidden_state)  # {-1, 0, +1}
        h_signs[h_signs == 0] = 1  # Map 0 to +1

        # Pack to bits
        h_bits = pack_input_signs(h_signs)  # [ceil(d_model/32)]

        # Compute per-hidden-state structural factor
        # This is the ||h|| * sqrt(d) term from calibration
        h_norm = np.linalg.norm(hidden_state)
        structural_factor = h_norm * np.sqrt(self.d_model)

        # Compute logits for each token
        logits = np.zeros(self.vocab_size, dtype=np.float32)

        for v in range(self.vocab_size):
            # Get row v from wte_bits
            row_v_bits = self.wte_bits[v, :]  # [ceil(d_model/32)]

            # Derive per-row seed
            row_seed = (seed + v) & 0xFFFFFFFFFFFFFFFF

            # BSDM-W dot product: wte[v, :] . hidden_state
            est, _ = bsdm_w_dot(
                row_v_bits,
                h_bits,
                k=self.config.T,
                cfg=self.sd_cfg,
                seed=row_seed,
                eps=self.config.eps,
                delta=self.config.delta / self.vocab_size,  # Split risk budget
                early_exit_enable=False,  # Disable early exit for more accurate estimates
                use_ctg=self.config.use_ctg,
            )

            # Apply calibrated scaling: ŷ = a * (structural * ẑ) + b
            logits[v] = self.config.scale_a * structural_factor * est + self.config.scale_b

        return logits


class TernaryLogitsHead:
    """1.53-bit ternary logits head using {-α, 0, +α} quantization.

    This is a strong baseline that uses 1.53 bits per weight on average.
    Uses optimal scaling factor α to minimize quantization error.
    """

    def __init__(self, wte_fp32: np.ndarray):
        """Initialize ternary logits head.

        Args:
            wte_fp32: Token embedding matrix [vocab_size, d_model]
        """
        self.vocab_size, self.d_model = wte_fp32.shape

        # Quantize to ternary {-1, 0, +1} (signs only)
        # Use threshold at ±0.75 * std (common choice for ternary quantization)
        threshold = 0.75 * np.std(wte_fp32)

        wte_signs = np.zeros_like(wte_fp32, dtype=np.float32)
        wte_signs[wte_fp32 > threshold] = 1.0
        wte_signs[wte_fp32 < -threshold] = -1.0
        # Middle values stay 0

        # Compute optimal scaling factor α per row
        # α = E[W * sign(W)] / E[sign(W)^2] where sign(W) is the ternary quantization
        # For simplicity, use global α = RMS(W) / RMS(sign(W))
        alpha = np.sqrt(np.mean(wte_fp32 ** 2)) / np.sqrt(np.mean(wte_signs ** 2) + 1e-8)

        self.wte_ternary = alpha * wte_signs
        
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Compute logits using ternary wte matrix.
        
        Args:
            hidden_states: Hidden states [seq_len, d_model] or [d_model]
            
        Returns:
            Logits [seq_len, vocab_size] or [vocab_size]
        """
        if hidden_states.ndim == 1:
            return self.wte_ternary @ hidden_states
        else:
            return hidden_states @ self.wte_ternary.T

