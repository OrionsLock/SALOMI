"""Adaptive CTG program selector with MLP and Gumbel-Softmax.

Phase III: Adaptive Selector for CTG-PROG v1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


@dataclass
class SelectorConfig:
    """Configuration for adaptive program selector."""
    K: int = 4  # Number of programs
    hidden_dim: int = 32  # Hidden layer size
    tau_init: float = 1.0  # Initial Gumbel temperature
    tau_final: float = 0.2  # Final Gumbel temperature
    tau_anneal_steps: int = 10000  # Steps to anneal temperature
    feature_dim: int = 8  # Input feature dimension


@dataclass
class SelectorState:
    """Runtime state for program selector."""
    step: int = 0  # Training step counter
    tau: float = 1.0  # Current Gumbel temperature
    phase_history: np.ndarray = None  # Recent phase history [history_len]
    
    def __post_init__(self):
        if self.phase_history is None:
            self.phase_history = np.zeros(8, dtype=np.int32)


class AdaptiveProgramSelector(nn.Module if TORCH_AVAILABLE else object):
    """2-layer MLP for adaptive CTG program selection.
    
    Architecture:
        Input features → Linear(hidden_dim) → ReLU → Linear(K) → Gumbel-Softmax
    
    Features:
        - Shortlist entropy (1D)
        - Top-2 margin (1D)
        - Attention entropy (1D)
        - Phase history statistics (5D: mean, std, min, max, last)
    
    Output:
        - Hard argmax at inference (deterministic)
        - Soft distribution during training (Gumbel-Softmax for gradients)
    """
    
    def __init__(self, cfg: SelectorConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for AdaptiveProgramSelector")
        super().__init__()
        self.cfg = cfg
        
        # 2-layer MLP
        self.fc1 = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.K)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            features: Input features [batch, feature_dim]
            tau: Gumbel temperature
            hard: If True, return hard one-hot (inference mode)
        
        Returns:
            (program_probs, program_id):
                - program_probs: Soft distribution [batch, K] (for training)
                - program_id: Hard argmax [batch] (for inference)
        """
        # MLP
        h = F.relu(self.fc1(features))
        logits = self.fc2(h)  # [batch, K]
        
        if hard:
            # Inference: hard argmax
            program_id = torch.argmax(logits, dim=-1)
            program_probs = F.one_hot(program_id, num_classes=self.cfg.K).float()
        else:
            # Training: Gumbel-Softmax
            program_probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            program_id = torch.argmax(program_probs, dim=-1)
        
        return program_probs, program_id


def extract_features(
    shortlist_logits: np.ndarray,
    top2_margin: float,
    attn_entropy: float,
    phase_history: np.ndarray,
) -> np.ndarray:
    """Extract feature vector for program selector.
    
    Args:
        shortlist_logits: Logits for shortlist candidates [shortlist_size]
        top2_margin: Margin between top-2 candidates
        attn_entropy: Attention distribution entropy
        phase_history: Recent CTG phase values [history_len]
    
    Returns:
        Feature vector [feature_dim=8]:
            [0]: Shortlist entropy
            [1]: Top-2 margin
            [2]: Attention entropy
            [3]: Phase history mean
            [4]: Phase history std
            [5]: Phase history min
            [6]: Phase history max
            [7]: Phase history last
    """
    # Shortlist entropy
    if len(shortlist_logits) > 0:
        probs = np.exp(shortlist_logits - shortlist_logits.max())
        probs = probs / probs.sum()
        shortlist_entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        shortlist_entropy = 0.0
    
    # Phase history statistics
    phase_mean = float(np.mean(phase_history))
    phase_std = float(np.std(phase_history))
    phase_min = float(np.min(phase_history))
    phase_max = float(np.max(phase_history))
    phase_last = float(phase_history[-1]) if len(phase_history) > 0 else 0.0
    
    features = np.array([
        shortlist_entropy,
        top2_margin,
        attn_entropy,
        phase_mean,
        phase_std,
        phase_min,
        phase_max,
        phase_last,
    ], dtype=np.float32)
    
    return features

