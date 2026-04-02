"""Training harness for adaptive CTG program selector.

Phase III: Training with composite loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore


@dataclass
class TrainingConfig:
    """Configuration for CTG selector training."""
    lambda_var: float = 0.1  # Variance reduction weight
    lambda_switch: float = 0.05  # Program stability weight
    lambda_entropy: float = 0.01  # Entropy regularization weight
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_steps: int = 50000
    log_interval: int = 100
    checkpoint_interval: int = 1000


class CTGTrainer:
    """Training harness for adaptive program selector.
    
    Loss components:
        - L_base: Base task loss (e.g., perplexity, accuracy)
        - L_var: Variance reduction (reward lower variance in k_used)
        - L_switch: Program stability (penalize frequent switching)
        - L_entropy: Entropy regularization (avoid collapse to single program)
    
    Total loss:
        L = L_base + λ₁ L_var + λ₂ L_switch + λ₃ L_entropy
    """
    
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CTGTrainer")
        
        self.model = model
        self.cfg = cfg
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        
        # Training state
        self.step = 0
        self.program_usage = np.zeros(4, dtype=np.int32)  # Track usage of K=4 programs
        self.prev_program_id = None
        
        # Metrics history
        self.loss_history: List[Dict[str, float]] = []
    
    def compute_loss(
        self,
        program_probs: torch.Tensor,
        program_id: torch.Tensor,
        base_loss: torch.Tensor,
        k_variance: float,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Compute composite loss.
        
        Args:
            program_probs: Soft program distribution [batch, K]
            program_id: Hard program selection [batch]
            base_loss: Base task loss (scalar)
            k_variance: Variance of k_used across batch
            batch_size: Batch size
        
        Returns:
            Dict with loss components and total loss
        """
        # L_var: Variance reduction (lower is better)
        # Normalize by baseline variance to make scale-invariant
        L_var = torch.tensor(k_variance, dtype=torch.float32, device=self.device)
        
        # L_switch: Program stability (penalize switching)
        if self.prev_program_id is not None:
            switch_count = (program_id != self.prev_program_id).float().sum()
            L_switch = switch_count / batch_size
        else:
            L_switch = torch.tensor(0.0, device=self.device)
        
        # L_entropy: Entropy regularization (avoid collapse)
        # Encourage uniform distribution over programs
        avg_probs = program_probs.mean(dim=0)  # [K]
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        target_entropy = np.log(program_probs.shape[1])  # log(K)
        L_entropy = -entropy  # Negative because we want to maximize entropy
        
        # Total loss
        total_loss = (
            base_loss
            + self.cfg.lambda_var * L_var
            + self.cfg.lambda_switch * L_switch
            + self.cfg.lambda_entropy * L_entropy
        )
        
        return {
            "total": total_loss,
            "base": base_loss,
            "var": L_var,
            "switch": L_switch,
            "entropy": L_entropy,
        }
    
    def train_step(
        self,
        features: torch.Tensor,
        base_loss: torch.Tensor,
        k_variance: float,
        tau: float,
    ) -> Dict[str, float]:
        """Single training step.
        
        Args:
            features: Input features [batch, feature_dim]
            base_loss: Base task loss
            k_variance: Variance of k_used
            tau: Current Gumbel temperature
        
        Returns:
            Dict with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        program_probs, program_id = self.model(features, tau=tau, hard=False)
        
        # Compute loss
        losses = self.compute_loss(
            program_probs, program_id, base_loss, k_variance,
            batch_size=features.shape[0]
        )
        
        # Backward pass
        losses["total"].backward()
        self.optimizer.step()
        
        # Update state
        self.step += 1
        self.prev_program_id = program_id.detach()
        
        # Track program usage
        for pid in program_id.cpu().numpy():
            self.program_usage[int(pid)] += 1
        
        # Convert to float for logging
        loss_dict = {k: float(v.item()) for k, v in losses.items()}
        self.loss_history.append(loss_dict)
        
        return loss_dict
    
    def get_program_usage_histogram(self) -> Dict[int, float]:
        """Get normalized program usage histogram."""
        total = self.program_usage.sum()
        if total == 0:
            return {i: 0.0 for i in range(4)}
        return {i: float(self.program_usage[i]) / total for i in range(4)}
    
    def reset_program_usage(self):
        """Reset program usage counters."""
        self.program_usage.fill(0)

