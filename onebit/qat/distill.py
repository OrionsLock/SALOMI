"""Knowledge distillation loss for KD-QAT (logit-KL + CE + optional hidden-state MSE)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combined frozen-teacher distillation + language modeling loss.

    Computes::

        loss = kl_weight * T^2 * KL(softmax(teacher / T) || softmax(student / T))
             + ce_weight * CE(student[:, :-1], input_ids[:, 1:])
             + hidden_mse_weight * mean_l MSE(student_h[l], teacher_h[l])

    All loss reductions are computed in float32 for numerical stability under
    ``bf16``/``fp16`` autocast.
    """

    def __init__(
        self,
        kl_weight: float = 1.0,
        ce_weight: float = 0.0,
        temperature: float = 1.0,
        hidden_mse_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if kl_weight < 0 or ce_weight < 0 or hidden_mse_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if kl_weight == 0 and ce_weight == 0 and hidden_mse_weight == 0:
            raise ValueError("At least one of kl_weight, ce_weight, hidden_mse_weight must be > 0")
        self.kl_weight = float(kl_weight)
        self.ce_weight = float(ce_weight)
        self.temperature = float(temperature)
        self.hidden_mse_weight = float(hidden_mse_weight)

    @property
    def needs_hidden_states(self) -> bool:
        return self.hidden_mse_weight > 0.0

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        student_hidden_states: Optional[Sequence[torch.Tensor]] = None,
        teacher_hidden_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = student_logits.new_zeros((), dtype=torch.float32)
        metrics: Dict[str, float] = {}

        if self.kl_weight > 0.0:
            if teacher_logits is None:
                raise ValueError("teacher_logits is required when kl_weight > 0")
            t = self.temperature
            s_logp = F.log_softmax(student_logits.float() / t, dim=-1)
            t_logp = F.log_softmax(teacher_logits.detach().float() / t, dim=-1)
            kl = F.kl_div(
                s_logp.flatten(0, 1),
                t_logp.flatten(0, 1),
                log_target=True,
                reduction="batchmean",
            ) * (t * t)
            total = total + self.kl_weight * kl
            metrics["kl_loss"] = float(kl.detach().item())
        else:
            metrics["kl_loss"] = 0.0

        if self.ce_weight > 0.0:
            shift_logits = student_logits[:, :-1].float().reshape(-1, student_logits.size(-1))
            shift_labels = input_ids[:, 1:].reshape(-1)
            ce = F.cross_entropy(shift_logits, shift_labels)
            total = total + self.ce_weight * ce
            metrics["ce_loss"] = float(ce.detach().item())
        else:
            metrics["ce_loss"] = 0.0

        if self.hidden_mse_weight > 0.0:
            if student_hidden_states is None or teacher_hidden_states is None:
                raise ValueError("hidden_states are required when hidden_mse_weight > 0")
            # Match transformer block outputs (skip index 0 raw embedding if >1 state exists).
            start = 1 if len(student_hidden_states) > 1 else 0
            s_states = student_hidden_states[start:]
            t_states = teacher_hidden_states[start:]
            if len(s_states) != len(t_states):
                raise ValueError(
                    f"Hidden state count mismatch: student={len(s_states)} vs teacher={len(t_states)}"
                )
            mse_sum = student_logits.new_zeros((), dtype=torch.float32)
            for sh, th in zip(s_states, t_states):
                mse_sum = mse_sum + F.mse_loss(sh.float(), th.detach().float())
            h_mse = mse_sum / max(len(s_states), 1)
            total = total + self.hidden_mse_weight * h_mse
            metrics["hidden_mse"] = float(h_mse.detach().item())
        else:
            metrics["hidden_mse"] = 0.0

        metrics["loss"] = float(total.detach().item())
        return total, metrics
