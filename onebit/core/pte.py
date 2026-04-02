"""Probability-Transformed Ensemble (PTE).

PTE applies variance-reducing transformations to probability distributions
before ensembling. This reduces variance in softmax outputs and improves
calibration.

Key techniques:
1. **Temperature scaling**: Scale logits before softmax
2. **Probability sharpening**: Raise probabilities to power α
3. **Ensemble averaging**: Average transformed probabilities

Typical usage:
    # Collect logits from multiple runs
    logits_ensemble = [
        run_logits(..., seed=1),
        run_logits(..., seed=2),
        run_logits(..., seed=3),
    ]
    
    # Apply PTE
    pte = PTEnsemble(temperature=1.2, alpha=0.9)
    probs_final = pte.ensemble(logits_ensemble)
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PTEConfig:
    """PTE configuration.
    
    Attributes:
        temperature: Temperature for softmax scaling (>1 = smoother, <1 = sharper)
        alpha: Sharpening exponent for probabilities (>1 = sharper, <1 = smoother)
        ensemble_method: Ensemble method ("mean", "geometric_mean", "median")
        clip_logits: Whether to clip logits to prevent overflow
        clip_range: Range for logit clipping
    """
    temperature: float = 1.0
    alpha: float = 1.0
    ensemble_method: str = "mean"  # "mean", "geometric_mean", "median"
    clip_logits: bool = True
    clip_range: Tuple[float, float] = (-20.0, 20.0)


class PTEnsemble:
    """Probability-Transformed Ensemble."""
    
    def __init__(self, cfg: Optional[PTEConfig] = None):
        """Initialize PTE.
        
        Args:
            cfg: PTE configuration
        """
        self.cfg = cfg or PTEConfig()
    
    def ensemble(
        self,
        logits_list: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Ensemble logits using PTE.

        Args:
            logits_list: List of logit arrays [n_samples, vocab_size]
            weights: Optional weights for each logit array (default: uniform)

        Returns:
            Ensembled probabilities [vocab_size]
        """
        if len(logits_list) == 0:
            raise ValueError("logits_list is empty")

        if len(logits_list) == 1:
            # Single logits, apply temperature and sharpening
            return self._logits_to_probs(logits_list[0])

        # Convert to probabilities
        probs_list = [self._logits_to_probs(logits) for logits in logits_list]

        # Ensemble probabilities
        if self.cfg.ensemble_method == "mean":
            probs_ensemble = self._ensemble_mean(probs_list, weights)
        elif self.cfg.ensemble_method == "geometric_mean":
            probs_ensemble = self._ensemble_geometric_mean(probs_list, weights)
        elif self.cfg.ensemble_method == "median":
            probs_ensemble = self._ensemble_median(probs_list)
        else:
            raise ValueError(f"Unknown ensemble method: {self.cfg.ensemble_method}")

        # Normalize
        probs_ensemble = probs_ensemble / probs_ensemble.sum()

        return probs_ensemble
    
    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities with temperature and sharpening.
        
        Args:
            logits: Logit array [vocab_size]
        
        Returns:
            Probability array [vocab_size]
        """
        # Clip logits
        if self.cfg.clip_logits:
            logits = np.clip(logits, self.cfg.clip_range[0], self.cfg.clip_range[1])
        
        # Apply temperature scaling
        logits_scaled = logits / self.cfg.temperature
        
        # Softmax
        probs = self._softmax(logits_scaled)
        
        # Apply sharpening
        if self.cfg.alpha != 1.0:
            probs = np.power(probs, self.cfg.alpha)
            probs = probs / probs.sum()  # Renormalize
        
        return probs
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.
        
        Args:
            logits: Logit array [vocab_size]
        
        Returns:
            Probability array [vocab_size]
        """
        # Subtract max for numerical stability
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()
        return probs
    
    def _ensemble_mean(
        self,
        probs_list: List[np.ndarray],
        weights: Optional[List[float]],
    ) -> np.ndarray:
        """Ensemble using arithmetic mean.
        
        Args:
            probs_list: List of probability arrays
            weights: Optional weights
        
        Returns:
            Ensembled probabilities
        """
        if weights is None:
            weights = [1.0 / len(probs_list)] * len(probs_list)
        else:
            # Normalize weights
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
        
        probs_ensemble = np.zeros_like(probs_list[0])
        for w, probs in zip(weights, probs_list):
            probs_ensemble += w * probs
        
        return probs_ensemble
    
    def _ensemble_geometric_mean(
        self,
        probs_list: List[np.ndarray],
        weights: Optional[List[float]],
    ) -> np.ndarray:
        """Ensemble using geometric mean.
        
        Geometric mean is more robust to outliers.
        
        Args:
            probs_list: List of probability arrays
            weights: Optional weights
        
        Returns:
            Ensembled probabilities
        """
        if weights is None:
            weights = [1.0 / len(probs_list)] * len(probs_list)
        else:
            # Normalize weights
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
        
        # Geometric mean: (p1^w1 * p2^w2 * ...)^(1/sum(w))
        # In log space: exp(w1*log(p1) + w2*log(p2) + ...)
        log_probs_ensemble = np.zeros_like(probs_list[0])
        for w, probs in zip(weights, probs_list):
            # Add small epsilon to avoid log(0)
            log_probs_ensemble += w * np.log(probs + 1e-10)
        
        probs_ensemble = np.exp(log_probs_ensemble)
        
        return probs_ensemble
    
    def _ensemble_median(
        self,
        probs_list: List[np.ndarray],
    ) -> np.ndarray:
        """Ensemble using median.
        
        Median is most robust to outliers.
        
        Args:
            probs_list: List of probability arrays
        
        Returns:
            Ensembled probabilities
        """
        # Stack and take median along axis 0
        probs_stacked = np.stack(probs_list, axis=0)
        probs_ensemble = np.median(probs_stacked, axis=0)
        
        return probs_ensemble


def calibrate_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    T_range: Tuple[float, float] = (0.5, 2.0),
    n_steps: int = 20,
) -> float:
    """Calibrate temperature using grid search.
    
    Finds temperature that minimizes cross-entropy loss.
    
    Args:
        logits: Logit array [n_samples, vocab_size]
        targets: Target indices [n_samples]
        T_range: Temperature range to search
        n_steps: Number of steps in grid search
    
    Returns:
        Optimal temperature
    """
    T_min, T_max = T_range
    T_values = np.linspace(T_min, T_max, n_steps)
    
    best_T = 1.0
    best_loss = float('inf')
    
    for T in T_values:
        # Apply temperature scaling
        logits_scaled = logits / T
        
        # Compute cross-entropy loss
        loss = 0.0
        for i, target in enumerate(targets):
            probs = np.exp(logits_scaled[i] - np.max(logits_scaled[i]))
            probs = probs / probs.sum()
            loss -= np.log(probs[target] + 1e-10)
        
        loss /= len(targets)
        
        if loss < best_loss:
            best_loss = loss
            best_T = T
    
    return float(best_T)


def compute_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy.
    
    Args:
        probs: Probability array [n_samples, vocab_size]
        targets: Target indices [n_samples]
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with calibration statistics
    """
    # Get predicted probabilities and predictions
    pred_probs = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    # Compute accuracy
    correct = (predictions == targets).astype(float)
    
    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(pred_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute ECE
    ece = 0.0
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_confidence = pred_probs[mask].mean()
            bin_accuracy = correct[mask].mean()
            bin_weight = mask.sum() / len(targets)
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
    
    return {
        "ece": float(ece),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(pred_probs.mean()),
    }


# Type alias for clarity
Dict = dict  # type: ignore

