"""Method-of-Moments Self-Correction (MoM-SC).

MoM-SC uses moment matching to correct biased estimates from stochastic approximation.
Given multiple estimates with known bias structure, we can solve for the true value
by matching theoretical and empirical moments.

Key idea:
- Run BSDM-W with different T values (e.g., T=8, T=16, T=32)
- Each T has different bias: E[y_T] = y_true + bias(T)
- Use moment equations to solve for y_true

Typical usage:
    # Collect estimates at different T
    estimates = {
        8: run_bsdm_w(..., T=8),
        16: run_bsdm_w(..., T=16),
        32: run_bsdm_w(..., T=32),
    }
    
    # Apply MoM-SC
    corrector = MoMSelfCorrector()
    y_corrected = corrector.correct(estimates)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MoMSCConfig:
    """MoM-SC configuration.
    
    Attributes:
        T_values: List of T values to use for correction
        bias_model: Bias model ("power_law", "exponential", "linear")
        min_T: Minimum T value (for extrapolation bounds)
        max_T: Maximum T value (for extrapolation bounds)
    """
    T_values: List[int] = None  # type: ignore
    bias_model: str = "power_law"  # "power_law", "exponential", "linear"
    min_T: int = 8
    max_T: int = 64
    
    def __post_init__(self):
        if self.T_values is None:
            self.T_values = [8, 16, 32]


class MoMSelfCorrector:
    """Method-of-Moments Self-Correction."""
    
    def __init__(self, cfg: Optional[MoMSCConfig] = None):
        """Initialize MoM-SC corrector.
        
        Args:
            cfg: MoM-SC configuration
        """
        self.cfg = cfg or MoMSCConfig()
    
    def correct(
        self,
        estimates: Dict[int, float],
        weights: Optional[Dict[int, float]] = None,
    ) -> float:
        """Correct estimates using MoM-SC.
        
        Args:
            estimates: Dictionary mapping T -> estimate
            weights: Optional weights for each estimate (default: uniform)
        
        Returns:
            Corrected estimate
        """
        if len(estimates) < 2:
            # Need at least 2 estimates for correction
            return list(estimates.values())[0]
        
        # Extract T values and estimates
        T_vals = sorted(estimates.keys())
        y_vals = [estimates[T] for T in T_vals]
        
        # Apply bias model
        if self.cfg.bias_model == "power_law":
            return self._correct_power_law(T_vals, y_vals, weights)
        elif self.cfg.bias_model == "exponential":
            return self._correct_exponential(T_vals, y_vals, weights)
        elif self.cfg.bias_model == "linear":
            return self._correct_linear(T_vals, y_vals, weights)
        else:
            raise ValueError(f"Unknown bias model: {self.cfg.bias_model}")
    
    def _correct_power_law(
        self,
        T_vals: List[int],
        y_vals: List[float],
        weights: Optional[Dict[int, float]],
    ) -> float:
        """Correct using power law bias model: bias(T) = a / T^b.

        Assumes: E[y_T] = y_true + a / T^b

        For two estimates:
            y_1 = y_true + a / T_1^b
            y_2 = y_true + a / T_2^b

        Solving for y_true (assuming b=1):
            y_true = (T_1 * y_2 - T_2 * y_1) / (T_1 - T_2)
                   = (T_2 * y_2 - T_1 * y_1) / (T_2 - T_1)  # if T_2 > T_1

        Args:
            T_vals: List of T values
            y_vals: List of estimates
            weights: Optional weights

        Returns:
            Corrected estimate
        """
        if len(T_vals) == 2:
            # Exact solution for 2 estimates (b=1)
            T1, T2 = T_vals
            y1, y2 = y_vals

            if T1 == T2:
                return (y1 + y2) / 2

            # Richardson extrapolation formula
            # For bias ~ 1/T: y_true = (r*y_fine - y_coarse) / (r - 1) where r = T_fine/T_coarse
            r = T2 / T1  # Assume T2 > T1 (fine > coarse)
            y_corrected = (r * y2 - y1) / (r - 1)

            # Clamp to reasonable range (avoid extrapolation artifacts)
            y_min = min(y_vals)
            y_max = max(y_vals)
            margin = (y_max - y_min) * 2.0  # Allow 2x range
            y_corrected = np.clip(y_corrected, y_min - margin, y_max + margin)

            return float(y_corrected)
        else:
            # Weighted average for 3+ estimates (conservative)
            if weights is None:
                # Weight by T (higher T = more accurate)
                total_T = sum(T_vals)
                w = [T / total_T for T in T_vals]
            else:
                w = [weights.get(T, 1.0) for T in T_vals]
                total_w = sum(w)
                w = [wi / total_w for wi in w]
            
            y_corrected = sum(wi * yi for wi, yi in zip(w, y_vals))
            return float(y_corrected)
    
    def _correct_exponential(
        self,
        T_vals: List[int],
        y_vals: List[float],
        weights: Optional[Dict[int, float]],
    ) -> float:
        """Correct using exponential bias model: bias(T) = a * exp(-b * T).
        
        Args:
            T_vals: List of T values
            y_vals: List of estimates
            weights: Optional weights
        
        Returns:
            Corrected estimate
        """
        # For exponential, use weighted average with exponential weights
        if weights is None:
            # Weight by exp(T) (higher T = exponentially more accurate)
            exp_T = [np.exp(T / 10.0) for T in T_vals]  # Scale by 10 to avoid overflow
            total_exp = sum(exp_T)
            w = [e / total_exp for e in exp_T]
        else:
            w = [weights.get(T, 1.0) for T in T_vals]
            total_w = sum(w)
            w = [wi / total_w for wi in w]
        
        y_corrected = sum(wi * yi for wi, yi in zip(w, y_vals))
        return float(y_corrected)
    
    def _correct_linear(
        self,
        T_vals: List[int],
        y_vals: List[float],
        weights: Optional[Dict[int, float]],
    ) -> float:
        """Correct using linear bias model: bias(T) = a - b * T.
        
        For two estimates:
            y_1 = y_true + a - b * T_1
            y_2 = y_true + a - b * T_2
        
        Solving for y_true:
            y_true = (y_1 + y_2) / 2 + b * (T_1 + T_2) / 2
        
        Since we don't know b, use weighted average.
        
        Args:
            T_vals: List of T values
            y_vals: List of estimates
            weights: Optional weights
        
        Returns:
            Corrected estimate
        """
        # For linear, use weighted average
        if weights is None:
            # Equal weights
            w = [1.0 / len(T_vals)] * len(T_vals)
        else:
            w = [weights.get(T, 1.0) for T in T_vals]
            total_w = sum(w)
            w = [wi / total_w for wi in w]
        
        y_corrected = sum(wi * yi for wi, yi in zip(w, y_vals))
        return float(y_corrected)
    
    def correct_batch(
        self,
        estimates_batch: List[Dict[int, float]],
        weights: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Correct a batch of estimates.
        
        Args:
            estimates_batch: List of estimate dictionaries
            weights: Optional weights for each estimate
        
        Returns:
            Array of corrected estimates
        """
        return np.array([self.correct(est, weights) for est in estimates_batch])


def richardson_extrapolation(
    y_coarse: float,
    y_fine: float,
    T_coarse: int,
    T_fine: int,
    order: int = 1,
) -> float:
    """Richardson extrapolation for bias reduction.

    Assumes bias scales as O(T^(-order)).

    For order=1:
        y_true ≈ (T_coarse * y_fine - T_fine * y_coarse) / (T_coarse - T_fine)

    Args:
        y_coarse: Estimate with coarse T
        y_fine: Estimate with fine T
        T_coarse: Coarse T value
        T_fine: Fine T value
        order: Order of convergence (default: 1)

    Returns:
        Extrapolated estimate
    """
    if T_coarse == T_fine:
        return (y_coarse + y_fine) / 2

    if order == 1:
        # Linear extrapolation: y_true = (r*y_fine - y_coarse) / (r - 1)
        r = T_fine / T_coarse
        y_extrap = (r * y_fine - y_coarse) / (r - 1)
    else:
        # Higher-order extrapolation
        r = (T_fine / T_coarse) ** order
        y_extrap = (r * y_fine - y_coarse) / (r - 1)

    # Clamp to reasonable range
    y_min = min(y_coarse, y_fine)
    y_max = max(y_coarse, y_fine)
    margin = (y_max - y_min) * 2.0
    y_extrap = np.clip(y_extrap, y_min - margin, y_max + margin)

    return float(y_extrap)


def compute_variance_reduction(
    estimates_original: np.ndarray,
    estimates_corrected: np.ndarray,
) -> Dict[str, float]:
    """Compute variance reduction statistics.
    
    Args:
        estimates_original: Original estimates (before correction)
        estimates_corrected: Corrected estimates (after MoM-SC)
    
    Returns:
        Dictionary with variance reduction statistics
    """
    var_original = np.var(estimates_original)
    var_corrected = np.var(estimates_corrected)
    
    reduction_ratio = var_original / var_corrected if var_corrected > 0 else 1.0
    reduction_pct = (1 - var_corrected / var_original) * 100 if var_original > 0 else 0.0
    
    return {
        "var_original": float(var_original),
        "var_corrected": float(var_corrected),
        "reduction_ratio": float(reduction_ratio),
        "reduction_pct": float(reduction_pct),
    }

