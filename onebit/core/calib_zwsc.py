"""Zero-Write Self-Calibration (ZWSC) and Stateless Domain-Adaptive Calibration (SDAC).

This module provides runtime calibration for 1-bit models without modifying weights.
Calibration statistics are collected from a domain-specific token stream and stored
in RAM only (no disk writes).

Key features:
- Collect per-layer mean/variance statistics from calibration tokens
- Compute affine transformations (a, b) to center and scale activations
- Apply transformations at runtime to improve stability
- Support domain-specific profiles (e.g., text vs code)
- Zero storage overhead (RAM-only, ephemeral)

Typical usage:
    # Collect statistics from calibration stream
    stats = collect_stats(calibration_tokens, model, n_tokens=50000)
    
    # Compute affine parameters
    affine_params = compute_affine_params(stats)
    
    # Apply at runtime
    y_calibrated = apply_affine(y, affine_params[layer_id])
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional, Iterator
from dataclasses import dataclass
import numpy as np


@dataclass
class LayerStats:
    """Statistics for a single layer.
    
    Attributes:
        mu_in: Mean of layer inputs (shape: [d])
        sigma_in: Std dev of layer inputs (shape: [d])
        mu_out: Mean of layer outputs (shape: [d])
        sigma_out: Std dev of layer outputs (shape: [d])
        n_samples: Number of samples used for statistics
    """
    mu_in: np.ndarray
    sigma_in: np.ndarray
    mu_out: np.ndarray
    sigma_out: np.ndarray
    n_samples: int


@dataclass
class AffineParams:
    """Affine transformation parameters for a layer.
    
    The transformation is: y_out = a * y_in + b
    
    Attributes:
        a: Scale parameter (shape: [d])
        b: Shift parameter (shape: [d])
    """
    a: np.ndarray
    b: np.ndarray


class OnlineStats:
    """Online computation of mean and variance using Welford's algorithm.
    
    This allows computing statistics in a single pass without storing all samples.
    """
    
    def __init__(self, dim: int):
        """Initialize online statistics tracker.
        
        Args:
            dim: Dimensionality of the data
        """
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)  # Sum of squared deviations
    
    def update(self, x: np.ndarray):
        """Update statistics with a new sample.
        
        Args:
            x: New sample (shape: [dim])
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
    
    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current mean and standard deviation.
        
        Returns:
            Tuple of (mean, std_dev), each with shape [dim]
        """
        if self.n < 2:
            return self.mean, np.ones(self.dim, dtype=np.float64)
        
        variance = self.m2 / self.n
        std_dev = np.sqrt(variance + 1e-8)  # Add epsilon for numerical stability
        return self.mean.astype(np.float32), std_dev.astype(np.float32)


def collect_stats(
    activation_stream: Iterator[Tuple[int, np.ndarray, np.ndarray]],
    n_tokens: int = 50000,
    layer_ids: Optional[list[int]] = None,
) -> Dict[int, LayerStats]:
    """Collect per-layer statistics from an activation stream.
    
    Args:
        activation_stream: Iterator yielding (layer_id, input, output) tuples
        n_tokens: Number of tokens to process for calibration
        layer_ids: List of layer IDs to track (None = track all)
    
    Returns:
        Dictionary mapping layer_id to LayerStats
    """
    # Initialize online statistics trackers
    input_stats: Dict[int, OnlineStats] = {}
    output_stats: Dict[int, OnlineStats] = {}
    
    tokens_processed = 0
    
    for layer_id, x_in, x_out in activation_stream:
        # Filter by layer_ids if specified
        if layer_ids is not None and layer_id not in layer_ids:
            continue
        
        # Initialize trackers for new layers
        if layer_id not in input_stats:
            dim = x_in.shape[-1] if x_in.ndim > 0 else 1
            input_stats[layer_id] = OnlineStats(dim)
            output_stats[layer_id] = OnlineStats(dim)
        
        # Update statistics
        input_stats[layer_id].update(x_in.flatten() if x_in.ndim > 1 else x_in)
        output_stats[layer_id].update(x_out.flatten() if x_out.ndim > 1 else x_out)
        
        tokens_processed += 1
        if tokens_processed >= n_tokens:
            break
    
    # Compile final statistics
    stats = {}
    for layer_id in input_stats.keys():
        mu_in, sigma_in = input_stats[layer_id].get_stats()
        mu_out, sigma_out = output_stats[layer_id].get_stats()
        
        stats[layer_id] = LayerStats(
            mu_in=mu_in,
            sigma_in=sigma_in,
            mu_out=mu_out,
            sigma_out=sigma_out,
            n_samples=input_stats[layer_id].n,
        )
    
    return stats


def compute_affine_params(
    stats: Dict[int, LayerStats],
    target_mean: float = 0.0,
    target_std: float = 1.0,
    mode: str = "output",
) -> Dict[int, AffineParams]:
    """Compute affine transformation parameters from statistics.
    
    The transformation normalizes activations to have target mean and std:
        y_normalized = (y - mu) / sigma * target_std + target_mean
        y_normalized = a * y + b
    where:
        a = target_std / sigma
        b = target_mean - a * mu
    
    Args:
        stats: Per-layer statistics from collect_stats()
        target_mean: Target mean for normalized activations
        target_std: Target standard deviation for normalized activations
        mode: Which statistics to use ("input", "output", or "both")
    
    Returns:
        Dictionary mapping layer_id to AffineParams
    """
    affine_params = {}
    
    for layer_id, layer_stats in stats.items():
        # Choose which statistics to use
        if mode == "input":
            mu = layer_stats.mu_in
            sigma = layer_stats.sigma_in
        elif mode == "output":
            mu = layer_stats.mu_out
            sigma = layer_stats.sigma_out
        elif mode == "both":
            # Average input and output statistics
            mu = (layer_stats.mu_in + layer_stats.mu_out) / 2
            sigma = (layer_stats.sigma_in + layer_stats.sigma_out) / 2
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Compute affine parameters
        # y_normalized = (y - mu) / sigma * target_std + target_mean
        # y_normalized = (target_std / sigma) * y + (target_mean - (target_std / sigma) * mu)
        a = target_std / (sigma + 1e-8)  # Avoid division by zero
        b = target_mean - a * mu
        
        affine_params[layer_id] = AffineParams(a=a, b=b)
    
    return affine_params


def apply_affine(
    y: np.ndarray,
    params: AffineParams,
) -> np.ndarray:
    """Apply affine transformation to activations.
    
    Args:
        y: Input activations (shape: [d] or [batch, d])
        params: Affine parameters (a, b)
    
    Returns:
        Transformed activations (same shape as y)
    """
    return params.a * y + params.b


def apply_affine_inplace(
    y: np.ndarray,
    params: AffineParams,
) -> np.ndarray:
    """Apply affine transformation in-place.
    
    Args:
        y: Input activations (shape: [d] or [batch, d])
        params: Affine parameters (a, b)
    
    Returns:
        Reference to y (modified in-place)
    """
    y *= params.a
    y += params.b
    return y


@dataclass
class CalibrationProfile:
    """A calibration profile for a specific domain.
    
    Attributes:
        name: Profile name (e.g., "wikitext", "code", "math")
        affine_params: Per-layer affine parameters
        n_tokens: Number of tokens used for calibration
        domain_tags: Optional tags describing the domain
    """
    name: str
    affine_params: Dict[int, AffineParams]
    n_tokens: int
    domain_tags: Optional[list[str]] = None


def create_profile(
    name: str,
    activation_stream: Iterator[Tuple[int, np.ndarray, np.ndarray]],
    n_tokens: int = 50000,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    mode: str = "output",
    domain_tags: Optional[list[str]] = None,
) -> CalibrationProfile:
    """Create a calibration profile from an activation stream.
    
    This is a convenience function that combines collect_stats() and
    compute_affine_params() into a single call.
    
    Args:
        name: Profile name
        activation_stream: Iterator yielding (layer_id, input, output) tuples
        n_tokens: Number of tokens to process
        target_mean: Target mean for normalized activations
        target_std: Target standard deviation
        mode: Which statistics to use ("input", "output", or "both")
        domain_tags: Optional tags describing the domain
    
    Returns:
        CalibrationProfile ready for use
    """
    # Collect statistics
    stats = collect_stats(activation_stream, n_tokens=n_tokens)
    
    # Compute affine parameters
    affine_params = compute_affine_params(
        stats,
        target_mean=target_mean,
        target_std=target_std,
        mode=mode,
    )
    
    return CalibrationProfile(
        name=name,
        affine_params=affine_params,
        n_tokens=n_tokens,
        domain_tags=domain_tags,
    )

