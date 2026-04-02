#!/usr/bin/env python3
"""
GELU-Aware Quantization for SALOMI
Implements quantization that accounts for GELU nonlinearity effects
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Callable, Optional
from tqdm import tqdm

class GELUAwareQuantizer:
    """
    GELU-aware quantizer that accounts for nonlinearity effects
    """

    def __init__(self, base_quantizer: Callable, sensitivity_threshold: float = 0.1):
        """
        Initialize GELU-aware quantizer

        Args:
            base_quantizer: Base quantizer to use for non-sensitive regions
            sensitivity_threshold: Threshold for GELU sensitivity
        """
        self.base_quantizer = base_quantizer
        self.sensitivity_threshold = sensitivity_threshold

    def quantize(self, weights: np.ndarray, activations: np.ndarray,
                hessian: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Quantize weights accounting for GELU amplification effects

        Args:
            weights: Weight matrix to quantize
            activations: Activation data for sensitivity analysis
            hessian: Optional Hessian diagonal for importance weighting

        Returns:
            Tuple of (quantized_weights, metrics)
        """
        # Identify GELU-sensitive weights
        sensitive_mask = self._identify_sensitive_weights(weights, activations, hessian)

        # Apply different quantization strategies
        quantized_weights = self._apply_gelu_aware_quantization(weights, sensitive_mask)

        # Compute metrics
        metrics = self._compute_quantization_metrics(weights, quantized_weights, sensitive_mask)

        return quantized_weights, metrics

    def _identify_sensitive_weights(self, weights: np.ndarray, activations: np.ndarray,
                                 hessian: np.ndarray = None) -> np.ndarray:
        """
        Identify weights in GELU-sensitive regions
        """
        # Weights near zero are most sensitive to sign flips
        weight_magnitudes = np.abs(weights)

        # Compute activation variance for each weight dimension
        activation_variance = np.var(activations, axis=0)

        # Combined sensitivity score
        # Sensitivity = weight_magnitude * activation_variance
        # Weights with small magnitude but high activation variance are most sensitive
        sensitivity = weight_magnitudes * activation_variance

        # Apply Hessian weighting if provided
        if hessian is not None:
            # Ensure Hessian matches dimensions
            if len(hessian) == activations.shape[1]:
                sensitivity *= hessian
            elif len(hessian) == 1:
                sensitivity *= hessian[0]
            else:
                # Use mean Hessian if dimensions don't match
                sensitivity *= np.mean(hessian)

        # Create sensitivity mask
        sensitive_mask = sensitivity < self.sensitivity_threshold

        return sensitive_mask

    def _apply_gelu_aware_quantization(self, weights: np.ndarray,
                                     sensitive_mask: np.ndarray) -> np.ndarray:
        """
        Apply different quantization strategies based on sensitivity
        """
        quantized_weights = np.zeros_like(weights)

        # High precision for sensitive weights
        sensitive_indices = np.where(sensitive_mask)
        quantized_weights[sensitive_indices] = self._high_precision_quantize(
            weights[sensitive_indices]
        )

        # Standard quantization for non-sensitive weights
        non_sensitive_indices = np.where(~sensitive_mask)
        quantized_weights[non_sensitive_indices] = self.base_quantizer(
            weights[non_sensitive_indices]
        )

        return quantized_weights

    def _high_precision_quantize(self, weights: np.ndarray) -> np.ndarray:
        """
        High precision quantization for sensitive weights
        """
        # Use 8-bit quantization for sensitive regions
        # Scale to [-1, 1] range
        max_val = np.max(np.abs(weights))
        if max_val > 1e-6:
            scaled = weights / max_val
            quantized = np.round(scaled * 127)  # 7 bits + sign
            return quantized / 127 * max_val
        else:
            return weights

    def _compute_quantization_metrics(self, original: np.ndarray, quantized: np.ndarray,
                                   sensitive_mask: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive quantization metrics
        """
        # Overall metrics
        overall_corr = np.corrcoef(original.flatten(), quantized.flatten())[0, 1]
        overall_mse = np.mean((original - quantized) ** 2)

        # Sensitive region metrics
        sensitive_orig = original[sensitive_mask]
        sensitive_quant = quantized[sensitive_mask]

        if len(sensitive_orig) > 0:
            sensitive_corr = np.corrcoef(sensitive_orig, sensitive_quant)[0, 1]
            sensitive_mse = np.mean((sensitive_orig - sensitive_quant) ** 2)
            sensitive_count = len(sensitive_orig)
        else:
            sensitive_corr = 0.0
            sensitive_mse = 0.0
            sensitive_count = 0

        # Non-sensitive region metrics
        non_sensitive_orig = original[~sensitive_mask]
        non_sensitive_quant = quantized[~sensitive_mask]

        if len(non_sensitive_orig) > 0:
            non_sensitive_corr = np.corrcoef(non_sensitive_orig, non_sensitive_quant)[0, 1]
            non_sensitive_mse = np.mean((non_sensitive_orig - non_sensitive_quant) ** 2)
            non_sensitive_count = len(non_sensitive_orig)
        else:
            non_sensitive_corr = 0.0
            non_sensitive_mse = 0.0
            non_sensitive_count = 0

        # Sensitivity analysis
        sensitivity_ratio = sensitive_count / len(original.flatten()) if len(original.flatten()) > 0 else 0

        return {
            "overall": {
                "correlation": overall_corr,
                "mse": overall_mse,
                "total_params": len(original.flatten())
            },
            "sensitive_region": {
                "correlation": sensitive_corr,
                "mse": sensitive_mse,
                "param_count": sensitive_count,
                "ratio": sensitivity_ratio
            },
            "non_sensitive_region": {
                "correlation": non_sensitive_corr,
                "mse": non_sensitive_mse,
                "param_count": non_sensitive_count,
                "ratio": 1.0 - sensitivity_ratio
            },
            "sensitivity_analysis": {
                "sensitive_ratio": sensitivity_ratio,
                "threshold": self.sensitivity_threshold,
                "method": "gelu_aware"
            }
        }

    def adaptive_gelu_quantization(self, weights: np.ndarray, activations: np.ndarray,
                                 hessian: np.ndarray = None,
                                 threshold_range: Tuple[float, float] = (0.05, 0.2)) -> Dict[str, Any]:
        """
        Adaptive GELU-aware quantization with threshold optimization

        Args:
            weights: Weight matrix to quantize
            activations: Activation data
            hessian: Optional Hessian diagonal
            threshold_range: Range of sensitivity thresholds to test

        Returns:
            Best quantization results
        """
        best_score = -np.inf
        best_result = None
        best_threshold = self.sensitivity_threshold

        # Test different thresholds
        thresholds = np.linspace(threshold_range[0], threshold_range[1], 5)

        for threshold in thresholds:
            # Update threshold
            self.sensitivity_threshold = threshold

            # Quantize with this threshold
            quantized_weights, metrics = self.quantize(weights, activations, hessian)

            # Score based on sensitive region quality
            score = metrics["sensitive_region"]["correlation"]

            # Update best if this is better
            if score > best_score:
                best_score = score
                best_result = {
                    "quantized_weights": quantized_weights,
                    "metrics": metrics,
                    "threshold": threshold
                }
                best_threshold = threshold

        return {
            "best_result": best_result,
            "best_threshold": best_threshold,
            "best_score": best_score,
            "thresholds_tested": list(thresholds)
        }

def create_gelu_aware_quantizer(base_quantizer: Callable, sensitivity_threshold: float = 0.1) -> GELUAwareQuantizer:
    """Factory function"""
    return GELUAwareQuantizer(base_quantizer, sensitivity_threshold)

# Example usage
if __name__ == "__main__":
    print("GELUAwareQuantizer ready for use")
    print("Usage: quantizer = create_gelu_aware_quantizer(base_quantizer)")
    print("       quantized_weights, metrics = quantizer.quantize(weights, activations)")