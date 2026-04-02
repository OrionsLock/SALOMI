#!/usr/bin/env python3
"""
Adaptive Block Sizing for SALOMI
Implements intelligent block size selection based on actual weight structure
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Callable
from tqdm import tqdm

class AdaptiveBlockSizer:
    """
    Adaptive block sizing that selects optimal block size based on weight structure
    """

    def __init__(self, min_block: int = 2, max_block: int = 8):
        """
        Initialize adaptive block sizer

        Args:
            min_block: Minimum block size to consider
            max_block: Maximum block size to consider
        """
        self.min_block = min_block
        self.max_block = max_block

    def find_optimal_block_size(self, weight_matrix: np.ndarray,
                               hessian: np.ndarray = None) -> Dict[str, Any]:
        """
        Find optimal block size for given weight matrix

        Args:
            weight_matrix: Weight matrix to analyze
            hessian: Optional Hessian diagonal for importance weighting

        Returns:
            Dictionary with optimal block size and analysis
        """
        analysis = {}

        # Test all block sizes
        for block_size in range(self.min_block, self.max_block + 1):
            block_analysis = self._analyze_block_size(weight_matrix, block_size, hessian)
            analysis[block_size] = block_analysis

        # Select best block size
        best_block, best_score = self._select_best_block_size(analysis)

        return {
            "analysis": analysis,
            "optimal_block_size": best_block,
            "optimal_score": best_score,
            "recommendation": self._generate_recommendation(best_block, analysis)
        }

    def _analyze_block_size(self, weights: np.ndarray, block_size: int,
                          hessian: np.ndarray = None) -> Dict[str, Any]:
        """
        Analyze how well weights fit a specific block size
        """
        h, w = weights.shape

        # Check if block size is compatible
        if h % block_size != 0 or w % block_size != 0:
            return {
                "compatible": False,
                "reason": "Incompatible dimensions"
            }

        # Calculate spatial correlation
        spatial_corr = self._compute_spatial_correlation(weights, block_size)

        # Calculate magnitude uniformity
        mag_uniformity = self._compute_magnitude_uniformity(weights, block_size)

        # Calculate Hessian-weighted importance (if Hessian provided)
        importance_score = 1.0
        if hessian is not None:
            importance_score = self._compute_importance_score(weights, hessian, block_size)

        # Combined score
        combined_score = 0.4 * spatial_corr + 0.4 * mag_uniformity + 0.2 * importance_score

        return {
            "compatible": True,
            "block_size": block_size,
            "spatial_correlation": spatial_corr,
            "magnitude_uniformity": mag_uniformity,
            "importance_score": importance_score,
            "combined_score": combined_score,
            "block_count": (h // block_size) * (w // block_size),
            "coverage": (block_size * block_size) / (h * w) if (h * w) > 0 else 0
        }

    def _compute_spatial_correlation(self, weights: np.ndarray, block_size: int) -> float:
        """
        Compute spatial correlation within blocks
        """
        h, w = weights.shape
        correlations = []

        # Iterate through all blocks
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = weights[i:i+block_size, j:j+block_size]

                # Only process complete blocks
                if block.shape == (block_size, block_size):
                    # Compute correlation within block
                    flat_block = block.flatten()

                    # Compute pairwise correlations
                    corr_matrix = np.corrcoef(flat_block)
                    avg_corr = np.mean(corr_matrix[np.triu_indices(block_size*block_size, k=1)])

                    correlations.append(avg_corr)

        return np.mean(correlations) if correlations else 0.0

    def _compute_magnitude_uniformity(self, weights: np.ndarray, block_size: int) -> float:
        """
        Compute magnitude uniformity within blocks
        """
        h, w = weights.shape
        magnitudes = np.abs(weights)
        uniformities = []

        # Iterate through all blocks
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = magnitudes[i:i+block_size, j:j+block_size]

                # Only process complete blocks
                if block.shape == (block_size, block_size):
                    # Compute coefficient of variation (std/mean)
                    block_std = np.std(block)
                    block_mean = np.mean(block)
                    cv = block_std / (block_mean + 1e-6)

                    # Uniformity is inverse of CV (lower CV = more uniform)
                    uniformity = 1.0 / (cv + 1.0)
                    uniformities.append(uniformity)

        return np.mean(uniformities) if uniformities else 0.0

    def _compute_importance_score(self, weights: np.ndarray, hessian: np.ndarray,
                                block_size: int) -> float:
        """
        Compute importance score using Hessian weighting
        """
        h, w = weights.shape
        importance_scores = []

        # Ensure Hessian matches dimensions
        if len(hessian) != w:
            # Reshape or repeat Hessian to match
            if len(hessian) == 1:
                hessian = np.tile(hessian, w)
            else:
                # Repeat last value
                hessian = np.pad(hessian, (0, w - len(hessian)), 'edge')

        # Iterate through all blocks
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = weights[i:i+block_size, j:j+block_size]

                # Only process complete blocks
                if block.shape == (block_size, block_size):
                    # Get corresponding Hessian values
                    hessian_block = hessian[j:j+block_size]

                    # Weight magnitudes by Hessian importance
                    weighted_magnitudes = np.abs(block) * hessian_block.reshape(1, -1)

                    # Importance is mean of weighted magnitudes
                    importance = np.mean(weighted_magnitudes)
                    importance_scores.append(importance)

        # Normalize importance scores
        if importance_scores:
            min_imp = min(importance_scores)
            max_imp = max(importance_scores)
            range_imp = max_imp - min_imp

            if range_imp > 1e-6:
                normalized = [(imp - min_imp) / range_imp for imp in importance_scores]
                return np.mean(normalized)

        return 0.5  # Default if no importance data

    def _select_best_block_size(self, analysis: Dict[int, Dict]) -> Tuple[int, float]:
        """
        Select best block size based on analysis
        """
        valid_analyses = [a for a in analysis.values() if a.get("compatible", False)]

        if not valid_analyses:
            # Fallback to middle size if none are compatible
            best_block = (self.min_block + self.max_block) // 2
            return best_block, 0.0

        # Find analysis with highest combined score
        best_analysis = max(valid_analyses, key=lambda x: x["combined_score"])
        return best_analysis["block_size"], best_analysis["combined_score"]

    def _generate_recommendation(self, best_block: int, analysis: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Generate recommendation based on analysis
        """
        # Get analysis for best block
        best_analysis = analysis.get(best_block, {})

        # Compare with other sizes
        comparisons = {}
        for block_size, block_analysis in analysis.items():
            if block_size != best_block and block_analysis.get("compatible", False):
                score_diff = best_analysis["combined_score"] - block_analysis["combined_score"]
                comparisons[block_size] = {
                    "score_difference": score_diff,
                    "percentage_better": (score_diff / block_analysis["combined_score"] * 100)
                    if block_analysis["combined_score"] > 1e-6 else 0
                }

        # Generate recommendation
        recommendation = {
            "optimal_block_size": best_block,
            "reason": self._explain_recommendation(best_analysis),
            "comparisons": comparisons,
            "confidence": self._calculate_confidence(best_analysis, analysis)
        }

        return recommendation

    def _explain_recommendation(self, analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for recommendation
        """
        spatial = analysis.get("spatial_correlation", 0)
        mag_unif = analysis.get("magnitude_uniformity", 0)
        importance = analysis.get("importance_score", 0)

        explanations = []

        if spatial > 0.7:
            explanations.append(f"high spatial correlation ({spatial:.3f})")
        elif spatial > 0.4:
            explanations.append(f"moderate spatial correlation ({spatial:.3f})")
        else:
            explanations.append(f"low spatial correlation ({spatial:.3f})")

        if mag_unif > 0.8:
            explanations.append(f"uniform magnitudes ({mag_unif:.3f})")
        elif mag_unif > 0.5:
            explanations.append(f"moderately uniform magnitudes ({mag_unif:.3f})")
        else:
            explanations.append(f"non-uniform magnitudes ({mag_unif:.3f})")

        if importance > 0.7:
            explanations.append(f"high importance weights ({importance:.3f})")
        elif importance > 0.4:
            explanations.append(f"moderate importance ({importance:.3f})")

        return "Block size selected due to " + ", ".join(explanations)

    def _calculate_confidence(self, best_analysis: Dict[str, Any],
                            all_analysis: Dict[int, Dict]) -> float:
        """
        Calculate confidence in recommendation
        """
        # Base confidence from best analysis
        base_confidence = best_analysis.get("combined_score", 0) * 0.8

        # Add confidence based on how much better it is than others
        other_scores = []
        for block_size, analysis in all_analysis.items():
            if block_size != best_analysis["block_size"] and analysis.get("compatible", False):
                other_scores.append(analysis.get("combined_score", 0))

        if other_scores:
            avg_other = np.mean(other_scores)
            if avg_other > 1e-6:
                relative_improvement = (best_analysis["combined_score"] - avg_other) / avg_other
                base_confidence += min(relative_improvement * 0.2, 0.2)  # Max 20% bonus

        # Ensure confidence is in [0, 1] range
        return max(0.0, min(1.0, base_confidence))

    def adaptive_block_quantization(self, weights: np.ndarray, quantizer: Callable,
                                  hessian: np.ndarray = None) -> Dict[str, Any]:
        """
        Perform adaptive block quantization using optimal block size

        Args:
            weights: Weight matrix to quantize
            quantizer: Quantizer function
            hessian: Optional Hessian diagonal

        Returns:
            Quantization results with adaptive block sizing
        """
        # Find optimal block size
        block_analysis = self.find_optimal_block_size(weights, hessian)
        optimal_block = block_analysis["optimal_block_size"]

        print(f"Using adaptive block size: {optimal_block}x{optimal_block}")

        # Quantize using optimal block size
        quantized_weights, quantization_info = quantizer(weights, block_size=optimal_block)

        return {
            "quantized_weights": quantized_weights,
            "quantization_info": quantization_info,
            "block_analysis": block_analysis,
            "optimal_block_size": optimal_block
        }

def create_adaptive_block_sizer(min_block: int = 2, max_block: int = 8) -> AdaptiveBlockSizer:
    """Factory function"""
    return AdaptiveBlockSizer(min_block, max_block)

# Example usage
if __name__ == "__main__":
    print("AdaptiveBlockSizer ready for use")
    print("Usage: sizer = create_adaptive_block_sizer()")
    print("       analysis = sizer.find_optimal_block_size(weights, hessian)")
    print("       optimal_block = analysis['optimal_block_size']")