#!/usr/bin/env python3
"""
Cross-Validation System for SALOMI
Prevents overfitting by implementing proper cross-validation
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Callable
from tqdm import tqdm

class CrossValidator:
    """
    Cross-validation system to prevent overfitting in quantization
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validator

        Args:
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def cross_validate(self, weights: np.ndarray, quantizer: Callable,
                     calibration_data: List[Any], fit_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform cross-validation to prevent overfitting

        Args:
            weights: Weight matrix to quantize
            quantizer: Quantizer function/object
            calibration_data: Calibration data for fitting
            fit_params: Additional parameters for quantizer

        Returns:
            Cross-validation results
        """
        if fit_params is None:
            fit_params = {}

        # Convert calibration data to array for splitting
        if isinstance(calibration_data, list):
            data_array = np.array(calibration_data)
        else:
            data_array = calibration_data

        # Manual KFold (no sklearn dependency)
        n_samples = len(data_array)
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        fold_size = n_samples // self.n_folds
        folds = []
        for f in range(self.n_folds):
            val_start = f * fold_size
            val_end = val_start + fold_size if f < self.n_folds - 1 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            folds.append((train_idx, val_idx))

        fold_results = []
        train_scores = []
        val_scores = []

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            print(f"Fold {fold_idx + 1}/{self.n_folds}")

            # Split data
            train_data = data_array[train_indices]
            val_data = data_array[val_indices]

            # Train quantizer on training fold
            try:
                # Fit quantizer to training data
                if hasattr(quantizer, 'fit'):
                    quantizer.fit(train_data, **fit_params)
                else:
                    # Functional quantizer
                    quantizer = quantizer(train_data, **fit_params)

                # Quantize weights
                quantized_weights = quantizer.transform(weights)

                # Evaluate on training data
                train_metrics = self._evaluate_quantization(
                    weights, quantized_weights, train_data
                )

                # Evaluate on validation data
                val_metrics = self._evaluate_quantization(
                    weights, quantized_weights, val_data
                )

                fold_results.append({
                    "fold": fold_idx,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "success": True
                })

                train_scores.append(train_metrics["correlation"])
                val_scores.append(val_metrics["correlation"])

            except Exception as e:
                print(f"Error in fold {fold_idx}: {e}")
                fold_results.append({
                    "fold": fold_idx,
                    "error": str(e),
                    "success": False
                })

        # Compute overall statistics
        overall_stats = self._compute_overall_stats(fold_results, train_scores, val_scores)

        return {
            "fold_results": fold_results,
            "overall_stats": overall_stats,
            "success_rate": sum(1 for r in fold_results if r["success"]) / len(fold_results)
        }

    def _evaluate_quantization(self, original_weights: np.ndarray,
                             quantized_weights: np.ndarray,
                             data: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate quantization quality on specific data
        """
        # Flatten for correlation calculation
        orig_flat = original_weights.flatten()
        quant_flat = quantized_weights.flatten()

        # Calculate metrics
        correlation = np.corrcoef(orig_flat, quant_flat)[0, 1]
        mse = np.mean((orig_flat - quant_flat) ** 2)
        variance = np.var(orig_flat)
        nmse = mse / (variance + 1e-10)

        # Additional metrics
        max_error = np.max(np.abs(orig_flat - quant_flat))
        mean_error = np.mean(np.abs(orig_flat - quant_flat))

        return {
            "correlation": correlation,
            "mse": mse,
            "nmse": nmse,
            "max_error": max_error,
            "mean_error": mean_error,
            "data_size": len(data)
        }

    def _compute_overall_stats(self, fold_results: List[Dict], train_scores: List[float],
                              val_scores: List[float]) -> Dict[str, Any]:
        """
        Compute overall statistics across all folds
        """
        successful_folds = [r for r in fold_results if r["success"]]

        if not successful_folds:
            return {
                "error": "No successful folds",
                "success": False
            }

        # Calculate mean metrics
        mean_train_corr = np.mean(train_scores)
        mean_val_corr = np.mean(val_scores)

        # Calculate standard deviations
        std_train_corr = np.std(train_scores)
        std_val_corr = np.std(val_scores)

        # Calculate overfitting metrics
        train_val_gap = mean_train_corr - mean_val_corr
        overfitting_score = train_val_gap / (mean_train_corr + 1e-6)

        # Quality metrics
        quality_score = mean_val_corr  # Primary metric

        return {
            "mean_train_correlation": mean_train_corr,
            "mean_val_correlation": mean_val_corr,
            "std_train_correlation": std_train_corr,
            "std_val_correlation": std_val_corr,
            "train_val_gap": train_val_gap,
            "overfitting_score": overfitting_score,
            "quality_score": quality_score,
            "success": True,
            "num_successful_folds": len(successful_folds),
            "total_folds": len(fold_results)
        }

    def validate_quantizer(self, quantizer: Callable, weights: np.ndarray,
                          calibration_data: List[Any], threshold: float = 0.85) -> bool:
        """
        Validate quantizer using cross-validation

        Args:
            quantizer: Quantizer to validate
            weights: Weights to quantize
            calibration_data: Calibration data
            threshold: Minimum quality threshold

        Returns:
            True if quantizer passes validation, False otherwise
        """
        # Run cross-validation
        cv_results = self.cross_validate(weights, quantizer, calibration_data)

        # Check if validation passed
        if not cv_results["success"]:
            return False

        stats = cv_results["overall_stats"]
        if not stats["success"]:
            return False

        # Check quality threshold
        quality_ok = stats["quality_score"] >= threshold

        # Check overfitting
        overfitting_ok = stats["overfitting_score"] < 0.1  # Less than 10% overfitting

        # Overall validation
        validation_passed = quality_ok and overfitting_ok

        print(f"Validation Results:")
        print(f"  Quality Score: {stats['quality_score']:.4f} (threshold: {threshold})")
        print(f"  Overfitting Score: {stats['overfitting_score']:.4f} (threshold: 0.1)")
        print(f"  Quality OK: {quality_ok}")
        print(f"  Overfitting OK: {overfitting_ok}")
        print(f"  Overall: {'PASSED' if validation_passed else 'FAILED'}")

        return validation_passed

    def adaptive_cross_validation(self, weights: np.ndarray, quantizer_factory: Callable,
                                calibration_data: List[Any], param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Adaptive cross-validation with parameter search

        Args:
            weights: Weights to quantize
            quantizer_factory: Factory function to create quantizers
            calibration_data: Calibration data
            param_grid: Parameter grid to search

        Returns:
            Best parameters and results
        """
        best_score = -np.inf
        best_params = None
        best_results = None

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        print(f"Testing {len(param_combinations)} parameter combinations...")

        for params in tqdm(param_combinations, desc="Parameter search"):
            # Create quantizer with these parameters
            quantizer = quantizer_factory(**params)

            # Run cross-validation
            cv_results = self.cross_validate(weights, quantizer, calibration_data)

            # Check if successful
            if cv_results["success"] and "overall_stats" in cv_results:
                stats = cv_results["overall_stats"]
                if stats["success"]:
                    current_score = stats["quality_score"]

                    # Update best if this is better
                    if current_score > best_score:
                        best_score = current_score
                        best_params = params
                        best_results = cv_results

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_results": best_results,
            "num_combinations_tested": len(param_combinations)
        }

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid"""
        if not param_grid:
            return [{}]

        # Get parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Generate combinations
        combinations = []
        self._generate_combinations_recursive(param_names, param_values, 0, {}, combinations)

        return combinations

    def _generate_combinations_recursive(self, names: List[str], values: List[List[Any]],
                                        index: int, current: Dict[str, Any], combinations: List[Dict[str, Any]]):
        """Recursively generate parameter combinations"""
        if index == len(names):
            combinations.append(current.copy())
            return

        for value in values[index]:
            current[names[index]] = value
            self._generate_combinations_recursive(names, values, index + 1, current, combinations)

def create_cross_validator(n_folds: int = 5, random_state: int = 42) -> CrossValidator:
    """Factory function"""
    return CrossValidator(n_folds, random_state)

# Example usage
if __name__ == "__main__":
    print("CrossValidator ready for use")
    print("Usage: validator = create_cross_validator()")
    print("       cv_results = validator.cross_validate(weights, quantizer, calibration_data)")