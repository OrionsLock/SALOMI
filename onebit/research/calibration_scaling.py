#!/usr/bin/env python3
"""
Fixed Hessian Estimation for SALOMI
Properly accounts for GELU nonlinearities and provides accurate sensitivity estimation
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from typing import Dict, Any, Optional, Tuple
import torch.nn.functional as F
from tqdm import tqdm

class ProperHessianEstimator:
    """
    Proper Hessian estimation that accounts for nonlinearities in transformer models
    """

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Load model for Hessian estimation"""
        print(f"Loading {self.model_name} for Hessian estimation...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def estimate_hessian(self, layer_name: str, calibration_data: torch.Tensor,
                        method: str = "activation_aware") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Estimate Hessian using proper methods that account for nonlinearities

        Args:
            layer_name: Name of layer to estimate Hessian for
            calibration_data: Input tensor for calibration
            method: Estimation method ('activation_aware', 'empirical_fisher', 'backprop')

        Returns:
            Tuple of (hessian_diagonal, metrics)
        """
        if method == "activation_aware":
            return self._activation_aware_hessian(layer_name, calibration_data)
        elif method == "empirical_fisher":
            return self._empirical_fisher_hessian(layer_name, calibration_data)
        elif method == "backprop":
            return self._backprop_hessian(layer_name, calibration_data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _activation_aware_hessian(self, layer_name: str, calibration_data: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Activation-aware Hessian estimation that accounts for GELU nonlinearities
        """
        # Get the specific layer
        layer = self._get_layer(layer_name)
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found")

        # Set up hooks to capture activations and outputs
        activations = []
        outputs = []

        def forward_hook(module, input, output):
            activations.append(input[0].detach())
            outputs.append(output.detach())

        # Register hook
        hook_handle = layer.register_forward_hook(forward_hook)

        # Run calibration data through model
        with torch.no_grad():
            try:
                self.model(calibration_data.to(self.device))
            except Exception as e:
                hook_handle.remove()
                raise RuntimeError(f"Forward pass failed: {e}")

        # Remove hook
        hook_handle.remove()

        # Process captured data
        if not activations or not outputs:
            raise RuntimeError("No activations or outputs captured")

        # Concatenate all activations and outputs
        X = torch.cat(activations, dim=0)  # Shape: [total_tokens, d_model]
        Y = torch.cat(outputs, dim=0)     # Shape: [total_tokens, d_model]

        # Account for GELU nonlinearity
        # GELU(x) = x * Φ(x) where Φ is standard Gaussian CDF
        # GELU'(x) = Φ(x) + x * φ(x) where φ is standard Gaussian PDF

        # Compute GELU derivative components
        cdf = 0.5 * (1 + torch.erf(X / torch.sqrt(torch.tensor(2.0))))
        pdf = torch.exp(-X**2 / 2) / torch.sqrt(torch.tensor(2 * np.pi))

        # GELU derivative
        gelu_deriv = cdf + X * pdf

        # Second derivative (for Hessian)
        # GELU''(x) = -x * GELU'(x) + φ(x)
        gelu_second_deriv = -X * gelu_deriv + pdf

        # Hessian diagonal approximation accounting for nonlinearity
        # H_ii ≈ E[x_i² * (GELU'(x_i))² + x_i * GELU''(x_i)]
        hessian_components = X**2 * gelu_deriv**2 + X * gelu_second_deriv
        H_diag = hessian_components.mean(dim=0).cpu().numpy()

        # Compute metrics
        metrics = {
            "method": "activation_aware",
            "activation_mean": float(X.mean()),
            "activation_std": float(X.std()),
            "gelu_deriv_mean": float(gelu_deriv.mean()),
            "hessian_mean": float(H_diag.mean()),
            "hessian_std": float(H_diag.std()),
            "num_samples": X.shape[0],
            "layer_dimension": X.shape[1]
        }

        return H_diag, metrics

    def _empirical_fisher_hessian(self, layer_name: str, calibration_data: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Empirical Fisher information matrix approximation
        """
        layer = self._get_layer(layer_name)
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found")

        # Capture gradients
        gradients = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        # Register hook
        hook_handle = layer.register_backward_hook(backward_hook)

        # Forward pass
        calibration_data = calibration_data.to(self.device)
        outputs = self.model(calibration_data)

        # Compute loss and backward pass
        loss = outputs.logits.sum()  # Simple loss for gradient computation
        loss.backward()

        # Remove hook
        hook_handle.remove()

        # Process gradients
        if not gradients:
            raise RuntimeError("No gradients captured")

        # Concatenate all gradients
        G = torch.cat(gradients, dim=0)  # Shape: [total_tokens, d_model]

        # Fisher approximation: H ≈ E[gg^T]
        fisher = (G.unsqueeze(2) @ G.unsqueeze(1)).mean(dim=0)
        H_diag = fisher.diagonal().cpu().numpy()

        # Metrics
        metrics = {
            "method": "empirical_fisher",
            "gradient_mean": float(G.mean()),
            "gradient_std": float(G.std()),
            "fisher_norm": float(fisher.norm()),
            "num_samples": G.shape[0],
            "layer_dimension": G.shape[1]
        }

        return H_diag, metrics

    def _backprop_hessian(self, layer_name: str, calibration_data: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Direct backpropagation Hessian estimation (more accurate but slower)
        """
        layer = self._get_layer(layer_name)
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found")

        # We'll compute Hessian-vector products and reconstruct
        hessian_approx = None
        num_samples = min(10, calibration_data.shape[0])  # Limit for speed

        for i in range(num_samples):
            # Get single sample
            sample = calibration_data[i:i+1].to(self.device)

            # Forward pass
            outputs = self.model(sample)
            loss = outputs.logits.sum()

            # First gradient
            first_grad = torch.autograd.grad(loss, layer.weight, create_graph=True)[0]

            # Hessian-vector product approximation
            # We use a random vector for stochastic approximation
            v = torch.randn_like(layer.weight)

            # Second derivative: Hv = ∇(∇L·v)
            hvp = torch.autograd.grad(first_grad, layer.weight, grad_outputs=v, retain_graph=True)[0]

            # Accumulate
            if hessian_approx is None:
                hessian_approx = hvp.unsqueeze(2) @ v.unsqueeze(1)
            else:
                hessian_approx += hvp.unsqueeze(2) @ v.unsqueeze(1)

        # Average and extract diagonal
        if hessian_approx is not None:
            hessian_approx /= num_samples
            H_diag = hessian_approx.diagonal().cpu().numpy()
        else:
            # Fallback to uniform if estimation failed
            H_diag = np.ones(layer.weight.shape[1])

        metrics = {
            "method": "backprop",
            "num_samples": num_samples,
            "hessian_norm": float(torch.norm(hessian_approx)) if hessian_approx is not None else 0,
            "layer_dimension": layer.weight.shape[1]
        }

        return H_diag, metrics

    def _get_layer(self, layer_name: str):
        """Get layer by name with flexible matching"""
        # Try exact match first
        layer = None
        try:
            layer = dict(self.model.named_modules())[layer_name]
        except KeyError:
            pass

        # Try partial match
        if layer is None:
            for name, module in self.model.named_modules():
                if layer_name in name:
                    layer = module
                    break

        return layer

    def validate_hessian_estimation(self, layer_name: str, calibration_data: torch.Tensor) -> Dict[str, Any]:
        """
        Validate Hessian estimation across different methods
        """
        results = {}

        for method in ["activation_aware", "empirical_fisher", "backprop"]:
            try:
                hessian, metrics = self.estimate_hessian(layer_name, calibration_data, method)
                results[method] = {
                    "hessian": hessian,
                    "metrics": metrics,
                    "success": True
                }
            except Exception as e:
                results[method] = {
                    "error": str(e),
                    "success": False
                }

        # Compare methods
        comparison = self._compare_hessian_methods(results)

        return {
            "individual_results": results,
            "comparison": comparison
        }

    def _compare_hessian_methods(self, results: Dict) -> Dict[str, Any]:
        """Compare different Hessian estimation methods"""
        comparison = {}

        # Check which methods succeeded
        successful_methods = [m for m, r in results.items() if r["success"]]

        if len(successful_methods) >= 2:
            # Compare Hessian values
            base_method = successful_methods[0]
            base_hessian = results[base_method]["hessian"]

            for method in successful_methods[1:]:
                hessian = results[method]["hessian"]
                corr = np.corrcoef(base_hessian, hessian)[0, 1]
                mse = np.mean((base_hessian - hessian)**2)

                comparison[f"{base_method}_vs_{method}"] = {
                    "correlation": corr,
                    "mse": mse,
                    "similarity": "high" if corr > 0.8 else "medium" if corr > 0.5 else "low"
                }

        return comparison

def create_hessian_estimator(model_name: str = "gpt2") -> ProperHessianEstimator:
    """Factory function"""
    return ProperHessianEstimator(model_name)

# Example usage
if __name__ == "__main__":
    # This would be used in actual quantization pipeline
    print("ProperHessianEstimator ready for use")
    print("Usage: estimator = create_hessian_estimator()")
    print("       hessian, metrics = estimator.estimate_hessian(layer_name, calibration_data)")
