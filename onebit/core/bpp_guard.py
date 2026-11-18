#!/usr/bin/env python3
"""
Accurate BPP Calculation for SALOMI
Fixes the BPP calculation to include ALL overhead (codebooks, metadata, etc.)

Phase-0 fixes:
  * Parameters are counted once per tensor.  Previously every ``add_*`` call
    added ``param_count`` again, so a tensor with signs + indices + data was
    counted 2-3x (halving/thirding the reported bpp).  Pass ``tensor=<name>``
    to the ``add_*`` methods to attach several components to one tensor;
    ``calculate_for_quantized_model`` does this automatically.
  * ``add_indices`` uses ``ceil(log2(n_codes))`` bits per index when the
    codebook size is known (instead of a fixed 8 bits regardless of K).
  * ``calculate_for_quantized_model`` counts unquantized parameters at their
    storage width (dtype size, or an explicit ``unquantized_bits``).
"""

import math
import numpy as np
from typing import Dict, Any, Union, List, Optional
import struct
import json


def count_bits_packed(arr: np.ndarray) -> int:
    """Count total bits in packed uint32 array."""
    return arr.size * 32


def index_bits_for(n_codes: int) -> int:
    """Fixed-width bits needed to address ``n_codes`` codebook entries."""
    if n_codes < 1:
        raise ValueError(f"n_codes must be >= 1, got {n_codes}")
    return max(1, math.ceil(math.log2(n_codes))) if n_codes > 1 else 0


class BPPCalculator:
    """
    Accurate bits-per-parameter calculator that includes all overhead
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters"""
        self.total_bits = 0
        self.total_params = 0
        self.component_bits = {}
        self._tensor_params: Dict[str, int] = {}
        self._anon_counter = 0

    def add_quantized_weights(self, quantized_data: Union[np.ndarray, List[int], bytes],
                             param_count: int, component: str = "weights",
                             tensor: Optional[str] = None):
        """
        Add quantized weight data to calculation

        Args:
            quantized_data: The quantized weight data
            param_count: Number of parameters this represents
            component: Component name for tracking
            tensor: Optional tensor id; params are counted once per tensor id
        """
        # Calculate bits
        if isinstance(quantized_data, np.ndarray):
            bits = quantized_data.nbytes * 8
        elif isinstance(quantized_data, (list, bytes)):
            if isinstance(quantized_data, list):
                # Assume list of integers
                bits = len(quantized_data) * 32  # 32 bits per int
            else:
                bits = len(quantized_data) * 8
        else:
            raise ValueError(f"Unsupported data type: {type(quantized_data)}")

        self._add_bits(bits, param_count, component, tensor)

    def add_codebook(self, codebook: Union[np.ndarray, List[float]], component: str = "codebook"):
        """
        Add codebook storage overhead
        """
        if isinstance(codebook, np.ndarray):
            bits = codebook.nbytes * 8
        elif isinstance(codebook, list):
            # Assume list of floats (64 bits each)
            bits = len(codebook) * 64
        else:
            raise ValueError(f"Unsupported codebook type: {type(codebook)}")

        # Codebooks are overhead, don't count toward params
        self.total_bits += bits
        self._track_component_bits("codebook", bits)

    def add_metadata(self, metadata: Dict[str, Any], component: str = "metadata"):
        """
        Add metadata storage overhead
        """
        # Serialize metadata to JSON and calculate size
        json_str = json.dumps(metadata)
        bits = len(json_str.encode('utf-8')) * 8

        self.total_bits += bits
        self._track_component_bits("metadata", bits)

    def add_indices(self, indices: Union[np.ndarray, List[int]], param_count: int,
                   bits_per_index: Optional[int] = None, component: str = "indices",
                   n_codes: Optional[int] = None, tensor: Optional[str] = None):
        """
        Add index storage for vector quantization

        Bits per index, in order of precedence:
          1. ``bits_per_index`` if given explicitly;
          2. ``ceil(log2(n_codes))`` if the codebook size ``n_codes`` is given;
          3. otherwise the actual storage width of ``indices`` (ndarray dtype
             size; 32 bits for a Python list).
        """
        if isinstance(indices, np.ndarray):
            index_count = indices.size
            storage_width = indices.dtype.itemsize * 8
        elif isinstance(indices, list):
            index_count = len(indices)
            storage_width = 32
        else:
            raise ValueError(f"Unsupported indices type: {type(indices)}")

        if bits_per_index is None:
            bits_per_index = index_bits_for(n_codes) if n_codes is not None else storage_width
        if not isinstance(bits_per_index, (int, np.integer)):
            raise TypeError(f"bits_per_index must be an int, got {type(bits_per_index).__name__}")

        bits = index_count * int(bits_per_index)
        self._add_bits(bits, param_count, component, tensor)

    def add_routing_bits(self, routing_data: Union[np.ndarray, List[int]],
                        param_count: int, component: str = "routing",
                        tensor: Optional[str] = None):
        """
        Add routing bit overhead
        """
        if isinstance(routing_data, np.ndarray):
            bit_count = routing_data.size
        elif isinstance(routing_data, list):
            bit_count = len(routing_data)
        else:
            raise ValueError(f"Unsupported routing data type: {type(routing_data)}")

        # Routing bits are typically 1 bit per decision
        bits = bit_count * 1
        self._add_bits(bits, param_count, component, tensor)

    def add_signs(self, signs: Union[np.ndarray, List[int]], param_count: int,
                 component: str = "signs", tensor: Optional[str] = None):
        """
        Add sign bit storage
        """
        if isinstance(signs, np.ndarray):
            bit_count = signs.size
        elif isinstance(signs, list):
            bit_count = len(signs)
        else:
            raise ValueError(f"Unsupported signs type: {type(signs)}")

        # Signs are typically 1 bit each
        bits = bit_count * 1
        self._add_bits(bits, param_count, component, tensor)

    def add_unquantized(self, param_count: int, bits_per_param: float,
                        component: str = "unquantized", tensor: Optional[str] = None):
        """Add a parameter tensor stored unquantized at ``bits_per_param`` bits each."""
        self._add_bits(param_count * bits_per_param, param_count, component, tensor)

    def _register_params(self, param_count: int, tensor: Optional[str]) -> None:
        """Count ``param_count`` once per tensor id (anonymous calls count every time)."""
        if tensor is None:
            self._anon_counter += 1
            tensor = f"__anon_{self._anon_counter}"
        prev = self._tensor_params.get(tensor)
        if prev is None:
            self._tensor_params[tensor] = param_count
            self.total_params += param_count
        elif prev != param_count:
            raise ValueError(
                f"Tensor {tensor!r} registered with {prev} params, now {param_count}"
            )

    def _add_bits(self, bits: int, param_count: int, component: str,
                  tensor: Optional[str] = None):
        """Internal method to add bits and track components"""
        self.total_bits += bits
        self._register_params(param_count, tensor)
        self._track_component_bits(component, bits)

    def _track_component_bits(self, component: str, bits: int):
        """Track bits by component"""
        if component not in self.component_bits:
            self.component_bits[component] = 0
        self.component_bits[component] += bits

    def calculate_bpp(self) -> float:
        """
        Calculate actual bits per parameter
        """
        if self.total_params == 0:
            return 0.0
        return self.total_bits / self.total_params

    def get_detailed_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed breakdown of BPP components
        """
        breakdown = {
            "total_bits": self.total_bits,
            "total_params": self.total_params,
            "bpp": self.calculate_bpp(),
            "component_breakdown": {},
            "component_percentages": {}
        }

        # Calculate component breakdown
        for component, bits in self.component_bits.items():
            percentage = (bits / self.total_bits * 100) if self.total_bits > 0 else 0
            breakdown["component_breakdown"][component] = {
                "bits": bits,
                "percentage": percentage
            }
            breakdown["component_percentages"][component] = percentage

        return breakdown

    def validate_quantization(self, target_bpp: float, tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Validate if quantization meets target BPP

        Args:
            target_bpp: Target bits per parameter
            tolerance: Allowed tolerance (e.g., 0.1 = 10%)

        Returns:
            Validation results
        """
        actual_bpp = self.calculate_bpp()
        ratio = actual_bpp / target_bpp
        within_tolerance = abs(ratio - 1.0) <= tolerance

        return {
            "target_bpp": target_bpp,
            "actual_bpp": actual_bpp,
            "ratio": ratio,
            "within_tolerance": within_tolerance,
            "tolerance": tolerance,
            "detailed_breakdown": self.get_detailed_breakdown()
        }

    def calculate_for_quantized_model(self, quantized_model,
                                      unquantized_bits: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate BPP for a complete quantized model

        Args:
            quantized_model: Quantized model with proper attributes
            unquantized_bits: bits/param for parameters without quantized
                attributes; default = their actual dtype width (e.g. 32 for fp32).

        Returns:
            Complete BPP analysis
        """
        self.reset()

        quant_attrs = ("quantized_data", "codebook", "metadata", "indices", "routing_bits", "signs")
        seen = set()

        # Iterate through all parameters (tied parameters are visited once)
        for name, param in quantized_model.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            param_count = param.numel()

            # Only attributes attached to this parameter instance count: torch.Tensor
            # has a built-in ``indices()`` method, so ``hasattr(param, "indices")``
            # is True for every tensor.
            attrs = {a: param.__dict__[a] for a in quant_attrs if a in getattr(param, "__dict__", {})}

            if not attrs:
                width = unquantized_bits if unquantized_bits is not None else param.element_size() * 8
                self.add_unquantized(param_count, width, f"{name}_unquantized", tensor=name)
                continue

            # Check for quantized attributes; params are registered once for this tensor.
            if "quantized_data" in attrs:
                self.add_quantized_weights(attrs["quantized_data"], param_count, f"{name}_weights", tensor=name)

            if "codebook" in attrs:
                self.add_codebook(attrs["codebook"], f"{name}_codebook")

            if "metadata" in attrs:
                self.add_metadata(attrs["metadata"], f"{name}_metadata")

            if "indices" in attrs:
                n_codes = len(attrs["codebook"]) if "codebook" in attrs else None
                self.add_indices(attrs["indices"], param_count, component=f"{name}_indices",
                                 n_codes=n_codes, tensor=name)

            if "routing_bits" in attrs:
                self.add_routing_bits(attrs["routing_bits"], param_count, f"{name}_routing", tensor=name)

            if "signs" in attrs:
                self.add_signs(attrs["signs"], param_count, f"{name}_signs", tensor=name)

            # A tensor with only codebook/metadata still has parameters.
            self._register_params(param_count, name)

        return self.get_detailed_breakdown()

def assert_bpp_one(W_bits: np.ndarray, n_params: int, tolerance: float = 1e-9) -> None:
    """Assert that packed weights are exactly 1.00 bpp.

    Args:
        W_bits: Packed weight array, shape [n_params_words] uint32
        n_params: Number of parameters
        tolerance: Tolerance for bpp check (default: 1e-9)

    Raises:
        AssertionError: If bpp is not within tolerance of 1.00
    """
    n_bits = count_bits_packed(W_bits)
    bpp = n_bits / n_params

    if abs(bpp - 1.0) >= tolerance:
        raise AssertionError(f"BPP check failed: {bpp:.10f} bpp (expected 1.00 ± {tolerance})")


def create_bpp_calculator() -> BPPCalculator:
    """Factory function"""
    return BPPCalculator()

# Example usage
if __name__ == "__main__":
    print("BPPCalculator ready for use")
    print("Usage: calculator = create_bpp_calculator()")
    print("       calculator.add_quantized_weights(quantized_data, param_count)")
    print("       actual_bpp = calculator.calculate_bpp()")
