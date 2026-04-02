#!/usr/bin/env python3
"""
Accurate BPP Calculation for SALOMI
Fixes the BPP calculation to include ALL overhead (codebooks, metadata, etc.)
"""

import numpy as np
from typing import Dict, Any, Union, List
import struct
import json

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

    def add_quantized_weights(self, quantized_data: Union[np.ndarray, List[int], bytes],
                             param_count: int, component: str = "weights"):
        """
        Add quantized weight data to calculation

        Args:
            quantized_data: The quantized weight data
            param_count: Number of parameters this represents
            component: Component name for tracking
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

        self._add_bits(bits, param_count, component)

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
                   bits_per_index: int = 8, component: str = "indices"):
        """
        Add index storage for vector quantization
        """
        if isinstance(indices, np.ndarray):
            index_count = indices.size
        elif isinstance(indices, list):
            index_count = len(indices)
        else:
            raise ValueError(f"Unsupported indices type: {type(indices)}")

        bits = index_count * bits_per_index
        self._add_bits(bits, param_count, component)

    def add_routing_bits(self, routing_data: Union[np.ndarray, List[int]],
                        param_count: int, component: str = "routing"):
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
        self._add_bits(bits, param_count, component)

    def add_signs(self, signs: Union[np.ndarray, List[int]], param_count: int,
                 component: str = "signs"):
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
        self._add_bits(bits, param_count, component)

    def _add_bits(self, bits: int, param_count: int, component: str):
        """Internal method to add bits and track components"""
        self.total_bits += bits
        self.total_params += param_count
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

    def calculate_for_quantized_model(self, quantized_model) -> Dict[str, Any]:
        """
        Calculate BPP for a complete quantized model

        Args:
            quantized_model: Quantized model with proper attributes

        Returns:
            Complete BPP analysis
        """
        self.reset()

        # Iterate through all parameters
        for name, param in quantized_model.named_parameters():
            param_count = param.numel()

            # Check for quantized attributes
            if hasattr(param, "quantized_data"):
                self.add_quantized_weights(param.quantized_data, param_count, f"{name}_weights")

            if hasattr(param, "codebook"):
                self.add_codebook(param.codebook, f"{name}_codebook")

            if hasattr(param, "metadata"):
                self.add_metadata(param.metadata, f"{name}_metadata")

            if hasattr(param, "indices"):
                self.add_indices(param.indices, param_count, f"{name}_indices")

            if hasattr(param, "routing_bits"):
                self.add_routing_bits(param.routing_bits, param_count, f"{name}_routing")

            if hasattr(param, "signs"):
                self.add_signs(param.signs, param_count, f"{name}_signs")

        return self.get_detailed_breakdown()

def create_bpp_calculator() -> BPPCalculator:
    """Factory function"""
    return BPPCalculator()

# Example usage
if __name__ == "__main__":
    print("BPPCalculator ready for use")
    print("Usage: calculator = create_bpp_calculator()")
    print("       calculator.add_quantized_weights(quantized_data, param_count)")
    print("       actual_bpp = calculator.calculate_bpp()")
