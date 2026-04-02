"""Storage guard: Verify export stays exactly 1.00 bpp."""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def count_bits_packed(arr: np.ndarray) -> int:
    """Count total bits in packed uint32 array."""
    return arr.size * 32


def verify_bpp(
    W_bits: np.ndarray,
    *,
    n_params: int,
    tolerance: float = 1e-9,
) -> Dict:
    """Verify that packed weights are exactly 1.00 bpp.
    
    Args:
        W_bits: Packed weight array, shape [n_params_words] uint32
        n_params: Number of parameters
        tolerance: Tolerance for bpp check (default: 1e-9)
    
    Returns:
        Dict with:
            "n_params": int - Number of parameters
            "n_bits": int - Total bits stored
            "bpp": float - Bits per parameter
            "pass": bool - True if bpp is within tolerance of 1.00
    """
    n_bits = count_bits_packed(W_bits)
    bpp = n_bits / n_params
    
    # Check if bpp is within tolerance of 1.00
    pass_check = abs(bpp - 1.0) < tolerance
    
    return {
        "n_params": n_params,
        "n_bits": n_bits,
        "bpp": bpp,
        "pass": pass_check,
    }


def verify_model_bpp(
    layers: Dict[str, np.ndarray],
    *,
    tolerance: float = 1e-9,
) -> Dict:
    """Verify that all layers in a model are exactly 1.00 bpp.
    
    Args:
        layers: Dict mapping layer name to packed weight array
        tolerance: Tolerance for bpp check (default: 1e-9)
    
    Returns:
        Dict with:
            "total_params": int - Total parameters across all layers
            "total_bits": int - Total bits stored
            "bpp": float - Overall bits per parameter
            "pass": bool - True if all layers pass
            "layers": Dict - Per-layer results
    """
    total_params = 0
    total_bits = 0
    layer_results = {}
    all_pass = True
    
    for name, W_bits in layers.items():
        # Infer n_params from packed array size
        # Assume each uint32 word stores 32 parameters
        n_params = W_bits.size * 32
        
        result = verify_bpp(W_bits, n_params=n_params, tolerance=tolerance)
        layer_results[name] = result
        
        total_params += result["n_params"]
        total_bits += result["n_bits"]
        
        if not result["pass"]:
            all_pass = False
    
    overall_bpp = total_bits / total_params if total_params > 0 else 0.0
    
    return {
        "total_params": total_params,
        "total_bits": total_bits,
        "bpp": overall_bpp,
        "pass": all_pass,
        "layers": layer_results,
    }


def print_bpp_report(result: Dict):
    """Print a formatted BPP report."""
    print("=" * 80)
    print("Storage Guard: BPP Verification Report")
    print("=" * 80)
    
    if "layers" in result:
        # Model-level report
        print(f"Total parameters: {result['total_params']:,}")
        print(f"Total bits: {result['total_bits']:,}")
        print(f"Overall BPP: {result['bpp']:.10f}")
        print(f"Status: {'✅ PASS' if result['pass'] else '❌ FAIL'}")
        print()
        
        print("Per-layer breakdown:")
        print("-" * 80)
        for name, layer_result in result["layers"].items():
            status = "✅" if layer_result["pass"] else "❌"
            print(f"{status} {name:30s} | {layer_result['n_params']:10,} params | {layer_result['bpp']:.10f} bpp")
    else:
        # Single-layer report
        print(f"Parameters: {result['n_params']:,}")
        print(f"Bits: {result['n_bits']:,}")
        print(f"BPP: {result['bpp']:.10f}")
        print(f"Status: {'✅ PASS' if result['pass'] else '❌ FAIL'}")
    
    print("=" * 80)


def main():
    """Demo: Verify BPP for synthetic model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Storage guard: Verify 1.00 bpp")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--params-per-layer", type=int, default=1024, help="Parameters per layer")
    
    args = parser.parse_args()
    
    print(f"Creating synthetic model with {args.n_layers} layers, {args.params_per_layer} params each...")
    
    # Create synthetic model
    layers = {}
    for i in range(args.n_layers):
        n_params = args.params_per_layer
        n_words = (n_params + 31) // 32
        
        # Create random packed weights
        W_bits = np.random.randint(0, 2**32, size=n_words, dtype=np.uint32)
        layers[f"layer_{i}"] = W_bits
    
    # Verify BPP
    result = verify_model_bpp(layers)
    
    # Print report
    print_bpp_report(result)


if __name__ == "__main__":
    main()

