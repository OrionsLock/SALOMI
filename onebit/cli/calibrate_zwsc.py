"""CLI tool for Zero-Write Self-Calibration (ZWSC).

This tool collects calibration statistics from a token stream and creates
a calibration profile that can be used at runtime.

Usage:
    # Calibrate on WikiText-103
    python -m onebit.cli.calibrate_zwsc \\
        --input data/wikitext-103.txt \\
        --profile-name wikitext \\
        --n-tokens 50000 \\
        --output profiles/wikitext.npz

    # Calibrate on code domain
    python -m onebit.cli.calibrate_zwsc \\
        --input data/code_samples.txt \\
        --profile-name code \\
        --n-tokens 50000 \\
        --output profiles/code.npz \\
        --domain-tags code python

    # Test calibration effect
    python -m onebit.cli.calibrate_zwsc \\
        --input data/test.txt \\
        --profile-name test \\
        --n-tokens 10000 \\
        --test-mode \\
        --output profiles/test.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Tuple, Optional

import numpy as np

from ..core.calib_zwsc import (
    collect_stats,
    compute_affine_params,
    create_profile,
    CalibrationProfile,
    AffineParams,
)


def mock_activation_stream(
    n_tokens: int,
    n_layers: int = 12,
    d_model: int = 768,
    seed: int = 42,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Generate mock activation stream for testing.
    
    In a real implementation, this would:
    1. Load a tokenizer and model
    2. Tokenize input text
    3. Run forward pass and capture activations
    4. Yield (layer_id, input_activation, output_activation) tuples
    
    For now, we generate synthetic data with realistic statistics.
    
    Args:
        n_tokens: Number of tokens to generate
        n_layers: Number of layers
        d_model: Model dimension
        seed: Random seed
    
    Yields:
        (layer_id, input_activation, output_activation) tuples
    """
    rng = np.random.RandomState(seed)
    
    for token_idx in range(n_tokens):
        for layer_id in range(n_layers):
            # Generate input activation
            # Simulate layer-dependent statistics (deeper layers have different distributions)
            layer_scale = 1.0 + 0.1 * layer_id
            layer_shift = 0.5 * layer_id
            
            x_in = rng.randn(d_model).astype(np.float32) * layer_scale + layer_shift
            
            # Simulate layer transformation (e.g., attention + FFN)
            # In reality, this would be the actual model forward pass
            x_out = x_in * 1.2 + 0.3 * rng.randn(d_model).astype(np.float32)
            
            yield layer_id, x_in, x_out


def save_profile(profile: CalibrationProfile, output_path: Path):
    """Save calibration profile to disk.
    
    Args:
        profile: Calibration profile to save
        output_path: Output file path (.npz format)
    """
    # Prepare data for saving
    data = {
        "name": profile.name,
        "n_tokens": profile.n_tokens,
        "domain_tags": json.dumps(profile.domain_tags) if profile.domain_tags else "",
    }
    
    # Save affine parameters for each layer
    for layer_id, params in profile.affine_params.items():
        data[f"layer_{layer_id}_a"] = params.a
        data[f"layer_{layer_id}_b"] = params.b
    
    # Save to npz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    
    print(f"✓ Saved profile to {output_path}")
    print(f"  - Name: {profile.name}")
    print(f"  - Tokens: {profile.n_tokens}")
    print(f"  - Layers: {len(profile.affine_params)}")
    if profile.domain_tags:
        print(f"  - Tags: {', '.join(profile.domain_tags)}")


def load_profile(input_path: Path) -> CalibrationProfile:
    """Load calibration profile from disk.
    
    Args:
        input_path: Input file path (.npz format)
    
    Returns:
        Loaded calibration profile
    """
    data = np.load(input_path, allow_pickle=True)
    
    # Load metadata
    name = str(data["name"])
    n_tokens = int(data["n_tokens"])
    domain_tags_str = str(data.get("domain_tags", ""))
    domain_tags = json.loads(domain_tags_str) if domain_tags_str else None
    
    # Load affine parameters
    affine_params = {}
    layer_ids = set()
    
    for key in data.keys():
        if key.startswith("layer_") and key.endswith("_a"):
            layer_id = int(key.split("_")[1])
            layer_ids.add(layer_id)
    
    for layer_id in sorted(layer_ids):
        a = data[f"layer_{layer_id}_a"]
        b = data[f"layer_{layer_id}_b"]
        affine_params[layer_id] = AffineParams(a=a, b=b)
    
    return CalibrationProfile(
        name=name,
        affine_params=affine_params,
        n_tokens=n_tokens,
        domain_tags=domain_tags,
    )


def test_calibration_effect(
    profile: CalibrationProfile,
    test_stream: Iterator[Tuple[int, np.ndarray, np.ndarray]],
    n_test_tokens: int = 1000,
):
    """Test the effect of calibration on activation statistics.
    
    Args:
        profile: Calibration profile to test
        test_stream: Test activation stream
        n_test_tokens: Number of test tokens
    """
    from ..core.calib_zwsc import apply_affine, OnlineStats
    
    print("\n=== Testing Calibration Effect ===")
    
    # Collect statistics before and after calibration
    stats_before = {}
    stats_after = {}
    
    for layer_id, x_in, x_out in test_stream:
        if layer_id not in profile.affine_params:
            continue
        
        # Initialize stats trackers
        if layer_id not in stats_before:
            dim = x_out.shape[-1]
            stats_before[layer_id] = OnlineStats(dim)
            stats_after[layer_id] = OnlineStats(dim)
        
        # Update before stats
        stats_before[layer_id].update(x_out)
        
        # Apply calibration
        x_out_calibrated = apply_affine(x_out, profile.affine_params[layer_id])
        
        # Update after stats
        stats_after[layer_id].update(x_out_calibrated)
        
        if stats_before[layer_id].n >= n_test_tokens:
            break
    
    # Report statistics
    print(f"\nLayer | Before (μ ± σ) | After (μ ± σ)")
    print("-" * 60)
    
    for layer_id in sorted(stats_before.keys()):
        mu_before, sigma_before = stats_before[layer_id].get_stats()
        mu_after, sigma_after = stats_after[layer_id].get_stats()
        
        # Average across dimensions for reporting
        mu_before_avg = np.mean(mu_before)
        sigma_before_avg = np.mean(sigma_before)
        mu_after_avg = np.mean(mu_after)
        sigma_after_avg = np.mean(sigma_after)
        
        print(f"{layer_id:5d} | {mu_before_avg:6.3f} ± {sigma_before_avg:5.3f} | "
              f"{mu_after_avg:6.3f} ± {sigma_after_avg:5.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Write Self-Calibration (ZWSC) tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input/output
    parser.add_argument("--input", type=str, help="Input text file (currently unused - using mock data)")
    parser.add_argument("--output", type=str, required=True, help="Output profile path (.npz)")
    
    # Profile settings
    parser.add_argument("--profile-name", type=str, required=True, help="Profile name")
    parser.add_argument("--n-tokens", type=int, default=50000, help="Number of calibration tokens")
    parser.add_argument("--domain-tags", nargs="*", help="Domain tags (e.g., 'code', 'math')")
    
    # Model settings (for mock data)
    parser.add_argument("--n-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Calibration settings
    parser.add_argument("--target-mean", type=float, default=0.0, help="Target mean")
    parser.add_argument("--target-std", type=float, default=1.0, help="Target std dev")
    parser.add_argument("--mode", choices=["input", "output", "both"], default="output",
                        help="Which statistics to use")
    
    # Testing
    parser.add_argument("--test-mode", action="store_true", help="Test calibration effect")
    parser.add_argument("--n-test-tokens", type=int, default=1000, help="Number of test tokens")
    
    args = parser.parse_args()
    
    print(f"=== ZWSC Calibration ===")
    print(f"Profile: {args.profile_name}")
    print(f"Tokens: {args.n_tokens}")
    print(f"Layers: {args.n_layers}")
    print(f"Dimension: {args.d_model}")
    print(f"Target: μ={args.target_mean}, σ={args.target_std}")
    print(f"Mode: {args.mode}")
    if args.domain_tags:
        print(f"Tags: {', '.join(args.domain_tags)}")
    
    # Generate activation stream
    print(f"\nGenerating activation stream...")
    activation_stream = mock_activation_stream(
        n_tokens=args.n_tokens,
        n_layers=args.n_layers,
        d_model=args.d_model,
        seed=args.seed,
    )
    
    # Create profile
    print(f"Collecting statistics from {args.n_tokens} tokens...")
    profile = create_profile(
        name=args.profile_name,
        activation_stream=activation_stream,
        n_tokens=args.n_tokens,
        target_mean=args.target_mean,
        target_std=args.target_std,
        mode=args.mode,
        domain_tags=args.domain_tags,
    )
    
    print(f"✓ Collected statistics for {len(profile.affine_params)} layers")
    
    # Save profile
    output_path = Path(args.output)
    save_profile(profile, output_path)
    
    # Test mode
    if args.test_mode:
        print(f"\nGenerating test stream...")
        test_stream = mock_activation_stream(
            n_tokens=args.n_test_tokens,
            n_layers=args.n_layers,
            d_model=args.d_model,
            seed=args.seed + 1,  # Different seed for test
        )
        test_calibration_effect(profile, test_stream, args.n_test_tokens)
    
    print(f"\n✓ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

