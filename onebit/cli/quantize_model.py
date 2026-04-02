"""CLI for quantizing GPT-2 to 1-bit.

Usage:
    python -m onebit.cli.quantize_model --model gpt2 --output models/gpt2-1bit.npz
    python -m onebit.cli.quantize_model --model gpt2-medium --output models/gpt2-medium-1bit.npz
    python -m onebit.cli.quantize_model --mock --output models/gpt2-mock-1bit.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from onebit.model.quantize_gpt2 import (
    GPT2Config,
    create_mock_gpt2_weights,
    quantize_gpt2,
    save_quantized_model,
    load_quantized_model,
    print_model_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Quantize GPT-2 to 1-bit")
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "mock"],
        help="Model to quantize (default: gpt2)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for quantized model (.npz)",
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock weights instead of loading from HuggingFace",
    )
    
    parser.add_argument(
        "--test-load",
        action="store_true",
        help="Test loading the saved model",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Model configs
    configs = {
        "gpt2": GPT2Config(n_layers=12, n_heads=12, d_model=768, d_ff=3072),
        "gpt2-medium": GPT2Config(n_layers=24, n_heads=16, d_model=1024, d_ff=4096),
        "gpt2-large": GPT2Config(n_layers=36, n_heads=20, d_model=1280, d_ff=5120),
        "gpt2-xl": GPT2Config(n_layers=48, n_heads=25, d_model=1600, d_ff=6400),
        "mock": GPT2Config(n_layers=12, n_heads=12, d_model=768, d_ff=3072),
    }
    
    cfg = configs[args.model]
    
    print(f"Quantizing {args.model} to 1-bit...")
    print(f"Config: {cfg.n_layers} layers, {cfg.d_model} d_model")
    print()
    
    # Load or create weights
    if args.mock or args.model == "mock":
        print("Creating mock weights...")
        weights_fp32 = create_mock_gpt2_weights(cfg)
    else:
        # TODO: Load from HuggingFace
        print("Loading from HuggingFace not yet implemented, using mock weights...")
        weights_fp32 = create_mock_gpt2_weights(cfg)
    
    print(f"Loaded {len(weights_fp32)} weight tensors")
    print()
    
    # Quantize
    print("Quantizing to 1-bit...")
    model = quantize_gpt2(weights_fp32, cfg)
    print()
    
    # Print summary
    print_model_summary(model)
    print()
    
    # Save
    print(f"Saving to {output_path}...")
    save_quantized_model(model, output_path)
    print()
    
    # Test load
    if args.test_load:
        print("Testing load...")
        model_loaded = load_quantized_model(output_path)
        print("✅ Load successful!")
        print()
        print_model_summary(model_loaded)


if __name__ == "__main__":
    main()

