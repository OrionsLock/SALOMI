"""Ingest official GPT-2 weights and quantize to 1-bit.

Usage:
    python -m onebit.cli.ingest_gpt2 [--model gpt2]
"""
import argparse
from pathlib import Path
from onebit.model.quantize_gpt2 import (
    load_gpt2_from_huggingface,
    quantize_gpt2,
    save_quantized_model,
    print_model_summary
)

def main():
    parser = argparse.ArgumentParser(description="Ingest GPT-2 weights")
    parser.add_argument("--model", type=str, default="gpt2", 
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: models/{model}-real-1bit.npz)")
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_path = Path(f"models/{args.model}-real-1bit.npz")
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ingesting {args.model}...")
    
    # 1. Load FP32 weights
    weights_fp32, cfg = load_gpt2_from_huggingface(args.model)
    
    # 2. Quantize
    print("Quantizing to 1-bit...")
    model = quantize_gpt2(weights_fp32, cfg)
    
    # 3. Print summary
    print_model_summary(model)
    
    # 4. Save
    print(f"Saving to {output_path}...")
    save_quantized_model(model, output_path)
    print("Done!")

if __name__ == "__main__":
    main()

