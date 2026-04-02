import argparse
import time
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from onebit.model.quantize_gpt2 import load_quantized_model, QuantizedGPT2
from onebit.model.runtime_transformer import RuntimeTransformer, InferenceConfig

def debug_logits_correlation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to quantized model")
    parser.add_argument("--T", type=int, default=16, help="Number of ticks")
    args = parser.parse_args()

    print("Loading resources...")
    # 1. Load Teacher (FP32)
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # 2. Load Student (1-bit)
    print(f"Loading SALOMI from {args.model}...")
    model_1bit = load_quantized_model(args.model)
    infer_cfg = InferenceConfig(
        T=args.T,
        backend="cpu",
        use_hcl_logits=False, 
        head_type="fp32",     # Use FP32 head to verify body quality
        walsh_N=2,
        antithetic=True,
        order=2
    )
    student = RuntimeTransformer(model_1bit, infer_cfg)

    # 3. Create a dummy input (short sequence)
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(text, return_tensors="pt")
    # Ensure length is multiple of 32 if needed, or just use what we have
    # Short sequence: 10 tokens.
    print(f"Input: '{text}'")
    print(f"Tokens: {tokens.shape}")

    # 4. Run Teacher
    with torch.no_grad():
        out_teacher = teacher(tokens)
        logits_fp32 = out_teacher.logits[0].numpy() # [Seq, V]

    # 5. Run Student
    print("Running Student...")
    start = time.time()
    input_ids_np = tokens[0].numpy()
    logits_1bit = student.forward(input_ids_np, return_all_logits=True)
    dt = time.time() - start
    print(f"Student forward took {dt:.2f}s")

    # 6. Analyze Correlation
    seq_len = logits_fp32.shape[0]
    vocab_size = logits_fp32.shape[1]
    
    print("\n=== Logits Analysis ===")
    
    # Global Stats
    print(f"FP32 Stats: Mean={logits_fp32.mean():.2f}, Std={logits_fp32.std():.2f}, Min={logits_fp32.min():.2f}, Max={logits_fp32.max():.2f}")
    print(f"1Bit Stats: Mean={logits_1bit.mean():.2f}, Std={logits_1bit.std():.2f}, Min={logits_1bit.min():.2f}, Max={logits_1bit.max():.2f}")
    
    # Per-token Correlation
    corrs = []
    top1_matches = 0
    top10_overlaps = 0
    
    for t in range(seq_len):
        l_fp = logits_fp32[t]
        l_1b = logits_1bit[t]
        
        # Pearson Correlation
        corr = np.corrcoef(l_fp, l_1b)[0, 1]
        corrs.append(corr)
        
        # Top-k
        top1_fp = np.argmax(l_fp)
        top1_1b = np.argmax(l_1b)
        if top1_fp == top1_1b:
            top1_matches += 1
            
        top10_fp = set(np.argsort(l_fp)[-10:])
        top10_1b = set(np.argsort(l_1b)[-10:])
        overlap = len(top10_fp.intersection(top10_1b))
        top10_overlaps += overlap
        
    avg_corr = np.mean(corrs)
    avg_overlap = top10_overlaps / seq_len
    acc = top1_matches / seq_len
    
    print(f"\nAverage Logit Correlation: {avg_corr:.4f}")
    print(f"Top-1 Accuracy vs FP32:    {acc:.4f}")
    print(f"Avg Top-10 Overlap:        {avg_overlap:.1f}/10")
    
    # Check Sign Flip?
    if avg_corr < 0:
        print("\nWARNING: Negative Correlation! Logits might be inverted.")
        
    # Check if broken
    if avg_corr < 0.1:
        print("\nCRITICAL: Correlation near zero. HCL head or features are scrambled.")

if __name__ == "__main__":
    debug_logits_correlation()

