import numpy as np
import argparse
import sys
from pathlib import Path
from onebit.model.quantize_gpt2 import load_quantized_model, GPT2Config
from onebit.model.runtime_transformer import RuntimeTransformer, InferenceConfig
from onebit.runtime.ctg_grammar import CTG, CTGState, CTGRule, make_default_programs
from transformers import GPT2Tokenizer

# Force UTF-8 for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to quantized model .npz")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt")
    parser.add_argument("--T", type=int, default=32, help="Compute budget (ticks)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hcl", action="store_true", help="Use HCL logits")
    parser.add_argument("--ctg", action="store_true", help="Enable CTG")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    model_path = Path(args.model)
    print(f"Loading model from {model_path}...")
    
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found.")
        return
    
    model = load_quantized_model(model_path)
    print("Model loaded.")

    # Create runtime
    infer_cfg = InferenceConfig(
        T=args.T, 
        backend="cpu", 
        order=2, 
        beta=0.30, 
        lambd=0.0, 
        walsh_N=2, 
        antithetic=True,
        use_ctg=args.ctg,
        use_hcl_logits=args.hcl
    )
    print(f"Creating runtime (T={infer_cfg.T})...")
    runtime = RuntimeTransformer(model, infer_cfg)
    
    # Initialize CTG if enabled
    ctg_engine = None
    ctg_state = CTGState()
    if args.ctg:
        print("Initializing CTG Engine...")
        # Define a custom program to inhibit common loop tokens
        # Punctuation: . (13), , (11), \n (198), etc.
        loop_tokens = np.array([13, 11, 198, 0], dtype=np.int32)
        
        # Program 0: PASS (Base)
        prog0 = [CTGRule(op="PASS")]
        
        # Program 1: Anti-Loop (Inhibit punctuation with high probability)
        # prob_num=3, prob_den=4 -> 75% duty cycle active
        prog1 = [
            CTGRule(op="INHIBIT", ids=loop_tokens, period=4, prob_num=3, prob_den=4)
        ]
        
        ctg_engine = CTG(programs=[prog0, prog1], vocab_size=model.config.vocab_size)
    
    print(f"Prompt: '{args.prompt}'")
    
    input_ids = tokenizer.encode(args.prompt)
    print(f"Input IDs: {input_ids}")
    
    curr_input = np.array(input_ids, dtype=np.int32)
    
    print("-" * 40)
    print("Generating 10 tokens...")
    
    # Apply a repetition penalty manually
    generated_tokens = []
    
    for i in range(10):
        # Forward pass
        logits = runtime.forward(curr_input, seed=args.seed+i) # [vocab_size]
        
        # --- Repetition Penalty (Still useful) ---
        for token_id in generated_tokens:
            logits[token_id] -= 5.0  # Strong penalty
            
        # Also penalize prompt tokens slightly less
        for token_id in input_ids:
             logits[token_id] -= 2.0
        
        # --- CTG Logic ---
        if ctg_engine is not None:
            # 1. Shortlist (Top-K)
            K = 32
            shortlist_idx = np.argsort(logits)[-K:]
            shortlist_idx = shortlist_idx.astype(np.int32) # Ensure correct type for CTG
            
            # 2. Apply CTG
            # Use Program 1 (Anti-Loop) for now
            ctg_state, mask, invert_flag = ctg_engine.apply(
                ctg_state, shortlist_idx, program_id=1
            )
            
            # 3. Filter
            valid_mask = (mask == 1)
            if np.any(valid_mask):
                valid_idx = shortlist_idx[valid_mask]
            else:
                # Fallback if all inhibited (shouldn't happen with just punctuation inhibit)
                valid_idx = shortlist_idx
                
            # 4. Select
            logits_subset = logits[valid_idx]
            if invert_flag:
                logits_subset = -logits_subset
                
            # Greedy on modified subset
            best_local_idx = np.argmax(logits_subset)
            next_token = valid_idx[best_local_idx]
            
            # Debug CTG action
            # filtered_count = K - len(valid_idx)
            # if filtered_count > 0:
            #     print(f"[CTG] Inhibited {filtered_count} tokens")
        else:
            # Standard Greedy
            next_token = np.argmax(logits)

        generated_tokens.append(next_token)
        
        # Debug top 5 tokens
        top_k = 5
        top_idx = np.argsort(logits)[-top_k:][::-1]
        # print("  Top 5 candidates:")
        # for idx in top_idx:
        #     try:
        #         token_str = tokenizer.decode([idx])
        #         token_repr = repr(token_str)
        #         print(f"    {idx}: {token_repr:<20} score={logits[idx]:.4f}")
        #     except Exception:
        #         pass

        # Append
        curr_input = np.append(curr_input, next_token)
        
        # Print generation so far
        try:
            text = tokenizer.decode(curr_input)
            print(f"Gen: {text}")
        except Exception:
            print(f"Gen: <encoding error>")

if __name__ == "__main__":
    main()
