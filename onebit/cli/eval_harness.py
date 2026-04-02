"""SALOMI Evaluation Harness.

This script evaluates the 1-bit GPT-2 model against an FP32 baseline
across multiple modes (1-bit HCL, 1-bit HCL + CTG).

Metrics:
- Perplexity (PPL)
- Top-k Overlap (Top-1, Top-5, Top-10) with FP32 teacher
- Entropy Statistics (Teacher vs Student)
- CTG Intervention Rate & Effectiveness
"""
import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from onebit.model.quantize_gpt2 import load_quantized_model
from onebit.model.runtime_transformer import RuntimeTransformer, InferenceConfig
from onebit.runtime.ctg_grammar import CTG, CTGState, CTGRule

# Force UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class Evaluator:
    def __init__(self, model_path: str, device: str = "cpu"):
        print("Loading resources...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.device = device
        
        # 1. FP32 Teacher (HuggingFace)
        print("Loading FP32 Teacher...")
        self.teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.teacher.eval()
        
        # 2. 1-bit Student (SALOMI)
        print(f"Loading SALOMI from {model_path}...")
        self.quant_model = load_quantized_model(Path(model_path))
        
        # Initialize CTG Engine (Program 1: Anti-Loop)
        # Punctuation: . (13), , (11), \n (198)
        loop_tokens = np.array([13, 11, 198], dtype=np.int32)
        prog1 = [CTGRule(op="INHIBIT", ids=loop_tokens, period=4, prob_num=3, prob_den=4)]
        # Dummy prog0
        prog0 = [CTGRule(op="PASS")]
        self.ctg_engine = CTG(programs=[prog0, prog1], vocab_size=self.quant_model.config.vocab_size)
        
    def _get_batch_logits_teacher(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get teacher logits for a batch of sequences."""
        with torch.no_grad():
            outputs = self.teacher(input_ids)
            return outputs.logits # [B, Seq, Vocab]

    def run_eval(self, text_data: List[str], T: int = 64, stride: int = 64, limit: int = -1, max_tokens: int = 4096):
        """Run evaluation on text data."""
        
        # Setup 1-bit Runtime with HCL head + CTG enabled (logically)
        infer_cfg = InferenceConfig(
            T=T, backend="cpu", order=2, beta=0.30, lambd=0.0, 
            walsh_N=2, antithetic=True, use_ctg=True, 
            head_type="1bit", # Placeholder, HCL takes precedence
            use_hcl_logits=True 
        )
        student = RuntimeTransformer(self.quant_model, infer_cfg)
        
        # Metrics Accumulators
        nll_sum_fp32 = 0.0
        nll_sum_1bit = 0.0
        nll_sum_ctg = 0.0
        
        n_tokens = 0
        
        stats = {
            "top1_match_1bit": 0, "top1_match_ctg": 0,
            "top10_overlap_1bit": 0.0, "top10_overlap_ctg": 0.0,
            "entropy_fp32": 0.0, "entropy_1bit": 0.0,
            "ctg_interventions": 0,
            "ctg_helpful": 0, # New top-1 matches teacher when CTG changed it
            "ctg_harmful": 0, # New top-1 mismatch teacher when CTG changed it
        }
        
        print(f"Starting Eval (Stride={stride}, Limit={limit})...")
        
        # Tokenize all text
        full_encodings = self.tokenizer("\n\n".join(text_data), return_tensors="pt")
        total_len = full_encodings.input_ids.size(1)
        
        # Sliding window
        pbar = tqdm(range(0, total_len, stride))
        
        chunk_count = 0
        for i in pbar:
            if limit > 0 and chunk_count >= limit:
                break
            if max_tokens > 0 and n_tokens >= max_tokens:
                break
            
            # Get chunk
            end_loc = min(i + stride, total_len)
            input_ids = full_encodings.input_ids[:, i:end_loc].to(self.device) # [1, Seq]
            seq_len = input_ids.size(1)
            
            if seq_len < 2: continue
            
            # 1. Teacher Forward
            teacher_logits = self._get_batch_logits_teacher(input_ids) # [1, Seq, V]
            
            # 2. Student Forward
            # Runtime expects numpy [Seq]
            input_ids_np = input_ids[0].cpu().numpy()
            # Get all logits: [Seq, V]
            student_logits = student.forward(input_ids_np, return_all_logits=True)
            
            # Debug Logits
            if chunk_count == 0:
                print(f"\n[DEBUG] Chunk 0 Logits Stats:")
                print(f"  FP32: Mean={teacher_logits.mean():.4f}, Std={teacher_logits.std():.4f}, Min={teacher_logits.min():.4f}, Max={teacher_logits.max():.4f}")
                print(f"  1Bit: Mean={student_logits.mean():.4f}, Std={student_logits.std():.4f}, Min={student_logits.min():.4f}, Max={student_logits.max():.4f}")
            
            # Compare token by token (predicting next token)
            # input: x_0 ... x_{N-1}
            # target: x_1 ... x_N
            # logits[t] predicts x_{t+1}
            
            # We evaluate positions 0 to Seq-2 (predicting 1 to Seq-1)
            # Because student.forward(return_all=True) returns logits for all positions 0..Seq-1
            
            # Targets
            targets = input_ids[0, 1:].cpu().numpy() # [Seq-1]
            
            # Valid range
            n_steps = seq_len - 1
            
            # CTG State reset per chunk (simplification)
            ctg_state = CTGState()
            
            for t in range(n_steps):
                target_id = targets[t]
                
                # Teacher Logits
                l_fp32 = teacher_logits[0, t].cpu().numpy()
                probs_fp32 = self._softmax(l_fp32)
                nll_sum_fp32 += -np.log(probs_fp32[target_id] + 1e-10)
                
                # Student Logits (1-bit HCL)
                l_1bit = student_logits[t]
                probs_1bit = self._softmax(l_1bit)
                nll_sum_1bit += -np.log(probs_1bit[target_id] + 1e-10)
                
                # --- CTG Logic Simulation ---
                # We simulate CTG decision on top of 1-bit logits
                K = 32
                top_k_idx = np.argsort(l_1bit)[-K:].astype(np.int32)
                
                # Apply CTG (Program 1)
                new_state, mask, invert = self.ctg_engine.apply(ctg_state, top_k_idx, program_id=1)
                ctg_state = new_state
                
                # Filter & Select
                valid_mask = (mask == 1)
                if np.any(valid_mask):
                    valid_idx = top_k_idx[valid_mask]
                else:
                    valid_idx = top_k_idx
                
                # Re-eval best token under CTG
                # Note: To compute NLL_CTG properly we'd need a full probability distribution.
                # CTG makes the distribution sparse. 
                # For PPL, we just use the 1-bit NLL (CTG is for generation quality/sampling, hard to PPL).
                # But we CAN track top-1 match.
                
                l_ctg_subset = l_1bit[valid_idx]
                if invert: l_ctg_subset = -l_ctg_subset
                
                best_ctg = valid_idx[np.argmax(l_ctg_subset)]
                
                # --- Metrics ---
                
                # 1. Top-1 Match vs Teacher
                best_fp32 = np.argmax(l_fp32)
                best_1bit = np.argmax(l_1bit)
                
                stats["top1_match_1bit"] += (best_1bit == best_fp32)
                stats["top1_match_ctg"] += (best_ctg == best_fp32)
                
                # 2. Top-10 Overlap
                top10_fp32 = set(np.argsort(l_fp32)[-10:])
                top10_1bit = set(np.argsort(l_1bit)[-10:])
                stats["top10_overlap_1bit"] += len(top10_fp32 & top10_1bit) / 10.0
                
                # 3. Entropy
                stats["entropy_fp32"] += -np.sum(probs_fp32 * np.log(probs_fp32 + 1e-10))
                stats["entropy_1bit"] += -np.sum(probs_1bit * np.log(probs_1bit + 1e-10))
                
                # 4. CTG Stats
                if best_1bit != best_ctg:
                    stats["ctg_interventions"] += 1
                    if best_ctg == best_fp32:
                        stats["ctg_helpful"] += 1
                    else:
                        stats["ctg_harmful"] += 1
                
                n_tokens += 1
            
            chunk_count += 1
            
            # Checkpoint Log
            if i % (stride * 5) == 0 and n_tokens > 0:
                curr_ppl_fp32 = np.exp(nll_sum_fp32 / n_tokens)
                curr_ppl_1bit = np.exp(nll_sum_1bit / n_tokens)
                pbar.set_description(f"PPL: FP32={curr_ppl_fp32:.1f} 1Bit={curr_ppl_1bit:.1f}")
        
        # Final Report
        print("\n=== Evaluation Results ===")
        print(f"Tokens Evaluated: {n_tokens}")
        print(f"FP32 PPL:    {np.exp(nll_sum_fp32 / n_tokens):.2f}")
        print(f"1-bit PPL:   {np.exp(nll_sum_1bit / n_tokens):.2f}")
        print("-" * 30)
        print(f"Top-1 Match (1-bit): {stats['top1_match_1bit'] / n_tokens:.2%}")
        print(f"Top-1 Match (CTG):   {stats['top1_match_ctg'] / n_tokens:.2%}")
        print(f"Top-10 Overlap:      {stats['top10_overlap_1bit'] / n_tokens:.2%}")
        print("-" * 30)
        print(f"Avg Entropy (FP32):  {stats['entropy_fp32'] / n_tokens:.2f}")
        print(f"Avg Entropy (1-bit): {stats['entropy_1bit'] / n_tokens:.2f}")
        print("-" * 30)
        print(f"CTG Interventions:   {stats['ctg_interventions']} ({stats['ctg_interventions']/n_tokens:.2%})")
        if stats["ctg_interventions"] > 0:
            print(f"  Helpful: {stats['ctg_helpful']} ({stats['ctg_helpful']/stats['ctg_interventions']:.2%})")
            print(f"  Harmful: {stats['ctg_harmful']} ({stats['ctg_harmful']/stats['ctg_interventions']:.2%})")
            
        results = {
            "ppl_fp32": np.exp(nll_sum_fp32 / n_tokens),
            "ppl_1bit": np.exp(nll_sum_1bit / n_tokens),
            "top1_match_1bit": stats["top1_match_1bit"] / n_tokens,
            "top1_match_ctg": stats["top1_match_ctg"] / n_tokens,
            "top10_overlap": stats["top10_overlap_1bit"] / n_tokens,
            "entropy_1bit": stats["entropy_1bit"] / n_tokens
        }
        return results

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, default="wikitext_tiny.txt", help="Text file to eval on")
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of chunks")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens to eval with 1-bit")
    args = parser.parse_args()
    
    # Generate dummy wikitext if not exists
    if not Path(args.input).exists():
        print("Generating dummy wikitext sample...")
        # Use some real-ish text from imported module or simple string
        text = "The capital of France is Paris. " * 1000
        with open(args.input, "w", encoding="utf-8") as f:
            f.write(text)
    
    # Load text
    with open(args.input, "r", encoding="utf-8") as f:
        data = [f.read()]
        
    evaluator = Evaluator(args.model)
    evaluator.run_eval(data, T=args.T, limit=args.limit, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
