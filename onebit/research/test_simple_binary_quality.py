"""Test true binary GPT-2 quality on actual language modeling tasks.

This measures perplexity on real text, comparing:
1. FP32 teacher
2. Naive binary (sign * global scale)
3. Binary with Row+Col scales
4. Binary with LowRank magnitude
"""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


def test_binary_perplexity():
    """Test binary quantization on actual language modeling."""
    print("=" * 70)
    print("BINARY GPT-2 PERPLEXITY TEST (TRUE LM QUALITY)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model and tokenizer
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Load some test text
    print("Loading WikiText-2 test set...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_texts = [t for t in dataset['text'][:100] if len(t) > 50][:20]
    except:
        print("WikiText-2 not available, using synthetic test data")
        test_texts = [
            "The transformer architecture has revolutionized natural language processing and machine learning. These models use self-attention mechanisms to process sequences in parallel, which makes them highly efficient for training on modern hardware.",
            "In the study of artificial intelligence, neural networks play a crucial role. They consist of layers of interconnected neurons that can learn complex patterns from data through backpropagation.",
            "The stock market experienced significant volatility today as investors reacted to economic news. Technology stocks led the decline while defensive sectors showed resilience.",
        ]
    
    print(f"Testing on {len(test_texts)} text samples")
    
    def compute_perplexity(model):
        """Compute average perplexity on test texts."""
        losses = []
        for text in test_texts:
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=128, truncation=True).to(device)
            if tokens.size(1) < 2:
                continue
            with torch.no_grad():
                outputs = model(tokens, labels=tokens)
                losses.append(outputs.loss.item())
        return np.exp(np.mean(losses)) if losses else float('inf')
    
    # 1. FP32 baseline
    print("\n1. FP32 Teacher:")
    fp32_ppl = compute_perplexity(model)
    print(f"   Perplexity: {fp32_ppl:.2f}")
    
    # Save original weights
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # 2. Naive binary (sign * global scale)
    print("\n2. Naive Binary (sign * global scale):")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            W = original_state[name]
            scale = W.abs().mean()
            sign = torch.sign(W)
            sign[sign == 0] = 1
            param.data = sign * scale
    
    binary_ppl = compute_perplexity(model)
    print(f"   Perplexity: {binary_ppl:.2f}")
    print(f"   vs FP32: {(binary_ppl / fp32_ppl - 1) * 100:+.1f}%")
    
    # 3. Binary with Row+Col scales
    print("\n3. Binary with Row+Col scales:")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            W = original_state[name]
            sign = torch.sign(W)
            sign[sign == 0] = 1
            
            # Row+Col magnitude approximation
            W_abs = W.abs()
            row_scales = W_abs.mean(dim=1, keepdim=True)
            col_scales = W_abs.mean(dim=0, keepdim=True)
            global_scale = W_abs.mean()
            
            # Rank-1 approximation
            magnitude = (row_scales @ col_scales) / global_scale
            param.data = sign * magnitude
    
    rowcol_ppl = compute_perplexity(model)
    print(f"   Perplexity: {rowcol_ppl:.2f}")
    print(f"   vs FP32: {(rowcol_ppl / fp32_ppl - 1) * 100:+.1f}%")
    
    # 4. Binary with LowRank magnitude (r=2)
    print("\n4. Binary with LowRank magnitude (r=2):")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            W = original_state[name]
            sign = torch.sign(W)
            sign[sign == 0] = 1
            
            # SVD of magnitude matrix
            W_abs = W.abs()
            U, S, Vh = torch.linalg.svd(W_abs, full_matrices=False)
            r = 2
            magnitude = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
            param.data = sign * magnitude
    
    lowrank_ppl = compute_perplexity(model)
    print(f"   Perplexity: {lowrank_ppl:.2f}")
    print(f"   vs FP32: {(lowrank_ppl / fp32_ppl - 1) * 100:+.1f}%")
    
    # Restore original weights
    for name, param in model.named_parameters():
        param.data = original_state[name]
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Binary GPT-2 Perplexity Results")
    print("=" * 70)
    print(f"{'Method':<25} {'PPL':>10} {'vs FP32':>12}")
    print("-" * 50)
    print(f"{'FP32 (baseline)':<25} {fp32_ppl:>10.2f} {'---':>12}")
    print(f"{'Binary (naive)':<25} {binary_ppl:>10.2f} {(binary_ppl/fp32_ppl-1)*100:>+11.1f}%")
    print(f"{'Binary + Row+Col':<25} {rowcol_ppl:>10.2f} {(rowcol_ppl/fp32_ppl-1)*100:>+11.1f}%")
    print(f"{'Binary + LowRank r=2':<25} {lowrank_ppl:>10.2f} {(lowrank_ppl/fp32_ppl-1)*100:>+11.1f}%")


if __name__ == '__main__':
    test_binary_perplexity()

