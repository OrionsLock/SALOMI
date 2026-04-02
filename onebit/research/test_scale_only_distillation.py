"""Test if optimizing ONLY the scales (freeze binary signs) can help.

This tests whether the binary signs from teacher initialization are 
already good enough, and we just need to tune the scales.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np


def test_scale_optimization():
    """Test optimizing per-layer scales with frozen binary signs."""
    print("=" * 70)
    print("SCALE-ONLY OPTIMIZATION TEST")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load teacher
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    teacher.eval()
    
    # Get all weight matrices and their binary versions
    print("Binarizing weights...")
    binary_weights = {}
    scales = {}
    
    for name, param in teacher.named_parameters():
        if 'weight' in name and param.dim() == 2:
            W = param.data.clone()
            sign = torch.sign(W)
            sign[sign == 0] = 1
            
            binary_weights[name] = sign
            scales[name] = nn.Parameter(torch.tensor([W.abs().mean()], device=device))
    
    print(f"Found {len(binary_weights)} weight matrices to binarize")
    
    # Make scales trainable
    scale_params = list(scales.values())
    optimizer = torch.optim.Adam(scale_params, lr=0.01)
    
    # Training data
    train_texts = [
        "The transformer architecture has revolutionized natural language processing.",
        "Machine learning models require large amounts of training data.",
        "Neural networks consist of interconnected layers of neurons.",
        "Deep learning has achieved remarkable results in computer vision.",
        "Language models predict the probability of word sequences.",
    ] * 20
    
    eval_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming many industries.",
    ]
    
    def get_teacher_ppl(texts):
        """Get teacher perplexity."""
        losses = []
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
            if tokens.size(1) < 2:
                continue
            with torch.no_grad():
                outputs = teacher(tokens, labels=tokens)
                losses.append(outputs.loss.item())
        return np.exp(np.mean(losses))
    
    def apply_binary_weights():
        """Replace teacher weights with binary * scale."""
        for name, param in teacher.named_parameters():
            if name in binary_weights:
                param.data = binary_weights[name] * scales[name]
    
    def restore_weights():
        """Placeholder - we don't restore in this test."""
        pass
    
    # Initial PPL
    teacher_ppl = get_teacher_ppl(eval_texts)
    print(f"\nTeacher FP32 PPL: {teacher_ppl:.2f}")
    
    # Apply binary weights
    apply_binary_weights()
    binary_ppl = get_teacher_ppl(eval_texts)
    print(f"Binary (initial scales) PPL: {binary_ppl:.2f}")
    print(f"Gap: {(binary_ppl / teacher_ppl - 1) * 100:+.1f}%")
    
    # Train scales
    print("\nTraining scales...")
    for epoch in range(20):
        np.random.shuffle(train_texts)
        epoch_losses = []
        
        for text in train_texts:
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
            if tokens.size(1) < 2:
                continue
            
            # Apply current scales
            apply_binary_weights()
            
            # Forward pass
            outputs = teacher(tokens, labels=tokens)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            apply_binary_weights()
            ppl = get_teacher_ppl(eval_texts)
            print(f"Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}, "
                  f"PPL={ppl:.2f}, gap={(ppl/teacher_ppl-1)*100:+.1f}%")
    
    # Final results
    apply_binary_weights()
    final_ppl = get_teacher_ppl(eval_texts)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Teacher FP32 PPL: {teacher_ppl:.2f}")
    print(f"Binary (optimized scales) PPL: {final_ppl:.2f}")
    print(f"Gap: {(final_ppl / teacher_ppl - 1) * 100:+.1f}%")
    
    # Show some scale values
    print("\nSample scale values:")
    for i, (name, scale) in enumerate(list(scales.items())[:5]):
        print(f"  {name}: {scale.item():.4f}")


if __name__ == '__main__':
    test_scale_optimization()

