"""Perplexity evaluation for language models.

Computes perplexity over a dataset using cross-entropy loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import time

from onebit.data.wikitext import WikiTextDataset


@dataclass
class PerplexityResult:
    """Perplexity evaluation result.
    
    Attributes:
        perplexity: Perplexity (exp(cross_entropy))
        cross_entropy: Average cross-entropy loss (bits per token)
        n_tokens: Number of tokens evaluated
        n_batches: Number of batches processed
        total_time: Total evaluation time (seconds)
        tokens_per_sec: Throughput (tokens/second)
    """
    perplexity: float
    cross_entropy: float
    n_tokens: int
    n_batches: int
    total_time: float
    tokens_per_sec: float


def compute_perplexity(
    dataset: WikiTextDataset,
    forward_fn: Callable[[np.ndarray], np.ndarray],
    seq_len: int = 512,
    stride: Optional[int] = None,
    max_tokens: Optional[int] = None,
    verbose: bool = True,
) -> PerplexityResult:
    """Compute perplexity over a dataset.
    
    Args:
        dataset: WikiText dataset
        forward_fn: Function that takes token_ids [seq_len] and returns logits [seq_len, vocab_size]
        seq_len: Sequence length for evaluation
        stride: Stride between batches (default: seq_len, no overlap)
        max_tokens: Maximum tokens to evaluate (None = all)
        verbose: Print progress
        
    Returns:
        PerplexityResult with metrics
    """
    if stride is None:
        stride = seq_len
    
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    start_time = time.time()
    
    for batch_idx, token_ids in enumerate(dataset.iter_batches(seq_len + 1, stride)):
        if len(token_ids) < 2:
            continue  # Need at least 2 tokens (input + target)
        
        # Split into input and targets
        input_ids = token_ids[:-1]  # [seq_len]
        target_ids = token_ids[1:]  # [seq_len]
        
        # Forward pass
        logits = forward_fn(input_ids)  # [seq_len, vocab_size] or [vocab_size] for last token
        
        # Handle different logits shapes
        if logits.ndim == 1:
            # Only last token logits returned
            logits = logits[np.newaxis, :]  # [1, vocab_size]
            target_ids = target_ids[-1:]  # [1]
        
        # Compute cross-entropy loss
        # loss = -log(p(target))
        for i, target_id in enumerate(target_ids):
            if i >= len(logits):
                break
            
            # Softmax
            logits_i = logits[i]
            logits_i = logits_i - np.max(logits_i)  # Numerical stability
            exp_logits = np.exp(logits_i)
            probs = exp_logits / np.sum(exp_logits)
            
            # Cross-entropy
            target_prob = probs[target_id]
            loss = -np.log(np.maximum(target_prob, 1e-10))  # Avoid log(0)
            
            total_loss += loss
            total_tokens += 1
        
        n_batches += 1
        
        if verbose and (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            ppl = np.exp(avg_loss)
            print(f"Batch {batch_idx + 1}: {total_tokens} tokens, PPL={ppl:.2f}")
        
        if max_tokens and total_tokens >= max_tokens:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Compute final metrics
    avg_cross_entropy = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_cross_entropy)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    
    if verbose:
        print(f"\nFinal Results:")
        print(f"  Tokens: {total_tokens:,}")
        print(f"  Batches: {n_batches}")
        print(f"  Cross-Entropy: {avg_cross_entropy:.4f} bits/token")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Throughput: {tokens_per_sec:.2f} tokens/s")
    
    return PerplexityResult(
        perplexity=perplexity,
        cross_entropy=avg_cross_entropy,
        n_tokens=total_tokens,
        n_batches=n_batches,
        total_time=total_time,
        tokens_per_sec=tokens_per_sec,
    )

