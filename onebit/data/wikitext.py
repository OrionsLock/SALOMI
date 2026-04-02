"""WikiText-103 dataset loader for SALOMI.

Provides tokenized WikiText-103 data for perplexity evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import numpy as np

DATASETS_AVAILABLE = False
try:
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    DATASETS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class WikiTextDataset:
    """WikiText-103 dataset with tokenization."""
    
    token_ids: np.ndarray  # [total_tokens] int32
    n_tokens: int
    vocab_size: int
    tokenizer_name: str = "gpt2"
    
    def get_batch(self, start_idx: int, seq_len: int) -> np.ndarray:
        """Get a batch of tokens.
        
        Args:
            start_idx: Starting token index
            seq_len: Sequence length
            
        Returns:
            Token IDs [seq_len] or shorter if at end
        """
        end_idx = min(start_idx + seq_len, self.n_tokens)
        return self.token_ids[start_idx:end_idx]
    
    def iter_batches(self, seq_len: int, stride: Optional[int] = None) -> Iterator[np.ndarray]:
        """Iterate over batches.
        
        Args:
            seq_len: Sequence length
            stride: Stride between batches (default: seq_len, no overlap)
            
        Yields:
            Token ID batches [seq_len] or shorter
        """
        if stride is None:
            stride = seq_len
        
        idx = 0
        while idx < self.n_tokens:
            batch = self.get_batch(idx, seq_len)
            if len(batch) > 0:
                yield batch
            idx += stride
    
    def save(self, output_path: Path):
        """Save dataset to disk."""
        np.savez_compressed(
            output_path,
            token_ids=self.token_ids,
            n_tokens=self.n_tokens,
            vocab_size=self.vocab_size,
            tokenizer_name=self.tokenizer_name,
        )
    
    @staticmethod
    def load(input_path: Path) -> WikiTextDataset:
        """Load dataset from disk."""
        data = np.load(input_path)
        return WikiTextDataset(
            token_ids=data["token_ids"],
            n_tokens=int(data["n_tokens"]),
            vocab_size=int(data["vocab_size"]),
            tokenizer_name=str(data["tokenizer_name"]),
        )


def load_wikitext103(
    split: str = "test",
    max_tokens: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> WikiTextDataset:
    """Load WikiText-103 dataset.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        max_tokens: Maximum number of tokens to load (None = all)
        cache_dir: Cache directory for HuggingFace datasets
        
    Returns:
        WikiTextDataset with tokenized data
        
    Raises:
        ImportError: If datasets or transformers not installed
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "datasets and transformers required for WikiText-103 loading. "
            "Install with: pip install datasets transformers"
        )
    
    # Load dataset
    print(f"Loading WikiText-103 {split} split...")
    dataset = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    
    # Load tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Tokenize all text
    print("Tokenizing...")
    all_token_ids = []
    
    for example in dataset:
        text = example["text"]
        if text.strip():  # Skip empty lines
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(tokens)
            
            if max_tokens and len(all_token_ids) >= max_tokens:
                all_token_ids = all_token_ids[:max_tokens]
                break
    
    token_ids = np.array(all_token_ids, dtype=np.int32)
    
    print(f"Loaded {len(token_ids):,} tokens")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    return WikiTextDataset(
        token_ids=token_ids,
        n_tokens=len(token_ids),
        vocab_size=tokenizer.vocab_size,
        tokenizer_name="gpt2",
    )


def load_wikitext103_cached(
    split: str = "test",
    max_tokens: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> WikiTextDataset:
    """Load WikiText-103 with caching.
    
    If cache_path exists, load from cache. Otherwise, download and cache.
    
    Args:
        split: Dataset split
        max_tokens: Maximum tokens
        cache_path: Path to cache file (.npz)
        
    Returns:
        WikiTextDataset
    """
    if cache_path and cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        return WikiTextDataset.load(cache_path)
    
    # Load from HuggingFace
    dataset = load_wikitext103(split=split, max_tokens=max_tokens)
    
    # Save to cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to cache: {cache_path}")
        dataset.save(cache_path)
    
    return dataset

