"""Streaming and local training data loaders for KD-QAT.

Strictly enforces the Phase 0 data hygiene rule: **never** load WikiText-2 test
or validation splits for training. General-corpus runs stream from
``HuggingFaceFW/fineweb-edu`` or ``openwebtext``; ``wikitext2-train`` is
provided solely for fast CPU smoke tests and debugging.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import torch

from onebit.eval.harness import load_eval_ids

_FORBIDDEN_SPLITS = {"test", "validation", "val", "eval"}


def _assert_train_split(split: str) -> None:
    if split.lower() in _FORBIDDEN_SPLITS:
        raise ValueError(
            f"Refusing to load split={split!r} for training. "
            "Training data must never touch evaluation/test splits."
        )


def _stream_hf_texts(
    dataset: str,
    dataset_config: Optional[str],
    seed: int,
    shuffle_buffer: int,
) -> Iterator[str]:
    """Yield raw text documents from a streaming Hugging Face dataset."""
    from datasets import load_dataset

    if dataset == "fineweb-edu":
        repo = "HuggingFaceFW/fineweb-edu"
        cfg = dataset_config or "sample-10BT"
        ds = load_dataset(repo, name=cfg, split="train", streaming=True)
        text_col = "text"
    elif dataset == "openwebtext":
        cfg = dataset_config
        try:
            ds = load_dataset("Skylion007/openwebtext", name=cfg, split="train", streaming=True)
        except Exception:
            ds = load_dataset("openwebtext", name=cfg, split="train", streaming=True)
        text_col = "text"
    else:
        raise ValueError(f"Unsupported streaming dataset: {dataset!r}")

    epoch = 0
    while True:
        cur = ds.shuffle(seed=seed + epoch, buffer_size=shuffle_buffer) if shuffle_buffer > 0 else ds
        for row in cur:
            text = row.get(text_col, "")
            if text:
                yield text
        epoch += 1


class PackedStreamingBatchIterator:
    """Packs streamed documents into fixed-length ``(batch_size, seqlen)`` token tensors.

    Appends ``eos_token_id`` between documents so sequences never bleed across
    document boundaries without an explicit separator and wastes zero padding tokens.
    """

    def __init__(
        self,
        text_iter: Iterator[str],
        tokenizer,
        batch_size: int,
        seqlen: int,
    ) -> None:
        self.text_iter = text_iter
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.seqlen = int(seqlen)
        self.eos_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)
        self._buf: List[int] = []

    def __iter__(self) -> "PackedStreamingBatchIterator":
        return self

    def __next__(self) -> torch.Tensor:
        needed = self.batch_size * self.seqlen
        while len(self._buf) < needed:
            text = next(self.text_iter)
            toks = self.tokenizer(text, add_special_tokens=False).input_ids
            if not toks:
                continue
            self._buf.extend(toks)
            if self.eos_id is not None:
                self._buf.append(self.eos_id)
        chunk = self._buf[:needed]
        del self._buf[:needed]
        return torch.tensor(chunk, dtype=torch.long).reshape(self.batch_size, self.seqlen)


class TokenBufferBatchIterator:
    """Random-window batch iterator over an in-memory 1-D token tensor (e.g., WikiText-2 train)."""

    def __init__(
        self,
        ids: torch.Tensor,
        batch_size: int,
        seqlen: int,
        seed: int = 0,
    ) -> None:
        if ids.ndim != 1:
            raise ValueError(f"Expected 1-D token tensor, got shape {tuple(ids.shape)}")
        if ids.numel() <= seqlen + 1:
            raise ValueError(f"Token buffer too short ({ids.numel()}) for seqlen={seqlen}")
        self.ids = ids
        self.batch_size = int(batch_size)
        self.seqlen = int(seqlen)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self) -> "TokenBufferBatchIterator":
        return self

    def __next__(self) -> torch.Tensor:
        max_start = self.ids.numel() - self.seqlen - 1
        starts = torch.randint(0, max_start, (self.batch_size,), generator=self.generator)
        return torch.stack([self.ids[s : s + self.seqlen] for s in starts.tolist()])


class SyntheticBatchIterator:
    """Deterministic synthetic token batch stream for unit tests and offline smoke checks."""

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seqlen: int,
        seed: int = 0,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.batch_size = int(batch_size)
        self.seqlen = int(seqlen)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self) -> "SyntheticBatchIterator":
        return self

    def __next__(self) -> torch.Tensor:
        return torch.randint(
            0, self.vocab_size, (self.batch_size, self.seqlen), generator=self.generator, dtype=torch.long
        )


def build_train_dataloader(
    dataset: str,
    tokenizer,
    batch_size: int,
    seqlen: int,
    seed: int = 0,
    dataset_config: Optional[str] = None,
    shuffle_buffer: int = 10_000,
    split: str = "train",
    vocab_size: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    """Create a training batch iterator yielding ``(batch_size, seqlen)`` ``LongTensor``s.

    Supported datasets:
      * ``"fineweb-edu"``: ``HuggingFaceFW/fineweb-edu`` streaming (default config ``sample-10BT``).
      * ``"openwebtext"``: ``Skylion007/openwebtext`` streaming.
      * ``"wikitext2-train"``: WikiText-2-raw-v1 **train** split only (fast CPU smoke tests).
      * ``"synthetic"``: Deterministic random tokens (unit tests / offline debugging).
    """
    _assert_train_split(split)
    if dataset in ("fineweb-edu", "openwebtext"):
        texts = _stream_hf_texts(dataset, dataset_config, seed=seed, shuffle_buffer=shuffle_buffer)
        return PackedStreamingBatchIterator(texts, tokenizer, batch_size=batch_size, seqlen=seqlen)
    if dataset == "wikitext2-train":
        train_ids = load_eval_ids(tokenizer, "wikitext2", split="train")
        return TokenBufferBatchIterator(train_ids, batch_size=batch_size, seqlen=seqlen, seed=seed)
    if dataset == "synthetic":
        v = vocab_size or getattr(tokenizer, "vocab_size", 50257)
        return SyntheticBatchIterator(vocab_size=v, batch_size=batch_size, seqlen=seqlen, seed=seed)
    raise ValueError(
        f"Unknown training dataset: {dataset!r}. "
        "Choose from: 'fineweb-edu', 'openwebtext', 'wikitext2-train', 'synthetic'."
    )
