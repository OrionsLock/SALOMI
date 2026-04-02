"""Step 2: Test Hybrid on Small Trained Model.

ChatGPT's validation: "Compare perplexity, not just correlation on one matrix."

This module:
1. Creates a tiny transformer for language modeling
2. Trains it on a simple text corpus
3. Compares raw 1-bit vs ternary vs hybrid_block4 on PERPLEXITY
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available - using numpy simulation")


@dataclass
class TinyConfig:
    """Config for tiny transformer."""
    vocab_size: int = 256  # Character-level
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    seq_len: int = 32
    batch_size: int = 32


def generate_synthetic_corpus(n_samples: int = 10000, seq_len: int = 32) -> np.ndarray:
    """Generate synthetic text-like data with patterns."""
    # Create repeating patterns (compressible)
    patterns = [
        list(range(10)),  # 0-9
        list(range(65, 75)),  # A-J
        list(range(97, 107)),  # a-j
    ]
    
    data = []
    for _ in range(n_samples):
        pattern = patterns[np.random.randint(len(patterns))]
        start = np.random.randint(len(pattern))
        seq = [(pattern[(start + i) % len(pattern)]) for i in range(seq_len)]
        data.append(seq)
    
    return np.array(data, dtype=np.int32)


class TinyLinear:
    """Tiny linear layer with multiple quantization modes."""
    
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        # Initialize weights
        self.W = np.random.randn(d_out, d_in).astype(np.float32) * 0.02
        self.b = np.zeros(d_out, dtype=np.float32)
    
    def forward_fp32(self, x: np.ndarray) -> np.ndarray:
        """FP32 forward pass."""
        return x @ self.W.T + self.b
    
    def forward_binary(self, x: np.ndarray) -> np.ndarray:
        """Binary quantized forward."""
        scale = np.mean(np.abs(self.W), axis=1, keepdims=True)
        W_binary = np.sign(self.W) * scale
        return x @ W_binary.T + self.b
    
    def forward_ternary(self, x: np.ndarray, threshold_pct: float = 30) -> np.ndarray:
        """Ternary quantized forward."""
        threshold = np.percentile(np.abs(self.W), threshold_pct)
        W_ternary = np.sign(self.W) * (np.abs(self.W) > threshold)
        scale = np.mean(np.abs(self.W[np.abs(self.W) > threshold]))
        W_ternary = W_ternary * scale
        return x @ W_ternary.T + self.b
    
    def forward_hybrid_block(self, x: np.ndarray, block_size: int = 4, 
                              n_levels: int = 2) -> np.ndarray:
        """Hybrid block-structured forward."""
        d_out, d_in = self.W.shape
        
        # Step 1: Compute block signs
        W_hybrid = np.zeros_like(self.W)
        n_blocks_h = d_out // block_size
        n_blocks_w = d_in // block_size
        
        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                block = self.W[bi*block_size:(bi+1)*block_size,
                              bj*block_size:(bj+1)*block_size]
                block_sign = np.sign(np.mean(block))
                if block_sign == 0:
                    block_sign = 1
                W_hybrid[bi*block_size:(bi+1)*block_size,
                        bj*block_size:(bj+1)*block_size] = block_sign
        
        # Step 2: Compute magnitude levels
        magnitudes = np.abs(self.W)
        thresholds = [np.percentile(magnitudes, 100 * i / n_levels) 
                      for i in range(1, n_levels)]
        
        level_scales = np.linspace(0, np.mean(np.abs(self.W)), n_levels)
        
        for i, thresh in enumerate(thresholds):
            mask = magnitudes >= thresh
            W_hybrid[mask] *= level_scales[i + 1] / np.maximum(np.abs(W_hybrid[mask]), 1e-10)
        
        # Set lowest level to small non-zero
        low_mask = magnitudes < thresholds[0] if thresholds else np.ones_like(magnitudes, dtype=bool)
        W_hybrid[low_mask] *= level_scales[0] / np.maximum(np.abs(W_hybrid[low_mask]), 1e-10)
        
        return x @ W_hybrid.T + self.b


class TinyTransformer:
    """Minimal transformer for testing quantization."""
    
    def __init__(self, cfg: TinyConfig):
        self.cfg = cfg
        
        # Embedding
        self.embed = np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float32) * 0.02
        
        # Attention layers
        self.layers = []
        for _ in range(cfg.n_layers):
            layer = {
                'q': TinyLinear(cfg.d_model, cfg.d_model),
                'k': TinyLinear(cfg.d_model, cfg.d_model),
                'v': TinyLinear(cfg.d_model, cfg.d_model),
                'o': TinyLinear(cfg.d_model, cfg.d_model),
                'ff1': TinyLinear(cfg.d_model, cfg.d_model * 4),
                'ff2': TinyLinear(cfg.d_model * 4, cfg.d_model),
            }
            self.layers.append(layer)
        
        # Output projection
        self.lm_head = TinyLinear(cfg.d_model, cfg.vocab_size)
    
    def forward(self, x: np.ndarray, mode: str = 'fp32') -> np.ndarray:
        """Forward pass with specified quantization mode."""
        # Embedding
        h = self.embed[x]  # [B, T, D]
        
        # Forward fn based on mode
        def linear_fn(layer: TinyLinear, x: np.ndarray) -> np.ndarray:
            if mode == 'fp32':
                return layer.forward_fp32(x)
            elif mode == 'binary':
                return layer.forward_binary(x)
            elif mode == 'ternary':
                return layer.forward_ternary(x)
            elif mode == 'hybrid':
                return layer.forward_hybrid_block(x, block_size=4, n_levels=2)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        # Transformer layers (simplified - no actual attention)
        for layer in self.layers:
            # Simplified: just FFN (attention is expensive in numpy)
            h_ff = linear_fn(layer['ff1'], h)
            h_ff = np.maximum(h_ff, 0)  # ReLU
            h_ff = linear_fn(layer['ff2'], h_ff)
            h = h + h_ff  # Residual
        
        # Output logits
        logits = linear_fn(self.lm_head, h)
        return logits
    
    def compute_loss(self, x: np.ndarray, mode: str = 'fp32') -> float:
        """Compute cross-entropy loss."""
        # x: [B, T] - predict next token
        inputs = x[:, :-1]  # [B, T-1]
        targets = x[:, 1:]   # [B, T-1]
        
        logits = self.forward(inputs, mode)  # [B, T-1, V]
        
        # Softmax + cross-entropy
        logits_flat = logits.reshape(-1, self.cfg.vocab_size)
        targets_flat = targets.flatten()
        
        # Stable softmax
        logits_max = np.max(logits_flat, axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy
        n = len(targets_flat)
        ce = -np.mean(np.log(probs[np.arange(n), targets_flat] + 1e-10))
        
        return ce
    
    def compute_perplexity(self, x: np.ndarray, mode: str = 'fp32') -> float:
        """Compute perplexity."""
        ce = self.compute_loss(x, mode)
        return np.exp(ce)

    def train_step(self, x: np.ndarray, lr: float = 0.01) -> float:
        """Simple training step (gradient descent on FP32)."""
        # This is a simplified training - just perturb weights to reduce loss
        loss_before = self.compute_loss(x, 'fp32')

        # Perturb each layer's weights slightly
        for layer in self.layers:
            for name in ['ff1', 'ff2']:
                lin = layer[name]
                # Random direction
                direction = np.random.randn(*lin.W.shape).astype(np.float32) * 0.001
                lin.W += direction
                loss_after = self.compute_loss(x, 'fp32')
                if loss_after > loss_before:
                    lin.W -= 2 * direction  # Go opposite direction

        return self.compute_loss(x, 'fp32')


def test_hybrid_on_tiny_lm():
    """Test hybrid scheme on tiny language model."""
    print("=" * 80)
    print("STEP 2: TEST HYBRID ON SMALL TRAINED MODEL")
    print("=" * 80)
    print("\nChatGPT's challenge: 'Compare perplexity, not just correlation'")
    print("-" * 80)

    # Config
    cfg = TinyConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        seq_len=32,
        batch_size=64
    )

    print(f"\nModel config:")
    print(f"  Vocab size: {cfg.vocab_size}")
    print(f"  Model dim: {cfg.d_model}")
    print(f"  Layers: {cfg.n_layers}")
    print(f"  Seq len: {cfg.seq_len}")

    # Generate data
    print("\nGenerating synthetic corpus...")
    train_data = generate_synthetic_corpus(n_samples=1000, seq_len=cfg.seq_len)
    test_data = generate_synthetic_corpus(n_samples=200, seq_len=cfg.seq_len)

    # Create model
    model = TinyTransformer(cfg)

    # Train for a few steps
    print("\nTraining (simplified)...")
    for epoch in range(5):
        batch_idx = np.random.randint(0, len(train_data), cfg.batch_size)
        batch = train_data[batch_idx]
        loss = model.train_step(batch)
        print(f"  Epoch {epoch+1}: Loss = {loss:.4f}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON (lower is better)")
    print("=" * 60)

    results = {}
    modes = ['fp32', 'binary', 'ternary', 'hybrid']
    bpp_map = {'fp32': 32.0, 'binary': 1.0, 'ternary': 1.58, 'hybrid': 1.06}

    for mode in modes:
        # Evaluate on multiple batches
        perplexities = []
        for i in range(0, len(test_data), cfg.batch_size):
            batch = test_data[i:i+cfg.batch_size]
            if len(batch) < cfg.batch_size:
                continue
            ppl = model.compute_perplexity(batch, mode)
            if not np.isnan(ppl) and ppl < 1e6:
                perplexities.append(ppl)

        avg_ppl = np.mean(perplexities) if perplexities else float('inf')
        results[mode] = avg_ppl

    print(f"\n{'Mode':<15} {'BPP':>8} {'Perplexity':>15} {'vs FP32':>12}")
    print("-" * 55)

    fp32_ppl = results['fp32']
    for mode in modes:
        ppl = results[mode]
        bpp = bpp_map[mode]
        vs_fp32 = (ppl / fp32_ppl - 1) * 100 if fp32_ppl > 0 else 0
        print(f"{mode:<15} {bpp:>8.2f} {ppl:>15.2f} {vs_fp32:>+11.1f}%")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    binary_ppl = results['binary']
    ternary_ppl = results['ternary']
    hybrid_ppl = results['hybrid']

    print(f"\nTernary vs Binary gap: {(binary_ppl / ternary_ppl - 1) * 100:.1f}%")
    print(f"Hybrid vs Ternary gap: {(hybrid_ppl / ternary_ppl - 1) * 100:.1f}%")

    if hybrid_ppl <= ternary_ppl * 1.05:  # Within 5%
        print("\n✓ HYBRID MATCHES TERNARY on perplexity!")
        print(f"  Hybrid: {hybrid_ppl:.2f} PPL @ 1.06 bpp")
        print(f"  Ternary: {ternary_ppl:.2f} PPL @ 1.58 bpp")
        print(f"  → 33% fewer bits for comparable quality")
    else:
        print(f"\n✗ Hybrid still trails ternary")
        print(f"  Gap: {(hybrid_ppl / ternary_ppl - 1) * 100:.1f}%")

    return results


if __name__ == "__main__":
    test_hybrid_on_tiny_lm()

