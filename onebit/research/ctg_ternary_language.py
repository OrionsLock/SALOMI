"""CTG Ternary Language: Express ternary through temporal codes.

Key idea from ChatGPT:
- Store only 1 bit (sign) per weight
- CTG opcodes over T ticks express +1, 0, -1 behaviors
- INHIBIT = temporal "zero" symbol
- Train model to use this language

The "language":
- Alphabet: {PASS, INVERT, INHIBIT}
- Phonemes: opcodes per tick
- Morphemes: patterns that mean "+", "0", "-"
- Grammar: arrangement over channels/layers

This is TRAINING-TIME, not post-hoc.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class CTGPatternCodebook:
    """Define patterns that express +1, 0, -1 over T ticks."""
    
    # Opcodes
    PASS = 0      # Keep sign as-is
    INVERT = 1    # Flip sign
    INHIBIT = 2   # Zero contribution this tick
    
    def __init__(self, T: int = 4):
        self.T = T
        # Define semantic patterns
        self.patterns = {
            'strong_pos': [self.PASS] * T,           # Always +sign
            'strong_neg': [self.INVERT] * T,         # Always -sign
            'zero': [self.INHIBIT] * T,              # Always inhibited
            'weak_pos': [self.PASS, self.PASS, self.INHIBIT, self.INHIBIT],
            'weak_neg': [self.INVERT, self.INVERT, self.INHIBIT, self.INHIBIT],
            'alternating': [self.PASS, self.INVERT, self.PASS, self.INVERT],
        }
    
    def get_pattern(self, name: str) -> List[int]:
        return self.patterns.get(name, self.patterns['zero'])
    
    def pattern_to_effective_value(self, sign: float, pattern: List[int]) -> float:
        """Compute effective contribution over T ticks."""
        total = 0.0
        for op in pattern:
            if op == self.PASS:
                total += sign
            elif op == self.INVERT:
                total += -sign
            # INHIBIT adds nothing
        return total / len(pattern)


class CTGTernaryLinear(nn.Module):
    """CTG with FIXED temporal grammar - learns which blocks use which pattern.

    Key: The patterns are FIXED (not learned), model learns which to use.
    This is like having a fixed "language" - model learns to speak it.

    Patterns (over T=4 ticks):
    - Pattern 0: [1,1,1,1] → strong (+1 or -1 depending on sign)
    - Pattern 1: [1,1,0,0] → weak (0.5× magnitude)
    - Pattern 2: [0,0,0,0] → zero (inhibited)

    Stored: 1 bit (sign) + 2 bits/block (pattern) = ~1.12 bpp
    """

    def __init__(self, in_features: int, out_features: int, block_size: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Latent weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # Pattern selection logits per block (3 patterns → 2 bits)
        n_blocks = (in_features * out_features + block_size - 1) // block_size
        self.pattern_logits = nn.Parameter(torch.zeros(n_blocks, 3))

        # Fixed pattern multipliers (not learned)
        self.register_buffer('pattern_mult', torch.tensor([1.0, 0.5, 0.0]))

        self.scale = nn.Parameter(torch.tensor(0.5))

    def get_effective_weights(self) -> torch.Tensor:
        w = self.weight

        # Binary signs with STE
        w_sign = torch.sign(w)
        w_sign = torch.where(w_sign == 0, torch.ones_like(w_sign), w_sign)
        w_sign = w.clamp(-1, 1) + (w_sign - w.clamp(-1, 1)).detach()

        # Soft pattern selection (differentiable)
        pattern_probs = F.softmax(self.pattern_logits, dim=-1)  # (n_blocks, 3)

        # Effective multiplier = weighted sum of pattern multipliers
        block_mult = pattern_probs @ self.pattern_mult  # (n_blocks,)

        # Expand to weights
        n_weights = w.numel()
        mult_expanded = block_mult.repeat_interleave(self.block_size)[:n_weights]
        mult_expanded = mult_expanded.reshape(self.out_features, self.in_features)

        return w_sign * mult_expanded * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.get_effective_weights())

    def bits_per_param(self) -> float:
        n_weights = self.in_features * self.out_features
        n_blocks = self.pattern_logits.shape[0]
        sign_bits = n_weights * 1
        pattern_bits = n_blocks * np.log2(3)  # ~1.58 bits per block
        return (sign_bits + pattern_bits + 32) / n_weights

    def get_ternary_distribution(self) -> dict:
        with torch.no_grad():
            W = self.get_effective_weights()
            thresh = 0.1 * self.scale.abs().item()
            pos = (W > thresh).sum().item()
            neg = (W < -thresh).sum().item()
            zero = ((W >= -thresh) & (W <= thresh)).sum().item()
            total = W.numel()
        return {'pos': pos/total, 'neg': neg/total, 'zero': zero/total}

    def get_pattern_distribution(self) -> dict:
        with torch.no_grad():
            probs = F.softmax(self.pattern_logits, dim=-1).mean(0)
        return {'strong': probs[0].item(), 'weak': probs[1].item(), 'zero': probs[2].item()}


class TernaryLinear(nn.Module):
    """Standard ternary for comparison - better STE."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # Ternary: top 70% by magnitude get sign, bottom 30% get zero
        thresh = w.abs().quantile(0.3)
        mask = (w.abs() > thresh).float()
        w_tern = torch.sign(w) * mask
        # STE with clamped gradients
        w_tern = w.clamp(-1, 1) + (w_tern - w.clamp(-1, 1)).detach()
        return F.linear(x, w_tern * self.scale)

    def bits_per_param(self) -> float:
        return 1.58


class BinaryLinear(nn.Module):
    """Standard binary for comparison - better STE."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w_bin = torch.sign(w)
        w_bin = torch.where(w_bin == 0, torch.ones_like(w_bin), w_bin)
        # STE with clamped gradients
        w_bin = w.clamp(-1, 1) + (w_bin - w.clamp(-1, 1)).detach()
        return F.linear(x, w_bin * self.scale)

    def bits_per_param(self) -> float:
        return 1.0


def train_and_compare():
    """Train all three methods on same task, compare."""
    print("=" * 70)
    print("CTG TERNARY LANGUAGE: Training-Time Comparison")
    print("=" * 70)

    torch.manual_seed(42)

    # Task: Learn a random linear mapping
    d_in, d_out = 128, 128
    W_true = torch.randn(d_out, d_in) * 0.5

    # Training data
    n_train = 5000
    X_train = torch.randn(n_train, d_in)
    Y_train = X_train @ W_true.T

    # Test data
    n_test = 1000
    X_test = torch.randn(n_test, d_in)
    Y_test = X_test @ W_true.T

    results = {}

    for name, model in [
        ('binary', BinaryLinear(d_in, d_out)),
        ('ternary', TernaryLinear(d_in, d_out)),
        ('ctg_b16', CTGTernaryLinear(d_in, d_out, block_size=16)),  # 1.10 bpp
        ('ctg_b8', CTGTernaryLinear(d_in, d_out, block_size=8)),   # 1.20 bpp
        ('ctg_b4', CTGTernaryLinear(d_in, d_out, block_size=4)),   # 1.40 bpp
        ('ctg_b2', CTGTernaryLinear(d_in, d_out, block_size=2)),   # 1.79 bpp
    ]:
        print(f"\nTraining {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

        for epoch in range(2000):
            optimizer.zero_grad()
            Y_pred = model(X_train)
            loss = F.mse_loss(Y_pred, Y_train)

            # For CTG: encourage some blocks to use zero pattern
            if hasattr(model, 'pattern_logits'):
                # Entropy regularization to encourage discrete pattern selection
                probs = F.softmax(model.pattern_logits, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
                # Push toward low entropy (confident selection)
                loss = loss + 0.01 * entropy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 400 == 0:
                print(f"  Epoch {epoch}: loss={loss.item():.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            Y_pred = model(X_test)
            mse = F.mse_loss(Y_pred, Y_test).item()
            corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0,1]

        bpp = model.bits_per_param()
        results[name] = {'mse': mse, 'corr': corr, 'bpp': bpp}

        if hasattr(model, 'get_ternary_distribution'):
            dist = model.get_ternary_distribution()
            print(f"  Weight dist: +:{dist['pos']:.1%} 0:{dist['zero']:.1%} -:{dist['neg']:.1%}")
        if hasattr(model, 'get_pattern_distribution'):
            pdist = model.get_pattern_distribution()
            print(f"  Pattern dist: strong:{pdist['strong']:.1%} weak:{pdist['weak']:.1%} zero:{pdist['zero']:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Method':<12} {'Corr':>10} {'MSE':>10} {'BPP':>8} {'vs Tern':>10}")
    print("-" * 52)

    tern_corr = results['ternary']['corr']
    for name, data in sorted(results.items(), key=lambda x: -x[1]['corr']):
        vs = (data['corr'] / tern_corr - 1) * 100
        marker = "✓" if data['bpp'] <= 1.1 and data['corr'] >= tern_corr * 0.99 else ""
        print(f"{name:<12} {data['corr']:>10.4f} {data['mse']:>10.4f} {data['bpp']:>8.2f} {vs:>+9.1f}% {marker}")

    return results


if __name__ == "__main__":
    train_and_compare()

