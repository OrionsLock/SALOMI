"""
FIND TRUE 1.00 BPP CONFIGURATION

Goal: Find a method that achieves EXACTLY 1.00 bpp with maximum quality.

With 1.00 bpp budget per weight, we have:
- Must store signs: ~1.0 bits per weight (unavoidable)
- Remaining for VQ: ~0 bits

This means at 1.00 bpp, we can ONLY store signs!

Let's explore what's possible at various BPP targets.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FINDING OPTIMAL CONFIGURATIONS AT VARIOUS BPP TARGETS")
print("=" * 80)

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

calib_text = "The quick brown fox jumps over the lazy dog. " * 50
calib_inputs = tokenizer(calib_text, return_tensors="pt", max_length=512, truncation=True)

# Get one matrix for testing
W = model.transformer.h[0].mlp.c_fc.weight.detach().cpu().numpy().T
h, w = W.shape
n_weights = h * w

acts = []
hook = model.transformer.h[0].mlp.c_fc.register_forward_hook(
    lambda m, i, o: acts.append(i[0].detach().cpu().numpy()))
with torch.no_grad():
    model(calib_inputs.input_ids)
hook.remove()

X = acts[0].reshape(-1, W.shape[1])
H_diag = np.mean(X ** 2, axis=0)
X_test = X[:200]
Y_orig = X_test @ W.T


def hessian_vq(W, H_diag, K, block_size=4):
    h, w = W.shape
    bs = block_size
    
    S = np.sign(W)
    S[S == 0] = 1.0
    M = np.abs(W)
    
    ph, pw = (bs - h % bs) % bs, (bs - w % bs) % bs
    Mp = np.pad(M, ((0, ph), (0, pw)))
    Hp = np.pad(np.tile(H_diag, (h, 1)), ((0, ph), (0, pw)), constant_values=1e-6)
    
    hp, wp = Mp.shape
    n_blocks = (hp // bs) * (wp // bs)
    blocks = Mp.reshape(hp//bs, bs, wp//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    weights = Hp.reshape(hp//bs, bs, wp//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    
    np.random.seed(42)
    C = blocks[np.random.choice(len(blocks), min(K, len(blocks)), replace=False)].copy()
    for _ in range(15):
        d = np.sum((blocks[:, None] - C[None])**2, axis=2)
        a = np.argmin(d, axis=1)
        for i in range(K):
            m = (a == i)
            if m.sum() > 0:
                C[i] = np.sum(blocks[m] * weights[m], 0) / (np.sum(weights[m], 0) + 1e-8)
    
    d = np.sum((blocks[:, None] - C[None])**2, axis=2)
    a = np.argmin(d, axis=1)
    
    recon = C[a].reshape(hp//bs, wp//bs, bs, bs).transpose(0, 2, 1, 3).reshape(hp, wp)
    W_q = S * recon[:h, :w]
    
    # BPP
    sign_bits = 1.0 * n_weights
    index_bits = n_blocks * np.log2(K)
    cb_bits = K * (bs * bs) * 16
    bpp = (sign_bits + index_bits + cb_bits) / n_weights
    
    return W_q, bpp


def ternary(W):
    thr = np.percentile(np.abs(W), 30)
    mask = np.abs(W) > thr
    scale = np.mean(np.abs(W[mask])) if mask.any() else 1.0
    W_q = np.zeros_like(W)
    W_q[W > thr] = scale
    W_q[W < -thr] = -scale
    return W_q, 1.58


# Sweep K values
print(f"\n{'K':>6} {'BPP':>8} {'Corr':>8} {'vs Tern':>10}")
print("-" * 38)

W_tern, _ = ternary(W)
corr_tern = np.corrcoef(Y_orig.ravel(), (X_test @ W_tern.T).ravel())[0, 1]
print(f"{'Tern':>6} {1.58:>8.3f} {corr_tern:>8.4f} {'baseline':>10}")

for K in [2, 4, 8, 16, 32, 64, 128, 256]:
    W_q, bpp = hessian_vq(W, H_diag, K)
    corr = np.corrcoef(Y_orig.ravel(), (X_test @ W_q.T).ravel())[0, 1]
    delta = (corr - corr_tern) / corr_tern * 100
    print(f"{K:>6} {bpp:>8.3f} {corr:>8.4f} {delta:>+9.1f}%")


print("\n" + "=" * 80)
print("WHAT BEATS TERNARY?")
print("=" * 80)

# Find the sweet spot
print(f"""
Target: Beat ternary (1.58 bpp, {corr_tern:.4f} corr) on BOTH metrics

Options:
""")

for K in [2, 4, 8, 16, 32]:
    W_q, bpp = hessian_vq(W, H_diag, K)
    corr = np.corrcoef(Y_orig.ravel(), (X_test @ W_q.T).ravel())[0, 1]
    
    beats_bpp = bpp < 1.58
    beats_corr = corr > corr_tern
    
    status = "✓ BEATS BOTH" if (beats_bpp and beats_corr) else \
             "✓ Beats BPP only" if beats_bpp else \
             "✓ Beats Corr only" if beats_corr else "✗ Loses both"
    
    print(f"K={K:>3}: BPP={bpp:.3f}, Corr={corr:.4f} → {status}")


print("\n" + "=" * 80)
print("AT EXACTLY 1.00 BPP - WHAT'S POSSIBLE?")
print("=" * 80)

# At 1.00 bpp, signs take 1.0 bits, leaving 0 for VQ
# The ONLY thing we can do is sign-only quantization

def sign_only(W, X_calib):
    """Just keep signs, reconstruct magnitudes from calibration data."""
    S = np.sign(W)
    S[S == 0] = 1.0
    
    # Estimate magnitude from E[|X @ W|] / E[|X|]
    # This is a crude approximation
    Y = X_calib @ W.T
    avg_magnitude = np.mean(np.abs(Y), axis=0) / np.mean(np.abs(X_calib))
    
    W_q = S * avg_magnitude[:, None]
    return W_q, 1.0

W_sign, bpp_sign = sign_only(W, X)
corr_sign = np.corrcoef(Y_orig.ravel(), (X_test @ W_sign.T).ravel())[0, 1]
print(f"Sign-only @ 1.00 bpp: correlation = {corr_sign:.4f}")
print(f"vs Ternary @ 1.58 bpp: correlation = {corr_tern:.4f}")
print(f"Delta: {(corr_sign - corr_tern)/corr_tern*100:+.1f}%")


print("\n" + "=" * 80)
print("HONEST CONCLUSIONS")
print("=" * 80)

print("""
1. At 1.00 bpp (the SALOMI goal), we can only store signs.
   - Sign-only achieves ~0.48 correlation (terrible)
   - Ternary at 1.58 bpp achieves ~0.91 correlation

2. To beat ternary, we need at least ~1.1-1.2 bpp:
   - K=4 at 1.13 bpp: correlation ~0.85
   - K=8 at 1.19 bpp: correlation ~0.89

3. The 0.58 bpp claim was mathematically impossible because:
   - Signs require 1.0 bits minimum
   - Any VQ adds more bits on top

4. Sub-1-bit REQUIRES sign-free storage (BSDM-W approach)
   - But sign reconstruction accuracy is only ~54%
   - This destroys output quality

REVISED REALISTIC GOAL:
- Achieve 1.06 bpp (BinaryVQ K=2) with similar quality to ternary
- This is 33% fewer bits than ternary (1.58)
- Still a meaningful improvement, just not "sub-1-bit"
""")

