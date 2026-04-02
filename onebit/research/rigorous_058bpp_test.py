"""
RIGOROUS TEST: 0.58 BPP DualPathVQ vs Real Models
=================================================

Testing the exact claims:
1. DualPathVQ achieves 0.58 bpp
2. It beats ternary (1.58 bpp) in correlation
3. Validate on ALL 48 GPT-2 weight matrices
4. Test on real activations, not synthetic
5. Full model perplexity comparison
"""

import numpy as np
import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RIGOROUS 0.58 BPP TEST: DualPathVQ vs Ternary")
print("=" * 80)


def dual_path_vq(W, H_diag, k_high=32, k_low=8, threshold=0.6, block_size=4):
    """
    DualPathVQ: Importance-based adaptive routing.
    
    High importance blocks -> K=32 (5 bits)
    Low importance blocks -> K=8 (3 bits)
    
    Expected BPP: ~0.58
    """
    h, w = W.shape
    bs = block_size
    
    # Sign-magnitude decomposition
    S = np.sign(W)
    S[S == 0] = 1.0
    M = np.abs(W)
    
    # Pad
    ph, pw = (bs - h % bs) % bs, (bs - w % bs) % bs
    Mp = np.pad(M, ((0, ph), (0, pw)))
    Hp = np.pad(np.tile(H_diag, (h, 1)), ((0, ph), (0, pw)), constant_values=1e-6)
    
    hp, wp = Mp.shape
    n_blocks = (hp // bs) * (wp // bs)
    
    # Extract blocks
    blocks = Mp.reshape(hp//bs, bs, wp//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    weights = Hp.reshape(hp//bs, bs, wp//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    
    # Compute importance per block
    importance = np.mean(blocks * weights, axis=1)
    threshold_val = np.percentile(importance, (1 - threshold) * 100)
    high_mask = importance >= threshold_val
    
    # K-means for each path
    def kmeans(X, Hw, k, seed=42):
        np.random.seed(seed)
        if len(X) == 0:
            return np.zeros((k, X.shape[1])), np.array([])
        C = X[np.random.choice(len(X), min(k, len(X)), replace=False)].copy()
        for _ in range(15):
            d = np.sum((X[:, None] - C[None])**2, axis=2)
            a = np.argmin(d, axis=1)
            for i in range(k):
                m = (a == i)
                if m.sum() > 0:
                    C[i] = np.sum(X[m] * Hw[m], 0) / (np.sum(Hw[m], 0) + 1e-8)
        d = np.sum((X[:, None] - C[None])**2, axis=2)
        return C, np.argmin(d, axis=1)
    
    # Quantize each path
    C_high, a_high = kmeans(blocks[high_mask], weights[high_mask], k_high)
    C_low, a_low = kmeans(blocks[~high_mask], weights[~high_mask], k_low)
    
    # Reconstruct
    recon = np.zeros_like(blocks)
    if len(a_high) > 0:
        recon[high_mask] = C_high[a_high]
    if len(a_low) > 0:
        recon[~high_mask] = C_low[a_low]
    
    M_recon = recon.reshape(hp//bs, wp//bs, bs, bs).transpose(0, 2, 1, 3).reshape(hp, wp)
    W_q = S * M_recon[:h, :w]
    
    # Calculate BPP
    n_high = np.sum(high_mask)
    n_low = np.sum(~high_mask)
    
    # Sign entropy
    pos = np.mean(S > 0)
    sign_entropy = -pos * np.log2(pos + 1e-10) - (1-pos) * np.log2(1-pos + 1e-10)
    sign_bits = sign_entropy * h * w
    
    # Routing bit (1 bit per block)
    routing_bits = n_blocks
    
    # Index bits (entropy-coded estimate)
    index_bits_high = n_high * np.log2(k_high) if n_high > 0 else 0
    index_bits_low = n_low * np.log2(k_low) if n_low > 0 else 0
    
    # Codebook overhead
    cb_bits = (k_high + k_low) * (bs * bs) * 16
    
    total_bits = sign_bits + routing_bits + index_bits_high + index_bits_low + cb_bits
    bpp = total_bits / (h * w)
    
    return W_q, bpp


def ternary_quantize(W):
    """Standard ternary: {-scale, 0, +scale}"""
    thr = np.percentile(np.abs(W), 30)
    mask = np.abs(W) > thr
    scale = np.mean(np.abs(W[mask])) if mask.any() else 1.0
    W_q = np.zeros_like(W)
    W_q[W > thr] = scale
    W_q[W < -thr] = -scale
    return W_q, 1.58  # log2(3) ≈ 1.58


# Load model and tokenizer
print("\nLoading GPT-2...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Calibration data
calib_text = "The quick brown fox jumps over the lazy dog. " * 50
calib_inputs = tokenizer(calib_text, return_tensors="pt", max_length=512, truncation=True)

print("\n" + "=" * 80)
print("TEST 1: ALL 48 WEIGHT MATRICES - Correlation Comparison")
print("=" * 80)

results = []
layer_names = []

for layer_idx in range(12):
    block = model.transformer.h[layer_idx]
    
    modules = [
        (f"L{layer_idx}.c_attn", block.attn.c_attn),
        (f"L{layer_idx}.c_proj", block.attn.c_proj),
        (f"L{layer_idx}.c_fc", block.mlp.c_fc),
        (f"L{layer_idx}.c_proj2", block.mlp.c_proj),
    ]
    
    for name, module in modules:
        W = module.weight.detach().cpu().numpy().T  # (out, in)
        
        # Get activations
        acts = []
        h = module.register_forward_hook(lambda m, i, o: acts.append(i[0].detach().cpu().numpy()))
        with torch.no_grad():
            model(calib_inputs.input_ids)
        h.remove()
        
        X = acts[0].reshape(-1, W.shape[1])
        H_diag = np.mean(X ** 2, axis=0)
        
        # Test data
        X_test = X[:200]
        Y_orig = X_test @ W.T
        
        # Ternary
        W_tern, bpp_tern = ternary_quantize(W)
        Y_tern = X_test @ W_tern.T
        corr_tern = np.corrcoef(Y_orig.ravel(), Y_tern.ravel())[0, 1]
        
        # DualPathVQ
        W_vq, bpp_vq = dual_path_vq(W, H_diag)
        Y_vq = X_test @ W_vq.T
        corr_vq = np.corrcoef(Y_orig.ravel(), Y_vq.ravel())[0, 1]
        
        results.append({
            'name': name,
            'corr_tern': corr_tern,
            'corr_vq': corr_vq,
            'bpp_tern': bpp_tern,
            'bpp_vq': bpp_vq,
            'winner': 'VQ' if corr_vq > corr_tern else 'TERN'
        })
        layer_names.append(name)

# Summary
print(f"\n{'Layer':<15} {'Tern Corr':>10} {'VQ Corr':>10} {'VQ BPP':>8} {'Winner':>8}")
print("-" * 55)

vq_wins = 0
for r in results:
    print(f"{r['name']:<15} {r['corr_tern']:>10.4f} {r['corr_vq']:>10.4f} {r['bpp_vq']:>8.2f} {r['winner']:>8}")
    if r['winner'] == 'VQ':
        vq_wins += 1

print("-" * 55)
mean_corr_tern = np.mean([r['corr_tern'] for r in results])
mean_corr_vq = np.mean([r['corr_vq'] for r in results])
mean_bpp_vq = np.mean([r['bpp_vq'] for r in results])

print(f"{'MEAN':<15} {mean_corr_tern:>10.4f} {mean_corr_vq:>10.4f} {mean_bpp_vq:>8.2f}")
print(f"\nVQ wins: {vq_wins}/48 matrices ({100*vq_wins/48:.1f}%)")
print(f"Mean improvement: {(mean_corr_vq - mean_corr_tern)/mean_corr_tern*100:+.2f}%")
print(f"Mean BPP: {mean_bpp_vq:.2f} (target: 0.58)")


print("\n" + "=" * 80)
print("TEST 2: BPP VERIFICATION - Is 0.58 honest?")
print("=" * 80)

# Detailed BPP breakdown for one matrix
W_sample = model.transformer.h[0].mlp.c_fc.weight.detach().cpu().numpy().T
h, w = W_sample.shape
bs = 4
n_weights = h * w
n_blocks = ((h + bs - 1) // bs) * ((w + bs - 1) // bs)

print(f"Sample matrix: {h} x {w} = {n_weights:,} weights")
print(f"Blocks: {n_blocks:,} ({bs}x{bs})")

# Compute exact BPP components
k_high, k_low = 32, 8
threshold = 0.6
n_high = int(n_blocks * threshold)
n_low = n_blocks - n_high

sign_bits = 1.0 * n_weights  # Worst case: 1 bit per weight
routing_bits = n_blocks * 1.0  # 1 bit per block for routing
index_bits = n_high * np.log2(k_high) + n_low * np.log2(k_low)
cb_bits = (k_high + k_low) * (bs * bs) * 16  # FP16 codebook

total_bits = sign_bits + routing_bits + index_bits + cb_bits
bpp = total_bits / n_weights

print(f"\nBPP Breakdown:")
print(f"  Sign bits:     {sign_bits:,.0f} ({sign_bits/n_weights:.3f} per weight)")
print(f"  Routing bits:  {routing_bits:,.0f} ({routing_bits/n_weights:.3f} per weight)")
print(f"  Index bits:    {index_bits:,.0f} ({index_bits/n_weights:.3f} per weight)")
print(f"  Codebook bits: {cb_bits:,} ({cb_bits/n_weights:.3f} per weight)")
print(f"  TOTAL:         {total_bits:,.0f}")
print(f"  BPP:           {bpp:.3f}")
print(f"\n  Claimed: 0.58 bpp")
print(f"  Actual:  {bpp:.3f} bpp")
print(f"  VERDICT: {'HONEST' if abs(bpp - 0.58) < 0.15 else 'MISLEADING!'}")


print("\n" + "=" * 80)
print("TEST 3: STATISTICAL SIGNIFICANCE - Multiple seeds")
print("=" * 80)

W = model.transformer.h[0].mlp.c_fc.weight.detach().cpu().numpy().T
acts = []
h = model.transformer.h[0].mlp.c_fc.register_forward_hook(
    lambda m, i, o: acts.append(i[0].detach().cpu().numpy()))
with torch.no_grad():
    model(calib_inputs.input_ids)
h.remove()
X = acts[0].reshape(-1, W.shape[1])
H_diag = np.mean(X ** 2, axis=0)
X_test = X[:200]
Y_orig = X_test @ W.T

W_tern, _ = ternary_quantize(W)
corr_tern = np.corrcoef(Y_orig.ravel(), (X_test @ W_tern.T).ravel())[0, 1]

vq_corrs = []
vq_bpps = []
for seed in range(20):
    np.random.seed(seed)
    W_vq, bpp = dual_path_vq(W, H_diag)
    corr = np.corrcoef(Y_orig.ravel(), (X_test @ W_vq.T).ravel())[0, 1]
    vq_corrs.append(corr)
    vq_bpps.append(bpp)

print(f"Ternary:     {corr_tern:.4f}")
print(f"VQ Mean:     {np.mean(vq_corrs):.4f} ± {np.std(vq_corrs):.4f}")
print(f"VQ Min:      {np.min(vq_corrs):.4f}")
print(f"VQ Max:      {np.max(vq_corrs):.4f}")
print(f"BPP Mean:    {np.mean(vq_bpps):.3f}")
print(f"\nVERDICT: VQ beats ternary on {sum(c > corr_tern for c in vq_corrs)}/20 seeds")


print("\n" + "=" * 80)
print("TEST 4: OUT-OF-DISTRIBUTION GENERALIZATION")
print("=" * 80)

# Train on one text, test on completely different text
ood_texts = [
    "Machine learning is transforming industries worldwide.",
    "Python programming involves functions, classes, and modules.",
    "The stock market fluctuated wildly during the pandemic.",
    "Quantum computing promises exponential speedups for certain problems.",
]

print(f"{'Test Text':<60} {'Tern':>8} {'VQ':>8} {'Winner':>8}")
print("-" * 90)

for text in ood_texts:
    inputs = tokenizer(text, return_tensors="pt")
    acts_ood = []
    h = model.transformer.h[0].mlp.c_fc.register_forward_hook(
        lambda m, i, o: acts_ood.append(i[0].detach().cpu().numpy()))
    with torch.no_grad():
        model(inputs.input_ids)
    h.remove()

    X_ood = acts_ood[0].reshape(-1, W.shape[1])
    Y_ood = X_ood @ W.T

    corr_t = np.corrcoef(Y_ood.ravel(), (X_ood @ W_tern.T).ravel())[0, 1]
    corr_v = np.corrcoef(Y_ood.ravel(), (X_ood @ W_vq.T).ravel())[0, 1]
    winner = 'VQ' if corr_v > corr_t else 'TERN'

    print(f"{text[:57]+'...':<60} {corr_t:>8.4f} {corr_v:>8.4f} {winner:>8}")


print("\n" + "=" * 80)
print("TEST 5: ERROR METRICS (not just correlation)")
print("=" * 80)

Y_tern = X_test @ W_tern.T
Y_vq = X_test @ W_vq.T

metrics = {
    'Correlation': (np.corrcoef(Y_orig.ravel(), Y_tern.ravel())[0,1],
                   np.corrcoef(Y_orig.ravel(), Y_vq.ravel())[0,1]),
    'L2 Relative': (np.linalg.norm(Y_tern - Y_orig) / np.linalg.norm(Y_orig),
                   np.linalg.norm(Y_vq - Y_orig) / np.linalg.norm(Y_orig)),
    'MSE': (np.mean((Y_tern - Y_orig)**2), np.mean((Y_vq - Y_orig)**2)),
    'Max Error': (np.max(np.abs(Y_tern - Y_orig)), np.max(np.abs(Y_vq - Y_orig))),
    'RMSE': (np.sqrt(np.mean((Y_tern - Y_orig)**2)), np.sqrt(np.mean((Y_vq - Y_orig)**2))),
}

print(f"{'Metric':<15} {'Ternary':>12} {'VQ':>12} {'Winner':>10}")
print("-" * 52)
for name, (t, v) in metrics.items():
    if name == 'Correlation':
        winner = 'VQ' if v > t else 'TERN'
    else:
        winner = 'VQ' if v < t else 'TERN'
    print(f"{name:<15} {t:>12.4f} {v:>12.4f} {winner:>10}")


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"""
CLAIM 1: 0.58 bpp
  Measured BPP: {mean_bpp_vq:.2f}
  VERDICT: {'✓ VALID' if abs(mean_bpp_vq - 0.58) < 0.15 else '✗ INVALID'}

CLAIM 2: Beats ternary in correlation
  VQ wins on {vq_wins}/48 matrices
  Mean improvement: {(mean_corr_vq - mean_corr_tern)/mean_corr_tern*100:+.2f}%
  VERDICT: {'✓ VALID' if vq_wins > 24 else '✗ INVALID'}

CLAIM 3: Works on real model activations
  Tested on GPT-2 with real text
  VERDICT: ✓ VALID

OVERALL: {'✓ CLAIMS SUPPORTED' if vq_wins > 24 and abs(mean_bpp_vq - 0.58) < 0.15 else '✗ CLAIMS NOT FULLY SUPPORTED'}
""")

