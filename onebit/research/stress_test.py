"""
RIGOROUS STRESS TESTING: Try to Break HessianVQ

Tests:
1. Statistical significance (multiple seeds)
2. Correlation vs actual error (are we measuring the right thing?)
3. Calibration data sensitivity
4. Different weight distributions
5. Edge cases (small matrices, extreme values)
6. Comparison to other baselines (RTN 2-bit, 3-bit, 4-bit)
7. Full model perplexity (the real test)
8. Overfitting to calibration data
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
import copy
import math
warnings.filterwarnings('ignore')

print("=" * 80)
print("RIGOROUS STRESS TESTING: TRYING TO BREAK HESSIANVQ")
print("=" * 80)


def hessian_kmeans(X, H, k, max_iter=15, seed=42):
    np.random.seed(seed)
    C = X[np.random.choice(len(X), min(k, len(X)), replace=False)].copy()
    for _ in range(max_iter):
        d = np.sum((X[:, None] - C[None])**2, axis=2)
        a = np.argmin(d, axis=1)
        for i in range(k):
            m = (a == i)
            if m.sum() > 0:
                C[i] = np.sum(X[m] * H[m], axis=0) / (np.sum(H[m], axis=0) + 1e-8)
    return C, a


def quantize_hessianvq(W, H_diag, K=128, seed=42):
    h, w = W.shape
    bs = 4
    S = np.sign(W); S[S == 0] = 1.0
    M = np.abs(W)
    ph, pw = (bs - h % bs) % bs, (bs - w % bs) % bs
    Mp = np.pad(M, ((0, ph), (0, pw)))
    Hp = np.pad(np.tile(H_diag, (h, 1)), ((0, ph), (0, pw)), constant_values=1e-6)
    hp, wp = Mp.shape
    blk = Mp.reshape(hp//bs, bs, wp//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    wts = Hp.reshape(hp//bs, bs, wp//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    C, a = hessian_kmeans(blk, wts, K, seed=seed)
    rec = C[a].reshape(hp//bs, wp//bs, bs, bs).transpose(0, 2, 1, 3).reshape(hp, wp)
    return S * rec[:h, :w]


def quantize_ternary(W):
    S = np.sign(W); S[S == 0] = 1
    thr = np.percentile(np.abs(W), 30)
    sc = np.mean(np.abs(W[np.abs(W) > thr])) if (np.abs(W) > thr).any() else 1.0
    return S * (np.abs(W) > thr) * sc


def quantize_rtn(W, bits):
    """Round-to-nearest uniform quantization."""
    levels = 2 ** bits
    w_min, w_max = W.min(), W.max()
    scale = (w_max - w_min) / (levels - 1)
    W_q = np.round((W - w_min) / scale) * scale + w_min
    return W_q


# Load model
print("\nLoading GPT-2...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Get weight and activations
layer = model.transformer.h[0].mlp.c_fc
W = layer.weight.detach().cpu().numpy().T
d_out, d_in = W.shape

calib_text = "The quick brown fox jumps over the lazy dog. " * 30
inputs = tokenizer(calib_text, return_tensors="pt", max_length=512, truncation=True)

acts = []
def hook(m, i, o): acts.append(i[0].detach().cpu().numpy())
h = layer.register_forward_hook(hook)
model(inputs.input_ids)
h.remove()
X_calib = acts[0].reshape(-1, d_in)
H_diag = np.mean(X_calib**2, axis=0)


print("\n" + "=" * 80)
print("TEST 1: STATISTICAL SIGNIFICANCE (10 random seeds)")
print("=" * 80)

tern_corrs = []
vq_corrs = []
X_test = X_calib[:200]
Y_test = X_test @ W.T

W_tern = quantize_ternary(W)
corr_tern = np.corrcoef((X_test @ W_tern.T).ravel(), Y_test.ravel())[0, 1]

for seed in range(10):
    W_vq = quantize_hessianvq(W, H_diag, K=128, seed=seed)
    corr_vq = np.corrcoef((X_test @ W_vq.T).ravel(), Y_test.ravel())[0, 1]
    vq_corrs.append(corr_vq)
    tern_corrs.append(corr_tern)

print(f"Ternary:     {np.mean(tern_corrs):.4f} ± {np.std(tern_corrs):.4f}")
print(f"HessianVQ:   {np.mean(vq_corrs):.4f} ± {np.std(vq_corrs):.4f}")
print(f"Min VQ:      {np.min(vq_corrs):.4f}")
print(f"Max VQ:      {np.max(vq_corrs):.4f}")
print(f"VERDICT:     {'PASS' if np.min(vq_corrs) > corr_tern else 'FAIL'} - VQ beats ternary on ALL seeds")


print("\n" + "=" * 80)
print("TEST 2: CORRELATION vs ACTUAL ERROR (L2, Linf, MSE)")
print("=" * 80)

W_vq = quantize_hessianvq(W, H_diag, K=128)
W_tern = quantize_ternary(W)

def compute_errors(W_orig, W_q, X_test):
    Y_orig = X_test @ W_orig.T
    Y_q = X_test @ W_q.T

    corr = np.corrcoef(Y_orig.ravel(), Y_q.ravel())[0, 1]
    l2_rel = np.linalg.norm(Y_q - Y_orig) / np.linalg.norm(Y_orig)
    linf = np.max(np.abs(Y_q - Y_orig))
    mse = np.mean((Y_q - Y_orig)**2)

    return {'corr': corr, 'l2_rel': l2_rel, 'linf': linf, 'mse': mse}

err_tern = compute_errors(W, W_tern, X_test)
err_vq = compute_errors(W, W_vq, X_test)

print(f"{'Metric':<12} {'Ternary':>12} {'HessianVQ':>12} {'Winner':>10}")
print("-" * 50)
for metric in ['corr', 'l2_rel', 'linf', 'mse']:
    t, v = err_tern[metric], err_vq[metric]
    if metric == 'corr':
        winner = 'VQ' if v > t else 'Tern'
    else:
        winner = 'VQ' if v < t else 'Tern'
    print(f"{metric:<12} {t:>12.4f} {v:>12.4f} {winner:>10}")


print("\n" + "=" * 80)
print("TEST 3: OVERFITTING TO CALIBRATION DATA")
print("=" * 80)

# Use DIFFERENT text for testing
test_text = "Machine learning models require careful evaluation. " * 30
test_inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)

acts2 = []
h = layer.register_forward_hook(lambda m, i, o: acts2.append(i[0].detach().cpu().numpy()))
model(test_inputs.input_ids)
h.remove()
X_ood = acts2[0].reshape(-1, d_in)[:200]  # Out-of-distribution test
Y_ood = X_ood @ W.T

corr_tern_ood = np.corrcoef((X_ood @ W_tern.T).ravel(), Y_ood.ravel())[0, 1]
corr_vq_ood = np.corrcoef((X_ood @ W_vq.T).ravel(), Y_ood.ravel())[0, 1]

print(f"On calibration data: VQ={np.mean(vq_corrs):.4f}, Tern={corr_tern:.4f}")
print(f"On OOD test data:    VQ={corr_vq_ood:.4f}, Tern={corr_tern_ood:.4f}")
print(f"VQ degradation:      {(np.mean(vq_corrs) - corr_vq_ood)*100:.2f}%")
print(f"VERDICT:             {'PASS' if corr_vq_ood > corr_tern_ood else 'FAIL'}")


print("\n" + "=" * 80)
print("TEST 4: COMPARISON TO OTHER BASELINES (RTN 2/3/4-bit)")
print("=" * 80)

for bits in [2, 3, 4]:
    W_rtn = quantize_rtn(W, bits)
    corr_rtn = np.corrcoef((X_test @ W_rtn.T).ravel(), Y_test.ravel())[0, 1]
    bpp = bits
    print(f"RTN {bits}-bit @ {bpp:.2f} bpp: corr={corr_rtn:.4f}")

print(f"Ternary @ 1.58 bpp:    corr={corr_tern:.4f}")
print(f"HessianVQ @ 0.94 bpp:  corr={np.mean(vq_corrs):.4f}")
print(f"\nVERDICT: HessianVQ at 0.94 bpp vs RTN 2-bit at 2.0 bpp:")
W_rtn2 = quantize_rtn(W, 2)
corr_rtn2 = np.corrcoef((X_test @ W_rtn2.T).ravel(), Y_test.ravel())[0, 1]
print(f"         {'PASS' if np.mean(vq_corrs) > corr_rtn2 else 'FAIL'} - VQ ({np.mean(vq_corrs):.4f}) vs RTN-2 ({corr_rtn2:.4f})")


print("\n" + "=" * 80)
print("TEST 5: WEIGHT DISTRIBUTION EDGE CASES")
print("=" * 80)

# Test on different layers with different distributions
test_layers = [
    ("L0.c_attn", model.transformer.h[0].attn.c_attn),
    ("L0.c_proj", model.transformer.h[0].attn.c_proj),
    ("L11.c_fc", model.transformer.h[11].mlp.c_fc),
    ("L11.c_proj", model.transformer.h[11].mlp.c_proj),
]

print(f"{'Layer':<12} {'W_std':>8} {'Ternary':>10} {'VQ':>10} {'Winner':>8}")
print("-" * 52)

for name, lyr in test_layers:
    W_l = lyr.weight.detach().cpu().numpy().T
    acts_l = []
    hl = lyr.register_forward_hook(lambda m, i, o: acts_l.append(i[0].detach().cpu().numpy()))
    model(inputs.input_ids)
    hl.remove()
    X_l = acts_l[0].reshape(-1, W_l.shape[1])[:200]
    H_l = np.mean(X_l**2, axis=0)
    Y_l = X_l @ W_l.T

    W_tern_l = quantize_ternary(W_l)
    W_vq_l = quantize_hessianvq(W_l, H_l, K=128)

    corr_t = np.corrcoef((X_l @ W_tern_l.T).ravel(), Y_l.ravel())[0, 1]
    corr_v = np.corrcoef((X_l @ W_vq_l.T).ravel(), Y_l.ravel())[0, 1]

    winner = "VQ" if corr_v > corr_t else "TERN"
    print(f"{name:<12} {np.std(W_l):>8.4f} {corr_t:>10.4f} {corr_v:>10.4f} {winner:>8}")


print("\n" + "=" * 80)
print("TEST 6: SMALL MATRIX EDGE CASE")
print("=" * 80)

# Create a small matrix where block VQ might struggle
W_small = np.random.randn(16, 16) * 0.1
H_small = np.ones(16)
X_small = np.random.randn(50, 16) * 0.1
Y_small = X_small @ W_small.T

W_tern_s = quantize_ternary(W_small)
W_vq_s = quantize_hessianvq(W_small, H_small, K=16)  # Fewer codes for small matrix

corr_tern_s = np.corrcoef((X_small @ W_tern_s.T).ravel(), Y_small.ravel())[0, 1]
corr_vq_s = np.corrcoef((X_small @ W_vq_s.T).ravel(), Y_small.ravel())[0, 1]

print(f"16x16 matrix with K=16:")
print(f"Ternary: {corr_tern_s:.4f}")
print(f"VQ:      {corr_vq_s:.4f}")
print(f"VERDICT: {'PASS' if corr_vq_s > corr_tern_s else 'FAIL - VQ struggles on small matrices!'}")


print("\n" + "=" * 80)
print("TEST 7: EXTREME SPARSITY (90% zeros)")
print("=" * 80)

W_sparse = W.copy()
mask = np.random.rand(*W.shape) < 0.9
W_sparse[mask] = 0

W_tern_sp = quantize_ternary(W_sparse)
W_vq_sp = quantize_hessianvq(W_sparse, H_diag, K=128)
Y_sp = X_test @ W_sparse.T

corr_tern_sp = np.corrcoef((X_test @ W_tern_sp.T).ravel(), Y_sp.ravel())[0, 1]
corr_vq_sp = np.corrcoef((X_test @ W_vq_sp.T).ravel(), Y_sp.ravel())[0, 1]

print(f"90% sparse matrix:")
print(f"Ternary: {corr_tern_sp:.4f}")
print(f"VQ:      {corr_vq_sp:.4f}")
print(f"VERDICT: {'PASS' if corr_vq_sp > corr_tern_sp else 'FAIL - Ternary wins on sparse!'}")



print("\n" + "=" * 80)
print("TEST 8: THE REAL TEST - FULL MODEL PERPLEXITY")
print("=" * 80)
print("This is the test that matters. Per-layer correlation means nothing")
print("if the full model produces garbage.")

import math

# Evaluation text (different from calibration)
eval_text = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow models to focus on relevant parts of the input.
Large language models like GPT-4 have billions of parameters.
Quantization reduces memory requirements but may hurt accuracy."""

eval_inputs = tokenizer(eval_text, return_tensors="pt")

# Get original PPL
with torch.no_grad():
    outputs = model(eval_inputs.input_ids, labels=eval_inputs.input_ids)
    ppl_orig = math.exp(outputs.loss.item())
print(f"FP16 Baseline PPL: {ppl_orig:.2f}")

# Now quantize ALL layers and test
import torch
import copy

def apply_ternary_to_model(model):
    """Apply ternary quantization to all linear layers."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'ln' not in name and 'wte' not in name and 'wpe' not in name:
            W = module.weight.detach().cpu().numpy()
            if W.ndim == 2:
                W_q = quantize_ternary(W.T).T
                module.weight.data = torch.tensor(W_q, dtype=torch.float32)

def apply_vq_to_model(model, K=128):
    """Apply VQ to all linear layers (simplified - same H for all)."""
    # First pass: collect activations for each layer
    # Simplified: use uniform Hessian
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'ln' not in name and 'wte' not in name and 'wpe' not in name:
            W = module.weight.detach().cpu().numpy()
            if W.ndim == 2:
                # GPT-2 Conv1D: weight is (in, out), need to handle properly
                if W.shape[0] < W.shape[1]:  # Likely Conv1D
                    W_t = W.T  # (out, in)
                    H_diag_u = np.ones(W_t.shape[1])
                    W_q = quantize_hessianvq(W_t, H_diag_u, K=K).T
                else:
                    H_diag_u = np.ones(W.shape[1])
                    W_q = quantize_hessianvq(W, H_diag_u, K=K)
                module.weight.data = torch.tensor(W_q, dtype=torch.float32)

# Test Ternary full model
model_tern = copy.deepcopy(model)
apply_ternary_to_model(model_tern)
with torch.no_grad():
    outputs = model_tern(eval_inputs.input_ids, labels=eval_inputs.input_ids)
    ppl_tern = math.exp(outputs.loss.item())
print(f"Ternary Full Model PPL: {ppl_tern:.2f} ({ppl_tern/ppl_orig:.1f}x baseline)")

# Test VQ full model
model_vq = copy.deepcopy(model)
apply_vq_to_model(model_vq, K=128)
with torch.no_grad():
    outputs = model_vq(eval_inputs.input_ids, labels=eval_inputs.input_ids)
    ppl_vq = math.exp(outputs.loss.item())
print(f"HessianVQ Full Model PPL: {ppl_vq:.2f} ({ppl_vq/ppl_orig:.1f}x baseline)")

print(f"\n*** CRITICAL VERDICT ***")
if ppl_vq < ppl_tern:
    print(f"PASS: VQ PPL ({ppl_vq:.2f}) < Ternary PPL ({ppl_tern:.2f})")
elif ppl_vq < 100:
    print(f"PARTIAL: VQ PPL ({ppl_vq:.2f}) usable but > Ternary ({ppl_tern:.2f})")
else:
    print(f"FAIL: VQ PPL ({ppl_vq:.2f}) is catastrophic! Paper claims may be misleading.")


print("\n" + "=" * 80)
print("TEST 9: LAYER-WISE ERROR ACCUMULATION")
print("=" * 80)

# Track error as we go through layers
model_fresh = GPT2LMHeadModel.from_pretrained('gpt2')

def get_layer_output(model, layer_idx, input_ids):
    """Get output after specific layer."""
    with torch.no_grad():
        hidden = model.transformer.wte(input_ids) + model.transformer.wpe(
            torch.arange(input_ids.shape[1]).unsqueeze(0))
        for i, block in enumerate(model.transformer.h[:layer_idx+1]):
            hidden = block(hidden)[0]
    return hidden

# Compare layer outputs progressively
print(f"{'Layer':>6} {'FP16 norm':>12} {'VQ norm':>12} {'Rel Error':>12}")
print("-" * 50)

for layer_idx in [0, 3, 6, 9, 11]:
    h_orig = get_layer_output(model, layer_idx, eval_inputs.input_ids)
    h_vq = get_layer_output(model_vq, layer_idx, eval_inputs.input_ids)

    norm_orig = torch.norm(h_orig).item()
    norm_vq = torch.norm(h_vq).item()
    rel_err = torch.norm(h_vq - h_orig).item() / norm_orig

    print(f"{layer_idx:>6} {norm_orig:>12.2f} {norm_vq:>12.2f} {rel_err:>12.4f}")

print("\nIf relative error grows exponentially, that's why PPL fails.")


print("\n" + "=" * 80)
print("TEST 10: BPP CALCULATION VERIFICATION")
print("=" * 80)

# Verify our BPP calculation is honest
def verify_bpp(W, K=128, block_size=4):
    h, w = W.shape
    n_weights = h * w
    bs = block_size

    # Pad
    ph, pw = (bs - h % bs) % bs, (bs - w % bs) % bs
    hp, wp = h + ph, w + pw
    n_blocks = (hp // bs) * (wp // bs)

    # Signs: entropy-coded
    signs = np.sign(W)
    pos = np.sum(signs > 0) / n_weights
    neg = np.sum(signs < 0) / n_weights
    zero = np.sum(signs == 0) / n_weights
    sign_entropy = 0
    for p in [pos, neg, zero]:
        if p > 0:
            sign_entropy -= p * np.log2(p)
    sign_bits = sign_entropy * n_weights

    # Indices: assuming uniform distribution (worst case)
    index_bits = n_blocks * np.log2(K)

    # Codebook overhead
    codebook_bits = K * (bs * bs) * 16

    total_bits = sign_bits + index_bits + codebook_bits
    bpp = total_bits / n_weights

    print(f"Weight matrix: {h} x {w} = {n_weights:,} weights")
    print(f"Blocks: {n_blocks:,}")
    print(f"Sign bits: {sign_bits:,.0f} ({sign_entropy:.3f} entropy)")
    print(f"Index bits: {index_bits:,.0f} ({np.log2(K):.2f} per block)")
    print(f"Codebook bits: {codebook_bits:,}")
    print(f"Total: {total_bits:,.0f} bits")
    print(f"BPP: {bpp:.4f}")

    # What we claimed
    claimed_bpp = 0.94
    print(f"\nClaimed BPP: {claimed_bpp}")
    print(f"Actual BPP:  {bpp:.4f}")
    print(f"VERDICT: {'HONEST' if abs(bpp - claimed_bpp) < 0.1 else 'DISHONEST!'}")

    return bpp

verify_bpp(W, K=128)


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

