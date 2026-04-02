"""Test Hybrid Block approach on REAL GPT-2 weights."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
np.random.seed(42)

def hybrid_block_quantize(W, block_size=4, n_levels=2):
    """Hybrid quantization: block signs + magnitude levels."""
    d_out, d_in = W.shape
    
    # Step 1: Block signs (majority sign per block)
    W_block_signs = np.zeros_like(W)
    n_blocks_h = d_out // block_size
    n_blocks_w = d_in // block_size
    
    for bi in range(n_blocks_h):
        for bj in range(n_blocks_w):
            block = W[bi*block_size:(bi+1)*block_size,
                      bj*block_size:(bj+1)*block_size]
            block_sign = np.sign(np.sum(block))  # majority sign
            if block_sign == 0:
                block_sign = 1
            W_block_signs[bi*block_size:(bi+1)*block_size,
                         bj*block_size:(bj+1)*block_size] = block_sign
    
    # Step 2: Magnitude levels
    magnitudes = np.abs(W)
    thresholds = [np.percentile(magnitudes, 100 * i / n_levels) 
                  for i in range(1, n_levels)]
    
    mag_levels = np.zeros_like(magnitudes)
    level_values = np.linspace(0.2, 1.0, n_levels)  # scaled values
    
    for i in range(n_levels):
        if i == 0:
            mask = magnitudes < thresholds[0]
        elif i == n_levels - 1:
            mask = magnitudes >= thresholds[-1]
        else:
            mask = (magnitudes >= thresholds[i-1]) & (magnitudes < thresholds[i])
        mag_levels[mask] = level_values[i]
    
    # Combine: sign * magnitude_level * scale
    scale = np.mean(np.abs(W))
    W_hybrid = W_block_signs * mag_levels * scale
    
    # Calculate BPP
    n_blocks = n_blocks_h * n_blocks_w
    sign_bits = n_blocks  # 1 bit per block
    mag_bits = np.log2(n_levels) * W.size  # log2(levels) bits per weight
    bpp = (sign_bits + mag_bits) / W.size
    
    return W_hybrid, bpp

print('Loading GPT-2...')
model = GPT2LMHeadModel.from_pretrained('gpt2')

weight_names = [
    ('h.0.mlp.c_fc', model.transformer.h[0].mlp.c_fc.weight),
    ('h.0.mlp.c_proj', model.transformer.h[0].mlp.c_proj.weight),
    ('h.5.mlp.c_fc', model.transformer.h[5].mlp.c_fc.weight),
    ('h.5.mlp.c_proj', model.transformer.h[5].mlp.c_proj.weight),
    ('h.11.mlp.c_fc', model.transformer.h[11].mlp.c_fc.weight),
]

all_results = []

for name, W_fp32 in weight_names:
    W = W_fp32.detach().numpy()
    if W.shape[0] < W.shape[1]:
        W = W.T
    
    d_out, d_in = W.shape
    
    # Test data
    n_samples = 500
    X = np.random.randn(n_samples, d_in).astype(np.float32)
    Y_fp32 = X @ W.T
    X_test, Y_test = X[400:], Y_fp32[400:]
    
    results = {}
    
    # Binary
    W_bin = np.sign(W)
    W_bin = np.where(W_bin == 0, 1, W_bin)
    scale = np.mean(np.abs(W))
    Y_pred = X_test @ (W_bin * scale).T
    corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
    results['Binary'] = (corr, 1.0)
    
    # Ternary
    thresh = 0.3 * scale
    W_tern = np.zeros_like(W)
    W_tern[W > thresh] = scale
    W_tern[W < -thresh] = -scale
    Y_pred = X_test @ W_tern.T
    corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
    results['Ternary'] = (corr, 1.58)
    
    # Hybrid Block-4
    W_hybrid, bpp = hybrid_block_quantize(W, block_size=4, n_levels=2)
    Y_pred = X_test @ W_hybrid.T
    corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
    results['Hybrid Block-4'] = (corr, bpp)
    
    # Hybrid Block-2
    W_hybrid, bpp = hybrid_block_quantize(W, block_size=2, n_levels=2)
    Y_pred = X_test @ W_hybrid.T
    corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
    results['Hybrid Block-2'] = (corr, bpp)
    
    # LowRank r=2 for comparison
    W_abs = np.abs(W)
    U, S, Vh = np.linalg.svd(W_abs, full_matrices=False)
    r = 2
    W_abs_approx = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
    W_lowrank = W_bin * W_abs_approx
    Y_pred = X_test @ W_lowrank.T
    corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
    n_params = r * (d_out + d_in + 1)
    bpp = 1.0 + 32 * n_params / W.size
    results['LowRank r=2'] = (corr, bpp)
    
    all_results.append((name, results))

# Aggregate
print()
print('=' * 80)
print('HYBRID BLOCK vs LOWRANK on GPT-2 WEIGHTS')
print('=' * 80)

agg = {}
for name, results in all_results:
    for method, (corr, bpp) in results.items():
        if method not in agg:
            agg[method] = []
        agg[method].append((corr, bpp))

tern_avg = np.mean([c for c, b in agg['Ternary']])

print()
print(f"{'Method':<20} {'Avg Corr':>10} {'BPP':>8} {'vs Ternary':>12}")
print('-' * 55)

for method in ['Binary', 'Hybrid Block-4', 'Hybrid Block-2', 'LowRank r=2', 'Ternary']:
    avg_corr = np.mean([c for c, b in agg[method]])
    avg_bpp = np.mean([b for c, b in agg[method]])
    vs_tern = ((avg_corr / tern_avg) - 1) * 100
    marker = ' ★' if vs_tern >= 0 else ''
    print(f'{method:<20} {avg_corr:>10.4f} {avg_bpp:>8.3f} {vs_tern:>+11.1f}%{marker}')

