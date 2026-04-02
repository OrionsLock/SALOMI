"""Test Block-Shared Magnitude on real GPT-2 weights."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
np.random.seed(42)

def block_magnitude_binary(W, block_size=16):
    """Binary signs + one magnitude scalar per block."""
    M, N = W.shape
    
    # Pad if needed
    M_pad = ((M + block_size - 1) // block_size) * block_size
    N_pad = ((N + block_size - 1) // block_size) * block_size
    W_padded = np.zeros((M_pad, N_pad))
    W_padded[:M, :N] = W
    
    # Signs (1 bit per weight)
    signs = np.sign(W_padded)
    signs[signs == 0] = 1
    
    # One magnitude per block
    n_blocks_m = M_pad // block_size
    n_blocks_n = N_pad // block_size
    block_mags = np.zeros((n_blocks_m, n_blocks_n))
    
    for i in range(n_blocks_m):
        for j in range(n_blocks_n):
            block = W_padded[i*block_size:(i+1)*block_size,
                            j*block_size:(j+1)*block_size]
            block_mags[i, j] = np.abs(block).mean()
    
    # Reconstruct
    W_recon = np.zeros_like(W_padded)
    for i in range(n_blocks_m):
        for j in range(n_blocks_n):
            W_recon[i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size] = signs[
                        i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size] * block_mags[i, j]
    
    # Calculate BPP
    n_weights = M * N
    sign_bits = n_weights  # 1 bit per weight
    mag_bits = n_blocks_m * n_blocks_n * 32  # 32 bits per block
    bpp = (sign_bits + mag_bits) / n_weights
    
    return W_recon[:M, :N], bpp

print('Loading GPT-2...')
model = GPT2LMHeadModel.from_pretrained('gpt2')

weight_matrices = [
    ('h.0.mlp.c_fc', model.transformer.h[0].mlp.c_fc.weight),
    ('h.0.mlp.c_proj', model.transformer.h[0].mlp.c_proj.weight),
    ('h.5.mlp.c_fc', model.transformer.h[5].mlp.c_fc.weight),
    ('h.5.mlp.c_proj', model.transformer.h[5].mlp.c_proj.weight),
    ('h.11.mlp.c_fc', model.transformer.h[11].mlp.c_fc.weight),
    ('h.11.mlp.c_proj', model.transformer.h[11].mlp.c_proj.weight),
]

all_results = {}

for name, W_t in weight_matrices:
    W = W_t.detach().numpy()
    if W.shape[0] < W.shape[1]:
        W = W.T
    
    M, N = W.shape
    
    # Test data
    n_samples = 500
    X = np.random.randn(n_samples, N).astype(np.float32)
    Y_fp32 = X @ W.T
    X_test, Y_test = X[400:], Y_fp32[400:]
    
    results = {}
    
    # Binary baseline
    W_bin = np.sign(W)
    W_bin[W_bin == 0] = 1
    scale = np.abs(W).mean()
    Y_pred = X_test @ (W_bin * scale).T
    corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
    results['Binary'] = (corr, 1.0)
    
    # Ternary baseline
    thresh = 0.3 * scale
    W_tern = np.zeros_like(W)
    W_tern[W > thresh] = scale
    W_tern[W < -thresh] = -scale
    Y_pred = X_test @ W_tern.T
    corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
    results['Ternary'] = (corr, 1.58)
    
    # Block-Shared Magnitude at different block sizes
    for block_size in [8, 16, 32, 64]:
        W_bm, bpp = block_magnitude_binary(W, block_size)
        Y_pred = X_test @ W_bm.T
        corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
        results[f'BlockMag b={block_size}'] = (corr, bpp)
    
    for method, (corr, bpp) in results.items():
        if method not in all_results:
            all_results[method] = []
        all_results[method].append((corr, bpp))

# Aggregate results
print()
print('=' * 80)
print('BLOCK-SHARED MAGNITUDE: Results on GPT-2 Weights')
print('=' * 80)

tern_avg = np.mean([c for c, b in all_results['Ternary']])

print()
print(f"{'Method':<20} {'Avg Corr':>10} {'BPP':>8} {'vs Ternary':>12} {'Overhead':>10}")
print('-' * 65)

methods = ['Binary', 'BlockMag b=64', 'BlockMag b=32', 'BlockMag b=16', 'BlockMag b=8', 'Ternary']
for method in methods:
    avg_corr = np.mean([c for c, b in all_results[method]])
    avg_bpp = np.mean([b for c, b in all_results[method]])
    vs_tern = ((avg_corr / tern_avg) - 1) * 100
    overhead = (avg_bpp - 1.0) * 100
    
    marker = ''
    if vs_tern >= -2 and avg_bpp < 1.58:
        marker = ' *** BEST'
    elif vs_tern >= -5 and avg_bpp < 1.3:
        marker = ' ** GOOD'
    
    print(f'{method:<20} {avg_corr:>10.4f} {avg_bpp:>8.3f} {vs_tern:>+11.1f}% {overhead:>+9.1f}%{marker}')

print()
print('Overhead = extra bits above 1.0 bpp')

