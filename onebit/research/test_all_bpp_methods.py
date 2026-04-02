"""Comprehensive test of all methods at various BPP levels."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel

torch.manual_seed(42)

print('Loading GPT-2...')
model = GPT2LMHeadModel.from_pretrained('gpt2')

weight_names = [
    ('h.0.mlp.c_fc', model.transformer.h[0].mlp.c_fc.weight),
    ('h.0.mlp.c_proj', model.transformer.h[0].mlp.c_proj.weight),
    ('h.5.mlp.c_fc', model.transformer.h[5].mlp.c_fc.weight),
    ('h.5.mlp.c_proj', model.transformer.h[5].mlp.c_proj.weight),
    ('h.11.mlp.c_fc', model.transformer.h[11].mlp.c_fc.weight),
    ('h.11.mlp.c_proj', model.transformer.h[11].mlp.c_proj.weight),
    ('h.0.attn.c_attn', model.transformer.h[0].attn.c_attn.weight),
    ('h.5.attn.c_attn', model.transformer.h[5].attn.c_attn.weight),
]

all_results = {}

for name, W_fp32 in weight_names:
    W = W_fp32.detach().float()
    if W.shape[0] < W.shape[1]:
        W = W.T
    
    out_f, in_f = W.shape
    n_weights = out_f * in_f
    
    # Test data
    n_samples = 500
    X = torch.randn(n_samples, in_f)
    Y_fp32 = X @ W.T
    X_test, Y_test = X[400:], Y_fp32[400:]
    
    results = {}
    
    # Binary baseline
    W_bin = torch.sign(W)
    W_bin = torch.where(W_bin == 0, torch.ones_like(W_bin), W_bin)
    scale = torch.mean(torch.abs(W))
    Y_pred = X_test @ (W_bin * scale).T
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    results['Binary'] = (corr, 1.0)
    
    # Ternary
    thresh = 0.3 * scale
    W_tern = torch.zeros_like(W)
    W_tern[W > thresh] = scale
    W_tern[W < -thresh] = -scale
    Y_pred = X_test @ W_tern.T
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    results['Ternary'] = (corr, 1.58)
    
    # Row+Col scales (32-bit)
    row_scales = torch.abs(W).mean(dim=1, keepdim=True)
    col_scales = torch.abs(W).mean(dim=0, keepdim=True)
    W_rowcol = W_bin * row_scales * col_scales / scale
    Y_pred = X_test @ W_rowcol.T
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    bpp = 1.0 + 32 * (out_f + in_f) / n_weights
    results['Row+Col'] = (corr, bpp)
    
    # LowRank magnitude approximation
    W_abs = torch.abs(W)
    for rank in [1, 2, 4, 8, 16]:
        U, S, Vh = torch.linalg.svd(W_abs, full_matrices=False)
        W_abs_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
        W_lowrank = W_bin * W_abs_approx
        Y_pred = X_test @ W_lowrank.T
        corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
        n_lowrank = rank * (out_f + in_f + 1)
        bpp = 1.0 + 32 * n_lowrank / n_weights
        results[f'LowRank r={rank}'] = (corr, bpp)
    
    for method, (corr, bpp) in results.items():
        if method not in all_results:
            all_results[method] = []
        all_results[method].append((corr, bpp))

# Print results
print()
print('=' * 80)
print('ALL METHODS: Quality vs BPP Trade-off')
print('=' * 80)

tern_avg = np.mean([c for c, b in all_results['Ternary']])
bin_avg = np.mean([c for c, b in all_results['Binary']])

print()
print(f"{'Method':<20} {'Avg Corr':>10} {'BPP':>8} {'vs Ternary':>12} {'vs Binary':>12}")
print('-' * 70)

methods_sorted = sorted(all_results.keys(), key=lambda m: np.mean([b for c, b in all_results[m]]))

for method in methods_sorted:
    avg_corr = np.mean([c for c, b in all_results[method]])
    avg_bpp = np.mean([b for c, b in all_results[method]])
    vs_tern = ((avg_corr / tern_avg) - 1) * 100
    vs_bin = ((avg_corr / bin_avg) - 1) * 100
    
    marker = ''
    if vs_tern >= -1:
        marker = ' *** MATCHES TERNARY'
    elif vs_tern >= -3:
        marker = ' ** CLOSE'
    elif vs_tern >= -5:
        marker = ' * GOOD'
    
    print(f'{method:<20} {avg_corr:>10.4f} {avg_bpp:>8.3f} {vs_tern:>+11.1f}% {vs_bin:>+11.1f}%{marker}')

print()
print('SUMMARY:')
print(f'  Binary baseline: {bin_avg:.4f} at 1.00 bpp')
print(f'  Ternary baseline: {tern_avg:.4f} at 1.58 bpp')
print(f'  Gap to close: {((tern_avg/bin_avg)-1)*100:.1f}%')

