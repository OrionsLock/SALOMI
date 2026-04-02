"""Test Row+Col scaling variants with different scale bit-widths."""
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
    ('h.11.mlp.c_proj', model.transformer.h[11].mlp.c_proj.weight),
    ('h.0.attn.c_attn', model.transformer.h[0].attn.c_attn.weight),
    ('h.5.attn.c_attn', model.transformer.h[5].attn.c_attn.weight),
]

all_results = {}

def quantize_scales(s, bits=8):
    s_min, s_max = s.min(), s.max()
    s_norm = (s - s_min) / (s_max - s_min + 1e-8)
    s_quant = torch.round(s_norm * (2**bits - 1)) / (2**bits - 1)
    return s_quant * (s_max - s_min) + s_min

for name, W_fp32 in weight_names:
    W = W_fp32.detach().float()
    if W.shape[0] < W.shape[1]:
        W = W.T
    
    out_f, in_f = W.shape
    n_weights = out_f * in_f
    
    n_samples = 500
    X = torch.randn(n_samples, in_f) * 0.5
    sparse_mask = torch.rand(n_samples, in_f) > 0.9
    X = X + sparse_mask.float() * torch.randn(n_samples, in_f) * 2.0
    Y_fp32 = X @ W.T
    X_test, Y_test = X[400:], Y_fp32[400:]
    
    results = {}
    
    # Binary
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
    
    # Row + Col scales (32-bit)
    row_scales = torch.abs(W).mean(dim=1, keepdim=True)
    col_scales = torch.abs(W).mean(dim=0, keepdim=True)
    W_rowcol = W_bin * row_scales * col_scales / scale
    Y_pred = X_test @ W_rowcol.T
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    bpp = 1.0 + 32 * (out_f + in_f) / n_weights
    results['Row+Col (32-bit)'] = (corr, bpp)
    
    # 8-bit scales
    row_q8 = quantize_scales(row_scales, 8)
    col_q8 = quantize_scales(col_scales, 8)
    W_q8 = W_bin * row_q8 * col_q8 / scale
    Y_pred = X_test @ W_q8.T
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    bpp = 1.0 + 8 * (out_f + in_f) / n_weights
    results['Row+Col (8-bit)'] = (corr, bpp)
    
    # 4-bit scales
    row_q4 = quantize_scales(row_scales, 4)
    col_q4 = quantize_scales(col_scales, 4)
    W_q4 = W_bin * row_q4 * col_q4 / scale
    Y_pred = X_test @ W_q4.T
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    bpp = 1.0 + 4 * (out_f + in_f) / n_weights
    results['Row+Col (4-bit)'] = (corr, bpp)
    
    # 2-bit scales
    row_q2 = quantize_scales(row_scales, 2)
    col_q2 = quantize_scales(col_scales, 2)
    W_q2 = W_bin * row_q2 * col_q2 / scale
    Y_pred = X_test @ W_q2.T
    corr = np.corrcoef(Y_pred.numpy().flatten(), Y_test.numpy().flatten())[0, 1]
    bpp = 1.0 + 2 * (out_f + in_f) / n_weights
    results['Row+Col (2-bit)'] = (corr, bpp)
    
    for method, (corr, bpp) in results.items():
        if method not in all_results:
            all_results[method] = []
        all_results[method].append((corr, bpp))

# Print results
print()
print('=' * 75)
print('ROW+COL SCALING: Quantized Scale Variants')
print('=' * 75)

tern_avg = np.mean([c for c, b in all_results['Ternary']])

print()
header = f"{'Method':<22} {'Avg Corr':>10} {'BPP':>8} {'vs Ternary':>12} {'vs 1.58':>10}"
print(header)
print('-' * 65)

for method in ['Binary', 'Row+Col (2-bit)', 'Row+Col (4-bit)', 
               'Row+Col (8-bit)', 'Row+Col (32-bit)', 'Ternary']:
    if method in all_results:
        avg_corr = np.mean([c for c, b in all_results[method]])
        avg_bpp = np.mean([b for c, b in all_results[method]])
        vs_tern = ((avg_corr / tern_avg) - 1) * 100
        savings = (1.58 - avg_bpp) / 1.58 * 100
        
        marker = ' *BEST*' if vs_tern > -3 and avg_bpp < 1.2 else ''
        print(f'{method:<22} {avg_corr:>10.4f} {avg_bpp:>8.3f} {vs_tern:>+11.1f}% {savings:>+9.1f}%{marker}')

print()
print('vs 1.58 = storage savings compared to ternary')

