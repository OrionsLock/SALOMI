"""Comprehensive 1-bit quantization analysis."""
from onebit.research.unified_1bit import *

print('=' * 80)
print('COMPREHENSIVE 1-BIT QUANTIZATION ANALYSIS')
print('=' * 80)

for dim in [128, 256]:
    print(f'\n\n### DIMENSION: {dim}x{dim} ###\n')
    cfg = UnifiedConfig(d_out=dim, d_in=dim)
    results = run_comparison(cfg, verbose=False)
    
    # Sort by BPP
    results_sorted = sorted(results, key=lambda x: x['bpp'])
    
    print(f"{'Mode':<15} {'BPP':>8} {'Corr':>8} {'Gap':>8}")
    print('-' * 45)
    
    ternary_corr = next(r['correlation'] for r in results if r['mode'] == 'ternary')
    
    for r in results_sorted:
        gap = ternary_corr - r['correlation']
        marker = 'WIN' if gap <= 0 and r['bpp'] < 1.58 else ''
        print(f"{r['mode']:<15} {r['bpp']:>8.3f} {r['correlation']:>8.4f} {gap:>+8.4f} {marker}")
    
    # Find best at each BPP threshold
    print(f'\nBest methods:')
    for bpp_thresh in [1.0, 1.25, 1.5]:
        candidates = [r for r in results if r['bpp'] <= bpp_thresh]
        if candidates:
            best = max(candidates, key=lambda x: x['correlation'])
            print(f"  At <={bpp_thresh:.2f} bpp: {best['mode']} (corr={best['correlation']:.4f})")

print('\n\n' + '=' * 80)
print('CONCLUSION')
print('=' * 80)
print("""
For POST-TRAINING quantization of random Gaussian weights:

1. Sign-only at ~1.0 bpp: ~0.80 correlation
2. Ternary at 1.58 bpp:   ~0.86 correlation
3. Gap:                   ~0.06 (6%)

The 0.58 bits in ternary encode WHERE the zeros are.
For random weights, this is random information - no structure to exploit.

To beat ternary at 1.0 bpp, you need TRAINING-AWARE approaches:
- Train with binary constraint (STE)
- Network learns to encode all info in signs
- Magnitudes become uniform or structured

This is how BitNet achieves good results - it's trained from scratch
with 1-bit weights, not post-training quantized.
""")

