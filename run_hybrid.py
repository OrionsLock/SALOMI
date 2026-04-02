"""Test hybrid approach: block-structured signs + magnitude levels.

ChatGPT's key insight: If we use block-structured signs (compressible),
we can use the freed bits for magnitude information. This is how we
can match ternary at 1.0 bpp effective.
"""
from onebit.research.unified_1bit import EntropyShapedBinary
import numpy as np

print("=" * 80)
print("HYBRID: BLOCK-STRUCTURED SIGNS + MAGNITUDE LEVELS")
print("=" * 80)
print("""
Budget allocation for 1.0 bpp target:
- Block signs (2x2): 0.25 bpp
- Magnitude levels: 0.75 bpp -> ~2-3 levels per weight

This combines:
1. Spatial structure (block signs) for compression
2. Magnitude information (levels) for quality
""")

for dim in [64, 128]:
    print(f"\n{'='*80}")
    print(f"DIMENSION: {dim}x{dim}")
    print("="*80)
    
    esb = EntropyShapedBinary(d_out=dim, d_in=dim, n_iters=3000, lr=0.05)
    results = esb.compare_hybrid(n_samples=2000)
    
    print(f"\n{'Method':<25} {'Corr':>10} {'BPP':>8} {'Levels':>8}")
    print("-" * 55)
    
    for key, r in results.items():
        levels = r.get('n_levels', '-')
        print(f"{key:<25} {r['correlation']:>10.4f} {r['bpp']:>8.3f} {str(levels):>8}")
    
    # Analysis
    ternary_corr = results['ternary']['correlation']
    binary_corr = results['binary']['correlation']
    
    print(f"\n--- Analysis ---")
    print(f"Ternary (1.58 bpp):     {ternary_corr:.4f}")
    print(f"Binary (1.0 bpp):       {binary_corr:.4f}")
    
    # Find best hybrid under 1.2 bpp (allowing some overhead)
    best_hybrid = None
    for key, r in results.items():
        if 'hybrid' in key:
            if best_hybrid is None or r['correlation'] > best_hybrid['correlation']:
                best_hybrid = {'key': key, **r}
    
    if best_hybrid:
        print(f"\nBest hybrid: {best_hybrid['key']}")
        print(f"  Correlation: {best_hybrid['correlation']:.4f}")
        print(f"  Effective BPP: {best_hybrid['bpp']:.3f}")
        print(f"  Magnitude levels: {best_hybrid['n_levels']}")
        
        gap_binary = ternary_corr - binary_corr
        gap_hybrid = ternary_corr - best_hybrid['correlation']
        
        if gap_hybrid < gap_binary:
            improvement = (gap_binary - gap_hybrid) / gap_binary * 100
            print(f"\n  ✓ Hybrid reduces gap by {improvement:.1f}%!")
        
        if best_hybrid['correlation'] >= ternary_corr and best_hybrid['bpp'] <= 1.1:
            print(f"\n  ★★★ HYBRID MATCHES TERNARY AT ~1.0 BPP! ★★★")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
The hybrid approach combines:
1. Block-structured signs (compressible -> saves bits)
2. Magnitude levels (uses freed bits for quality)

If block signs compress to 0.25 bpp (2x2 blocks), we have 0.75 bpp
for magnitude levels, which allows 2-4 levels per weight.

This is ChatGPT's key insight: total file size / weights = 1.0 bpp,
not "1 raw bit per weight independently".
""")

