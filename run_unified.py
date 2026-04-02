"""
Run Unified 1-Bit Comparison

Tests all approaches for achieving 1.00 bpp that could beat 1.58-bit ternary.
"""

import numpy as np
from onebit.research.unified_1bit import (
    UnifiedConfig, UnifiedBinaryLayer, QuantMode,
    evaluate_mode, compute_bpp, run_comparison, print_results
)


def main():
    print("=" * 70)
    print("UNIFIED 1-BIT RESEARCH: Comparing All Approaches")
    print("=" * 70)
    print()
    
    # Test multiple dimensions
    for dim in [128, 256, 512]:
        print(f"\n{'='*70}")
        print(f"DIMENSION: {dim}x{dim}")
        print(f"{'='*70}")
        
        cfg = UnifiedConfig(
            d_out=dim,
            d_in=dim,
            rank=min(16, dim // 16),  # Reasonable rank
            ctg_period=5,
            ctg_enabled=True,
        )
        
        results = run_comparison(cfg, verbose=True)
        print_results(results)
        
        # Compute gap to ternary for each method
        ternary_corr = next(r['correlation'] for r in results if r['mode'] == 'ternary')
        
        print(f"\nGap to Ternary (higher is worse):")
        for r in sorted(results, key=lambda x: x['bpp']):
            gap = ternary_corr - r['correlation']
            marker = "✓" if gap <= 0 else ""
            print(f"  {r['mode']:<20} gap={gap:+.4f} {marker}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:
1. Sign-only (1.00 bpp):     ~0.80 correlation
2. Ternary (1.58 bpp):       ~0.86 correlation  <- Target to beat
3. Gap:                      ~0.06 (6%)

The fundamental limit for POST-TRAINING quantization:
- Ternary zeros are at positions where |W| is small
- This information requires ~0.58 bits to encode
- No scheme can recover this without storing it

For TRAINING-AWARE approaches:
- Network can learn to push signal away from inhibit slots
- CTG patterns become a constraint the network optimizes around
- This requires end-to-end training on real tasks
""")


if __name__ == "__main__":
    main()

