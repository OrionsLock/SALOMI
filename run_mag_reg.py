"""Test magnitude regularization for 1-bit training."""
from onebit.research.unified_1bit import TrainingAwareSimulation
import numpy as np

print("=" * 70)
print("MAGNITUDE REGULARIZATION FOR 1-BIT TRAINING")
print("=" * 70)
print("""
Key insight: If we regularize magnitudes to be uniform during training,
then sign-only quantization loses less information.

This is the key to making 1-bit work: train the network to put all
information in signs, not magnitudes.
""")

for dim in [64, 128]:
    print(f"\n### Dimension: {dim}x{dim} ###\n")
    
    sim = TrainingAwareSimulation(d_out=dim, d_in=dim, n_iters=3000, lr=0.05)
    results = sim.compare_with_mag_reg(n_samples=2000)
    
    print(f"{'Method':<20} {'Correlation':>12} {'Mag Var':>12}")
    print("-" * 50)
    
    for method, data in results.items():
        mag_var = data.get('mag_var', 0)
        print(f"{method:<20} {data['correlation']:>12.4f} {mag_var:>12.6f}")
    
    # Analysis
    fp32_corr = results['fp32']['correlation']
    binary_corr = results['binary']['correlation']
    
    print(f"\nRelative to FP32 ({fp32_corr:.4f}):")
    print(f"  Binary (no reg):  {binary_corr/fp32_corr*100:.1f}%")
    
    for mag_reg in [0.01, 0.1, 1.0]:
        key = f'binary_reg_{mag_reg}'
        if key in results:
            reg_corr = results[key]['correlation']
            print(f"  Binary (reg={mag_reg}): {reg_corr/fp32_corr*100:.1f}%")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Magnitude regularization encourages uniform magnitudes, which means
sign-only quantization loses less information.

If this works, it validates the approach:
1. Train with magnitude uniformity constraint
2. Quantize to signs only
3. Achieve ternary-level quality at 1.0 bpp
""")

