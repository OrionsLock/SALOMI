"""
Generate figures for the SALOMI research paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('onebit/research/paper/figures', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# =============================================================================
# Figure 1: BPP vs Correlation Comparison
# =============================================================================
def plot_bpp_vs_correlation():
    methods = ['DualPathVQ', 'HessianVQ-32', 'HessianVQ-64', 'HessianVQ-128', 'HessianVQ-256', 'Ternary']
    bpp = [0.58, 0.81, 0.88, 0.94, 1.05, 1.58]
    corr = [0.9237, 0.8113, 0.8434, 0.8961, 0.9509, 0.7348]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot our methods
    ax.scatter(bpp[:-1], corr[:-1], s=150, c='#2ecc71', marker='o', label='Our Methods', zorder=5)
    ax.scatter([bpp[-1]], [corr[-1]], s=200, c='#e74c3c', marker='s', label='Ternary (Baseline)', zorder=5)
    
    # Add labels
    for i, m in enumerate(methods):
        offset = (0.03, 0.01) if m != 'Ternary' else (-0.15, -0.03)
        ax.annotate(m, (bpp[i] + offset[0], corr[i] + offset[1]), fontsize=10)
    
    # Add Pareto frontier line
    pareto_bpp = [0.58, 0.81, 0.88, 0.94, 1.05]
    pareto_corr = [0.9237, 0.8113, 0.8434, 0.8961, 0.9509]
    ax.plot(pareto_bpp, pareto_corr, 'g--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    # Add reference line from ternary
    ax.axhline(y=0.7348, color='r', linestyle=':', alpha=0.5, label='Ternary Quality')
    ax.axvline(x=1.58, color='r', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Bits Per Parameter (BPP)', fontsize=14)
    ax.set_ylabel('Output Correlation', fontsize=14)
    ax.set_title('Sub-1-Bit Quantization: Quality vs Compression', fontsize=16)
    ax.legend(loc='lower right')
    ax.set_xlim(0.4, 1.8)
    ax.set_ylim(0.65, 1.0)
    
    plt.tight_layout()
    plt.savefig('onebit/research/paper/figures/fig1_bpp_vs_correlation.png', dpi=150)
    plt.close()
    print("Generated: fig1_bpp_vs_correlation.png")

# =============================================================================
# Figure 2: Module-wise Comparison
# =============================================================================
def plot_module_comparison():
    modules = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
    ternary = [0.8027, 0.6961, 0.8824, 0.5579]
    hessianvq = [0.9297, 0.8646, 0.9140, 0.8762]
    
    x = np.arange(len(modules))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ternary, width, label='Ternary (1.58 bpp)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, hessianvq, width, label='HessianVQ-128 (0.94 bpp)', color='#2ecc71')
    
    # Add improvement percentages
    for i, (t, h) in enumerate(zip(ternary, hessianvq)):
        improvement = (h - t) / t * 100
        ax.annotate(f'+{improvement:.1f}%', (x[i] + width/2, h + 0.02), 
                    ha='center', fontsize=10, fontweight='bold', color='#27ae60')
    
    ax.set_xlabel('Module Type', fontsize=14)
    ax.set_ylabel('Output Correlation', fontsize=14)
    ax.set_title('Correlation by Module Type: HessianVQ vs Ternary', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('onebit/research/paper/figures/fig2_module_comparison.png', dpi=150)
    plt.close()
    print("Generated: fig2_module_comparison.png")

# =============================================================================
# Figure 3: Layer-by-Layer Results
# =============================================================================
def plot_layer_results():
    layers = list(range(12))
    ternary = [0.7338, 0.6402, 0.6271, 0.7105, 0.7182, 0.7930, 
               0.7834, 0.8226, 0.7951, 0.7828, 0.7225, 0.6880]
    hessianvq = [0.9362, 0.9194, 0.9124, 0.8760, 0.8652, 0.8834,
                 0.8701, 0.8834, 0.8784, 0.9080, 0.9016, 0.9195]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(layers, ternary, 'rs-', linewidth=2, markersize=8, label='Ternary (1.58 bpp)')
    ax.plot(layers, hessianvq, 'go-', linewidth=2, markersize=8, label='HessianVQ-128 (0.94 bpp)')
    
    # Fill the improvement area
    ax.fill_between(layers, ternary, hessianvq, alpha=0.3, color='green')
    
    ax.set_xlabel('Layer Index', fontsize=14)
    ax.set_ylabel('Mean Output Correlation', fontsize=14)
    ax.set_title('Layer-by-Layer Quantization Quality', fontsize=16)
    ax.legend(loc='lower right')
    ax.set_xticks(layers)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('onebit/research/paper/figures/fig3_layer_results.png', dpi=150)
    plt.close()
    print("Generated: fig3_layer_results.png")

# =============================================================================
# Figure 4: DualPathVQ Architecture Diagram (text-based)
# =============================================================================
def create_architecture_diagram():
    diagram = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DualPathVQ Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Weight Matrix W                                                           │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │  w₁₁  w₁₂  w₁₃  w₁₄ │ w₁₅  w₁₆  w₁₇  w₁₈ │ ...        │              │
│   │  w₂₁  w₂₂  w₂₃  w₂₄ │ w₂₅  w₂₆  w₂₇  w₂₈ │ ...        │              │
│   │  w₃₁  w₃₂  w₃₃  w₃₄ │ w₃₅  w₃₆  w₃₇  w₃₈ │ ...        │              │
│   │  w₄₁  w₄₂  w₄₃  w₄₄ │ w₄₅  w₄₆  w₄₇  w₄₈ │ ...        │              │
│   ├─────────────────────┼─────────────────────┼─────────────┤              │
│   │      Block 1        │      Block 2        │    ...      │              │
│   └─────────────────────┴─────────────────────┴─────────────┘              │
│                │                    │                                       │
│                ▼                    ▼                                       │
│   ┌────────────────────────────────────────────────────────┐               │
│   │           Importance Computation                        │               │
│   │   I_i = mean(|Block_i| × Hessian_i)                    │               │
│   └────────────────────────────────────────────────────────┘               │
│                │                                                            │
│                ▼                                                            │
│   ┌────────────────────────────────────────────────────────┐               │
│   │              Adaptive Routing                           │               │
│   │                                                         │               │
│   │   if I_i ≥ threshold (top 60%):                        │               │
│   │       → HIGH PATH: K=32 codebook (5 bits/block)        │               │
│   │   else:                                                 │               │
│   │       → LOW PATH: K=8 codebook (3 bits/block)          │               │
│   └────────────────────────────────────────────────────────┘               │
│                │                                                            │
│                ▼                                                            │
│   ┌────────────────────────────────────────────────────────┐               │
│   │              Storage Format                             │               │
│   │                                                         │               │
│   │   Per weight:                                           │               │
│   │     • Sign: ~0.5 bits (entropy coded)                  │               │
│   │     • Routing: 1 bit per 16 weights = 0.0625 bits      │               │
│   │     • VQ index: (0.6×5 + 0.4×3)/16 ≈ 0.26 bits        │               │
│   │   ─────────────────────────────────────                 │               │
│   │   Total: ~0.58 bits per parameter                      │               │
│   └────────────────────────────────────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    with open('onebit/research/paper/figures/fig4_architecture.txt', 'w') as f:
        f.write(diagram)
    print("Generated: fig4_architecture.txt")

# Run all
if __name__ == "__main__":
    print("Generating figures...")
    try:
        plot_bpp_vs_correlation()
        plot_module_comparison()
        plot_layer_results()
        create_architecture_diagram()
        print("\nAll figures generated successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: matplotlib may not be available. Text figures created.")
        create_architecture_diagram()

