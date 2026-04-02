# SALOMI: True 1-Bit Runtime with OpenCL
## Complete End-to-End Project Plan

**Project Name:** SALOMI (Stochastic Approximation for Low-Memory Inference)  
**Goal:** Run neural network inference using exactly **1.00 bits per parameter** (bpp)  
**Status:** Tier-4 (Performance & Advanced Features) - 57% complete  
**Last Updated:** 2025-11-07

---

## Executive Summary

SALOMI is a research project implementing a complete inference runtime for neural networks quantized to exactly 1 bit per weight. The system uses stochastic binary matrix multiplication (BSDM-W), sigma-delta modulation (ΣΔ), sequential probability ratio tests (SPRT), and OpenCL acceleration to achieve competitive accuracy while maintaining strict 1.00 bpp export.

**Key Innovation:** Unlike traditional 1-bit quantization that stores only sign bits, SALOMI uses:
- **Procedural bitstream generation** via PRF-seeded ΣΔ modulation
- **Early-exit SPRT** for adaptive compute budgets
- **Zero storage overhead** for contrast enhancement (CTG)
- **Deterministic outputs** for reproducibility

**Target Hardware:** AMD RX 6750 XT (gfx1031) with OpenCL backend

---

## Core Concepts

### 1. 1-Bit Quantization

**Storage Format:**
- Each weight stored as **1 bit**: sign only (±1)
- Bit packing: LSB-first within 32-bit words, row-major for matrices
- Export constraint: Exactly **1.00 bpp** (no overhead allowed)

**Bit Packing Convention:**
```
bit = 1 for weight ≥ 0 (positive)
bit = 0 for weight < 0 (negative)
```

### 2. BSDM-W (Binary Stochastic Dot-product with Modulation)

**Core Operation:** Binary matrix multiplication using XNOR + popcount
```
result = 2 * popcount(XNOR(A_bits, B_bits)) - K
```

**ΣΔ Modulation (MASH-1-1):**
- **Purpose:** Convert 1-bit weights to stochastic bitstreams
- **Parameters:** β=0.30, λ=1/256, E1=±4, E2=±8
- **Determinism:** SplitMix64 PRF for all randomness
- **Normalization:** base_u ∈ [-1, 1]

**Early-Exit:**
- Adaptive tick budgets (k) based on SPRT confidence
- Controlled by `early_exit_enable` flag
- Stage-A probe: fixed kA=16 ticks (no early-exit)
- SPRT certification: adaptive k ∈ [k0, k_max]

### 3. CTG (Carrier Toggle Guard)

**Purpose:** Reduce variance by toggling carrier signs
- **Zero storage:** Operations generated on-the-fly from PRF seed
- **Deterministic:** Identical seeds → identical outputs
- **Current status:** Compiled but gated by `use_ctg` flag
- **Default:** `use_ctg=False` (disabled until proven safe)

### 4. SPRT (Sequential Probability Ratio Test)

**Purpose:** Certify ordering of attention scores with adaptive budgets
- **Wald SPRT:** Binary observations with log-likelihood ratios
- **DAG pruning:** Reduce pair tests by 30%+ vs all-pairs
- **Risk budgets:** α = β = (delta_total * share) / 2
- **Early-exit:** Stop when confidence threshold reached

---

## Architecture Overview

### Inference Pipeline (Per Token)

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Attention (Stage-A + SPRT)                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Stage-A Probe (fixed kA=16 ticks)                        │
│    - Compute means for all L keys                           │
│    - Elbow detection → select T ∈ {8, 12, 16}               │
│    - Output: {T_sel, idx_top[T], mu}                        │
│                                                              │
│ 2. SPRT Certification (adaptive k ∈ [k0, k_max])            │
│    - Certify ordering of top-T candidates                   │
│    - DAG pruning to reduce pair tests                       │
│    - Output: {top1, certificate, k_used}                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: KV Retrieval (LDP-KV)                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Encode KV cache with expander graphs                     │
│    - d_left × d_right = d_kv (e.g., 64 × 64 = 4096)         │
│    - Bipartite graph with degree d_deg                      │
│                                                              │
│ 2. Retrieve top-1 attention position                        │
│    - Decode K, V for selected position                      │
│    - Output: {kv_positions, K_dec, V_dec}                   │
│                                                              │
│ 3. Periodic Repair (PR-4.0)                                 │
│    - Refresh 1-bit KV signal every R tokens                 │
│    - Zero additional resident bits                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Logits (HCL + SPRT)                                │
├─────────────────────────────────────────────────────────────┤
│ 1. Shortlist warm-start (PR-4.2)                            │
│    - Carry over top-k from previous token                   │
│    - Reduce HCL chunk evaluations                           │
│                                                              │
│ 2. HCL chunked energies                                     │
│    - Partition vocabulary into chunks                       │
│    - Compute chunk energies with adaptive budgets           │
│                                                              │
│ 3. SPRT certification                                       │
│    - Certify top-1 token from shortlist                     │
│    - Output: {top1, certificate, k_used}                    │
└─────────────────────────────────────────────────────────────┘
```

### Backend Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ CPU Backend (Reference Implementation)                      │
├─────────────────────────────────────────────────────────────┤
│ - Pure Python/NumPy                                         │
│ - Bit-exact reference for validation                        │
│ - Used for golden logs and parity checks                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ OpenCL Backend (Production)                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Naive Kernel (bsdm_w_naive_norm.cl)                      │
│    - Row-parallel: 1 work-item per row                      │
│    - Simple, easy to verify                                 │
│    - Baseline for parity checks                             │
│                                                              │
│ 2. Tiled Kernel (bsdm_w_tiled.cl)                           │
│    - Tile-parallel: TILE_KW_WORDS ∈ {32, 64, 128}           │
│    - Local memory optimization                              │
│    - 1.25×+ speedup on large shapes                         │
│    - Auto-tuned per device                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tier Structure (Tier-0 → Tier-9)

### Tier-0: Foundation (Complete ✅)
- Project structure and build system
- Basic NumPy/OpenCL setup
- Bit packing utilities

### Tier-1: Core Operators (Complete ✅)

**PR-1.1: BSDM-W Operator**
- CPU implementation with ΣΔ modulation (MASH-1-1)
- OpenCL naive kernel
- Byte-parity CPU↔OpenCL

**PR-1.2: Golden Logs**
- Deterministic test cases with fixed seeds
- RLE compression for tick streams (56% median reduction)
- JSONL format for reproducibility

**PR-1.3: Walsh N=2 + Antithetic Pairs**
- Walsh-Hadamard transform for variance reduction
- Antithetic pairs: (ε, -ε) for bias cancellation
- CTG compiled and gated by `use_ctg` flag

**PR-1.4: Stage-A Probe**
- Fixed kA=16 ticks (no early-exit)
- Elbow detection → T ∈ {8, 12, 16}
- Per-row PRF seeding

**PR-1.5: SPRT-DAG Certification**
- Wald SPRT for Top-T ordering
- DAG pruning (30%+ reduction in pair tests)
- Adaptive k budgets

**PR-1.6: Controller + Certificates**
- Per-token orchestrator
- Risk budget allocation
- Certificate generation

### Tier-2: End-to-End Pipeline (Complete ✅)

**PR-2.0: Hadamard Transform Utilities**
- Fast Walsh-Hadamard transform
- Order-2 support

**PR-2.1: HCL Logits**
- Chunked energy computation
- Adaptive budgets

**PR-2.2: LDP-KV**
- Expander graph encoding
- Bipartite graph decoding

**PR-2.3: E2E Controller**
- Attention → KV → Logits pipeline
- Status codes and certificates

**PR-2.4: Bake-off Harness**
- CSV output with per-token metrics
- Summary JSON with P50/P95 latency

**PR-2.5: CTG A/B Testing**
- Shadow sampling infrastructure
- Comparison metrics

### Tier-3: TSR/BRE and DCVB (Complete ✅)
- Temporal smoothing and budget refinement
- Dynamic context-aware vocabulary budgets
- Early-exit mechanisms

### Tier-4: Performance & Advanced Features (In Progress 🔄)

**Completed (9/9 PRs):**

**PR-3.0: Rebase and Invariants**
- Locked invariants: 1.00 bpp, byte-parity, determinism
- No math changes allowed

**PR-3.1: Tiled OpenCL BSDM-W**
- Tile-parallel kernel with local memory
- 1.25×+ speedup on large shapes
- Byte-parity naive↔tiled

**PR-3.2: Auto-tuner**
- Microbench grid: TILE_KW_WORDS × local_size
- Cache best config per device
- JSON persistence

**PR-3.3: Run-Length Encoding**
- Logging/serialization layer only
- 56% median compression
- LEB128 + base64 wire format

**PR-3.4: Bench Harness**
- Kernel comparison tool
- CSV output with P50/P95 latency
- Parity checks

**PR-3.5: E2E Latency Histograms**
- Per-token CSV: time_ms, k_used, status
- Summary JSON: P50/P95, unsure_rate
- Soft gates: unsure_rate ≤ 2%, median_k ≤ 32

**PR-3.6: Deterministic Golden Matrix**
- 3 fixed test cases
- OpenCL naive↔tiled digest equality
- Reproducibility verification

**PR-3.7: Toggle/Energy Proxy**
- Energy counter formulas
- No-op verification

**PR-3.8: Stress, Soak, and Fuzz**
- Determinism soak (100 smoke, 1000 nightly)
- Seed/shape sweeps (256 seeds × 6 shapes)
- Triage bundles for failing cases

**PR-3.9: Docs and Release** (Pending)
- Full documentation
- Flags audit
- Release checks

**Sprint PR-4.0 → PR-4.6 (4/7 Complete):**

**PR-4.0: KV Refresh Without Extra Live Bits** ✅
- Periodic repair every R tokens
- Zero additional resident bits
- Deterministic schedule

**PR-4.1: Per-Head Dynamic Budgets (EMA)** ✅
- Exponential moving average smoothing
- 98.8% variance reduction
- Hysteresis for T quantization

**PR-4.2: Shortlist Carry-Over for Logits** ✅ (Unit Tests)
- Warm-start from previous token
- TopKHeap for efficient updates
- Deterministic shortlist

**PR-4.3: CTG Policy with 1% Shadow Sampling** ✅ (Unit Tests + Integration)
- Shadow A/B testing (1% sample rate)
- Promotion criteria: agree_rate ≥ 99.9%, overhead ≤ 15%
- Demotion on regressions
- 21/21 tests passing

**PR-4.4: HCL Tiled Kernel + Autotune** 🔲 (Pending)
- Tiled HCL chunked energies
- Auto-tune chunk size and tile params

**PR-4.5: LDP-KV Stage-2 OpenCL Kernel** 🔲 (Pending)
- GPU-accelerated KV encoding/decoding
- Expander graph on device

**PR-4.6: Pulse Scheduler + Chain Repair** 🔲 (Pending)
- Adaptive scheduling for long contexts
- Chain repair for stability

### Tier-5 through Tier-9 (Planned)
- Tier-5: Multi-layer transformers
- Tier-6: Full model integration
- Tier-7: Production deployment
- Tier-8: Optimization and tuning
- Tier-9: Release and documentation

---

## Invariants (Non-Negotiable)

All changes must preserve these invariants:

1. **Export = 1.00 bpp** - No storage overhead allowed
2. **Byte-parity** - CPU↔OpenCL and naive↔tiled must be bit-exact
3. **Determinism** - Fixed seeds → identical outputs across runs
4. **No math changes** - Core BSDM-W, ΣΔ, SPRT formulas locked
5. **Normalized domain** - base_u ∈ [-1, 1]
6. **MASH-1-1 params** - β=0.30, λ=1/256, E1=±4, E2=±8
7. **SplitMix64 seeding** - All randomness via PRF
8. **LSB-first packing** - Bit layout unchanged

---

## Test Coverage

**Current Status:** 110/120 tests passing (92%)

**Test Categories:**
- 31 RLE tests ✅
- 7 auto-tune tests ✅
- 3 tiled parity tests ✅
- 9 kernel comparison tests ✅
- 4 e2e latency tests ✅
- 3 golden matrix tests ✅
- 3 energy no-op tests ✅
- 17 stress tests ✅ (15 smoke + 2 nightly)
- 24 PR-2.x tests ✅
- 12 PR-4.0/4.1 tests ✅
- 21 PR-4.3 tests ✅

**Pre-existing failures:** 10 (unrelated to Tier-4 work)

---

## Performance Targets

### Soft Gates (Expected with Real Data)

**Attention:**
- unsure_rate ≤ 2%
- median_k ≤ 32
- P95_k ≤ 64

**Logits:**
- unsure_rate ≤ 2%
- median_pairs ≤ 1000

**Latency:**
- P50 ≤ 10ms per token
- P95 ≤ 50ms per token

**Speedup:**
- Tiled ≥ 1.25× naive on large shapes (M≥128, Kw≥64)

### Hard Gates (Blocking)

- Byte-parity: naive == tiled (exact)
- Determinism: run1 == run2 (exact)
- Export: exactly 1.00 bpp

---

## File Structure

```
SALOMI/
├── onebit/
│   ├── core/              # Core utilities
│   │   ├── elbow.py       # Elbow detection
│   │   ├── golden_bits.py # RLE encoding/decoding
│   │   ├── prf.py         # SplitMix64 PRF
│   │   └── hadamard.py    # Walsh-Hadamard transform
│   ├── ops/               # Operators
│   │   ├── bsdm_w.py      # BSDM-W CPU implementation
│   │   ├── attention_probe.py  # Stage-A probe
│   │   ├── logits_sprt.py # HCL logits
│   │   └── ldpkv.py       # LDP-KV encoding/decoding
│   ├── attn/              # Attention
│   │   ├── sprt_dag.py    # SPRT-DAG certification
│   │   └── runner.py      # Attention runner
│   ├── runtime/           # Runtime
│   │   ├── controller_e2e.py  # E2E controller
│   │   ├── ctg_policy.py  # CTG policy manager
│   │   ├── shortlist.py   # Shortlist cache
│   │   └── budget.py      # Budget smoothing
│   ├── backends/          # Backends
│   │   └── opencl/
│   │       ├── bsdm_w_naive_norm.cl  # Naive kernel
│   │       ├── bsdm_w_tiled.cl       # Tiled kernel
│   │       └── host_opencl.py        # OpenCL host
│   ├── cli/               # CLI tools
│   │   ├── bench_kernels.py  # Kernel benchmarking
│   │   ├── bench_e2e.py      # E2E benchmarking
│   │   ├── stress.py         # Stress testing
│   │   └── check_golden_matrix.py  # Golden matrix check
│   ├── metrics/           # Metrics
│   │   └── summarize.py   # Metrics summarization
│   ├── tools/             # Tools
│   │   └── triage.py      # Failing-case packer
│   └── ci/                # CI
│       └── golden_matrix.json  # Golden test cases
├── tests/                 # Tests
│   ├── attn/              # Attention tests
│   ├── runtime/           # Runtime tests
│   ├── stress/            # Stress tests
│   └── ci/                # CI tests
└── docs/                  # Documentation
    ├── SALOMI-PROJECT-PLAN.md  # This file
    ├── TIER-4-STATUS.md   # Tier-4 status
    └── PR-*.md            # PR summaries
```

---

## Next Steps

### Immediate (Sprint PR-4.0 → PR-4.6)

1. **PR-4.4: HCL Tiled Kernel + Autotune**
   - Implement tiled HCL chunked energies
   - Auto-tune chunk size and tile params
   - Target: 1.5×+ speedup on vocabulary

2. **PR-4.5: LDP-KV Stage-2 OpenCL Kernel**
   - GPU-accelerated KV encoding/decoding
   - Expander graph on device
   - Target: 2×+ speedup on KV ops

3. **PR-4.6: Pulse Scheduler + Chain Repair**
   - Adaptive scheduling for long contexts
   - Chain repair for stability
   - Target: stable 8k+ context

### Short-Term (Tier-4 Completion)

4. **PR-3.9: Docs and Release**
   - Complete Tier-4 documentation
   - Flags audit
   - Release checks
   - Archive CSVs and golden logs

5. **Full CI Golden Matrix**
   - Run all 3 golden test cases
   - Verify byte-parity across all backends
   - Lock determinism

6. **Benchmark on RX 6750 XT**
   - End-to-end latency profiling
   - Memory bandwidth analysis
   - Power consumption measurement

### Medium-Term (Tier-5+)

7. **Multi-Layer Transformers**
   - Stack attention + FFN layers
   - Layer-wise budget allocation
   - Residual connections

8. **Full Model Integration**
   - Load pretrained weights
   - Quantize to 1-bit
   - Validate accuracy

9. **Production Deployment**
   - Optimize for throughput
   - Batch processing
   - Model serving

---

## Success Criteria

### Technical Milestones

- ✅ 1.00 bpp export maintained
- ✅ CPU↔OpenCL byte-parity verified
- ✅ Deterministic outputs for fixed seeds
- ✅ SPRT-DAG reduces pair tests by 30%+
- ✅ Tiled kernel 1.25×+ speedup
- ✅ RLE compression 56% median reduction
- ✅ 110/120 tests passing (92%)
- 🔄 CTG policy with shadow sampling (integration complete)
- 🔲 HCL tiled kernel with autotune
- 🔲 LDP-KV Stage-2 OpenCL kernel
- 🔲 Pulse scheduler + chain repair

### Research Goals

- Demonstrate competitive accuracy at 1.00 bpp
- Achieve <10ms P50 latency per token
- Support 8k+ context length
- Validate on real transformer models
- Publish results and open-source code

---

## References

**Key Papers:**
- Wald SPRT (1945) - Sequential probability ratio tests
- Sigma-Delta Modulation - Stochastic bitstream generation
- Expander Graphs - LDP-KV encoding

**Hardware:**
- AMD RX 6750 XT (gfx1031)
- OpenCL 2.0+

**Dependencies:**
- Python 3.13+
- NumPy
- PyOpenCL
- pytest

---

---

## Quick Reference

### Common Commands

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific tier tests:**
```bash
pytest tests/runtime/test_ctg_policy.py -v          # PR-4.3 unit tests
pytest tests/runtime/test_ctg_wiring_smoke.py -v    # PR-4.3 integration
pytest tests/attn/test_ctg_stageA_shadow_agrees.py -v  # PR-4.3 shadow
```

**Benchmark kernels:**
```bash
python -m onebit.cli.bench_kernels --M 128 --Kw 64 --T 16 --runs 50
```

**E2E benchmarking:**
```bash
python -m onebit.cli.bench_e2e --tokens 100 --output results.csv
```

**Check golden matrix:**
```bash
python -m onebit.cli.check_golden_matrix --out golden.jsonl
```

**Stress testing:**
```bash
python -m onebit.cli.stress --preset smoke  # Quick smoke test
python -m onebit.cli.stress --preset nightly  # Full nightly test
```

### Environment Variables

```bash
ONEBIT_SKIP_PERF=1        # Skip performance tests
ONEBIT_AUTOTUNE=0         # Disable auto-tuner
ONEBIT_TILE=<N>           # Override tile size
ONEBIT_LS=<N>             # Override local size
ONEBIT_FULL_MATRIX=1      # Run full golden matrix
ONEBIT_ALPHA_T=<float>    # Override EMA alpha for T
ONEBIT_ALPHA_K=<float>    # Override EMA alpha for k
```

### Key Metrics

**Attention:**
- `k_attn_used`: Number of ticks used in SPRT
- `T_sel`: Selected top-T size (8, 12, or 16)
- `attn_top1`: Top-1 attention position

**Logits:**
- `k_logits_used`: Number of pairs evaluated
- `logits_top1`: Top-1 token prediction
- `pairs_evaluated`: Total pair comparisons

**Status Codes:**
- `CERT_OK`: Successfully certified
- `ATTN_UNSURE`: Attention exhausted k_max
- `LOGITS_UNSURE`: Logits exhausted k_max

**CTG Fields:**
- `ctg_shadow`: 1 if shadow sampling active
- `ctg_pol_stageA`: 1 if CTG enabled for Stage-A
- `ctg_pol_attn`: 1 if CTG enabled for Attention
- `ctg_pol_logits`: 1 if CTG enabled for Logits

---

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Maintained By:** SALOMI Development Team

