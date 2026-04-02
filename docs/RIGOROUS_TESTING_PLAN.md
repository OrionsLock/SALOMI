# SALOMI Rigorous Testing & Improvement Plan

## Goal: Achieve 1.00 bpp, 1.0 correlation, Good PPL, Good Quality, Good Speed

---

## Executive Summary

Based on 21+ experiment rounds documented in SALOMI-RESEARCH-FINDINGS.md, we've identified critical gaps between claimed results and reality. This plan rigorously tests current best methods, identifies failure modes, and designs experiments to close the gap to our ultimate targets.

### Current State (Honest Assessment)

| Metric | Claimed Best | Verified Reality | Gap |
|--------|-------------|------------------|-----|
| BPP | 0.58 bpp | 1.33 bpp (with overhead) | +130% |
| Correlation | 0.9509 | 0.8416 (end-to-end) | -12% |
| PPL | "Good" | 2,926 (vs FP32: 44.73) | +6,440% |
| Speed | "Fast" | Unvalidated | Unknown |

### Ultimate Targets

| Metric | Target | Status |
|--------|--------|--------|
| BPP | Exactly 1.00 | Achievable with proper accounting |
| Correlation | ≥0.95 | Very challenging |
| PPL | <100 (≤2x FP32) | Extremely challenging |
| Speed | ≥100 tokens/sec | Need benchmarking |

---

## Phase 1: Validate Current Claims (Week 1-2)

### 1.1 BPP Validation

**Problem**: Previous BPP claims excluded critical overhead (codebooks, scales, metadata).

**Tests**:
```python
def test_bpp_accounting():
    """Verify ALL bits are counted."""
    components = {
        "weight_signs": count_sign_bits(),
        "codebook": count_codebook_bits(),
        "scales_per_row": count_row_scales(),
        "scales_per_col": count_col_scales(),
        "block_indices": count_block_indices(),
        "metadata": count_metadata_bits(),
    }
    
    total_bits = sum(components.values())
    total_params = count_parameters()
    actual_bpp = total_bits / total_params
    
    assert actual_bpp <= 1.00, f"BPP {actual_bpp} exceeds 1.00!"
```

**File**: `tests/test_bpp_strict.py`

### 1.2 Correlation Validation

**Problem**: Single-layer correlation doesn't translate to end-to-end quality.

**Tests**:
```python
def test_correlation_e2e():
    """Test correlation at multiple levels."""
    results = {
        "single_layer": [],
        "full_model": [],
        "with_nonlinearity": [],
    }
    
    # Test each layer independently
    for layer_id in range(12):
        corr = measure_layer_correlation(layer_id)
        results["single_layer"].append(corr)
    
    # Test full forward pass correlation
    output_fp32 = fp32_model(test_input)
    output_binary = binary_model(test_input)
    results["full_model"] = correlation(output_fp32, output_binary)
    
    # Test with nonlinearity (GELU)
    results["with_nonlinearity"] = measure_post_gelu_correlation()
    
    # The gap should be known
    print(f"Single-layer avg: {mean(results['single_layer']):.4f}")
    print(f"Full model: {results['full_model']:.4f}")
    print(f"Post-GELU: {results['with_nonlinearity']:.4f}")
```

**File**: `tests/test_correlation_levels.py`

### 1.3 Perplexity Validation

**Problem**: PPL degrades 2,000,000%+ on real text generation.

**Tests**:
```python
def test_perplexity_realistic():
    """Measure perplexity on held-out validation data."""
    
    # Load WikiText-2 validation set (not used in calibration!)
    val_data = load_wikitext2_val()
    
    # Measure true perplexity
    ppl = compute_perplexity(
        model=quantized_model,
        data=val_data,
        stride=512,  # Standard striding
        max_tokens=10000,
    )
    
    # Compare to baselines
    ppl_fp32 = compute_perplexity(fp32_model, val_data)
    ppl_ternary = compute_perplexity(ternary_model, val_data)
    
    print(f"FP32:    {ppl_fp32:.2f}")
    print(f"Ternary: {ppl_ternary:.2f} ({(ppl_ternary/ppl_fp32-1)*100:.1f}%)")
    print(f"Binary:  {ppl:.2f} ({(ppl/ppl_fp32-1)*100:.1f}%)")
```

**File**: `tests/test_perplexity_real.py`

### 1.4 Speed Validation

**Problem**: Speed claims are untested.

**Tests**:
```python
def test_speed_comprehensive():
    """Benchmark end-to-end inference speed."""
    results = {
        "fp32_latency_ms": [],
        "binary_latency_ms": [],
        "tokens_per_sec": [],
        "memory_mb": [],
    }
    
    # Warm-up
    for _ in range(10):
        model(dummy_input)
    
    # Benchmark
    for _ in range(100):
        start = time.perf_counter()
        output = model(test_input)
        end = time.perf_counter()
        results["latency_ms"].append((end - start) * 1000)
    
    print(f"P50 latency: {percentile(results['latency_ms'], 50):.2f} ms")
    print(f"P95 latency: {percentile(results['latency_ms'], 95):.2f} ms")
    print(f"Throughput: {1000 / mean(results['latency_ms']):.1f} tok/sec")
```

**File**: `tests/test_speed_benchmark.py`

---

## Phase 2: Identify Failure Modes (Week 2-3)

### 2.1 GELU Sensitivity Analysis

**Problem**: GELU amplifies quantization errors 200x in MLP layers.

```python
def analyze_gelu_sensitivity():
    """Map exact failure regions in GELU."""
    
    # Collect pre-GELU activations from real data
    activations = collect_pre_gelu_activations(val_data)
    
    # Analyze distribution
    in_sensitive_region = (activations.abs() < 1.0).float().mean()
    print(f"Activations in sensitive region: {in_sensitive_region*100:.1f}%")
    
    # Measure error amplification
    for threshold in [0.1, 0.5, 1.0, 2.0]:
        mask = activations.abs() < threshold
        error_before = compute_quantization_error(activations[mask])
        error_after = compute_post_gelu_error(activations[mask])
        amplification = error_after / error_before
        print(f"|x| < {threshold}: {amplification:.1f}x amplification")
```

**File**: `tests/test_gelu_failure.py`

### 2.2 Layer-by-Layer Error Propagation

```python
def trace_error_propagation():
    """Track how errors compound through layers."""
    
    errors = []
    for layer_id in range(12):
        # Run model up to this layer
        output_fp32 = run_layers(fp32_model, 0, layer_id, test_input)
        output_quant = run_layers(quant_model, 0, layer_id, test_input)
        
        # Measure error
        error = (output_fp32 - output_quant).abs().mean()
        errors.append(error)
        
        print(f"Layer {layer_id}: Error = {error:.6f}")
    
    # Fit exponential growth model
    growth_rate = fit_exponential(errors)
    print(f"Error growth rate: {growth_rate:.2f}x per layer")
```

**File**: `tests/test_error_propagation.py`

### 2.3 Weight Importance Distribution

```python
def analyze_importance_distribution():
    """Understand which weights matter most."""
    
    for layer_name, weight in model.named_parameters():
        if weight.dim() != 2:
            continue
            
        # Compute importance (Hessian-based)
        importance = compute_hessian_diag(weight, calibration_data)
        
        # Analyze distribution
        print(f"{layer_name}:")
        print(f"  Top 10% weights account for {importance.topk(int(0.1*len(importance))).values.sum()/importance.sum()*100:.1f}% of importance")
        print(f"  Bottom 50% account for {importance.sort().values[:int(0.5*len(importance))].sum()/importance.sum()*100:.1f}%")
```

**File**: `tests/test_importance_analysis.py`

### 2.4 Calibration Overfitting Detection

```python
def detect_calibration_overfitting():
    """Check if calibration generalizes."""
    
    # Split data
    train_data, val_data = split_calibration_data()
    
    # Calibrate on train
    calibrated_model = calibrate(model, train_data)
    
    # Evaluate on both
    ppl_train = compute_perplexity(calibrated_model, train_data)
    ppl_val = compute_perplexity(calibrated_model, val_data)
    
    overfit_ratio = ppl_val / ppl_train
    print(f"Train PPL: {ppl_train:.2f}")
    print(f"Val PPL: {ppl_val:.2f}")
    print(f"Overfit ratio: {overfit_ratio:.2f}x")
    
    if overfit_ratio > 2.0:
        print("❌ SEVERE OVERFITTING DETECTED")
```

**File**: `tests/test_overfit_detection.py`

---

## Phase 3: Design Experiments to Close the Gap (Week 3-4)

### 3.1 Experiment: GELU-Aware Quantization

**Hypothesis**: Protecting sensitive regions reduces error amplification.

```python
class GELUAwareQuantizer:
    def quantize(self, W, activations, hessian):
        # Identify weights that affect GELU-sensitive activations
        sensitivity = compute_gelu_sensitivity(W, activations)
        
        # Use higher precision for sensitive weights
        for row_idx, row in enumerate(W):
            if sensitivity[row_idx] > threshold:
                # Keep more bits for sensitive weights
                W_quant[row_idx] = quantize_to_4bit(row)
            else:
                # Binary for non-sensitive
                W_quant[row_idx] = binarize(row)
        
        return W_quant
```

**Targets**:
- Reduce post-GELU error by 50%
- Maintain average BPP < 1.5

### 3.2 Experiment: Iterative Error Correction

**Hypothesis**: Multiple correction passes reduce accumulated error.

```python
class IterativeCorrector:
    def correct(self, W_binary, W_fp32, n_iterations=3):
        W_current = W_binary
        
        for i in range(n_iterations):
            # Compute residual
            residual = W_fp32 - W_current
            
            # Compute optimal correction (low-rank)
            U, S, V = svd(residual)
            correction = U[:, :rank] @ diag(S[:rank]) @ V[:rank, :]
            
            # Apply correction
            W_current = W_current + correction
        
        return W_current
```

**Targets**:
- Improve correlation by 10% per iteration
- Total BPP overhead < 0.2

### 3.3 Experiment: Training-Aware Quantization

**Hypothesis**: Training with quantization constraints from scratch avoids GELU sensitivity.

```python
class QuantizationAwareTraining:
    def train_step(self, batch):
        # Forward with quantization noise
        with quantization_simulation():
            output = model(batch)
        
        # Loss includes quantization robustness term
        loss = cross_entropy(output, target)
        loss += lambda_quant * quantization_robustness_loss(model)
        
        # Backward and update
        loss.backward()
        optimizer.step()
```

**Targets**:
- PPL < 100 at 1.0 bpp
- Train from scratch or fine-tune efficiently

### 3.4 Experiment: Hybrid Precision Allocation

**Hypothesis**: Allocate precision budget optimally per layer.

```python
class HybridPrecisionAllocator:
    def allocate(self, model, total_bpp_budget=1.0):
        # Measure sensitivity per layer
        sensitivities = {}
        for name, layer in model.named_modules():
            sensitivities[name] = measure_layer_sensitivity(layer)
        
        # Allocate bits proportionally to sensitivity
        bit_allocation = optimize_allocation(
            sensitivities, 
            total_budget=total_bpp_budget * total_params,
            min_bits=1,
            max_bits=4,
        )
        
        return bit_allocation
```

**Targets**:
- Achieve FP32-level quality at 1.0 average bpp
- Allow 4-bit for critical layers, sub-1-bit for non-critical

---

## Phase 4: Implementation Checklist (Week 4-5)

### 4.1 Core Tests to Implement

- [ ] `tests/test_bpp_strict.py` - Verify all overhead is counted
- [ ] `tests/test_correlation_e2e.py` - Multi-level correlation
- [ ] `tests/test_perplexity_real.py` - Held-out perplexity
- [ ] `tests/test_speed_benchmark.py` - Comprehensive speed
- [ ] `tests/test_gelu_failure.py` - GELU sensitivity mapping
- [ ] `tests/test_error_propagation.py` - Layer-by-layer errors
- [ ] `tests/test_importance_analysis.py` - Weight importance
- [ ] `tests/test_overfit_detection.py` - Calibration overfitting

### 4.2 Improved Algorithms to Implement

- [ ] `onebit/research/gelu_aware_v2.py` - Enhanced GELU protection
- [ ] `onebit/research/iterative_correction.py` - Multi-pass correction
- [ ] `onebit/research/hybrid_allocation.py` - Optimal bit allocation
- [ ] `onebit/research/qat_finetuning.py` - Quantization-aware fine-tuning

### 4.3 Benchmarking Infrastructure

- [ ] `benchmark/e2e_benchmark.py` - End-to-end speed
- [ ] `benchmark/memory_benchmark.py` - Memory usage
- [ ] `benchmark/quality_benchmark.py` - Quality metrics
- [ ] `benchmark/compare_baselines.py` - vs ternary, FP32

---

## Phase 5: Success Metrics (Week 5-6)

### 5.1 Minimum Viable Results

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| BPP | ≤1.10 | 1.00 | <1.00 |
| Correlation | >0.85 | >0.95 | >0.99 |
| PPL | <500 | <100 | <60 |
| Speed | >50 tok/s | >100 tok/s | >200 tok/s |

### 5.2 Final Benchmark Table

```
┌─────────────────────────────────────────────────────────────────┐
│ SALOMI v2.0 Final Results                                       │
├──────────────────┬──────┬──────┬──────┬─────────┬──────────────┤
│ Method           │ BPP  │ Corr │ PPL  │ tok/sec │ Memory (MB) │
├──────────────────┼──────┼──────┼──────┼─────────┼──────────────┤
│ FP32 Baseline    │32.00 │ 1.00 │ 44.7 │   150   │    480      │
│ Ternary (BitNet) │ 1.58 │ 0.89 │ 52.3 │   180   │     95      │
│ Binary (Current) │ 1.00 │ 0.76 │2,926 │   200   │     60      │
│ SALOMI v2.0      │ 1.00 │  ?   │  ?   │    ?    │     60      │
└──────────────────┴──────┴──────┴──────┴─────────┴──────────────┘
```

---

## Risk Analysis

### High Risk Issues

1. **GELU Amplification is Fundamental**
   - May require training-time solution
   - Post-hoc methods may hit a wall

2. **Calibration Overfitting**
   - Need massive calibration data
   - Or fundamentally different approach

3. **Information Theoretic Limits**
   - 1.0 bpp may not contain enough info
   - May need to accept 1.1-1.2 bpp

### Mitigation Strategies

1. **Accept 1.1 bpp target** if 1.0 proves impossible
2. **Hybrid approach**: 1-bit for attention, 2-bit for MLP
3. **Training-aware**: Fine-tune with quantization constraints
4. **Architecture search**: Find models naturally robust to binary

---

## Timeline

```
Week 1-2: Phase 1 - Validate Claims
├── Implement strict BPP tests
├── Implement multi-level correlation tests
├── Implement perplexity validation
└── Implement speed benchmarks

Week 2-3: Phase 2 - Identify Failure Modes
├── GELU sensitivity analysis
├── Error propagation tracing
├── Importance distribution mapping
└── Overfitting detection

Week 3-4: Phase 3 - Design Experiments
├── GELU-aware quantization v2
├── Iterative error correction
├── Training-aware quantization
└── Hybrid precision allocation

Week 4-5: Phase 4 - Implementation
├── Implement all tests
├── Implement improved algorithms
├── Build benchmarking infrastructure
└── Run comprehensive experiments

Week 5-6: Phase 5 - Final Benchmarks
├── Measure final metrics
├── Compare to baselines
├── Document results
└── Identify next steps
```

---

## Appendix: Test Execution Commands

```bash
# Phase 1: Validation
pytest tests/test_bpp_strict.py -v
pytest tests/test_correlation_e2e.py -v
pytest tests/test_perplexity_real.py -v
pytest tests/test_speed_benchmark.py -v

# Phase 2: Failure Mode Analysis
python -m onebit.research.gelu_failure_analysis
python -m onebit.research.error_propagation_trace
python -m onebit.research.importance_analysis
python -m onebit.research.overfit_detection

# Phase 4: New Experiments
python -m onebit.research.gelu_aware_v2 --evaluate
python -m onebit.research.iterative_correction --iterations=3
python -m onebit.research.hybrid_allocation --budget=1.0
python -m onebit.research.qat_finetuning --epochs=10

# Phase 5: Final Benchmarks
python -m benchmark.e2e_benchmark --model=salomi_v2
python -m benchmark.compare_baselines
```

---

*Document created: 2025-12-02*
*SALOMI Project - Rigorous Testing Plan v1.0*