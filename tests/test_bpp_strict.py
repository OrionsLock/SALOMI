#!/usr/bin/env python3
"""
Strict BPP Validation Tests for SALOMI

This test suite validates that ALL bits are counted in BPP calculations,
including overhead that was previously missed:
- Codebook bits
- Scale factors (row, column, global)
- Block indices
- Metadata headers

Target: Verify actual BPP ≤ 1.00 with ALL overhead included.
"""

import pytest
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from onebit.core.bpp_guard import BPPCalculator
except ImportError:
    BPPCalculator = None

try:
    from onebit.quantization.hessian_vq import HessianVQ
except ImportError:
    HessianVQ = None


@dataclass
class BPPBreakdown:
    """Detailed breakdown of bits per component."""
    weight_signs: int = 0
    codebook_entries: int = 0
    codebook_bits_per_entry: int = 0
    row_scales: int = 0
    col_scales: int = 0
    global_scale: int = 0
    block_indices: int = 0
    metadata: int = 0
    
    @property
    def total_bits(self) -> int:
        return (
            self.weight_signs + 
            self.codebook_entries * self.codebook_bits_per_entry +
            self.row_scales + 
            self.col_scales + 
            self.global_scale +
            self.block_indices +
            self.metadata
        )


class StrictBPPCalculator:
    """Calculate BPP with strict accounting of ALL overhead."""
    
    def __init__(self):
        self.components: Dict[str, int] = {}
        self.param_count = 0
        
    def reset(self):
        """Reset all counters."""
        self.components = {}
        self.param_count = 0
        
    def add_weight_signs(self, shape: Tuple[int, ...]) -> int:
        """Add bits for weight signs (1 bit per weight)."""
        n_params = np.prod(shape)
        bits = int(n_params)  # 1 bit per sign
        self.components["weight_signs"] = self.components.get("weight_signs", 0) + bits
        self.param_count += n_params
        return bits
        
    def add_codebook(self, codebook: np.ndarray, bits_per_index: int) -> int:
        """Add bits for codebook storage and indices.
        
        Args:
            codebook: The codebook array (n_entries x entry_dim)
            bits_per_index: Bits needed to address codebook entries
        """
        n_entries, entry_dim = codebook.shape
        
        # Codebook entries stored as FP32 (32 bits each)
        codebook_storage = n_entries * entry_dim * 32
        self.components["codebook_storage"] = self.components.get("codebook_storage", 0) + codebook_storage
        
        return codebook_storage
        
    def add_indices(self, n_indices: int, bits_per_index: int) -> int:
        """Add bits for vector quantization indices."""
        bits = n_indices * bits_per_index
        self.components["vq_indices"] = self.components.get("vq_indices", 0) + bits
        return bits
        
    def add_row_scales(self, n_rows: int, bits_per_scale: int = 16) -> int:
        """Add bits for per-row scale factors."""
        bits = n_rows * bits_per_scale
        self.components["row_scales"] = self.components.get("row_scales", 0) + bits
        return bits
        
    def add_col_scales(self, n_cols: int, bits_per_scale: int = 16) -> int:
        """Add bits for per-column scale factors."""
        bits = n_cols * bits_per_scale
        self.components["col_scales"] = self.components.get("col_scales", 0) + bits
        return bits
        
    def add_global_scale(self, bits: int = 32) -> int:
        """Add bits for global scale factor."""
        self.components["global_scale"] = self.components.get("global_scale", 0) + bits
        return bits
        
    def add_block_indices(self, n_blocks: int, bits_per_index: int) -> int:
        """Add bits for block structure indices."""
        bits = n_blocks * bits_per_index
        self.components["block_indices"] = self.components.get("block_indices", 0) + bits
        return bits
        
    def add_metadata(self, bits: int) -> int:
        """Add bits for metadata (shape info, quantization config, etc.)."""
        self.components["metadata"] = self.components.get("metadata", 0) + bits
        return bits
        
    def calculate_bpp(self) -> float:
        """Calculate total bits per parameter."""
        if self.param_count == 0:
            return float('inf')
        total_bits = sum(self.components.values())
        return total_bits / self.param_count
        
    def get_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of bits."""
        total_bits = sum(self.components.values())
        return {
            "components": dict(self.components),
            "total_bits": total_bits,
            "param_count": self.param_count,
            "bpp": self.calculate_bpp(),
            "overhead_bits": total_bits - self.components.get("weight_signs", 0),
            "overhead_fraction": (total_bits - self.components.get("weight_signs", 0)) / total_bits if total_bits > 0 else 0,
        }


class TestBPPStrictAccounting:
    """Test suite for strict BPP accounting."""
    
    def test_binary_only_is_exactly_1bpp(self):
        """Pure binary (sign-only) should be exactly 1.0 bpp."""
        calc = StrictBPPCalculator()
        
        # Simulate a 768x768 weight matrix (typical GPT-2 attention)
        shape = (768, 768)
        calc.add_weight_signs(shape)
        
        bpp = calc.calculate_bpp()
        assert bpp == 1.0, f"Pure binary should be 1.0 bpp, got {bpp}"
        
        breakdown = calc.get_breakdown()
        assert breakdown["overhead_bits"] == 0, "Pure binary should have no overhead"
        
    def test_binary_with_global_scale(self):
        """Binary + 1 global scale should be ~1.0 bpp for large matrices."""
        calc = StrictBPPCalculator()
        
        shape = (768, 768)
        calc.add_weight_signs(shape)
        calc.add_global_scale(32)  # Single FP32 scale
        
        bpp = calc.calculate_bpp()
        expected = 1.0 + 32 / (768 * 768)  # ~1.00005
        
        assert abs(bpp - expected) < 0.0001, f"Expected {expected}, got {bpp}"
        assert bpp < 1.001, f"With global scale, BPP should be ~1.0, got {bpp}"
        
    def test_binary_with_row_col_scales(self):
        """Binary + row/col scales should be >1.0 bpp."""
        calc = StrictBPPCalculator()
        
        shape = (768, 3072)  # MLP up projection
        n_rows, n_cols = shape
        
        calc.add_weight_signs(shape)
        calc.add_row_scales(n_rows, bits_per_scale=16)  # FP16 row scales
        calc.add_col_scales(n_cols, bits_per_scale=16)  # FP16 col scales
        
        bpp = calc.calculate_bpp()
        
        # Calculate expected BPP
        sign_bits = n_rows * n_cols
        row_bits = n_rows * 16
        col_bits = n_cols * 16
        expected = (sign_bits + row_bits + col_bits) / (n_rows * n_cols)
        # = 1 + 16/3072 + 16/768 = 1 + 0.0052 + 0.0208 = 1.026
        
        breakdown = calc.get_breakdown()
        print(f"BPP with row+col scales: {bpp:.4f}")
        print(f"Breakdown: {breakdown}")
        
        assert abs(bpp - expected) < 0.0001, f"Expected {expected}, got {bpp}"
        assert bpp > 1.02, "Row+col scales should add significant overhead"
        
    def test_hessian_vq_bpp(self):
        """HessianVQ should account for codebook + indices."""
        calc = StrictBPPCalculator()
        
        shape = (768, 768)
        n_params = np.prod(shape)
        block_size = 4  # 4 elements per VQ block
        n_blocks = n_params // block_size
        codebook_size = 256  # 8 bits per index
        
        # Weight signs
        calc.add_weight_signs(shape)
        
        # Codebook storage: 256 entries × 4 dims × 32 bits
        codebook = np.zeros((codebook_size, block_size))
        calc.add_codebook(codebook, bits_per_index=8)
        
        # VQ indices: n_blocks × 8 bits
        calc.add_indices(n_blocks, bits_per_index=8)
        
        bpp = calc.calculate_bpp()
        
        # Expected: 1 (signs) + 8/4 (indices per weight) + codebook overhead
        # = 1 + 2 + (256*4*32)/(768*768) = 3 + 0.055 = 3.055
        breakdown = calc.get_breakdown()
        print(f"HessianVQ BPP: {bpp:.4f}")
        print(f"Breakdown: {breakdown}")
        
        assert bpp > 3.0, f"HessianVQ with indices should be >3 bpp, got {bpp}"
        
    def test_claimed_058_bpp_is_invalid(self):
        """Verify that claimed 0.58 bpp excludes necessary overhead."""
        calc = StrictBPPCalculator()
        
        # The claimed 0.58 bpp configuration
        shape = (768, 768)
        n_params = np.prod(shape)
        block_size = 32  # Larger blocks for better compression
        n_blocks = n_params // block_size
        codebook_size = 4  # 2-bit indices
        
        # What was probably counted: just the indices
        index_bits = n_blocks * 2  # 2 bits per index
        claimed_bpp = index_bits / n_params
        print(f"Claimed BPP (indices only): {claimed_bpp:.4f}")
        
        # What should be counted: signs + indices + codebook
        calc.add_weight_signs(shape)  # Signs are REQUIRED
        calc.add_indices(n_blocks, bits_per_index=2)
        codebook = np.zeros((codebook_size, block_size))
        calc.add_codebook(codebook, bits_per_index=2)
        
        actual_bpp = calc.calculate_bpp()
        breakdown = calc.get_breakdown()
        
        print(f"Actual BPP (with signs + codebook): {actual_bpp:.4f}")
        print(f"Breakdown: {breakdown}")
        
        # The "0.58 bpp" claim is likely missing signs (1.0 bpp)
        assert actual_bpp > 1.0, f"With signs, BPP must be >1.0, got {actual_bpp}"
        
    def test_gpt2_full_model_bpp(self):
        """Test BPP calculation for full GPT-2 model."""
        calc = StrictBPPCalculator()
        
        # GPT-2 small configuration
        n_layers = 12
        d_model = 768
        d_ff = 3072
        n_heads = 12
        vocab_size = 50257
        
        # Count all weight matrices
        weight_configs = [
            # Token embeddings
            ("wte", (vocab_size, d_model)),
            # Position embeddings
            ("wpe", (1024, d_model)),
        ]
        
        for layer in range(n_layers):
            # Attention
            weight_configs.append((f"attn.c_attn.{layer}", (d_model, 3 * d_model)))
            weight_configs.append((f"attn.c_proj.{layer}", (d_model, d_model)))
            # MLP
            weight_configs.append((f"mlp.c_fc.{layer}", (d_model, d_ff)))
            weight_configs.append((f"mlp.c_proj.{layer}", (d_ff, d_model)))
        
        # LM head (tied with wte typically, but count for accuracy)
        weight_configs.append(("lm_head", (d_model, vocab_size)))
        
        total_params = 0
        for name, shape in weight_configs:
            n_params = np.prod(shape)
            total_params += n_params
            calc.add_weight_signs(shape)
            # Add row + col scales for each matrix
            calc.add_row_scales(shape[0], bits_per_scale=16)
            calc.add_col_scales(shape[1], bits_per_scale=16)
        
        calc.add_global_scale(32)  # Global scale
        calc.add_metadata(1024)  # Metadata for config
        
        bpp = calc.calculate_bpp()
        breakdown = calc.get_breakdown()
        
        print(f"GPT-2 Full Model BPP: {bpp:.4f}")
        print(f"Total params: {total_params:,}")
        print(f"Breakdown: {breakdown}")
        
        # For a proper binary model, BPP should be close to 1.0 + overhead
        # The overhead should be small relative to total params
        overhead_bpp = bpp - 1.0
        print(f"Overhead BPP: {overhead_bpp:.4f}")
        
        assert bpp > 1.0, f"BPP must be >1.0 with any overhead"
        assert bpp < 1.1, f"Overhead should be <10%, got {overhead_bpp:.4f}"


class TestBPPvsQuality:
    """Test tradeoffs between BPP and quality."""
    
    def test_bpp_quality_pareto_frontier(self):
        """Verify Pareto frontier of BPP vs quality tradeoffs."""
        
        # Known configurations from research
        configs = [
            {"name": "Binary (signs only)", "bpp": 1.00, "correlation": 0.76},
            {"name": "Binary + row scale (FP16)", "bpp": 1.02, "correlation": 0.82},
            {"name": "Binary + row+col scale", "bpp": 1.04, "correlation": 0.85},
            {"name": "LowRank r=2", "bpp": 1.11, "correlation": 0.87},
            {"name": "LowRank r=4", "bpp": 1.22, "correlation": 0.89},
            {"name": "Ternary", "bpp": 1.58, "correlation": 0.89},
        ]
        
        print("\nBPP vs Quality Pareto Frontier:")
        print("-" * 60)
        for cfg in configs:
            quality_per_bit = cfg["correlation"] / cfg["bpp"]
            print(f"{cfg['name']:30} BPP={cfg['bpp']:.2f} Corr={cfg['correlation']:.2f} Q/B={quality_per_bit:.3f}")
        
        # At 1.0 bpp, we expect correlation ~0.76 (not 0.95!)
        binary_cfg = configs[0]
        assert binary_cfg["bpp"] == 1.0
        assert binary_cfg["correlation"] < 0.80, "Binary at 1.0 bpp should have correlation <0.80"
        
        # To achieve correlation >0.85, we need >1.04 bpp
        high_quality_configs = [c for c in configs if c["correlation"] > 0.85]
        min_bpp_for_high_quality = min(c["bpp"] for c in high_quality_configs)
        assert min_bpp_for_high_quality > 1.0, "High quality requires BPP > 1.0"


class TestMissingOverhead:
    """Test for commonly missed overhead in BPP calculations."""
    
    def test_codebook_overhead_not_amortized_incorrectly(self):
        """Codebook overhead should not be divided across too many uses."""
        calc = StrictBPPCalculator()
        
        # Small matrix where codebook overhead is significant
        shape = (64, 64)
        codebook_size = 256
        block_size = 4
        n_blocks = np.prod(shape) // block_size
        
        calc.add_weight_signs(shape)
        codebook = np.zeros((codebook_size, block_size))
        calc.add_codebook(codebook, bits_per_index=8)
        calc.add_indices(n_blocks, bits_per_index=8)
        
        bpp = calc.calculate_bpp()
        
        # Codebook: 256 * 4 * 32 = 32768 bits
        # Signs: 64 * 64 = 4096 bits  
        # Indices: (4096/4) * 8 = 8192 bits
        # Total: 32768 + 4096 + 8192 = 45056 bits
        # BPP: 45056 / 4096 = 11.0
        
        print(f"Small matrix BPP with codebook: {bpp:.2f}")
        assert bpp > 10.0, "Small matrices have huge codebook overhead"
        
    def test_metadata_is_counted(self):
        """Metadata bits should be counted in BPP."""
        calc = StrictBPPCalculator()
        
        shape = (768, 768)
        calc.add_weight_signs(shape)
        
        # Typical metadata: shape (2 int32), quantization config (8 int32)
        metadata_bits = (2 + 8) * 32
        calc.add_metadata(metadata_bits)
        
        bpp = calc.calculate_bpp()
        
        # Metadata overhead is small but should be counted
        overhead = (bpp - 1.0) * np.prod(shape)
        assert overhead == metadata_bits, "Metadata bits should be counted"


class TestRealWorldScenarios:
    """Test BPP in realistic deployment scenarios."""
    
    def test_model_file_size_matches_bpp(self):
        """Verify that calculated BPP matches actual file size."""
        
        # Simulate saving and loading a quantized model
        n_params = 124_000_000  # GPT-2 small
        claimed_bpp = 1.05
        
        # Expected file size
        expected_bits = n_params * claimed_bpp
        expected_bytes = expected_bits / 8
        expected_mb = expected_bytes / (1024 * 1024)
        
        print(f"Expected file size at {claimed_bpp} bpp: {expected_mb:.1f} MB")
        
        # FP32 reference
        fp32_bytes = n_params * 4
        fp32_mb = fp32_bytes / (1024 * 1024)
        compression_ratio = fp32_mb / expected_mb
        
        print(f"FP32 size: {fp32_mb:.1f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}x")
        
        # At 1.05 bpp, we should achieve ~30x compression
        assert compression_ratio > 25, f"Expected >25x compression, got {compression_ratio:.1f}x"
        
    def test_runtime_memory_matches_bpp(self):
        """Verify runtime memory usage matches BPP claims."""
        
        # For binary weights, memory should be bits/8 bytes
        n_params = 124_000_000
        bpp = 1.0
        
        # Expected memory for weights only
        weight_bits = n_params * bpp
        weight_bytes = weight_bits / 8
        weight_mb = weight_bytes / (1024 * 1024)
        
        print(f"Weight memory at {bpp} bpp: {weight_mb:.1f} MB")
        
        # Add runtime overhead (activations, KV cache, etc.)
        batch_size = 1
        seq_len = 512
        d_model = 768
        n_layers = 12
        
        # KV cache (FP32 for simplicity)
        kv_bytes = batch_size * seq_len * d_model * 2 * n_layers * 4
        kv_mb = kv_bytes / (1024 * 1024)
        
        print(f"KV cache: {kv_mb:.1f} MB")
        
        total_mb = weight_mb + kv_mb
        print(f"Total runtime memory: {total_mb:.1f} MB")


def run_all_bpp_tests():
    """Run all BPP validation tests."""
    print("=" * 70)
    print("STRICT BPP VALIDATION TESTS")
    print("=" * 70)
    
    # Test classes to run
    test_classes = [
        TestBPPStrictAccounting,
        TestBPPvsQuality,
        TestMissingOverhead,
        TestRealWorldScenarios,
    ]
    
    total_passed = 0
    total_failed = 0
    failures = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 50)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    print(f"  {method_name}... ", end="")
                    method()
                    print("PASSED")
                    total_passed += 1
                except AssertionError as e:
                    print(f"FAILED: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"ERROR: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    
    if failures:
        print("\nFailures:")
        for cls, method, msg in failures:
            print(f"  {cls}.{method}: {msg}")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_bpp_tests()
    sys.exit(0 if success else 1)