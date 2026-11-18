#!/usr/bin/env python3
"""
COMPREHENSIVE REAL-WORLD VALIDATION
Tests multiple quantization methods and bit-widths on real GPT-2.

NOTE (Phase 0): the previous version of this script produced the invalid table
in docs/HONEST_ASSESSMENT.md.  Its INT2/INT4/INT8 quantizers used a single
per-tensor absmax scale with wrong level counts, ``apply_quantization`` also
quantized the token/position embeddings (wte/wpe, and through weight tying the
LM head), bpp ignored scale overhead, and perplexity was measured on a
five-sentence string.  This version:

  * uses the reference quantizers in :mod:`onebit.quantization.baselines`
    (RTN with g128 or per-channel min-max scales, absmean binary / ternary),
  * quantizes only transformer-block linears
    (``harness.block_linear_modules`` / ``get_weight`` / ``set_weight``),
  * reports bits from the quantizers themselves (incl. fp16 scales/zeros) via
    ``harness.BitReport``,
  * evaluates perplexity with ``harness.perplexity`` on WikiText-2 test.

Usage:
    python tests/comprehensive_real_test.py                    # full test set (slow on CPU)
    python tests/comprehensive_real_test.py --max-windows 20   # quick, NOT reportable
    SALOMI_MAX_WINDOWS=20 python tests/comprehensive_real_test.py
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Install: pip install torch transformers")
    sys.exit(1)

from onebit.eval import harness
from onebit.quantization import baselines as B


@dataclass
class Result:
    name: str
    bpp: float  # bpp over quantized block linears (incl. scales)
    bpp_whole: float  # whole model, non-quantized params at 16 bits
    ppl: float
    ratio: float  # vs FP32
    speed_tokps: float
    passed: bool


def measure_speed(model, tokenizer, device: str, n_tokens: int = 30) -> float:
    """Measure generation speed (fake-quant weights: not representative of packed kernels)."""
    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=5, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)

    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=n_tokens, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start

    return n_tokens / elapsed


class QuantizationMethod:
    """Base class for quantization methods.

    ``quantize_weight`` takes an ``(out, in)`` float weight and returns
    ``(dequantized_weight, total_storage_bits)``.
    """

    def quantize_weight(self, W: torch.Tensor) -> Tuple[torch.Tensor, float]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class BinaryPerRow(QuantizationMethod):
    """Binary with per-row absmean scaling (1 bit + fp16 scale per row)."""

    def quantize_weight(self, W: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return B.binary_absmean(W, group=None)

    @property
    def name(self) -> str:
        return "Binary (per-row)"


class Ternary(QuantizationMethod):
    """Absmean ternary (-1, 0, +1) per row, 1.6 bits/weight packed + fp16 scale per row."""

    def quantize_weight(self, W: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return B.ternary_absmean(W, group=None)

    @property
    def name(self) -> str:
        return "Ternary (per-row)"


class RTN(QuantizationMethod):
    """Asymmetric min-max round-to-nearest, ``bits`` bits, group size ``group`` (None = per-row)."""

    def __init__(self, bits: int, group: Optional[int] = 128):
        self.bits = bits
        self.group = group

    def quantize_weight(self, W: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return B.rtn(W, self.bits, group=self.group)

    @property
    def name(self) -> str:
        g = f"g{self.group}" if self.group else "per-row"
        return f"INT{self.bits} RTN {g}"


class INT2(RTN):
    """2-bit RTN, group 128."""

    def __init__(self):
        super().__init__(2, 128)


class INT4(RTN):
    """4-bit RTN, group 128."""

    def __init__(self):
        super().__init__(4, 128)


class INT8(RTN):
    """8-bit RTN, per output channel."""

    def __init__(self):
        super().__init__(8, None)


def apply_quantization(model, method: QuantizationMethod, layers_to_quantize: str = "all") -> harness.BitReport:
    """Quantize transformer-block linear weights in place; embeddings / LM head stay FP."""
    report = harness.BitReport(total_params=harness.unique_param_count(model))
    with torch.no_grad():
        for name, mod in harness.block_linear_modules(model):
            if layers_to_quantize == "mlp" and ".mlp." not in name:
                continue
            if layers_to_quantize == "attn" and ".attn." not in name:
                continue
            w = harness.get_weight(mod)
            wq, bits = method.quantize_weight(w)
            harness.set_weight(mod, wq)
            report.add(name, w.numel(), float(bits))
    return report


def _default_max_windows() -> Optional[int]:
    v = os.environ.get("SALOMI_MAX_WINDOWS")
    return int(v) if v else None


def run_comprehensive_test(max_windows: Optional[int] = None, measure_gen_speed: bool = False):
    """Run comprehensive tests across multiple methods."""
    print("=" * 70)
    print("COMPREHENSIVE REAL-WORLD QUANTIZATION VALIDATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    scope = "full WikiText-2 test set" if max_windows is None else f"first {max_windows} windows (NOT reportable)"
    print(f"Eval: {scope}")

    # Load tokenizer once
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    ids = harness.load_eval_ids(tokenizer, "wikitext2", "test")

    results = []

    # Test methods
    methods = [
        BinaryPerRow(),
        Ternary(),
        INT2(),
        RTN(3, 128),
        INT4(),
        INT8(),
    ]

    # FP32 Baseline
    print("\n" + "-" * 60)
    print("FP32 BASELINE")
    print("-" * 60)

    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()

    ppl_fp32 = harness.perplexity(model, ids, max_windows=max_windows)["ppl"]
    speed_fp32 = measure_speed(model, tokenizer, device) if measure_gen_speed else float("nan")

    print(f"Perplexity: {ppl_fp32:.2f}")
    print(f"Speed: {speed_fp32:.1f} tok/s")

    results.append(Result(
        name="FP32",
        bpp=32.0,
        bpp_whole=32.0,
        ppl=ppl_fp32,
        ratio=1.0,
        speed_tokps=speed_fp32,
        passed=True
    ))

    # Test each quantization method
    for method in methods:
        print("\n" + "-" * 60)
        print(f"{method.name.upper()}")
        print("-" * 60)

        # Fresh model
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        model.eval()

        # Apply quantization
        report = apply_quantization(model, method)

        # Evaluate
        ppl = harness.perplexity(model, ids, max_windows=max_windows)["ppl"]
        speed = measure_speed(model, tokenizer, device) if measure_gen_speed else float("nan")
        ratio = ppl / ppl_fp32

        passed = ratio < 1.5  # informational only

        print(f"BPP (block linears): {report.bpp_linear:.3f}  whole model: {report.bpp_whole_model:.3f}")
        print(f"Perplexity: {ppl:.2f} ({ratio:.2f}x baseline)")
        print(f"Speed: {speed:.1f} tok/s")

        results.append(Result(
            name=method.name,
            bpp=report.bpp_linear,
            bpp_whole=report.bpp_whole_model,
            ppl=ppl,
            ratio=ratio,
            speed_tokps=speed,
            passed=passed
        ))

        # Memory cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<22} {'BPP':>6} {'BPPwhole':>9} {'PPL':>12} {'Ratio':>9} {'<1.5x':>6}")
    print("-" * 70)

    for r in results:
        status = "yes" if r.passed else "no"
        print(f"{r.name:<22} {r.bpp:>6.3f} {r.bpp_whole:>9.3f} {r.ppl:>12.2f} {r.ratio:>8.2f}x {status:>6}")

    print("\nAll numbers above come from this run; bpp includes fp16 scales/zero-points.")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-windows", type=int, default=_default_max_windows(),
                   help="cap on 1024-token eval windows (default: full test set, or $SALOMI_MAX_WINDOWS)")
    p.add_argument("--speed", action="store_true", help="also time greedy generation (fake-quant)")
    a = p.parse_args()
    run_comprehensive_test(max_windows=a.max_windows, measure_gen_speed=a.speed)