#!/usr/bin/env python3
"""
Proper End-to-End Evaluation System for SALOMI
Fixes the validation flaws by implementing proper perplexity evaluation

Phase-0 rewrite: the previous implementation could not run (no ``set_format``
on the tokenized dataset, wrong ``input=`` kwarg to the model, forward-pass
exceptions silently swallowed, pad tokens scored, line-level WikiText rows).
Perplexity is now computed exclusively by the shared harness
(:mod:`onebit.eval.harness`: WikiText-2-raw test split joined with "\\n\\n",
non-overlapping windows, token-weighted NLL).  Exceptions are never swallowed.

Bits-per-parameter are no longer "measured" from ad-hoc tensor attributes
(that returned 0 for real models); use the :class:`onebit.eval.harness.BitReport`
returned by the harness quantizers instead.
"""

import time
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onebit.eval import harness

_DATASET_ALIASES = {"wikitext": "wikitext2", "wikitext2": "wikitext2", "wikitext-2": "wikitext2"}


class EndToEndEvaluator:
    """
    Proper end-to-end evaluation system that measures actual perplexity
    instead of just correlation metrics
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize evaluator with model and tokenizer
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ids_cache: Dict[tuple, torch.Tensor] = {}
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with proper configuration"""
        print(f"Loading {self.model_name} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model in evaluation mode
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float32)
        self.model.eval()
        self.model.to(self.device)

    def _eval_ids(self, dataset_name: str, split: str) -> torch.Tensor:
        ds = _DATASET_ALIASES.get(dataset_name)
        if ds is None:
            raise ValueError(f"Unsupported dataset: {dataset_name!r} (supported: wikitext2)")
        key = (ds, split)
        if key not in self._ids_cache:
            self._ids_cache[key] = harness.load_eval_ids(self.tokenizer, ds, split)
        return self._ids_cache[key]

    def evaluate_perplexity(self, quantized_model, dataset_name: str = "wikitext2",
                           split: str = "test", max_windows: Optional[int] = None,
                           seq_length: int = 1024) -> Dict[str, Any]:
        """
        Evaluate a model with the shared harness perplexity protocol.

        Args:
            quantized_model: (Quantized) HF causal LM to evaluate
            dataset_name: "wikitext2" (alias "wikitext")
            split: Dataset split (test by default; use "validation" for tuning)
            max_windows: Debug-only cap on the number of ``seq_length`` windows;
                None = full split
            seq_length: Window length (non-overlapping)

        Returns:
            Dictionary containing evaluation metrics
        """
        ids = self._eval_ids(dataset_name, split)
        quantized_model.to(self.device)
        t0 = time.time()
        res = harness.perplexity(quantized_model, ids, seqlen=seq_length,
                                 max_windows=max_windows, device=str(self.device))
        elapsed = time.time() - t0
        return {
            "perplexity": res["ppl"],
            "average_loss": res["nll"],
            "total_tokens": res["tokens"],
            "windows": res["windows"],
            "seq_length": seq_length,
            "split": split,
            "full_split": max_windows is None,
            "eval_seconds": elapsed,
            "tokens_per_second": res["tokens"] / elapsed if elapsed > 0 else float("nan"),
        }

    def evaluate_quality_metrics(self, original_model, quantized_model,
                                dataset_name: str = "wikitext2",
                                max_windows: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare original vs quantized model quality
        """
        print("Evaluating quality metrics...")

        # Evaluate both models
        original_results = self.evaluate_perplexity(original_model, dataset_name, max_windows=max_windows)
        quantized_results = self.evaluate_perplexity(quantized_model, dataset_name, max_windows=max_windows)

        # Calculate quality metrics
        ppl_ratio = quantized_results["perplexity"] / original_results["perplexity"]
        quality_loss = (ppl_ratio - 1) * 100  # Percentage increase

        return {
            "original": original_results,
            "quantized": quantized_results,
            "ppl_ratio": ppl_ratio,
            "quality_loss_percent": quality_loss,
            "relative_quality": 100 - quality_loss if quality_loss < 100 else 0
        }

    def validate_quantization(self, quantized_model, bit_report: "harness.BitReport",
                             dataset_name: str = "wikitext2",
                             max_windows: Optional[int] = None) -> Dict[str, Any]:
        """
        Quality + size report for a quantized model.

        ``bit_report`` must be the :class:`onebit.eval.harness.BitReport` returned
        by the harness quantizer that produced ``quantized_model``; bits are not
        inferred from the model object.
        """
        if not isinstance(bit_report, harness.BitReport):
            raise TypeError(
                "validate_quantization needs the harness BitReport returned by the quantizer "
                "(e.g. harness.quantize_model_rtn_style / gptq_quantize_model)."
            )
        quality_results = self.evaluate_quality_metrics(self.model, quantized_model, dataset_name, max_windows)
        return {
            "bits": bit_report.summary(),
            "quality_metrics": quality_results,
        }

    def _measure_actual_bpp(self, model) -> float:
        """Removed: bpp cannot be recovered from a dequantized model object."""
        raise NotImplementedError(
            "_measure_actual_bpp was removed (it returned 0 for real models). Use the "
            "onebit.eval.harness.BitReport returned by the harness quantizers "
            "(report.bpp_linear / report.bpp_whole_model)."
        )


def create_evaluator(model_name: str = "gpt2") -> EndToEndEvaluator:
    """Factory function to create evaluator"""
    return EndToEndEvaluator(model_name)

# Example usage
if __name__ == "__main__":
    # Create evaluator
    evaluator = create_evaluator("gpt2")

    # This would be used with actual quantized models
    print("EndToEndEvaluator ready for use")
    print("Usage: evaluator.evaluate_perplexity(quantized_model)")
