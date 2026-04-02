#!/usr/bin/env python3
"""
Proper End-to-End Evaluation System for SALOMI
Fixes the validation flaws by implementing proper perplexity evaluation
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time
from typing import Dict, Any, Optional

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
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with proper configuration"""
        print(f"Loading {self.model_name} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model in evaluation mode
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def evaluate_perplexity(self, quantized_model, dataset_name: str = "wikitext",
                           split: str = "validation", max_samples: int = 100,
                           seq_length: int = 512) -> Dict[str, Any]:
        """
        Evaluate quantized model using proper perplexity metric

        Args:
            quantized_model: Quantized model to evaluate
            dataset_name: Name of dataset to use
            split: Dataset split (train/validation/test)
            max_samples: Maximum number of samples to evaluate
            seq_length: Sequence length for evaluation

        Returns:
            Dictionary containing evaluation metrics
        """
        # Load dataset
        dataset = self._load_dataset(dataset_name, split, max_samples, seq_length)

        # Create data loader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Evaluate
        results = self._run_evaluation(quantized_model, dataloader)

        return results

    def _load_dataset(self, dataset_name: str, split: str, max_samples: int, seq_length: int):
        """Load and preprocess dataset"""
        print(f"Loading {dataset_name} dataset, split: {split}")

        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        elif dataset_name == "ptb":
            dataset = load_dataset("ptb_text_only", split=split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Tokenize and preprocess
        def tokenize_function(examples):
            # Tokenize with padding
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=seq_length,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            }

        # Apply tokenization
        dataset = dataset.map(tokenize_function, batched=True)

        # Limit samples
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return dataset

    def _run_evaluation(self, quantized_model, dataloader) -> Dict[str, Any]:
        """Run actual evaluation loop"""
        print("Running evaluation...")

        total_loss = 0.0
        total_tokens = 0
        batch_times = []
        token_count = 0

        quantized_model.eval()
        quantized_model.to(self.device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Measure inference time
                start_time = time.time()

                # Forward pass
                try:
                    outputs = quantized_model(
                        input=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    continue

                # Measure time
                batch_time = time.time() - start_time
                batch_times.append(batch_time)

                # Compute loss (shifted for autoregressive)
                labels = input_ids[:, 1:].contiguous()  # Shift for autoregressive
                shift_logits = logits[:, :-1, :].contiguous()

                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='sum'
                )

                # Accumulate statistics
                total_loss += loss.item()
                total_tokens += labels.numel()
                token_count += labels.numel()

        # Calculate final metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Calculate latency metrics
        avg_latency = np.mean(batch_times) * 1000  # Convert to ms
        tokens_per_second = token_count / np.sum(batch_times) if batch_times else 0

        return {
            "perplexity": perplexity,
            "average_loss": avg_loss,
            "total_tokens": total_tokens,
            "average_latency_ms": avg_latency,
            "tokens_per_second": tokens_per_second,
            "batch_count": len(batch_times),
            "successful_batches": len(batch_times)
        }

    def evaluate_quality_metrics(self, original_model, quantized_model,
                                dataset_name: str = "wikitext") -> Dict[str, Any]:
        """
        Compare original vs quantized model quality
        """
        print("Evaluating quality metrics...")

        # Evaluate both models
        original_results = self.evaluate_perplexity(original_model, dataset_name)
        quantized_results = self.evaluate_perplexity(quantized_model, dataset_name)

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

    def validate_quantization(self, quantized_model, bpp_target: float,
                             dataset_name: str = "wikitext") -> Dict[str, Any]:
        """
        Comprehensive validation of quantization method
        """
        print(f"Validating quantization at {bpp_target} bpp...")

        # Evaluate quality
        quality_results = self.evaluate_quality_metrics(self.model, quantized_model, dataset_name)

        # Measure actual BPP
        actual_bpp = self._measure_actual_bpp(quantized_model)

        # Check if meets target
        bpp_ratio = actual_bpp / bpp_target
        meets_target = abs(bpp_ratio - 1.0) < 0.1  # 10% tolerance

        return {
            "bpp_target": bpp_target,
            "actual_bpp": actual_bpp,
            "bpp_ratio": bpp_ratio,
            "meets_bpp_target": meets_target,
            "quality_metrics": quality_results,
            "validation_passed": meets_target and quality_results["relative_quality"] > 80
        }

    def _measure_actual_bpp(self, model) -> float:
        """Measure actual bits per parameter including all overhead"""
        total_bits = 0
        total_params = 0

        for name, param in model.named_parameters():
            if hasattr(param, "quantized_data"):
                # Base quantization data
                total_bits += len(param.quantized_data) * 8

                # Codebook overhead
                if hasattr(param, "codebook"):
                    total_bits += len(param.codebook) * 8

                # Metadata
                if hasattr(param, "metadata"):
                    total_bits += len(param.metadata) * 8

                # Sign bits
                if hasattr(param, "signs"):
                    total_bits += len(param.signs) * 8

                # Index bits
                if hasattr(param, "indices"):
                    total_bits += len(param.indices) * 8

                # Routing bits
                if hasattr(param, "routing_bits"):
                    total_bits += len(param.routing_bits) * 8

                total_params += param.numel()

        return total_bits / total_params if total_params > 0 else 0

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
