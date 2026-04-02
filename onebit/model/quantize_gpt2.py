"""GPT-2 weight quantization to 1-bit.

This module handles:
1. Loading GPT-2 weights (from HuggingFace or mock)
2. Quantizing to 1-bit (sign only)
3. Packing into uint32 arrays
4. Verifying 1.00 bpp export constraint
5. Saving quantized model

Quantization strategy:
- Weights: Sign-only quantization (bit=1 for >=0, bit=0 for <0)
- Embeddings: Keep as FP32 (not quantized)
- Layer norms: Keep as FP32 (not quantized)
- Biases: Keep as FP32 (not quantized)

Export format:
- All weight matrices packed to 1-bit
- Metadata (shapes, layer names, etc.) stored separately
- Total storage: ~1.00 bpp for weights + FP32 for non-quantized params
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path

from onebit.core.packbits import pack_signs_rowmajor
from onebit.tools.export_guard import verify_model_bpp

try:
    from transformers import GPT2LMHeadModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class GPT2Config:
    """GPT-2 model configuration.
    
    Attributes:
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension
        d_ff: FFN hidden dimension (typically 4 * d_model)
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
    """
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    vocab_size: int = 50257
    max_seq_len: int = 1024


@dataclass
class QuantizedGPT2:
    """Quantized GPT-2 model.
    
    Attributes:
        config: Model configuration
        weights_1bit: Dict of 1-bit packed weight matrices
        weights_fp32: Dict of FP32 weights (embeddings, norms, biases)
        scales: Dict of per-tensor scales (mean absolute value)
        means: Dict of per-tensor original means (for DC correction)
        metadata: Additional metadata (shapes, etc.)
    """
    config: GPT2Config
    weights_1bit: Dict[str, np.ndarray]  # Packed uint32 arrays
    weights_fp32: Dict[str, np.ndarray]  # FP32 arrays
    scales: Dict[str, float]             # Per-tensor scales
    means: Dict[str, float]              # Per-tensor original means
    metadata: Dict


def create_mock_gpt2_weights(cfg: GPT2Config) -> Dict[str, np.ndarray]:
    """Create mock GPT-2 weights for testing.
    
    Args:
        cfg: Model configuration
    
    Returns:
        Dict mapping weight name to FP32 array
    """
    weights = {}
    
    # Token embeddings [vocab_size, d_model]
    weights["wte"] = np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float32) * 0.02
    
    # Position embeddings [max_seq_len, d_model]
    weights["wpe"] = np.random.randn(cfg.max_seq_len, cfg.d_model).astype(np.float32) * 0.01
    
    # Per-layer weights
    for layer_idx in range(cfg.n_layers):
        prefix = f"h.{layer_idx}"
        
        # Attention
        # Q, K, V projections: [d_model, 3*d_model]
        # NOTE: In GPT-2, weights are [d_in, d_out] = [d_model, 3*d_model]
        # We will transpose them during quantization to be [d_out, d_in] for efficient matmul
        weights[f"{prefix}.attn.c_attn.w"] = np.random.randn(cfg.d_model, 3 * cfg.d_model).astype(np.float32) * 0.02
        weights[f"{prefix}.attn.c_attn.b"] = np.zeros(3 * cfg.d_model, dtype=np.float32)
        
        # Attention output projection: [d_model, d_model]
        weights[f"{prefix}.attn.c_proj.w"] = np.random.randn(cfg.d_model, cfg.d_model).astype(np.float32) * 0.02
        weights[f"{prefix}.attn.c_proj.b"] = np.zeros(cfg.d_model, dtype=np.float32)
        
        # Layer norm 1
        weights[f"{prefix}.ln_1.g"] = np.ones(cfg.d_model, dtype=np.float32)
        weights[f"{prefix}.ln_1.b"] = np.zeros(cfg.d_model, dtype=np.float32)
        
        # FFN
        # Up projection: [d_model, d_ff]
        weights[f"{prefix}.mlp.c_fc.w"] = np.random.randn(cfg.d_model, cfg.d_ff).astype(np.float32) * 0.02
        weights[f"{prefix}.mlp.c_fc.b"] = np.zeros(cfg.d_ff, dtype=np.float32)
        
        # Down projection: [d_ff, d_model]
        weights[f"{prefix}.mlp.c_proj.w"] = np.random.randn(cfg.d_ff, cfg.d_model).astype(np.float32) * 0.02
        weights[f"{prefix}.mlp.c_proj.b"] = np.zeros(cfg.d_model, dtype=np.float32)
        
        # Layer norm 2
        weights[f"{prefix}.ln_2.g"] = np.ones(cfg.d_model, dtype=np.float32)
        weights[f"{prefix}.ln_2.b"] = np.zeros(cfg.d_model, dtype=np.float32)
    
    # Final layer norm
    weights["ln_f.g"] = np.ones(cfg.d_model, dtype=np.float32)
    weights["ln_f.b"] = np.zeros(cfg.d_model, dtype=np.float32)
    
    return weights


def load_gpt2_from_huggingface(model_name: str = "gpt2") -> Tuple[Dict[str, np.ndarray], GPT2Config]:
    """Load GPT-2 weights from HuggingFace.

    Args:
        model_name: HuggingFace model name ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

    Returns:
        Tuple of (weights dict, config)

    Raises:
        ImportError: If transformers not installed
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers required for HuggingFace loading. "
            "Install with: pip install transformers"
        )

    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_config = model.config

    # Create SALOMI config
    cfg = GPT2Config(
        n_layers=hf_config.n_layer,
        n_heads=hf_config.n_head,
        d_model=hf_config.n_embd,
        d_ff=hf_config.n_inner if hf_config.n_inner else 4 * hf_config.n_embd,
        vocab_size=hf_config.vocab_size,
        max_seq_len=hf_config.n_positions,
    )

    # Extract weights
    weights = {}
    state_dict = model.state_dict()

    # Token embeddings
    weights["wte"] = state_dict["transformer.wte.weight"].cpu().numpy()

    # Position embeddings
    weights["wpe"] = state_dict["transformer.wpe.weight"].cpu().numpy()

    # Per-layer weights
    for layer_idx in range(cfg.n_layers):
        prefix = f"h.{layer_idx}"
        hf_prefix = f"transformer.h.{layer_idx}"

        # Attention (c_attn contains Q, K, V concatenated)
        # HF shape: [d_model, 3*d_model] -> Keep as is, we will transpose during quantization
        weights[f"{prefix}.attn.c_attn.w"] = state_dict[f"{hf_prefix}.attn.c_attn.weight"].cpu().numpy()
        weights[f"{prefix}.attn.c_attn.b"] = state_dict[f"{hf_prefix}.attn.c_attn.bias"].cpu().numpy()

        # Attention output projection
        # HF shape: [d_model, d_model]
        weights[f"{prefix}.attn.c_proj.w"] = state_dict[f"{hf_prefix}.attn.c_proj.weight"].cpu().numpy()
        weights[f"{prefix}.attn.c_proj.b"] = state_dict[f"{hf_prefix}.attn.c_proj.bias"].cpu().numpy()

        # Layer norm 1
        weights[f"{prefix}.ln_1.g"] = state_dict[f"{hf_prefix}.ln_1.weight"].cpu().numpy()
        weights[f"{prefix}.ln_1.b"] = state_dict[f"{hf_prefix}.ln_1.bias"].cpu().numpy()

        # FFN up projection
        # HF shape: [d_model, d_ff]
        weights[f"{prefix}.mlp.c_fc.w"] = state_dict[f"{hf_prefix}.mlp.c_fc.weight"].cpu().numpy()
        weights[f"{prefix}.mlp.c_fc.b"] = state_dict[f"{hf_prefix}.mlp.c_fc.bias"].cpu().numpy()

        # FFN down projection
        # HF shape: [d_ff, d_model]
        weights[f"{prefix}.mlp.c_proj.w"] = state_dict[f"{hf_prefix}.mlp.c_proj.weight"].cpu().numpy()
        weights[f"{prefix}.mlp.c_proj.b"] = state_dict[f"{hf_prefix}.mlp.c_proj.bias"].cpu().numpy()

        # Layer norm 2
        weights[f"{prefix}.ln_2.g"] = state_dict[f"{hf_prefix}.ln_2.weight"].cpu().numpy()
        weights[f"{prefix}.ln_2.b"] = state_dict[f"{hf_prefix}.ln_2.bias"].cpu().numpy()

    # Final layer norm
    weights["ln_f.g"] = state_dict["transformer.ln_f.weight"].cpu().numpy()
    weights["ln_f.b"] = state_dict["transformer.ln_f.bias"].cpu().numpy()

    print(f"Loaded {len(weights)} weight tensors")
    print(f"Config: {cfg.n_layers} layers, {cfg.d_model} dim, {cfg.vocab_size} vocab")

    return weights, cfg


def quantize_gpt2(
    weights_fp32: Dict[str, np.ndarray],
    cfg: GPT2Config,
) -> QuantizedGPT2:
    """Quantize GPT-2 weights to 1-bit.
    
    Args:
        weights_fp32: Dict of FP32 weights
        cfg: Model configuration
    
    Returns:
        Quantized model
    """
    weights_1bit = {}
    weights_fp32_keep = {}
    scales = {}
    means = {}
    metadata = {}
    
    for name, W_fp32 in weights_fp32.items():
        # Decide whether to quantize
        # Quantize: weight matrices (*.w)
        # Keep FP32: embeddings (wte, wpe), layer norms (*.g, *.b), biases (*.b)
        
        if name.endswith(".w") and W_fp32.ndim == 2:
            # GPT-2 uses Conv1D layers where weights are [d_in, d_out].
            # For efficient matmul y = W_eff @ x, we need W_eff to be [d_out, d_in].
            # So we MUST transpose the weights.
            W_target = W_fp32.T  # [d_out, d_in]
            
            # Center weights per row (maximize entropy per neuron)
            # Each row corresponds to one output neuron.
            # We want each neuron to use its full dynamic range.
            mean = np.mean(W_target, axis=1, keepdims=True)
            W_centered = W_target - mean
            
            # Calculate scale per row (mean absolute value of centered weights)
            # Each row has its own scale factor.
            # scale: [d_out]
            scale = np.mean(np.abs(W_centered), axis=1)
            scales[name] = scale
            
            # Store original mean per row for DC correction
            means[name] = mean.flatten()
            
            # Quantize weight matrix to 1-bit (packed row-major)
            W_bits = pack_signs_rowmajor(W_centered)
            weights_1bit[name] = W_bits
            
            # Store metadata
            metadata[f"{name}.shape"] = W_target.shape
            metadata[f"{name}.dtype"] = "1bit"
        elif name.endswith(".w") and W_fp32.ndim == 1:
             # Some weights might be 1D? No, GPT-2 weights are 2D or 1D biases.
             # Biases are kept as FP32.
             # Just in case, keep as FP32
            weights_fp32_keep[name] = W_fp32
            metadata[f"{name}.dtype"] = "fp32"
        else:
            # Keep as FP32
            weights_fp32_keep[name] = W_fp32
            metadata[f"{name}.dtype"] = "fp32"
    
    # Verify 1.00 bpp for quantized weights
    bpp_result = verify_model_bpp(weights_1bit)
    if not bpp_result["pass"]:
        raise ValueError(f"BPP check failed: {bpp_result['bpp']:.6f} bpp (expected 1.00)")
    
    metadata["bpp_check"] = bpp_result
    
    return QuantizedGPT2(
        config=cfg,
        weights_1bit=weights_1bit,
        weights_fp32=weights_fp32_keep,
        scales=scales,
        means=means,
        metadata=metadata,
    )


def save_quantized_model(model: QuantizedGPT2, output_path: Path) -> None:
    """Save quantized model to disk.
    
    Args:
        model: Quantized model
        output_path: Output file path (.npz format)
    """
    # Prepare data for saving
    save_dict = {}
    
    # Config
    save_dict["config.n_layers"] = model.config.n_layers
    save_dict["config.n_heads"] = model.config.n_heads
    save_dict["config.d_model"] = model.config.d_model
    save_dict["config.d_ff"] = model.config.d_ff
    save_dict["config.vocab_size"] = model.config.vocab_size
    save_dict["config.max_seq_len"] = model.config.max_seq_len
    
    # 1-bit weights
    for name, W_bits in model.weights_1bit.items():
        save_dict[f"1bit.{name}"] = W_bits
    
    # FP32 weights
    for name, W_fp32 in model.weights_fp32.items():
        save_dict[f"fp32.{name}"] = W_fp32
    
    # Scales
    for name, scale in model.scales.items():
        # Handle both scalar and array scales
        if isinstance(scale, np.ndarray):
            save_dict[f"scale.{name}"] = scale
        else:
            save_dict[f"scale.{name}"] = float(scale)
    
    # Means
    for name, mean in model.means.items():
        if isinstance(mean, np.ndarray):
            save_dict[f"mean.{name}"] = mean
        else:
            save_dict[f"mean.{name}"] = float(mean)
    
    # Metadata
    for key, value in model.metadata.items():
        if isinstance(value, (int, float, str, tuple, list)):
            save_dict[f"meta.{key}"] = value
    
    # Save
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved quantized model to {output_path}")
    print(f"  1-bit weights: {len(model.weights_1bit)}")
    print(f"  FP32 weights: {len(model.weights_fp32)}")
    print(f"  BPP: {model.metadata['bpp_check']['bpp']:.6f}")


def load_quantized_model(input_path: Path) -> QuantizedGPT2:
    """Load quantized model from disk.
    
    Args:
        input_path: Input file path (.npz format)
    
    Returns:
        Quantized model
    """
    data = np.load(input_path, allow_pickle=True)
    
    # Load config
    cfg = GPT2Config(
        n_layers=int(data["config.n_layers"]),
        n_heads=int(data["config.n_heads"]),
        d_model=int(data["config.d_model"]),
        d_ff=int(data["config.d_ff"]),
        vocab_size=int(data["config.vocab_size"]),
        max_seq_len=int(data["config.max_seq_len"]),
    )
    
    # Load 1-bit weights
    weights_1bit = {}
    for key in data.keys():
        if key.startswith("1bit."):
            name = key[5:]  # Remove "1bit." prefix
            weights_1bit[name] = data[key]
    
    # Load FP32 weights
    weights_fp32 = {}
    for key in data.keys():
        if key.startswith("fp32."):
            name = key[5:]  # Remove "fp32." prefix
            weights_fp32[name] = data[key]
    
    # Load scales
    scales = {}
    for key in data.keys():
        if key.startswith("scale."):
            name = key[6:]  # Remove "scale." prefix
            val = data[key]
            # Check if scalar or array
            if val.ndim == 0:
                scales[name] = float(val)
            else:
                scales[name] = val
    
    # Load means
    means = {}
    for key in data.keys():
        if key.startswith("mean."):
            name = key[5:]  # Remove "mean." prefix
            val = data[key]
            # Check if scalar or array
            if val.ndim == 0:
                means[name] = float(val)
            else:
                means[name] = val
    
    # Load metadata
    metadata = {}
    for key in data.keys():
        if key.startswith("meta."):
            name = key[5:]  # Remove "meta." prefix
            metadata[name] = data[key]
    
    return QuantizedGPT2(
        config=cfg,
        weights_1bit=weights_1bit,
        weights_fp32=weights_fp32,
        scales=scales,
        means=means,
        metadata=metadata,
    )


def print_model_summary(model: QuantizedGPT2) -> None:
    """Print summary of quantized model.
    
    Args:
        model: Quantized model
    """
    print("=" * 60)
    print("Quantized GPT-2 Model Summary")
    print("=" * 60)
    print(f"Config:")
    print(f"  Layers: {model.config.n_layers}")
    print(f"  Heads: {model.config.n_heads}")
    print(f"  d_model: {model.config.d_model}")
    print(f"  d_ff: {model.config.d_ff}")
    print(f"  Vocab: {model.config.vocab_size}")
    print(f"  Max seq len: {model.config.max_seq_len}")
    print()
    print(f"Weights:")
    print(f"  1-bit matrices: {len(model.weights_1bit)}")
    print(f"  FP32 params: {len(model.weights_fp32)}")
    print(f"  Scales: {len(model.scales)}")
    print(f"  Means: {len(model.means)}")
    print()
    
    # Compute total parameters
    total_1bit = sum(W.size * 32 for W in model.weights_1bit.values())
    total_fp32 = sum(W.size for W in model.weights_fp32.values())
    total_params = total_1bit + total_fp32
    
    print(f"Parameters:")
    print(f"  1-bit: {total_1bit:,} ({total_1bit / 1e6:.2f}M)")
    print(f"  FP32: {total_fp32:,} ({total_fp32 / 1e6:.2f}M)")
    print(f"  Total: {total_params:,} ({total_params / 1e6:.2f}M)")
    print()
    
    # Compute storage
    storage_1bit = sum(W.nbytes for W in model.weights_1bit.values())
    storage_fp32 = sum(W.nbytes for W in model.weights_fp32.values())
    storage_total = storage_1bit + storage_fp32
    
    print(f"Storage:")
    print(f"  1-bit: {storage_1bit / 1024:.2f} KB")
    print(f"  FP32: {storage_fp32 / 1024:.2f} KB")
    print(f"  Total: {storage_total / 1024:.2f} KB ({storage_total / 1024 / 1024:.2f} MB)")
    print()
    
    if "bpp_check" in model.metadata:
        bpp = model.metadata["bpp_check"]["bpp"]
        print(f"BPP Check: {bpp:.6f} (target: 1.00) {'[PASS]' if abs(bpp - 1.0) < 1e-6 else '[FAIL]'}")
    print("=" * 60)
