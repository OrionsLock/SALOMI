"""HuggingFace model loader with architecture-agnostic adapter.

Supports the GPT-2 family (124M-1.5B) and any AutoModelForCausalLM
model (OPT, Pythia, Llama, Mistral, etc.) via ``ModelAdapter`` which
provides a uniform interface for iterating layers and their sub-modules.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

GPT2_FAMILY: Dict[str, str] = {
    "124M": "gpt2",
    "355M": "gpt2-medium",
    "774M": "gpt2-large",
    "1.5B": "gpt2-xl",
}

MODEL_CATALOG: Dict[str, str] = {
    **GPT2_FAMILY,
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
}


@dataclass
class ModelInfo:
    """Metadata extracted from a loaded model."""
    name: str
    n_params: int
    n_layers: int
    d_model: int
    d_ff: int
    n_heads: int
    vocab_size: int


# ======================================================================
# Architecture-agnostic adapter
# ======================================================================

class ModelAdapter:
    """Uniform interface over different transformer architectures.

    Abstracts away the differences in layer access, sub-module naming,
    and weight transposition between GPT-2, OPT, GPT-NeoX/Pythia, etc.
    """

    def __init__(self, model):
        self.model = model
        self.arch = self._detect_arch(model)

    @staticmethod
    def _detect_arch(model) -> str:
        cls_name = type(model).__name__.lower()
        cfg_type = type(model.config).__name__.lower()
        if "gpt2" in cls_name or "gpt2" in cfg_type:
            return "gpt2"
        if "opt" in cls_name or "opt" in cfg_type:
            return "opt"
        if "neox" in cls_name or "neox" in cfg_type or "pythia" in cfg_type:
            return "neox"
        if "llama" in cls_name or "llama" in cfg_type:
            return "llama"
        if "mistral" in cls_name or "mistral" in cfg_type:
            return "mistral"
        return "generic"

    @property
    def n_layers(self) -> int:
        cfg = self.model.config
        for attr in ("n_layer", "num_hidden_layers", "num_layers"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise AttributeError("Cannot determine n_layers from config")

    def get_layer_targets(self, layer_idx: int) -> List[Tuple[str, nn.Module]]:
        """Return ``[(comp_name, submodule), ...]`` for one transformer block.

        comp_name is one of:  mlp_fc, mlp_proj, attn_qkv, attn_proj
        (or attn_q/attn_k/attn_v when the arch splits Q/K/V).
        """
        block = self._get_block(layer_idx)
        if self.arch == "gpt2":
            return [
                ("mlp_fc", block.mlp.c_fc),
                ("mlp_proj", block.mlp.c_proj),
                ("attn_qkv", block.attn.c_attn),
                ("attn_proj", block.attn.c_proj),
            ]
        if self.arch == "opt":
            targets = [
                ("mlp_fc", block.fc1),
                ("mlp_proj", block.fc2),
                ("attn_proj", block.self_attn.out_proj),
            ]
            if hasattr(block.self_attn, "q_proj"):
                targets.append(("attn_q", block.self_attn.q_proj))
                targets.append(("attn_k", block.self_attn.k_proj))
                targets.append(("attn_v", block.self_attn.v_proj))
            return targets
        if self.arch == "neox":
            targets = [
                ("mlp_fc", block.mlp.dense_h_to_4h),
                ("mlp_proj", block.mlp.dense_4h_to_h),
                ("attn_proj", block.attention.dense),
            ]
            if hasattr(block.attention, "query_key_value"):
                targets.append(("attn_qkv", block.attention.query_key_value))
            return targets
        if self.arch in ("llama", "mistral"):
            targets = [
                ("mlp_fc", block.mlp.gate_proj),
                ("mlp_up", block.mlp.up_proj),
                ("mlp_proj", block.mlp.down_proj),
                ("attn_q", block.self_attn.q_proj),
                ("attn_k", block.self_attn.k_proj),
                ("attn_v", block.self_attn.v_proj),
                ("attn_proj", block.self_attn.o_proj),
            ]
            return targets
        return self._generic_targets(block)

    def get_weight_numpy(self, submod: nn.Module) -> np.ndarray:
        """Return weight as (d_out, d_in) numpy array.

        GPT-2 Conv1D stores weights as (d_in, d_out) -- we transpose.
        All other architectures use standard (d_out, d_in) Linear.
        """
        W = submod.weight.detach().cpu().float().numpy()
        if self.arch == "gpt2" and W.shape[0] < W.shape[1]:
            return W.T
        if self.arch == "gpt2":
            return W.T
        return W

    def set_weight_numpy(self, submod: nn.Module, W_np: np.ndarray):
        """Write a (d_out, d_in) numpy array back into the submodule."""
        if self.arch == "gpt2":
            t = torch.from_numpy(W_np.T.astype(np.float32))
        else:
            t = torch.from_numpy(W_np.astype(np.float32))
        with torch.no_grad():
            submod.weight.copy_(t)

    def _get_block(self, idx: int):
        m = self.model
        if self.arch == "gpt2":
            return m.transformer.h[idx]
        if self.arch == "opt":
            return m.model.decoder.layers[idx]
        if self.arch == "neox":
            return m.gpt_neox.layers[idx]
        if self.arch in ("llama", "mistral"):
            return m.model.layers[idx]
        for path in ("transformer.h", "model.layers", "model.decoder.layers",
                      "gpt_neox.layers", "transformer.layers"):
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                return obj[idx]
            except (AttributeError, IndexError):
                continue
        raise AttributeError(f"Cannot locate layer {idx} in {type(m).__name__}")

    @staticmethod
    def _generic_targets(block) -> List[Tuple[str, nn.Module]]:
        """Fallback: find all Linear modules in the block."""
        targets = []
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.dim() == 2:
                prefix = "mlp" if "mlp" in name or "fc" in name else "attn"
                targets.append((f"{prefix}_{name.replace('.', '_')}", mod))
        return targets


# ======================================================================
# Model info extraction (multi-arch)
# ======================================================================

def get_model_info(model, name: str = "") -> ModelInfo:
    """Extract structural metadata from any supported model."""
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())

    n_layers = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None) or 0
    d_model = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", None) or 0
    d_ff = (getattr(cfg, "n_inner", None)
            or getattr(cfg, "ffn_dim", None)
            or getattr(cfg, "intermediate_size", None)
            or 4 * d_model)
    n_heads = getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads", None) or 0
    vocab_size = getattr(cfg, "vocab_size", 0)

    return ModelInfo(
        name=name, n_params=n_params, n_layers=n_layers,
        d_model=d_model, d_ff=d_ff, n_heads=n_heads, vocab_size=vocab_size,
    )


# ======================================================================
# Loading
# ======================================================================

def load_proxy_family(
    sizes: Optional[list] = None,
    device: str = "cpu",
    dtype: Optional[str] = None,
) -> Dict[str, Tuple]:
    """Load model family members and their tokenizers.

    Args:
        sizes: list of labels from ``MODEL_CATALOG``
               (e.g. ``["124M", "opt-6.7b"]``).
               Default loads the four GPT-2 variants.
        device: torch device string.
        dtype: optional ``"float16"`` or ``"bfloat16"`` for large models.

    Returns:
        ``{size_label: (model, tokenizer, ModelInfo, ModelAdapter)}``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if sizes is None:
        sizes = list(GPT2_FAMILY.keys())

    torch_dtype = None
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    family: Dict[str, Tuple] = {}
    for label in sizes:
        hf_name = MODEL_CATALOG.get(label, label)
        print(f"  Loading {label} ({hf_name})...")

        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict = {"trust_remote_code": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kwargs)
        model.eval()
        model.to(device)

        info = get_model_info(model, name=label)
        adapter = ModelAdapter(model)
        family[label] = (model, tokenizer, info, adapter)
        print(f"    {info.n_params:,} params, {info.n_layers} layers, "
              f"d={info.d_model}, arch={adapter.arch}")

    return family


def create_scaled_variant(
    base_config,
    width_mult: float = 1.0,
    depth_mult: float = 1.0,
) -> Tuple:
    """Create a randomly-initialised GPT-2 with custom width/depth."""
    from transformers import GPT2Config, GPT2LMHeadModel

    new_n_embd = int(base_config.n_embd * width_mult)
    new_n_embd = max(64, (new_n_embd // 64) * 64)
    new_n_layer = max(1, int(base_config.n_layer * depth_mult))
    new_n_inner = 4 * new_n_embd

    cfg = GPT2Config(
        vocab_size=base_config.vocab_size,
        n_positions=base_config.n_positions,
        n_embd=new_n_embd,
        n_layer=new_n_layer,
        n_head=max(1, new_n_embd // 64),
        n_inner=new_n_inner,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model, cfg
