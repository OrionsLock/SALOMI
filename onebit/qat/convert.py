"""Model conversion, exact BitReport accounting, and optimizer parameter grouping for KD-QAT."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from onebit.eval.harness import BitReport, block_linear_modules, set_weight, unique_param_count
from onebit.qat.modules import (
    BinaryLinear,
    QuantizedEmbedding,
    QuantizedLinearInt,
    QuantizedModule,
    TernaryLinear,
    TiedQuantizedLMHead,
)


@dataclass
class QATConfig:
    """Configuration for converting an FP causal LM into a Phase 1 QAT student."""

    mode: str = "ternary"  # "ternary" | "binary" | "binary_paretoq"
    group_size: Optional[int] = None  # None or <=0 -> per-output-channel
    init_method: str = "twn"  # ternary: "twn" | "absmean"; binary: "svid" | "absmean"
    learnable_scale: bool = True
    use_shift: bool = False
    lsq_scale_grad: bool = False
    grad_clip: Optional[float] = None  # binary sign STE gradient clip (None = pure STE)
    train_signs: bool = True  # False = freeze latent weights, train scales + LN/biases only
    embed_bits: int = 0  # 0 = FP16 embeddings (standard); 4 or 8 = fake-quant embeddings + tied lm_head
    embed_symmetric: bool = True
    embed_group_size: Optional[int] = None
    quantize_pos_embeddings: bool = True  # also quantize wpe when embed_bits > 0
    freeze_embeddings: bool = False  # freeze embedding weights (typically True when embed_bits == 0)

    def to_dict(self) -> Dict:
        return asdict(self)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap ``torch.compile`` (``_orig_mod``) or DDP (``module``) wrappers."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        else:
            return model


def _get_parent_and_attr(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _find_named_attr(model: nn.Module, candidates: Tuple[str, ...]) -> Optional[Tuple[str, nn.Module]]:
    for path in candidates:
        obj = model
        found = True
        for part in path.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and isinstance(obj, nn.Module):
            return path, obj
    return None


def _replace_submodule(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent, attr = _get_parent_and_attr(model, dotted_name)
    setattr(parent, attr, new_module)


def convert_model_to_qat(model: nn.Module, cfg: QATConfig) -> Tuple[nn.Module, BitReport]:
    """Convert transformer-block linears (and optionally embeddings/LM head) in place.

    Supports HF GPT-2 (``Conv1D``), OPT, Pythia/GPT-NeoX, LLaMA, and plain ``nn.Linear``
    modules. Returns ``(model, bit_report)`` where ``bit_report.total_params`` equals the
    exact base FP model parameter count (excluding extra scale/shift metadata).
    """
    base_total_params = unique_param_count(model)
    group = cfg.group_size if cfg.group_size and cfg.group_size > 0 else None

    # 1. Replace every transformer-block linear weight.
    targets = block_linear_modules(model)
    if not targets:
        raise ValueError("No transformer-block linear modules found to convert")

    for name, mod in targets:
        if cfg.mode == "ternary":
            qmod: QuantizedModule = TernaryLinear.from_module(
                mod,
                group_size=group,
                init_method=cfg.init_method if cfg.init_method in ("twn", "absmean") else "twn",
                learnable_scale=cfg.learnable_scale,
                use_shift=cfg.use_shift,
                lsq_scale_grad=cfg.lsq_scale_grad,
                train_weights=cfg.train_signs,
            )
        elif cfg.mode == "binary":
            qmod = BinaryLinear.from_module(
                mod,
                mode="onebit",
                group_size=None,
                use_shift=cfg.use_shift,
                grad_clip=cfg.grad_clip,
                lsq_scale_grad=cfg.lsq_scale_grad,
                train_signs=cfg.train_signs,
            )
        elif cfg.mode == "binary_paretoq":
            qmod = BinaryLinear.from_module(
                mod,
                mode="paretoq",
                group_size=group,
                use_shift=cfg.use_shift,
                grad_clip=cfg.grad_clip,
                lsq_scale_grad=cfg.lsq_scale_grad,
                train_signs=cfg.train_signs,
            )
        else:
            raise ValueError(f"Unsupported QAT mode: {cfg.mode!r}")
        _replace_submodule(model, name, qmod)

    # 2. Optional embedding / LM-head quantization (handles tied and untied LM heads).
    wte_found = _find_named_attr(
        model,
        ("transformer.wte", "gpt_neox.embed_in", "model.decoder.embed_tokens", "model.embed_tokens", "wte"),
    )
    wpe_found = _find_named_attr(
        model,
        ("transformer.wpe", "model.decoder.embed_positions", "wpe"),
    )
    lm_head_found = _find_named_attr(model, ("lm_head", "embed_out"))

    if cfg.embed_bits and cfg.embed_bits > 0:
        emb_group = cfg.embed_group_size if cfg.embed_group_size and cfg.embed_group_size > 0 else None
        q_wte: Optional[QuantizedEmbedding] = None
        orig_wte_weight: Optional[nn.Parameter] = None

        if wte_found is not None and isinstance(wte_found[1], nn.Embedding):
            wte_name, wte_mod = wte_found
            orig_wte_weight = wte_mod.weight
            q_wte = QuantizedEmbedding.from_embedding(
                wte_mod,
                bits=cfg.embed_bits,
                symmetric=cfg.embed_symmetric,
                group_size=emb_group,
            )
            _replace_submodule(model, wte_name, q_wte)

        if lm_head_found is not None:
            head_name, head_mod = lm_head_found
            head_weight = getattr(head_mod, "weight", None)
            is_tied = (
                q_wte is not None
                and orig_wte_weight is not None
                and head_weight is not None
                and head_weight is orig_wte_weight
            )
            if is_tied:
                tied_head = TiedQuantizedLMHead(q_wte, bias=getattr(head_mod, "bias", None))
                _replace_submodule(model, head_name, tied_head)
            elif isinstance(head_mod, nn.Linear):
                q_head = QuantizedLinearInt.from_linear(
                    head_mod,
                    bits=cfg.embed_bits,
                    symmetric=cfg.embed_symmetric,
                    group_size=emb_group,
                )
                _replace_submodule(model, head_name, q_head)

        if cfg.quantize_pos_embeddings and wpe_found is not None and isinstance(wpe_found[1], nn.Embedding):
            wpe_name, wpe_mod = wpe_found
            q_wpe = QuantizedEmbedding.from_embedding(
                wpe_mod,
                bits=cfg.embed_bits,
                symmetric=cfg.embed_symmetric,
                group_size=emb_group,
            )
            _replace_submodule(model, wpe_name, q_wpe)

    # 3. Optional embedding freezing (commonly enabled when embeddings stay FP16).
    if cfg.freeze_embeddings:
        for found in (wte_found, wpe_found):
            if found is not None:
                path, _ = found
                parent, attr = _get_parent_and_attr(model, path)
                mod = getattr(parent, attr)
                for p in mod.parameters():
                    p.requires_grad_(False)
        if lm_head_found is not None:
            path, _ = lm_head_found
            parent, attr = _get_parent_and_attr(model, path)
            mod = getattr(parent, attr)
            for p in mod.parameters():
                p.requires_grad_(False)

    model._salomi_base_total_params = base_total_params  # type: ignore[attr-defined]
    model._salomi_qat_config = cfg  # type: ignore[attr-defined]
    report = build_bit_report(model, base_total_params=base_total_params)
    return model, report


def build_bit_report(model: nn.Module, base_total_params: Optional[int] = None) -> BitReport:
    """Compute an exact ``BitReport`` for a QAT model.

    Counts every quantized linear weight bit + scale/shift vector, every quantized
    embedding bit + scale/zero-point vector (counted once when tied with LM head),
    and every remaining unquantized parameter at 16 bits.
    """
    raw = unwrap_model(model)
    meta_params = 0
    seen_modules = set()
    q_modules: List[Tuple[str, QuantizedModule]] = []

    for name, mod in raw.named_modules():
        if id(mod) in seen_modules:
            continue
        seen_modules.add(id(mod))
        if isinstance(mod, QuantizedModule):
            q_modules.append((name, mod))
            meta_params += mod.meta_num_params()

    if base_total_params is None:
        stored = getattr(raw, "_salomi_base_total_params", None)
        if stored is not None:
            base_total_params = int(stored)
        else:
            base_total_params = unique_param_count(raw) - meta_params

    report = BitReport(total_params=base_total_params)
    for name, mod in q_modules:
        if mod.quant_kind == "embedding":
            report.add_embedding(name, mod.orig_num_params(), mod.storage_bits())
        else:
            report.add(name, mod.orig_num_params(), mod.storage_bits())
    return report


def parameter_groups(
    model: nn.Module,
    lr_weights: float,
    lr_scales: float,
    lr_other: Optional[float] = None,
    weight_decay: float = 0.0,
    scale_weight_decay: float = 0.0,
) -> List[Dict]:
    """Partition trainable parameters into separate optimizer groups:

    1. ``latent_weights``: latent FP shadow weights inside quantized linear layers.
    2. ``scales``: learnable quantizer scales and shifts (separate LR, typically 0 WD).
    3. ``other``: LayerNorms, biases, and (if trainable) embeddings.

    Every unique trainable parameter in ``model`` appears in exactly one group.
    """
    raw = unwrap_model(model)
    latent_ids = set()
    scale_ids = set()

    for mod in raw.modules():
        if isinstance(mod, QuantizedModule):
            if mod.quant_kind == "linear":
                for p in mod.latent_parameters():
                    latent_ids.add(id(p))
            for p in mod.scale_parameters():
                scale_ids.add(id(p))

    latent_params: List[nn.Parameter] = []
    scale_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    seen = set()

    for p in raw.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in scale_ids:
            scale_params.append(p)
        elif id(p) in latent_ids:
            latent_params.append(p)
        else:
            other_params.append(p)

    groups: List[Dict] = []
    if latent_params:
        groups.append({
            "name": "latent_weights",
            "params": latent_params,
            "lr": float(lr_weights),
            "weight_decay": float(weight_decay),
        })
    if scale_params:
        groups.append({
            "name": "scales",
            "params": scale_params,
            "lr": float(lr_scales),
            "weight_decay": float(scale_weight_decay),
        })
    if other_params:
        groups.append({
            "name": "other",
            "params": other_params,
            "lr": float(lr_other if lr_other is not None else lr_weights),
            "weight_decay": 0.0,
        })
    return groups


@torch.no_grad()
def copy_qat_weights_to_fp_model(
    qat_model: nn.Module,
    fp_model: nn.Module,
    source: str = "latent",
) -> nn.Module:
    """Copy trained weights from a QAT model into an FP base model (e.g., for ternary -> binary staging).

    Args:
        qat_model: Trained QAT model (e.g., ``TernaryLinear`` blocks).
        fp_model: Fresh FP model of the same architecture.
        source: ``"latent"`` copies ``w_lat`` (continuous shadow weights, best for SVID init);
            ``"dequantized"`` copies quantized ``dequantized_weight()``.
    """
    if source not in ("latent", "dequantized"):
        raise ValueError(f"source must be 'latent' or 'dequantized', got {source!r}")

    q_raw = unwrap_model(qat_model)
    q_mods = dict(q_raw.named_modules())
    fp_mods = dict(fp_model.named_modules())

    # Copy quantized linear weights and biases back into Conv1D / nn.Linear.
    for name, qmod in q_mods.items():
        if name not in fp_mods:
            continue
        fmod = fp_mods[name]
        if isinstance(qmod, (TernaryLinear, BinaryLinear)):
            w = qmod.w_lat.detach().float() if source == "latent" else qmod.dequantized_weight().float()
            set_weight(fmod, w)
            if qmod.bias is not None and getattr(fmod, "bias", None) is not None:
                fmod.bias.data.copy_(qmod.bias.detach().to(fmod.bias.dtype))
        elif isinstance(qmod, QuantizedEmbedding) and isinstance(fmod, nn.Embedding):
            w = qmod.weight.detach().float() if source == "latent" else qmod.dequantized_weight().float()
            fmod.weight.data.copy_(w.to(fmod.weight.dtype))
        elif isinstance(qmod, QuantizedLinearInt) and isinstance(fmod, nn.Linear):
            w = qmod.weight.detach().float() if source == "latent" else qmod.quantized_weight().detach().float()
            fmod.weight.data.copy_(w.to(fmod.weight.dtype))
            if qmod.bias is not None and fmod.bias is not None:
                fmod.bias.data.copy_(qmod.bias.detach().to(fmod.bias.dtype))
        elif isinstance(qmod, nn.LayerNorm) and isinstance(fmod, nn.LayerNorm):
            fmod.weight.data.copy_(qmod.weight.detach().to(fmod.weight.dtype))
            if qmod.bias is not None and fmod.bias is not None:
                fmod.bias.data.copy_(qmod.bias.detach().to(fmod.bias.dtype))
        elif isinstance(qmod, nn.Embedding) and isinstance(fmod, nn.Embedding):
            fmod.weight.data.copy_(qmod.weight.detach().to(fmod.weight.dtype))

    return fp_model
