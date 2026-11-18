"""Shared, standard evaluation harness for SALOMI.

Every number that goes into the docs should come from this module.

Protocol (matches GPTQ / HF "non-overlapping window" convention):
  * Model: HF causal LM (GPT-2 family is the primary target).
  * Data: WikiText-2-raw *test* split, documents joined with "\\n\\n", tokenized once.
  * Windows: non-overlapping ``seqlen``-token windows (default 1024); every target token
    in each window is scored; NLL is token-weighted across windows.
  * Quantized tensors: transformer-block linear weights only (attention + MLP).
    Embeddings, LayerNorms, biases and the (tied) LM head are kept in floating point,
    which is the convention used by GPTQ / BiLLM / OneBit / ParetoQ.
  * Bits are reported two ways:
      - ``bpp_linear``: bits / param over quantized block linear weights (incl. all
        scales, zero-points, codebooks, low-rank factors -- whatever the quantizer
        reports).
      - ``bpp_whole_model``: same numerator plus every other parameter stored at
        16 bits, divided by the total parameter count.
  * Results are written as JSON with git commit, config and environment so that any
    number in the docs can be traced to a file.

CLI example::

    python -m onebit.eval.harness --method fp16
    python -m onebit.eval.harness --method rtn --bits 4 --group 128
    python -m onebit.eval.harness --method gptq --bits 3 --group 128
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "phase0")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_WIKITEXT_ID = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def _load_wikitext_text(split: str) -> str:
    from datasets import load_dataset

    try:
        ds = load_dataset(*_WIKITEXT_ID, split=split)
    except Exception:  # older `datasets` versions only know the legacy id
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


def _tokenize_long(tokenizer, text: str, chunk_chars: int = 200_000) -> torch.Tensor:
    """Tokenize a long string in chunks (bounded memory) and return a 1-D LongTensor.

    Chunk boundaries are placed on newlines so no word is split across chunks.
    """
    ids: List[int] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl
        ids.extend(tokenizer(text[start:end]).input_ids)
        start = end
    return torch.tensor(ids, dtype=torch.long)


def load_eval_ids(tokenizer, dataset: str = "wikitext2", split: str = "test") -> torch.Tensor:
    if dataset != "wikitext2":
        raise ValueError(f"Unsupported eval dataset: {dataset}")
    text = _load_wikitext_text(split)
    if split == "test":
        # Single call keeps byte-exact parity with the common reference implementation.
        return tokenizer(text, return_tensors="pt").input_ids[0]
    return _tokenize_long(tokenizer, text)


def load_calibration(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 512,
    seed: int = 0,
    dataset: str = "wikitext2",
) -> torch.Tensor:
    """Random ``(n_samples, seqlen)`` windows from the *train* split (never the test split)."""
    ids = load_eval_ids(tokenizer, dataset, split="train")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seqlen - 1, (n_samples,), generator=g)
    return torch.stack([ids[s: s + seqlen] for s in starts.tolist()])


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


@torch.no_grad()
def perplexity(
    model: nn.Module,
    ids: torch.Tensor,
    seqlen: int = 1024,
    max_windows: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Token-weighted perplexity over non-overlapping windows."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    n_windows = ids.numel() // seqlen
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    nll, count = 0.0, 0
    for i in range(n_windows):
        x = ids[i * seqlen: (i + 1) * seqlen].unsqueeze(0).to(device)
        logits = model(x).logits[:, :-1].float()
        nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum"
        ).item()
        count += seqlen - 1
    if was_training:
        model.train()
    mean_nll = nll / max(count, 1)
    return {"ppl": math.exp(mean_nll), "nll": mean_nll, "tokens": count, "windows": n_windows}


# ---------------------------------------------------------------------------
# Model surgery helpers
# ---------------------------------------------------------------------------


def _is_conv1d(module: nn.Module) -> bool:
    return type(module).__name__ == "Conv1D"  # transformers.pytorch_utils.Conv1D


def block_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Linear layers inside transformer blocks (excludes embeddings and LM head)."""
    out = []
    for name, mod in model.named_modules():
        if not (isinstance(mod, nn.Linear) or _is_conv1d(mod)):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        out.append((name, mod))
    return out


def get_weight(module: nn.Module) -> torch.Tensor:
    """Return weight as a float32 ``(out_features, in_features)`` tensor (a copy)."""
    w = module.weight.data
    w = w.t() if _is_conv1d(module) else w
    return w.detach().float().clone()


def set_weight(module: nn.Module, w_out_in: torch.Tensor) -> None:
    w = w_out_in.t() if _is_conv1d(module) else w_out_in
    module.weight.data.copy_(w.to(module.weight.dtype).contiguous())


def transformer_blocks(model: nn.Module) -> nn.ModuleList:
    for attr in ("transformer.h", "gpt_neox.layers", "model.decoder.layers", "model.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("Could not locate transformer blocks for this architecture")


# ---------------------------------------------------------------------------
# Bit accounting
# ---------------------------------------------------------------------------


@dataclass
class BitReport:
    linear_params: int = 0
    linear_bits: float = 0.0
    total_params: int = 0
    per_layer: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add(self, name: str, n_params: int, bits: float) -> None:
        self.linear_params += n_params
        self.linear_bits += bits
        self.per_layer[name] = {"params": n_params, "bits": bits, "bpp": bits / max(n_params, 1)}

    @property
    def bpp_linear(self) -> float:
        return self.linear_bits / max(self.linear_params, 1)

    @property
    def other_params(self) -> int:
        return self.total_params - self.linear_params

    @property
    def bpp_whole_model(self) -> float:
        """Quantized linears + every other parameter at 16 bits, over all parameters."""
        return (self.linear_bits + 16.0 * self.other_params) / max(self.total_params, 1)

    def summary(self) -> Dict[str, float]:
        return {
            "bpp_linear": self.bpp_linear,
            "bpp_whole_model": self.bpp_whole_model,
            "linear_params": self.linear_params,
            "other_params_fp16": self.other_params,
            "total_params": self.total_params,
        }


def unique_param_count(model: nn.Module) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


# A quantizer maps an (out, in) float weight to (dequantized weight, total storage bits).
Quantizer = Callable[[torch.Tensor, Dict], Tuple[torch.Tensor, float]]


def quantize_model_rtn_style(model: nn.Module, quantizer: Quantizer, ctx: Optional[Dict] = None) -> BitReport:
    """Apply a data-free / per-layer quantizer to every block linear, in place."""
    report = BitReport(total_params=unique_param_count(model))
    ctx = dict(ctx or {})
    for name, mod in block_linear_modules(model):
        w = get_weight(mod)
        layer_ctx = dict(ctx, name=name)
        if "h_diag" in ctx:
            layer_ctx["h_diag"] = ctx["h_diag"][name]
        wq, bits = quantizer(w, layer_ctx)
        set_weight(mod, wq)
        report.add(name, w.numel(), float(bits))
    return report


# ---------------------------------------------------------------------------
# Calibration statistics
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_input_diag(model: nn.Module, calib: torch.Tensor, batch_size: int = 8) -> Dict[str, torch.Tensor]:
    """Return ``{layer_name: mean(x**2) over calibration tokens}`` for every block linear."""
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    hooks = []
    for name, mod in block_linear_modules(model):
        def hook(m, inp, out, name=name):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            sums[name] = sums.get(name, 0) + (x ** 2).sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        hooks.append(mod.register_forward_hook(hook))
    device = next(model.parameters()).device
    try:
        for i in range(0, calib.shape[0], batch_size):
            model(calib[i: i + batch_size].to(device))
    finally:
        for h in hooks:
            h.remove()
    return {k: (sums[k] / counts[k]).cpu() for k in sums}


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def environment_info() -> Dict[str, str]:
    import transformers

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def write_result(record: Dict, out_dir: str = DEFAULT_RESULTS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{record['run_name']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    method: str = "fp16"
    model: str = "gpt2"
    bits: int = 4
    group: int = 128  # <=0 means per-output-channel
    seqlen: int = 1024
    max_windows: Optional[int] = None
    calib_samples: int = 128
    calib_seqlen: int = 512
    seed: int = 0
    run_name: Optional[str] = None
    out_dir: str = DEFAULT_RESULTS_DIR
    extra: Dict = field(default_factory=dict)

    def default_name(self) -> str:
        g = f"g{self.group}" if self.group and self.group > 0 else "perchan"
        if self.method in ("fp16", "fp32"):
            base = self.method
        elif self.method in ("rtn", "gptq"):
            base = f"{self.method}_w{self.bits}_{g}"
        else:
            base = f"{self.method}_{g}"
        tag = f"_{self.model.replace('/', '-')}"
        win = f"_win{self.max_windows}" if self.max_windows else ""
        return base + tag + win


def run(cfg: RunConfig) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from onebit.quantization import baselines as B

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.float32).to(device).eval()
    ids = load_eval_ids(tok, "wikitext2", "test")

    t0 = time.time()
    group = cfg.group if cfg.group and cfg.group > 0 else None
    report: Optional[BitReport] = None
    if cfg.method in ("fp16", "fp32"):
        report = BitReport(total_params=unique_param_count(model))
        for name, mod in block_linear_modules(model):
            n = mod.weight.numel()
            report.add(name, n, n * (16.0 if cfg.method == "fp16" else 32.0))
        if cfg.method == "fp16":
            model = model.half().float()  # round-trip weights through fp16
    elif cfg.method == "rtn":
        report = quantize_model_rtn_style(model, lambda w, c: B.rtn(w, cfg.bits, group))
    elif cfg.method == "ternary":
        report = quantize_model_rtn_style(model, lambda w, c: B.ternary_absmean(w, group))
    elif cfg.method == "ternary_twn":
        report = quantize_model_rtn_style(model, lambda w, c: B.ternary_twn(w, group))
    elif cfg.method == "binary":
        report = quantize_model_rtn_style(model, lambda w, c: B.binary_absmean(w, group))
    elif cfg.method == "binary_svid":
        report = quantize_model_rtn_style(model, lambda w, c: B.binary_svid(w))
    elif cfg.method == "gptq":
        from onebit.quantization.gptq import gptq_quantize_model

        calib = load_calibration(tok, cfg.calib_samples, cfg.calib_seqlen, cfg.seed)
        report = gptq_quantize_model(model, calib, bits=cfg.bits, group_size=group, **cfg.extra)
    elif cfg.method.startswith("legacy_"):
        from onebit.quantization import legacy_adapters as L

        ctx = {}
        if cfg.extra.get("use_calibration_hessian", True):
            calib = load_calibration(tok, cfg.calib_samples, cfg.calib_seqlen, cfg.seed)
            ctx["h_diag"] = collect_input_diag(model, calib)
        report = quantize_model_rtn_style(model, L.get(cfg.method), ctx)
    else:
        raise ValueError(f"Unknown method {cfg.method}")
    quant_seconds = time.time() - t0

    t1 = time.time()
    res = perplexity(model, ids, cfg.seqlen, cfg.max_windows, device)
    eval_seconds = time.time() - t1

    record = {
        "run_name": cfg.run_name or cfg.default_name(),
        "protocol": {
            "dataset": "wikitext-2-raw-v1/test",
            "join": "\\n\\n",
            "seqlen": cfg.seqlen,
            "windows": res["windows"],
            "scored_tokens": res["tokens"],
            "full_test_set": cfg.max_windows is None,
            "quantized": "transformer-block linear weights; embeddings/LN/biases/lm_head kept FP",
        },
        "config": dataclasses.asdict(cfg),
        "result": {"ppl": res["ppl"], "nll": res["nll"], **report.summary()},
        "timing_seconds": {"quantize": quant_seconds, "eval": eval_seconds},
        "env": environment_info(),
    }
    record["config"].pop("out_dir", None)
    path = write_result(record, cfg.out_dir)
    print(
        f"[harness] {record['run_name']}: ppl={res['ppl']:.3f}  bpp_linear={report.bpp_linear:.4f}  "
        f"bpp_whole={report.bpp_whole_model:.4f}  -> {os.path.relpath(path, REPO_ROOT)}",
        flush=True,
    )
    return record


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="fp16",
                   help="fp16 | fp32 | rtn | gptq | ternary | ternary_twn | binary | binary_svid | legacy_<name>")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=128, help="group size; <=0 for per-output-channel")
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=None, help="debug only; omit for full test set")
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=DEFAULT_RESULTS_DIR)
    a = p.parse_args(argv)
    run(RunConfig(method=a.method, model=a.model, bits=a.bits, group=a.group, seqlen=a.seqlen,
                  max_windows=a.max_windows, calib_samples=a.calib_samples, calib_seqlen=a.calib_seqlen,
                  seed=a.seed, run_name=a.run_name, out_dir=a.out_dir))


if __name__ == "__main__":
    main()
