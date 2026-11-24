"""GPU-ready KD-QAT training loop with resumable checkpoints, JSONL logs, and harness eval."""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from onebit.eval.harness import (
    REPO_ROOT,
    BitReport,
    evaluate_model,
    load_eval_ids,
    perplexity as harness_perplexity,
)
from onebit.qat.convert import (
    QATConfig,
    build_bit_report,
    convert_model_to_qat,
    copy_qat_weights_to_fp_model,
    parameter_groups,
    unwrap_model,
)
from onebit.qat.data import build_train_dataloader
from onebit.qat.distill import DistillationLoss


@dataclass
class TrainConfig:
    """Hyperparameters and runtime configuration for a KD-QAT run."""

    # Model & data
    model: str = "gpt2"
    dataset: str = "fineweb-edu"
    dataset_config: Optional[str] = None
    shuffle_buffer: int = 10_000
    seqlen: int = 1024
    batch_size: int = 8
    grad_accum: int = 4

    # Budget (max_steps takes precedence if > 0; otherwise computed from max_tokens)
    max_tokens: int = 3_000_000_000
    max_steps: int = 0

    # Optimization
    lr: float = 3e-4
    scale_lr: float = 1e-3
    other_lr: Optional[float] = None
    weight_decay: float = 0.0
    scale_weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0

    # Distillation
    kl_weight: float = 1.0
    ce_weight: float = 0.0
    temperature: float = 1.0
    hidden_mse_weight: float = 0.0

    # Precision / runtime
    dtype: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"
    compile_model: bool = False
    seed: int = 0
    num_threads: Optional[int] = None

    # Evaluation & checkpointing
    eval_every: int = 500
    eval_windows: Optional[int] = 20  # mid-run eval windows (None = full test set)
    final_eval_windows: Optional[int] = None  # None = full WikiText-2 test set
    eval_seqlen: int = 1024
    log_every: int = 20
    save_every: int = 500

    # Paths & resume / staging
    run_name: Optional[str] = None
    work_dir: str = os.path.join(REPO_ROOT, "checkpoints", "phase1")
    results_dir: str = os.path.join(REPO_ROOT, "results", "phase1")
    resume: Optional[str] = None
    init_from_checkpoint: Optional[str] = None
    init_source: str = "latent"  # "latent" | "dequantized"

    def tokens_per_step(self) -> int:
        return self.batch_size * self.seqlen * self.grad_accum

    def resolved_max_steps(self) -> int:
        if self.max_steps and self.max_steps > 0:
            return int(self.max_steps)
        return max(1, math.ceil(self.max_tokens / max(self.tokens_per_step(), 1)))

    def resolved_warmup_steps(self, total_steps: int) -> int:
        if self.warmup_ratio and self.warmup_ratio > 0:
            return int(round(total_steps * self.warmup_ratio))
        return min(int(self.warmup_steps), max(0, total_steps - 1))

    def to_dict(self) -> Dict:
        return asdict(self)


def resolve_amp_dtype(dtype_name: str, device: torch.device) -> Tuple[Optional[torch.dtype], bool]:
    """Return ``(amp_dtype, use_grad_scaler)`` for the given device and requested precision."""
    if dtype_name == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, False
            return torch.float16, True
        return None, False
    if dtype_name == "fp32":
        return None, False
    if dtype_name == "bf16":
        return torch.bfloat16, False
    if dtype_name == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 autocast requires CUDA; use fp32 or bf16 on CPU")
        return torch.float16, True
    raise ValueError(f"Unsupported dtype: {dtype_name!r} (expected 'auto', 'bf16', 'fp16', 'fp32')")


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to ``min_lr_ratio * base_lr``."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    tokens_seen: int,
    best_ppl: float,
    qat_cfg: QATConfig,
    train_cfg: TrainConfig,
    report: BitReport,
    scaler: Optional[torch.amp.GradScaler] = None,
    dataloader: Optional[Iterator[torch.Tensor]] = None,
    micro_batches_seen: int = 0,
) -> str:
    """Atomically save a full resumable KD-QAT training checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw = unwrap_model(model)
    payload: Dict = {
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "micro_batches_seen": int(micro_batches_seen),
        "best_ppl": float(best_ppl),
        "qat_config": qat_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "bit_report": report.summary(),
        "base_total_params": int(report.total_params),
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_rng_state": dataloader.generator.get_state() if hasattr(dataloader, "generator") else None,
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    dataloader: Optional[Iterator[torch.Tensor]] = None,
    restore_rng: bool = True,
    map_location: str = "cpu",
) -> Dict:
    """Restore model, optimizer, scheduler, RNG, and counters from a checkpoint."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    raw = unwrap_model(model)
    raw.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and scaler.is_enabled() and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if restore_rng:
        if ckpt.get("rng_state_cpu") is not None:
            torch.set_rng_state(ckpt["rng_state_cpu"])
        if torch.cuda.is_available() and ckpt.get("rng_state_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_state_cuda"])
        if dataloader is not None and hasattr(dataloader, "generator") and ckpt.get("data_rng_state") is not None:
            dataloader.generator.set_state(ckpt["data_rng_state"])
    return ckpt


def _append_jsonl(path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def default_run_name(qat_cfg: QATConfig, train_cfg: TrainConfig) -> str:
    model_tag = train_cfg.model.replace("/", "-")
    g = f"g{qat_cfg.group_size}" if qat_cfg.group_size and qat_cfg.group_size > 0 else "perchan"
    emb = f"_e{qat_cfg.embed_bits}" if qat_cfg.embed_bits and qat_cfg.embed_bits > 0 else "_efp16"
    shift = "_shift" if qat_cfg.use_shift else ""
    win = f"_win{train_cfg.final_eval_windows}" if train_cfg.final_eval_windows else ""
    return f"kdqat_{qat_cfg.mode}_{g}{shift}{emb}_{model_tag}{win}"


def run_kd_qat_training(
    student: nn.Module,
    teacher: Optional[nn.Module],
    report: BitReport,
    qat_cfg: QATConfig,
    train_cfg: TrainConfig,
    dataloader: Iterator[torch.Tensor],
    tokenizer=None,
    eval_ids: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    write_final_harness_json: bool = True,
) -> Dict:
    """Core KD-QAT training loop decoupled from HF Hub downloads for easy testing and reuse."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_cfg.num_threads is not None:
        torch.set_num_threads(int(train_cfg.num_threads))

    run_name = train_cfg.run_name or default_run_name(qat_cfg, train_cfg)
    run_dir = os.path.join(train_cfg.work_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")
    latest_ckpt_path = os.path.join(run_dir, "latest.pt")
    best_ckpt_path = os.path.join(run_dir, "best.pt")

    amp_dtype, use_scaler = resolve_amp_dtype(train_cfg.dtype, device)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    student = student.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16):
            teacher = teacher.to(dtype=amp_dtype)

    if train_cfg.compile_model:
        student = torch.compile(student)

    criterion = DistillationLoss(
        kl_weight=train_cfg.kl_weight,
        ce_weight=train_cfg.ce_weight,
        temperature=train_cfg.temperature,
        hidden_mse_weight=train_cfg.hidden_mse_weight,
    )

    param_groups = parameter_groups(
        student,
        lr_weights=train_cfg.lr,
        lr_scales=train_cfg.scale_lr,
        lr_other=train_cfg.other_lr,
        weight_decay=train_cfg.weight_decay,
        scale_weight_decay=train_cfg.scale_weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=train_cfg.betas)

    total_steps = train_cfg.resolved_max_steps()
    warmup_steps = train_cfg.resolved_warmup_steps(total_steps)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=train_cfg.min_lr_ratio,
    )

    start_step = 0
    tokens_seen = 0
    micro_batches_seen = 0
    best_ppl = float("inf")

    if train_cfg.resume:
        ckpt = load_checkpoint(
            train_cfg.resume,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            dataloader=dataloader,
            restore_rng=True,
            map_location=str(device),
        )
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt["tokens_seen"])
        micro_batches_seen = int(ckpt.get("micro_batches_seen", 0))
        best_ppl = float(ckpt.get("best_ppl", float("inf")))
        # For streaming datasets without a torch.Generator, advance past already-consumed micro-batches.
        if not hasattr(dataloader, "generator") and micro_batches_seen > 0:
            for _ in range(micro_batches_seen):
                next(dataloader)
        print(
            f"[kd-qat] resumed {run_name} from step={start_step} tokens={tokens_seen} best_ppl={best_ppl:.3f}",
            flush=True,
        )

    trainable_params = sum(p.numel() for p in unwrap_model(student).parameters() if p.requires_grad)
    print(
        f"[kd-qat] run={run_name} mode={qat_cfg.mode} device={device} amp={amp_dtype} "
        f"bpp_linear={report.bpp_linear:.4f} bpp_whole={report.bpp_whole_model:.4f} "
        f"trainable={trainable_params / 1e6:.2f}M steps={start_step}->{total_steps} "
        f"tok/step={train_cfg.tokens_per_step()}",
        flush=True,
    )

    init_ppl: Optional[float] = None
    teacher_ppl: Optional[float] = None
    if eval_ids is not None and start_step == 0:
        if teacher is not None:
            t_res = harness_perplexity(
                teacher, eval_ids, seqlen=train_cfg.eval_seqlen, max_windows=train_cfg.eval_windows, device=str(device)
            )
            teacher_ppl = t_res["ppl"]
        s_res = harness_perplexity(
            unwrap_model(student),
            eval_ids,
            seqlen=train_cfg.eval_seqlen,
            max_windows=train_cfg.eval_windows,
            device=str(device),
        )
        init_ppl = s_res["ppl"]
        best_ppl = min(best_ppl, init_ppl)
        _append_jsonl(
            log_path,
            {
                "event": "eval",
                "step": 0,
                "tokens_seen": 0,
                "eval_ppl": init_ppl,
                "eval_nll": s_res["nll"],
                "eval_windows": s_res["windows"],
                "teacher_ppl": teacher_ppl,
            },
        )
        print(
            f"[kd-qat] init eval: student_ppl={init_ppl:.3f}"
            + (f" teacher_ppl={teacher_ppl:.3f}" if teacher_ppl is not None else ""),
            flush=True,
        )

    all_trainable = [p for g in optimizer.param_groups for p in g["params"]]
    group_idx = {g["name"]: i for i, g in enumerate(optimizer.param_groups)}

    student.train()
    t_start = time.time()
    t_interval = t_start
    toks_interval = 0
    step = start_step
    last_metrics: Dict[str, float] = {}
    loss_history = []

    for step in range(start_step + 1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_kl = 0.0
        accum_ce = 0.0
        accum_hmse = 0.0

        for _ in range(train_cfg.grad_accum):
            batch = next(dataloader).to(device, non_blocking=(device.type == "cuda"))
            micro_batches_seen += 1

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                teacher_logits = None
                teacher_hs = None
                if teacher is not None and (criterion.kl_weight > 0 or criterion.needs_hidden_states):
                    with torch.no_grad():
                        t_out = teacher(batch, output_hidden_states=criterion.needs_hidden_states)
                        teacher_logits = t_out.logits
                        teacher_hs = t_out.hidden_states if criterion.needs_hidden_states else None

                s_out = student(batch, output_hidden_states=criterion.needs_hidden_states)
                student_logits = s_out.logits
                student_hs = s_out.hidden_states if criterion.needs_hidden_states else None

                loss, metrics = criterion(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    input_ids=batch,
                    student_hidden_states=student_hs,
                    teacher_hidden_states=teacher_hs,
                )
                scaled_loss = loss / float(train_cfg.grad_accum)

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += metrics["loss"] / float(train_cfg.grad_accum)
            accum_kl += metrics["kl_loss"] / float(train_cfg.grad_accum)
            accum_ce += metrics["ce_loss"] / float(train_cfg.grad_accum)
            accum_hmse += metrics["hidden_mse"] / float(train_cfg.grad_accum)
            tokens_seen += int(batch.numel())
            toks_interval += int(batch.numel())

        if use_scaler:
            scaler.unscale_(optimizer)

        if train_cfg.grad_clip and train_cfg.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(all_trainable, train_cfg.grad_clip).item())
        else:
            grad_norm = 0.0

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        last_metrics = {
            "loss": accum_loss,
            "kl_loss": accum_kl,
            "ce_loss": accum_ce,
            "hidden_mse": accum_hmse,
            "grad_norm": grad_norm,
        }
        loss_history.append(accum_loss)

        if train_cfg.log_every > 0 and (step % train_cfg.log_every == 0 or step == total_steps or step == start_step + 1):
            now = time.time()
            dt = max(now - t_interval, 1e-6)
            tok_s = toks_interval / dt
            lr_w = optimizer.param_groups[group_idx["latent_weights"]]["lr"] if "latent_weights" in group_idx else optimizer.param_groups[0]["lr"]
            lr_s = optimizer.param_groups[group_idx["scales"]]["lr"] if "scales" in group_idx else lr_w
            rec = {
                "event": "train",
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": accum_loss,
                "kl_loss": accum_kl,
                "ce_loss": accum_ce,
                "hidden_mse": accum_hmse,
                "grad_norm": grad_norm,
                "lr_weights": lr_w,
                "lr_scales": lr_s,
                "tok_per_sec": tok_s,
                "elapsed_s": now - t_start,
            }
            _append_jsonl(log_path, rec)
            print(
                f"[kd-qat] step {step}/{total_steps} tokens={tokens_seen:,} "
                f"loss={accum_loss:.4f} kl={accum_kl:.4f} ce={accum_ce:.4f} "
                f"lr_w={lr_w:.2e} lr_s={lr_s:.2e} gnorm={grad_norm:.2f} tok/s={tok_s:,.0f}",
                flush=True,
            )
            t_interval = now
            toks_interval = 0

        if eval_ids is not None and train_cfg.eval_every > 0 and step % train_cfg.eval_every == 0 and step < total_steps:
            e_res = harness_perplexity(
                unwrap_model(student),
                eval_ids,
                seqlen=train_cfg.eval_seqlen,
                max_windows=train_cfg.eval_windows,
                device=str(device),
            )
            e_ppl = e_res["ppl"]
            is_best = e_ppl < best_ppl
            if is_best:
                best_ppl = e_ppl
                save_checkpoint(
                    best_ckpt_path,
                    student,
                    optimizer,
                    scheduler,
                    step,
                    tokens_seen,
                    best_ppl,
                    qat_cfg,
                    train_cfg,
                    report,
                    scaler=scaler,
                    dataloader=dataloader,
                    micro_batches_seen=micro_batches_seen,
                )
            _append_jsonl(
                log_path,
                {
                    "event": "eval",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "eval_ppl": e_ppl,
                    "eval_nll": e_res["nll"],
                    "eval_windows": e_res["windows"],
                    "best_ppl": best_ppl,
                    "is_best": is_best,
                },
            )
            print(
                f"[kd-qat] EVAL step {step} tokens={tokens_seen:,}: ppl={e_ppl:.3f} "
                f"(best={best_ppl:.3f}, win={e_res['windows']})",
                flush=True,
            )

        if train_cfg.save_every > 0 and step % train_cfg.save_every == 0 and step < total_steps:
            save_checkpoint(
                latest_ckpt_path,
                student,
                optimizer,
                scheduler,
                step,
                tokens_seen,
                best_ppl,
                qat_cfg,
                train_cfg,
                report,
                scaler=scaler,
                dataloader=dataloader,
                micro_batches_seen=micro_batches_seen,
            )

    train_seconds = time.time() - t_start

    # Save final latest checkpoint.
    save_checkpoint(
        latest_ckpt_path,
        student,
        optimizer,
        scheduler,
        step,
        tokens_seen,
        best_ppl,
        qat_cfg,
        train_cfg,
        report,
        scaler=scaler,
        dataloader=dataloader,
        micro_batches_seen=micro_batches_seen,
    )

    training_summary = {
        "steps": step,
        "tokens_seen": tokens_seen,
        "dataset": train_cfg.dataset,
        "dataset_config": train_cfg.dataset_config,
        "init_ppl": init_ppl,
        "teacher_ppl": teacher_ppl,
        "best_mid_eval_ppl": best_ppl if math.isfinite(best_ppl) else None,
        "final_train_loss": last_metrics.get("loss"),
        "initial_train_loss": loss_history[0] if loss_history else None,
        "checkpoint_latest": os.path.relpath(latest_ckpt_path, REPO_ROOT),
        "train_log_jsonl": os.path.relpath(log_path, REPO_ROOT),
    }

    harness_record: Optional[Dict] = None
    if write_final_harness_json and (eval_ids is not None or tokenizer is not None):
        cfg_dump = {
            "method": f"kd_qat_{qat_cfg.mode}",
            "model": train_cfg.model,
            "qat": qat_cfg.to_dict(),
            "train": {
                k: v
                for k, v in train_cfg.to_dict().items()
                if k not in ("work_dir", "results_dir")
            },
        }
        harness_record = evaluate_model(
            model=unwrap_model(student),
            tokenizer=tokenizer,
            report=report,
            run_name=run_name,
            config=cfg_dump,
            out_dir=train_cfg.results_dir,
            seqlen=train_cfg.eval_seqlen,
            max_windows=train_cfg.final_eval_windows,
            device=str(device),
            eval_ids=eval_ids,
            extra_timing={"train": train_seconds},
            training_summary=training_summary,
        )
        if harness_record["result"]["ppl"] < best_ppl:
            best_ppl = harness_record["result"]["ppl"]
            save_checkpoint(
                best_ckpt_path,
                student,
                optimizer,
                scheduler,
                step,
                tokens_seen,
                best_ppl,
                qat_cfg,
                train_cfg,
                report,
                scaler=scaler,
                dataloader=dataloader,
                micro_batches_seen=micro_batches_seen,
            )

    return {
        "run_name": run_name,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_ppl": best_ppl,
        "loss_history": loss_history,
        "latest_checkpoint": latest_ckpt_path,
        "log_path": log_path,
        "bit_report": report.summary(),
        "harness_record": harness_record,
    }


def train_from_pretrained(qat_cfg: QATConfig, train_cfg: TrainConfig) -> Dict:
    """End-to-end entry point: load HF teacher, build/convert student, train, and evaluate."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model)

    # Always keep the original pretrained FP model as the frozen distillation teacher.
    teacher = AutoModelForCausalLM.from_pretrained(train_cfg.model).float().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_base = copy.deepcopy(teacher)

    # Optional staged init (e.g., Stage 1 ternary checkpoint -> Stage 2 binary SVID init).
    if train_cfg.init_from_checkpoint:
        src_ckpt = torch.load(train_cfg.init_from_checkpoint, map_location="cpu", weights_only=False)
        src_qat_cfg = QATConfig(**src_ckpt["qat_config"])
        src_model = copy.deepcopy(teacher)
        src_model, _ = convert_model_to_qat(src_model, src_qat_cfg)
        src_model.load_state_dict(src_ckpt["model_state_dict"], strict=True)
        copy_qat_weights_to_fp_model(src_model, student_base, source=train_cfg.init_source)
        del src_model, src_ckpt
        print(
            f"[kd-qat] initialized student base weights from {train_cfg.init_from_checkpoint} "
            f"(source={train_cfg.init_source})",
            flush=True,
        )

    student, report = convert_model_to_qat(student_base, qat_cfg)

    dataloader = build_train_dataloader(
        dataset=train_cfg.dataset,
        tokenizer=tokenizer,
        batch_size=train_cfg.batch_size,
        seqlen=train_cfg.seqlen,
        seed=train_cfg.seed,
        dataset_config=train_cfg.dataset_config,
        shuffle_buffer=train_cfg.shuffle_buffer,
    )
    eval_ids = load_eval_ids(tokenizer, "wikitext2", split="test")

    return run_kd_qat_training(
        student=student,
        teacher=teacher,
        report=report,
        qat_cfg=qat_cfg,
        train_cfg=train_cfg,
        dataloader=dataloader,
        tokenizer=tokenizer,
        eval_ids=eval_ids,
        device=device,
        write_final_harness_json=True,
    )
