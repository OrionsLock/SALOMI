#!/usr/bin/env python3
"""Phase 1 CLI runner for GPU-ready KD-QAT (~1-bit binary and ternary GPT-2).

Examples:
    # Stage 1: Ternary KD-QAT on FineWeb-Edu (3B tokens, single GPU, bf16)
    python scripts/train_kd_qat.py \\
        --model gpt2 --mode ternary --init-method twn --embed-bits 8 \\
        --dataset fineweb-edu --max-tokens 3000000000 \\
        --seqlen 1024 --batch-size 16 --grad-accum 4 --dtype bf16 --compile

    # Stage 2: Binary OneBit KD-QAT initialized from Stage 1 ternary checkpoint
    python scripts/train_kd_qat.py \\
        --model gpt2 --mode binary --embed-bits 8 \\
        --init-from-checkpoint checkpoints/phase1/<ternary_run>/best.pt \\
        --dataset fineweb-edu --max-tokens 2000000000 \\
        --seqlen 1024 --batch-size 16 --grad-accum 4 --dtype bf16 --compile

    # Fast CPU smoke test (WikiText-2 train only; eval on first 2 test windows)
    python scripts/train_kd_qat.py \\
        --model gpt2 --mode ternary --dataset wikitext2-train \\
        --max-steps 20 --seqlen 128 --batch-size 2 --grad-accum 1 \\
        --eval-every 10 --eval-windows 2 --final-eval-windows 2 --dtype fp32
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onebit.eval.harness import REPO_ROOT  # noqa: E402
from onebit.qat.convert import QATConfig  # noqa: E402
from onebit.qat.trainer import TrainConfig, train_from_pretrained  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model & quantization
    p.add_argument("--model", default="gpt2", help="HF causal LM id (e.g., gpt2, gpt2-medium)")
    p.add_argument("--mode", default="ternary", choices=["ternary", "binary", "binary_paretoq"])
    p.add_argument("--group-size", type=int, default=0, help="<=0 means per-output-channel")
    p.add_argument("--init-method", default="twn", choices=["twn", "absmean", "svid"])
    p.add_argument("--no-learnable-scale", action="store_true", help="use dynamic absmean scales (ternary only)")
    p.add_argument("--use-shift", action="store_true", help="enable learnable per-channel/group shift (ParetoQ)")
    p.add_argument("--lsq-scale-grad", action="store_true", help="apply LSQ 1/sqrt(N) gradient scaling to scales")
    p.add_argument("--grad-clip-ste", type=float, default=None, help="optional |w|<=c clip on binary SignSTE")
    p.add_argument("--scale-only", action="store_true", help="freeze latent weights; train scales + LN/biases only")

    # Embeddings
    p.add_argument("--embed-bits", type=int, default=0, choices=[0, 4, 8],
                   help="0 = keep embeddings/lm_head FP16; 4 or 8 = fake-quantize embeddings & tied lm_head with STE")
    p.add_argument("--embed-asymmetric", action="store_true", help="asymmetric embedding quant (scale + zero-point)")
    p.add_argument("--embed-group-size", type=int, default=0, help="<=0 means per-row (per-token) embedding scales")
    p.add_argument("--keep-pos-embed-fp", action="store_true", help="keep wpe FP16 even when --embed-bits > 0")
    p.add_argument("--freeze-embeddings", action="store_true", help="freeze embedding parameters during training")

    # Data
    p.add_argument("--dataset", default="fineweb-edu",
                   choices=["fineweb-edu", "openwebtext", "wikitext2-train", "synthetic"])
    p.add_argument("--dataset-config", default=None, help="HF dataset config (default sample-10BT for fineweb-edu)")
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)

    # Budget
    p.add_argument("--max-tokens", type=int, default=3_000_000_000)
    p.add_argument("--max-steps", type=int, default=0, help=">0 overrides --max-tokens")

    # Optimizer & schedule
    p.add_argument("--lr", type=float, default=3e-4, help="LR for latent weights")
    p.add_argument("--scale-lr", type=float, default=1e-3, help="LR for quantizer scale/shift parameters")
    p.add_argument("--other-lr", type=float, default=None, help="LR for LayerNorms/biases/embeddings (default: --lr)")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scale-weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Distillation
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--hidden-mse-weight", type=float, default=0.0)

    # Runtime & precision
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--compile", action="store_true", help="enable torch.compile on student model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=None)

    # Eval, logging & checkpoints
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-windows", type=int, default=20, help="mid-run WikiText-2 test windows (0 = full)")
    p.add_argument("--final-eval-windows", type=int, default=0,
                   help="final WikiText-2 test windows (0 = full test set; use >0 for smoke tests only)")
    p.add_argument("--eval-seqlen", type=int, default=1024)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--run-name", default=None)
    p.add_argument("--work-dir", default=os.path.join(REPO_ROOT, "checkpoints", "phase1"))
    p.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results", "phase1"))
    p.add_argument("--resume", default=None, help="path to checkpoint to resume training from")
    p.add_argument("--init-from-checkpoint", default=None,
                   help="path to prior QAT checkpoint (e.g., ternary) to initialize a new stage (e.g., binary)")
    p.add_argument("--init-source", default="latent", choices=["latent", "dequantized"])

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    a = parse_args(argv)
    qat_cfg = QATConfig(
        mode=a.mode,
        group_size=a.group_size if a.group_size > 0 else None,
        init_method=a.init_method,
        learnable_scale=not a.no_learnable_scale,
        use_shift=a.use_shift,
        lsq_scale_grad=a.lsq_scale_grad,
        grad_clip=a.grad_clip_ste,
        train_signs=not a.scale_only,
        embed_bits=a.embed_bits,
        embed_symmetric=not a.embed_asymmetric,
        embed_group_size=a.embed_group_size if a.embed_group_size > 0 else None,
        quantize_pos_embeddings=not a.keep_pos_embed_fp,
        freeze_embeddings=a.freeze_embeddings,
    )
    train_cfg = TrainConfig(
        model=a.model,
        dataset=a.dataset,
        dataset_config=a.dataset_config,
        shuffle_buffer=a.shuffle_buffer,
        seqlen=a.seqlen,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        max_tokens=a.max_tokens,
        max_steps=a.max_steps,
        lr=a.lr,
        scale_lr=a.scale_lr,
        other_lr=a.other_lr,
        weight_decay=a.weight_decay,
        scale_weight_decay=a.scale_weight_decay,
        warmup_steps=a.warmup_steps,
        warmup_ratio=a.warmup_ratio,
        min_lr_ratio=a.min_lr_ratio,
        grad_clip=a.grad_clip,
        kl_weight=a.kl_weight,
        ce_weight=a.ce_weight,
        temperature=a.temperature,
        hidden_mse_weight=a.hidden_mse_weight,
        dtype=a.dtype,
        compile_model=a.compile,
        seed=a.seed,
        num_threads=a.num_threads,
        eval_every=a.eval_every,
        eval_windows=a.eval_windows if a.eval_windows > 0 else None,
        final_eval_windows=a.final_eval_windows if a.final_eval_windows > 0 else None,
        eval_seqlen=a.eval_seqlen,
        log_every=a.log_every,
        save_every=a.save_every,
        run_name=a.run_name,
        work_dir=a.work_dir,
        results_dir=a.results_dir,
        resume=a.resume,
        init_from_checkpoint=a.init_from_checkpoint,
        init_source=a.init_source,
    )
    train_from_pretrained(qat_cfg, train_cfg)


if __name__ == "__main__":
    main()
