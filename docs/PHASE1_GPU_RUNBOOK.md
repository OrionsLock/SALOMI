# Phase 1 GPU Runbook: KD-QAT for Ternary & ~1-Bit GPT-2

This runbook documents the exact commands, hardware sizing, estimated wall-clock time/cost, and equal-byte evaluation workflow for running Phase 1 Knowledge-Distillation Quantization-Aware Training (KD-QAT) on a single rented GPU.

> [!NOTE]
> All throughput, VRAM, wall-clock, and rental cost numbers below are **engineering estimates** for planning GPU rentals. Every reported perplexity and bit-rate claim in SALOMI must come from a JSON file written by [`onebit/eval/harness.py`](../onebit/eval/harness.py) under `results/`.

---

## 1. Environment Setup (Single Rented GPU)

Tested stack: Ubuntu 22.04/24.04, Python 3.10–3.12, CUDA 12.1+, PyTorch 2.3+.

```bash
git clone https://github.com/OrionsLock/SALOMI.git
cd SALOMI
git checkout phase1-kd-qat

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Verify Phase 0 + Phase 1 unit tests before starting a paid GPU run
pytest tests/test_phase0_correctness.py tests/test_bpp_guard.py tests/test_bpp_strict.py tests/test_phase1_qat.py -q
```

---

## 2. Recommended Experimental Plan (3–5B Token Budget)

Follow the staged **Ternary → Binary** progression recommended by ParetoQ / BitNet / OneBit:

### Stage 1: Ternary KD-QAT (~1.62 bpp linear, 3B tokens)

**Variant 1A — FP16 embeddings (direct comparison to Phase 0 `bpp_linear` convention):**
- `bpp_linear` = 1.616, `bpp_whole_model` = 6.182 (96.2 MB whole / 17.2 MB linear).

```bash
python scripts/train_kd_qat.py \
    --model gpt2 \
    --mode ternary \
    --init-method twn \
    --embed-bits 0 \
    --freeze-embeddings \
    --dataset fineweb-edu \
    --dataset-config sample-10BT \
    --max-tokens 3000000000 \
    --seqlen 1024 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 3e-4 \
    --scale-lr 1e-3 \
    --warmup-steps 500 \
    --min-lr-ratio 0.1 \
    --kl-weight 1.0 \
    --ce-weight 0.0 \
    --dtype bf16 \
    --compile \
    --eval-every 1000 \
    --eval-windows 20 \
    --save-every 1000 \
    --run-name kdqat_ternary_twn_efp16_3B_gpt2
```

**Variant 1B — INT8 embeddings & tied LM head (realistic whole-model compression):**
- `bpp_linear` = 1.616, `bpp_whole_model` = 3.658 (56.9 MB whole / 17.2 MB linear).

```bash
python scripts/train_kd_qat.py \
    --model gpt2 \
    --mode ternary \
    --init-method twn \
    --embed-bits 8 \
    --dataset fineweb-edu \
    --dataset-config sample-10BT \
    --max-tokens 3000000000 \
    --seqlen 1024 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 3e-4 \
    --scale-lr 1e-3 \
    --other-lr 1e-4 \
    --warmup-steps 500 \
    --min-lr-ratio 0.1 \
    --kl-weight 1.0 \
    --ce-weight 0.0 \
    --dtype bf16 \
    --compile \
    --eval-every 1000 \
    --eval-windows 20 \
    --save-every 1000 \
    --run-name kdqat_ternary_twn_e8_3B_gpt2
```

### Stage 2: Binary OneBit KD-QAT (~1.03 bpp linear, 2B tokens from Stage 1)

Initializes student weights from the trained Stage 1 ternary checkpoint (`--init-from-checkpoint`), recomputes SVID scales (`diag(a) · sign(W) · diag(b)`), and distills against the **original pretrained FP GPT-2 teacher**.

```bash
python scripts/train_kd_qat.py \
    --model gpt2 \
    --mode binary \
    --embed-bits 8 \
    --init-from-checkpoint checkpoints/phase1/kdqat_ternary_twn_e8_3B_gpt2/best.pt \
    --init-source latent \
    --dataset fineweb-edu \
    --dataset-config sample-10BT \
    --max-tokens 2000000000 \
    --seqlen 1024 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 2e-4 \
    --scale-lr 1e-3 \
    --other-lr 1e-4 \
    --warmup-steps 400 \
    --min-lr-ratio 0.1 \
    --kl-weight 1.0 \
    --ce-weight 0.0 \
    --dtype bf16 \
    --compile \
    --eval-every 1000 \
    --eval-windows 20 \
    --save-every 1000 \
    --run-name kdqat_binary_onebit_e8_staged5B_gpt2
```

> [!TIP]
> To ablate **ParetoQ shift** (`W_q = diag(a)·sign(W)·diag(b) + shift`, 1.041 bpp linear) or **hidden-state MSE distillation**, add `--use-shift` and/or `--hidden-mse-weight 0.1`.
> To resume an interrupted run on a spot/preemptible GPU, re-run the exact same command with `--resume checkpoints/phase1/<run_name>/latest.pt`.

### Stage 3 (Equal-Byte Hero Experiment): `gpt2-medium` (355M) at ~1-bit vs `gpt2` (124M) GPTQ INT4

Because `gpt2` (124M) at GPTQ INT4 g128 occupies **124.1 MB** total (45.1 MB linear), a **355M `gpt2-medium`** model at Binary OneBit + INT8 embeddings fits in **91.8 MB** total (38.5 MB linear) — **26% smaller in total bytes** than 124M INT4 while having 3.5× as many transformer parameters! (Even `gpt2-medium` Ternary + INT8 embeddings fits in **114.1 MB**, 8% smaller than 124M INT4.)

First record the Phase 0 FP16 and GPTQ baselines for `gpt2-medium` (optional reference):
```bash
python -m onebit.eval.harness --method fp16 --model gpt2-medium --out-dir results/phase1
python -m onebit.eval.harness --method gptq --bits 4 --group 128 --model gpt2-medium --out-dir results/phase1
```

Then run Binary OneBit KD-QAT on `gpt2-medium` (use `--batch-size 8 --grad-accum 8` on 24 GB GPUs):
```bash
python scripts/train_kd_qat.py \
    --model gpt2-medium \
    --mode binary \
    --embed-bits 8 \
    --dataset fineweb-edu \
    --dataset-config sample-10BT \
    --max-tokens 3000000000 \
    --seqlen 1024 \
    --batch-size 8 \
    --grad-accum 8 \
    --lr 2e-4 \
    --scale-lr 1e-3 \
    --other-lr 1e-4 \
    --warmup-steps 500 \
    --dtype bf16 \
    --compile \
    --eval-every 1000 \
    --eval-windows 20 \
    --save-every 1000 \
    --run-name kdqat_binary_onebit_e8_3B_gpt2-medium
```

---

## 3. Hardware, Throughput & Cost Estimates (Engineering Planning Only)

For GPT-2 124M KD-QAT (frozen `bf16` teacher forward + `bf16` student forward/backward + AdamW on ~85M–124M params, `seqlen=1024`, `--compile`):
- Effective batch size: `16 × 1024 × 4 = 65,536` tokens/step → 3B tokens ≈ 45,776 steps; 5B tokens ≈ 76,294 steps.
- Peak VRAM (`gpt2` 124M, `batch_size=16`, `seqlen=1024`, `bf16`): ~10–14 GB (comfortably fits on a 24 GB RTX 4090; increase `--batch-size 32 --grad-accum 2` on 80 GB A100/H100).
- Peak VRAM (`gpt2-medium` 355M, `batch_size=8`, `seqlen=1024`, `bf16`): ~16–20 GB (fits on 24 GB RTX 4090; use `--batch-size 16 --grad-accum 4` on A100/H100).

| GPU | Typical On-Demand Rate | Est. Throughput (`gpt2` 124M KD-QAT, `bf16` + compile) | Est. Time (3B / 5B tok) | Est. Cost (3B / 5B tok) |
|---|---:|---:|---:|---:|
| 1× RTX 4090 (24 GB) | ~\$0.40–\$0.65 / hr | ~55k–85k tok/s | ~10–15 hr / ~16–25 hr | ~\$5–\$10 / ~\$8–\$16 |
| 1× A100 SXM (80 GB) | ~\$1.30–\$1.90 / hr | ~90k–140k tok/s | ~6–9 hr / ~10–15 hr | ~\$8–\$17 / ~\$14–\$28 |
| 1× H100 SXM (80 GB) | ~\$2.50–\$3.50 / hr | ~180k–260k tok/s | ~3–5 hr / ~5–8 hr | ~\$8–\$17 / ~\$14–\$28 |

*Notes:*
- Teacher forward pass adds ~25–30% overhead compared to standard CE-only GPT-2 training, plus ~10–15% for STE fake-quantization (`--compile` fuses most of the STE elementwise ops).
- Streaming `fineweb-edu` over a fast datacenter link has negligible bottleneck at `shuffle_buffer=10000`; set `HF_TOKEN` if you hit Hub rate limits.
- Always run a 50-step benchmark first (`--max-steps 50 --eval-windows 2 --final-eval-windows 2`) to measure exact `tok/s` on your rented box before committing to a multi-hour run.

---

## 4. Equal-Byte Comparison Workflow

After any Phase 1 run finishes, its canonical full-test-set WikiText-2 result is automatically written to `results/phase1/<run_name>.json`. Compare all Phase 0 and Phase 1 runs at **equal total model bytes** and **equal linear bytes**:

```bash
# Sort by whole-model bytes (includes embeddings/LN/biases) + show GPT-2 family equal-byte matchups
python scripts/compare_equal_bytes.py

# Sort by transformer-block linear bytes only
python scripts/compare_equal_bytes.py --sort linear
```

### Phase 1 Success Criteria (from `CURRENT_STATE.md` & `docs/PHASE0_AUDIT.md`)
1. **Near-1-bit claim**: `bpp_linear ≤ 1.1` (with `bpp_whole_model` explicitly stated), full WikiText-2 test PPL within ~2× of FP (`≤ ~60` on GPT-2 124M), backed by a committed script and canonical harness JSON in `results/phase1/`.
2. **Equal-byte claim**: Beat or match Phase 0 **GPTQ INT4** (`31.28` PPL @ 124.1 MB whole / 45.1 MB linear) or **GPTQ INT3** (`41.43` PPL @ 113.5 MB whole / 34.5 MB linear) at equal or fewer total bytes (e.g., via `gpt2-medium` ternary/binary with INT8 embeddings).
3. **Data hygiene**: All training tokens come from a general corpus (`fineweb-edu` or `openwebtext`); WikiText-2 `test` is used strictly for evaluation via `onebit/eval/harness.py`.
