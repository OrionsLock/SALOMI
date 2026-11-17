# Validated Results

This is the shortest summary of the claims in SALOMI that are defensible from the current repository state.
**Every number here comes from a JSON record written by [`onebit/eval/harness.py`](../onebit/eval/harness.py)**; nothing is
copied from older logs. For what was retracted and why, see [`PHASE0_AUDIT.md`](PHASE0_AUDIT.md).

## Protocol

| | |
|---|---|
| Model | GPT-2 124M (`gpt2` on the Hugging Face Hub) |
| Eval data | WikiText-2-raw-v1 **test** split, joined with `"\n\n"`, tokenized once (287,644 tokens) |
| Windows | 1024 tokens, non-overlapping, token-weighted mean NLL → PPL (full test set unless the run name ends in `_winN`) |
| Calibration (GPTQ, legacy) | 128 random 512-token windows from the WikiText-2 **train** split, seed 0 (`RunConfig` defaults) |
| Quantized tensors | All linear layers inside the 12 transformer blocks (attn `c_attn`, `c_proj`; MLP `c_fc`, `c_proj`). Embeddings, LM head (tied), LayerNorm and biases stay in FP |
| `bpp_linear` | Bits of the quantized block weights **including** scales, zero-points, codebooks and factors (FP metadata at 16 bits), divided by their parameter count |
| `bpp_whole` | All model bits (non-quantized parameters at 16 bits) divided by all parameters |

## Canonical results (Phase 0)

<!-- PHASE0_TABLE -->

Records: [`results/phase0/`](../results/phase0/). Regenerate this table with `python scripts/run_phase0_baselines.py --table`.

<!-- PHASE0_PROVENANCE -->

## Validated or defensible claims

### 1. Post-training quantization at ~1–1.6 bpp does not work on GPT-2 124M

<!-- CLAIM1 -->

This matches the literature: even at 1.3B parameters, the best PTQ binarization (BiLLM) is several times worse than FP.

### 2. Correct low-bit baselines are much stronger than the repo previously reported

<!-- CLAIM2 -->

### 3. Correlation and perplexity are not interchangeable

Layer-wise output correlation of ~0.9 (binary, HVQ, low-rank) coexists with end-to-end PPL in the thousands. Two-Stage VQ
reached 0.98 average correlation and was still unusable end to end. Correlation and MSE are diagnostics only.

### 4. Early sub-1-bit claims were wrong under strict accounting

The 0.58-bpp DualPath and other sub-1-bit figures omitted sign bits, codebooks, indices or metadata
(`onebit/research/RIGOROUS_TEST_RESULTS.md`). In addition, the Two-Stage VQ bpp in `lowrank_residual.py` was understated by
exactly 1.0 bpp (now fixed), and embeddings were silently excluded from earlier whole-model bpp.

### 5. Training is the viable route to ~1 bit (preliminary)

A OneBit-style KD-QAT proof-of-concept (`diag(a)·sign(W)·diag(b)`, 1.03 bpp linear) trained on CPU for 40 minutes
(298k WikiText-2 train tokens, single seed) reduced PPL from 81,759 to **209.5** on the first 20 test windows.
This is preliminary: partial test set, tiny token budget, no tuning. Record:
[`results/experiments/onebit_kd_poc_cpu40min_preliminary.json`](../results/experiments/onebit_kd_poc_cpu40min_preliminary.json).

## Claims withdrawn

| Former claim | Status |
|---|---|
| FP32 GPT-2 PPL = 5.92; "1,208× / 1,458× / 158,000× FP32" ratios | Withdrawn — eval text was one paragraph repeated; FP32 on WikiText-2 is 29.95 |
| INT4 PPL = 8,812 / 16,881; INT2 = 7,913 | Withdrawn — per-tensor absmax quantizer that also quantized embeddings |
| "Hessian-weighted VQ +1.6% correlation" | Withdrawn — the weighting was a no-op, and the end-to-end runs passed `H = ones` |
| "HVQ + GPTQ refinement +2.0%" | Withdrawn — the refinement was not GPTQ (fixed noise); now raises `NotImplementedError` |
| "Mixed precision improves PPL 17% (8,629 → 7,152)" | Withdrawn as stated — both numbers were on the repeated-paragraph text; see the legacy rows above for the harness numbers |
| "MLP 77–200× more sensitive than attention" | Unverified — not re-measured with the harness |
| Proxy-SR-VQ Redun Score transfer / 70B scaling law | Withdrawn — score is dominated by magnitude CV; 2 model sizes, R² ≈ 0.01 |

## Best current one-sentence summary

SALOMI now has a trustworthy evaluation harness and correct baselines for GPT-2 124M; with them, every post-training method in
the repo fails at ~1 bpp, while a first quantization-aware distillation run already improves on all of them by more than an order
of magnitude — so the research continues as a training-based ~1-bit project.
