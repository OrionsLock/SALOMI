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

| Method | Config | bpp (linear) | bpp (whole) | WikiText-2 PPL | × FP32 | Record |
|---|---|---:|---:|---:|---:|---|
| FP32 | — | 32.000 | 26.921 | 29.95 | 1.00 | `fp32_gpt2` |
| FP16 | — | 16.000 | 16.000 | 29.95 | 1.00 | `fp16_gpt2` |
| RTN INT8 | per-channel | 8.031 | 10.561 | 29.99 | 1.00 | `rtn_w8_perchan_gpt2` |
| **GPTQ INT4** | group 128 | 4.250 | 7.980 | 31.28 | 1.04 | `gptq_w4_g128_gpt2` |
| RTN INT4 | group 128 | 4.250 | 7.980 | 33.26 | 1.11 | `rtn_w4_g128_gpt2` |
| RTN INT4 | per-channel | 4.031 | 7.831 | 38.35 | 1.28 | `rtn_w4_perchan_gpt2` |
| **GPTQ INT3** | group 128 | 3.250 | 7.298 | 41.43 | 1.38 | `gptq_w3_g128_gpt2` |
| RTN INT3 | group 128 | 3.250 | 7.298 | 66.19 | 2.21 | `rtn_w3_g128_gpt2` |
| **GPTQ INT2** | group 128 | 2.250 | 6.615 | 689.11 | 23 | `gptq_w2_g128_gpt2` |
| RTN INT2 | group 128 | 2.250 | 6.615 | 3,017 | 101 | `rtn_w2_g128_gpt2` |
| Ternary TWN | per-channel, thr 0.7·mean | 1.616 | 6.182 | 34,466 | 1,151 | `ternary_twn_perchan_gpt2` |
| Ternary absmean (BitNet rule) | per-channel | 1.616 | 6.182 | 1,696,137 | 56,631 | `ternary_perchan_gpt2` |
| Legacy mixed (HVQ + low-rank) | L0/L11 protected | 1.164 | 5.874 | 17,266 | 576 | `legacy_mixed_perchan_gpt2` |
| Binary absmean | group 128 | 1.125 | 5.847 | 293,959 | 9,815 | `binary_g128_gpt2` |
| Legacy low-rank residual r=8 | INT8 factors | 1.111 | 5.838 | 15,444 | 516 | `legacy_lowrank_r8_perchan_gpt2` |
| Binary SVID (OneBit init, no training) | row + col vectors | 1.028 | 5.781 | 92,925 | 3,103 | `binary_svid_perchan_gpt2` |
| Binary absmean | per-channel | 1.016 | 5.773 | 190,302 | 6,354 | `binary_perchan_gpt2` |

Records: [`results/phase0/`](../results/phase0/). Regenerate this table with `python scripts/run_phase0_baselines.py --table`.

Provenance: each JSON stores the commit it ran from. Runs flagged `+dirty` in the `--table` output were made from the
working tree while this branch was being built; the code in `onebit/` and `scripts/` that produced them is identical to commit
`fdf4ad5`, except that `ternary_twn` and the table printer were added later (`f68e2a9`, `0ab2888`). In most cases the dirty flag
was caused only by untracked result files; the harness now ignores `results/` when computing it.

## Validated or defensible claims

### 1. Post-training quantization at ~1–1.6 bpp does not work on GPT-2 124M

| | bpp (linear) | PPL | × FP32 |
|---|---:|---:|---:|
| Best ≤1.6 bpp post-training result (legacy low-rank r=8) | 1.111 | 15,444 | 516 |
| Fair ternary (TWN) | 1.616 | 34,466 | 1,151 |
| OneBit SVID init, no training | 1.028 | 92,925 | 3,103 |
| Binary absmean, per-channel | 1.016 | 190,302 | 6,354 |

For reference, guessing uniformly over the 50,257-token vocabulary gives PPL 50,257. Every ≤1.6 bpp post-training model is
within ~3× of that or worse.

This matches the literature: even at 1.3B parameters, the best PTQ binarization (BiLLM) is several times worse than FP.

### 2. Correct low-bit baselines are much stronger than the repo previously reported

With correct group-wise scales and a real GPTQ, INT4 is close to lossless (GPTQ 31.28, +4.5%; RTN 33.26, +11%) and INT3
is usable with GPTQ (41.43, 1.38×), versus the earlier claims of INT4 = 8,812 and "usable quality needs ~8 bits". GPTQ beats
RTN at every width, and the gap widens as bits drop (INT2: 689 vs 3,017). These are the baselines any ~1-bit method should be
compared with **at equal bytes**: a 1-bit model of a larger network should be compared with an INT3/INT4 GPTQ model of the
same size in bytes, not with FP16.

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
| "Mixed precision improves PPL 17% (8,629 → 7,152)" | Withdrawn — both numbers came from the repeated-paragraph text. Under the harness, mixed (17,266) is *worse* than low-rank r=8 alone (15,444) |
| "MLP 77–200× more sensitive than attention" | Unverified — not re-measured with the harness |
| Proxy-SR-VQ Redun Score transfer / 70B scaling law | Withdrawn — score is dominated by magnitude CV; 2 model sizes, R² ≈ 0.01 |

## Best current one-sentence summary

SALOMI now has a trustworthy evaluation harness and correct baselines for GPT-2 124M; with them, every post-training method in
the repo fails at ~1 bpp, while a first quantization-aware distillation run already improves on all of them by more than an order
of magnitude — so the research continues as a training-based ~1-bit project.
