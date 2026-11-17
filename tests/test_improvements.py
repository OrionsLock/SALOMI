#!/usr/bin/env python3
"""Comprehensive test of all SALOMI quantization improvements.

Runs every method on real GPT-2 weights, reports per-layer correlation
and effective BPP.  Also runs end-to-end perplexity for key configs.

Phase-0 changes:
  * PART 1: the Hessian diagonal comes from WikiText-2 *train* calibration
    windows (``harness.load_calibration``); X_test comes from held-out
    WikiText-2 *validation* tokens (previously the same 8 sentences were used
    for both H and X_test).
  * PART 2: perplexity is computed with ``harness.perplexity`` on WikiText-2
    test (previously a paragraph repeated 20x), quantization touches block
    linears only (harness convention), and every quantizer receives the real
    calibration Hessian diagonal (previously ``H_diag = np.ones``).
    ``--max-windows`` (default 40 windows of 1024 tokens) keeps it CPU-friendly;
    use ``--max-windows 0`` for the full test set.
  * The "HVQ + GPTQ" entry was removed: ``HessianVQ(gptq_refine=True)`` was not
    GPTQ and now raises; see ``onebit/quantization/gptq.py``.

Usage:
    python tests/test_improvements.py [--max-windows 40] [--calib-samples 128]
"""

import argparse
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Load GPT-2 once
# ---------------------------------------------------------------------------

def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("Loading GPT-2 …", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("  Model + tokenizer ready.", flush=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Held-out activations for ONE submodule (lightweight)
# ---------------------------------------------------------------------------

def collect_inputs(model, submodule, windows):
    """Run ``windows`` (n, seqlen) through the model and return the inputs to ``submodule``."""
    import torch
    acts = []

    def hook(mod, inp, out):
        acts.append(inp[0].detach().cpu().float().numpy().reshape(-1, inp[0].shape[-1]))

    h = submodule.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(windows.shape[0]):
                model(windows[i: i + 1])
    finally:
        h.remove()
    return np.concatenate(acts, axis=0)


def correlation(Y_ref, Y_test):
    a, b = Y_ref.flatten(), Y_test.flatten()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Quantization methods
# ---------------------------------------------------------------------------

def binary_baseline(W, _H):
    S = np.sign(W); S[S == 0] = 1.0
    return S * np.mean(np.abs(W)), 1.0 + 32.0 / W.size  # + one fp32 scale

def ternary_baseline(W, _H):
    S = np.sign(W); S[S == 0] = 1.0
    thresh = np.percentile(np.abs(W), 30)
    mask = np.abs(W) > thresh
    scale = np.mean(np.abs(W[mask])) if mask.any() else 1.0
    return S * mask * scale, np.log2(3) + 32.0 / W.size  # ideal ternary + one fp32 scale

def old_hvq(W, H):
    from onebit.quantization.hessian_vq import HessianVQ
    vq = HessianVQ(n_codes=32, block_size=4, max_iter=5, use_hessian_weight=False)
    return vq.quantize(W, H), vq.effective_bpp(W.size)

def new_hvq_k64(W, H):
    from onebit.quantization.hessian_vq import HessianVQ
    vq = HessianVQ(n_codes=64, block_size=4, max_iter=15, use_hessian_weight=True)
    return vq.quantize(W, H), vq.effective_bpp(W.size)

def lr_r4_fp32(W, H):
    from onebit.quantization.lowrank_residual import LowRankResidual
    lr = LowRankResidual(rank=4, factor_bits=32)
    return lr.quantize(W, H), lr.effective_bpp()

def lr_r8_int8(W, H):
    from onebit.quantization.lowrank_residual import LowRankResidual
    lr = LowRankResidual(rank=8, factor_bits=8)
    return lr.quantize(W, H), lr.effective_bpp()

def lr_r12_int8(W, H):
    from onebit.quantization.lowrank_residual import LowRankResidual
    lr = LowRankResidual(rank=12, factor_bits=8)
    return lr.quantize(W, H), lr.effective_bpp()

def two_stage(W, H):
    from onebit.quantization.lowrank_residual import ResidualHessianVQ
    rvq = ResidualHessianVQ(n_codes_coarse=64, n_codes_fine=32, block_size=4, max_iter=10)
    return rvq.quantize(W, H), rvq.effective_bpp()


METHODS = [
    ("Binary (sign*scale)          ", binary_baseline),
    ("Ternary (30% sparse)         ", ternary_baseline),
    ("OLD HVQ  K=32 unwt 5it       ", old_hvq),
    ("NEW HVQ  K=64 wt   15it      ", new_hvq_k64),
    ("LowRank r=4  FP32  (old)     ", lr_r4_fp32),
    ("LowRank r=8  INT8  (new)     ", lr_r8_int8),
    ("LowRank r=12 INT8  (new)     ", lr_r12_int8),
    ("Two-Stage VQ 64+32           ", two_stage),
]


# ---------------------------------------------------------------------------
# PART 2 quantizers (harness Quantizer signature: (w_out_in, ctx) -> (w_q, bits))
# ---------------------------------------------------------------------------

def _harness_binary(w, ctx):
    import torch
    W = w.detach().cpu().numpy().astype(np.float64)
    W_q, bpp = binary_baseline(W, None)
    return torch.tensor(W_q, dtype=torch.float32), bpp * W.size


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-windows", type=int, default=40,
                   help="1024-token WikiText-2 test windows for PART 2 (0 = full test set)")
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--test-windows", type=int, default=4,
                   help="held-out validation windows (of --calib-seqlen tokens) for PART 1 X_test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-part1", action="store_true")
    p.add_argument("--skip-part2", action="store_true")
    p.add_argument("--out", default=None, help="optional path for a PART 1 summary text file")
    args = p.parse_args(argv)
    max_windows = args.max_windows if args.max_windows and args.max_windows > 0 else None

    import torch, copy
    from onebit.eval import harness
    from onebit.quantization import legacy_adapters as L

    print("=" * 90)
    print("SALOMI QUANTIZATION IMPROVEMENTS — COMPREHENSIVE TEST")
    print("=" * 90, flush=True)

    model, tokenizer = load_gpt2()

    # Calibration Hessian diagonal (train split) for every block linear, collected once.
    print(f"Collecting calibration H_diag ({args.calib_samples}x{args.calib_seqlen} train tokens) …", flush=True)
    calib = harness.load_calibration(tokenizer, args.calib_samples, args.calib_seqlen, args.seed)
    h_diag = harness.collect_input_diag(model, calib)

    all_results = {name: {"corrs": [], "bpps": []} for name, _ in METHODS}

    # ------------------------------------------------------------------
    # PART 1: Per-layer output correlation on MLP c_fc (hardest case)
    # ------------------------------------------------------------------
    if not args.skip_part1:
        print("\n" + "=" * 90)
        print("PART 1: LAYER-WISE OUTPUT CORRELATION  (MLP c_fc — hardest component)")
        print("=" * 90)

        val_ids = harness.load_eval_ids(tokenizer, "wikitext2", "validation")
        sl = args.calib_seqlen
        n_test = min(args.test_windows, val_ids.numel() // sl)
        test_windows = val_ids[: n_test * sl].reshape(n_test, sl)

        test_layers = [0, 5, 11]
        for layer_idx in test_layers:
            sub = model.transformer.h[layer_idx].mlp.c_fc
            W = harness.get_weight(sub).numpy().astype(np.float64)
            d_out, d_in = W.shape
            H_diag = h_diag[f"transformer.h.{layer_idx}.mlp.c_fc"].numpy().astype(np.float32)

            print(f"\n  Layer {layer_idx}  ({d_out}x{d_in})  collecting held-out inputs …", end="", flush=True)
            X_test = collect_inputs(model, sub, test_windows).astype(np.float64)
            Y_ref = X_test @ W.T
            print(f"  H shape {H_diag.shape}, X_test {X_test.shape}", flush=True)

            for name, fn in METHODS:
                t0 = time.time()
                W_q, bpp = fn(W, H_diag)
                elapsed = time.time() - t0
                Y_q = X_test @ W_q.T
                corr = correlation(Y_ref, Y_q)
                all_results[name]["corrs"].append(corr)
                all_results[name]["bpps"].append(bpp)
                print(f"    {name.strip():<30} corr={corr:.5f}  bpp={bpp:.3f}  ({elapsed:.1f}s)", flush=True)

        # Summary table
        print("\n" + "-" * 90)
        header = f"{'Method':<35} {'Avg Corr':>9} {'Min Corr':>9} {'BPP':>7} {'vs OldHVQ':>10}"
        print(header)
        print("-" * 90)

        old_avg = np.mean(all_results["OLD HVQ  K=32 unwt 5it       "]["corrs"])

        for name, _ in METHODS:
            corrs = all_results[name]["corrs"]
            bpps = all_results[name]["bpps"]
            avg_c = np.mean(corrs)
            min_c = np.min(corrs)
            avg_bpp = np.mean(bpps)
            delta = ((avg_c - old_avg) / (abs(old_avg) + 1e-10)) * 100
            marker = " ***" if delta > 1 else ""
            print(f"{name:<35} {avg_c:>9.5f} {min_c:>9.5f} {avg_bpp:>7.2f} {delta:>+9.1f}%{marker}")

    # ------------------------------------------------------------------
    # PART 2: End-to-end perplexity (all transformer-block linears)
    # ------------------------------------------------------------------
    if not args.skip_part2:
        print("\n" + "=" * 90)
        scope = "full test set" if max_windows is None else f"first {max_windows} x 1024-token windows, NOT reportable"
        print(f"PART 2: END-TO-END PERPLEXITY  (WikiText-2 test, {scope})")
        print("        block linears quantized; embeddings / LN / biases / tied LM head kept FP")
        print("=" * 90, flush=True)

        ids = harness.load_eval_ids(tokenizer, "wikitext2", "test")
        fp = harness.perplexity(model, ids, seqlen=1024, max_windows=max_windows)
        fp_ppl = fp["ppl"]
        print(f"FP32 Baseline:  PPL = {fp_ppl:.2f}  ({fp['tokens']} tokens)\n", flush=True)

        ppl_configs = [
            ("Binary (sign*scale)", _harness_binary),
            ("NEW HVQ K=64", L.get("legacy_hvq_k64")),
            ("LowRank r=8 INT8", L.get("legacy_lowrank_r8")),
            ("Mixed (L0/11 prot.)", L.get("legacy_mixed")),
        ]

        for config_name, quantizer in ppl_configs:
            print(f"  Quantizing: {config_name} … ", end="", flush=True)
            t0 = time.time()
            qmodel = copy.deepcopy(model)
            report = harness.quantize_model_rtn_style(qmodel, quantizer, {"h_diag": h_diag, "seed": args.seed})
            ppl = harness.perplexity(qmodel, ids, seqlen=1024, max_windows=max_windows)["ppl"]
            ratio = ppl / fp_ppl
            elapsed = time.time() - t0
            print(f"PPL={ppl:>12.2f}  ratio={ratio:>8.1f}x  bpp_linear={report.bpp_linear:.3f}  "
                  f"bpp_whole={report.bpp_whole_model:.3f}  ({elapsed:.0f}s)", flush=True)
            del qmodel

    # ------------------------------------------------------------------
    # Save results (optional)
    # ------------------------------------------------------------------
    if args.out and not args.skip_part1:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("SALOMI Quantization Improvements — Test Results (PART 1)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Method':<35} {'Avg Corr':>9} {'Min Corr':>9} {'BPP':>7}\n")
            f.write("-" * 70 + "\n")
            for name, _ in METHODS:
                avg_c = np.mean(all_results[name]["corrs"])
                min_c = np.min(all_results[name]["corrs"])
                avg_bpp = np.mean(all_results[name]["bpps"])
                f.write(f"{name:<35} {avg_c:>9.5f} {min_c:>9.5f} {avg_bpp:>7.2f}\n")
        print(f"\nResults saved to {args.out}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
