#!/usr/bin/env python3
"""Comprehensive test of all SALOMI quantization improvements.

Runs every method on real GPT-2 weights, reports per-layer correlation
and effective BPP.  Also runs end-to-end perplexity for key configs.

Usage:
    python tests/test_improvements.py
"""

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
# Collect Hessian diagonal for ONE submodule (lightweight)
# ---------------------------------------------------------------------------

def get_hessian_diag(model, tokenizer, submodule, n_samples=16):
    """Run n_samples forward passes, return mean(X**2, axis=0)."""
    import torch
    acts = []

    def hook(mod, inp, out):
        acts.append(inp[0].detach().cpu().numpy().reshape(-1, inp[0].shape[-1]))

    h = submodule.register_forward_hook(hook)

    calib = [
        "Robert Boyle was an Anglo-Irish natural philosopher, chemist, physicist, and inventor.",
        "The transformer architecture was introduced in Attention Is All You Need in 2017.",
        "Quantization reduces model size by representing weights with fewer bits.",
        "Deep learning models typically use 32-bit floating point for parameters.",
        "The softmax function is exponentially sensitive to small logit changes.",
        "Matrix multiplication is the core operation in transformer inference.",
        "The Hessian diagonal measures loss sensitivity to each weight perturbation.",
        "Residual connections allow the original signal to pass through unchanged.",
        "Vector quantization groups weights and uses a shared codebook entry.",
        "GELU activation has high curvature near zero amplifying quantization errors.",
        "Cross-validation prevents overfitting by evaluating on held-out data.",
        "Mixed-precision assigns different bit-widths based on layer sensitivity.",
        "Low-rank factorization decomposes weights into two smaller matrices.",
        "Entropy coding exploits non-uniform distributions for better compression.",
        "Perplexity measures how well a model predicts a held-out test set.",
        "K-means++ initialization improves convergence by choosing diverse centroids.",
    ]

    with torch.no_grad():
        for text in calib[:n_samples]:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            model(**ids)

    h.remove()

    if acts:
        X = np.concatenate(acts, axis=0)
        return np.mean(X ** 2, axis=0).astype(np.float32), X
    return np.ones(768, dtype=np.float32), np.random.randn(16, 768).astype(np.float32)


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
    return S * np.mean(np.abs(W)), 1.0

def ternary_baseline(W, _H):
    S = np.sign(W); S[S == 0] = 1.0
    thresh = np.percentile(np.abs(W), 30)
    mask = np.abs(W) > thresh
    scale = np.mean(np.abs(W[mask])) if mask.any() else 1.0
    return S * mask * scale, 1.58

def old_hvq(W, H):
    from onebit.quantization.hessian_vq import HessianVQ
    vq = HessianVQ(n_codes=32, block_size=4, max_iter=5, use_hessian_weight=False)
    return vq.quantize(W, H), vq.effective_bpp(W.size)

def new_hvq_k64(W, H):
    from onebit.quantization.hessian_vq import HessianVQ
    vq = HessianVQ(n_codes=64, block_size=4, max_iter=15, use_hessian_weight=True)
    return vq.quantize(W, H), vq.effective_bpp(W.size)

def new_hvq_k64_gptq(W, H):
    from onebit.quantization.hessian_vq import HessianVQ
    vq = HessianVQ(n_codes=64, block_size=4, max_iter=15, use_hessian_weight=True, gptq_refine=True)
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
    ("NEW HVQ  K=64 wt   25it      ", new_hvq_k64),
    ("NEW HVQ  K=64 wt+GPTQ        ", new_hvq_k64_gptq),
    ("LowRank r=4  FP32  (old)     ", lr_r4_fp32),
    ("LowRank r=8  INT8  (new)     ", lr_r8_int8),
    ("LowRank r=12 INT8  (new)     ", lr_r12_int8),
    ("Two-Stage VQ 64+32           ", two_stage),
]


# ---------------------------------------------------------------------------
# Perplexity evaluation (no download needed)
# ---------------------------------------------------------------------------

def evaluate_perplexity(model, tokenizer, max_tokens=2048):
    import torch, torch.nn.functional as F

    eval_text = (
        "Robert Boyle was an Anglo-Irish natural philosopher, chemist, physicist, "
        "and inventor. He is largely regarded as the first modern chemist. "
        "The transformer architecture was introduced in 2017. It relies entirely "
        "on self-attention mechanisms. Machine learning quantization reduces the "
        "numerical precision of model parameters. Neural network pruning removes "
        "unnecessary parameters from a trained model. Large language models have "
        "demonstrated remarkable capabilities in text generation, translation, "
        "summarization, and question answering. The scaling laws suggest that "
        "performance improves predictably with model size. "
    ) * 20

    enc = tokenizer(eval_text, return_tensors="pt", truncation=True, max_length=max_tokens + 1)
    input_ids = enc["input_ids"][0][:max_tokens + 1]

    total_loss = 0.0
    total_tokens = 0
    seq_len = 256

    with torch.no_grad():
        for start in range(0, len(input_ids) - 1, seq_len):
            end = min(start + seq_len + 1, len(input_ids))
            chunk = input_ids[start:end].unsqueeze(0)
            logits = model(chunk).logits[:, :-1, :].contiguous()
            labels = chunk[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.numel()

    return float(np.exp(total_loss / total_tokens)), total_tokens


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("SALOMI QUANTIZATION IMPROVEMENTS — COMPREHENSIVE TEST")
    print("=" * 90, flush=True)

    model, tokenizer = load_gpt2()

    # ------------------------------------------------------------------
    # PART 1: Per-layer output correlation on MLP c_fc (hardest case)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PART 1: LAYER-WISE OUTPUT CORRELATION  (MLP c_fc — hardest component)")
    print("=" * 90)

    all_results = {name: {"corrs": [], "bpps": []} for name, _ in METHODS}

    test_layers = [0, 5, 11]
    for layer_idx in test_layers:
        W = model.transformer.h[layer_idx].mlp.c_fc.weight.detach().cpu().numpy().T
        d_out, d_in = W.shape

        print(f"\n  Layer {layer_idx}  ({d_out}x{d_in})  collecting Hessian …", end="", flush=True)
        H_diag, X_calib = get_hessian_diag(model, tokenizer,
                                           model.transformer.h[layer_idx].mlp.c_fc, n_samples=8)
        X_test = X_calib[:min(200, len(X_calib))]
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
    # PART 2: End-to-end perplexity (quantize everything)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PART 2: END-TO-END PERPLEXITY  (all 12 layers, all weights)")
    print("=" * 90, flush=True)

    import torch, copy

    fp_ppl, n_tok = evaluate_perplexity(model, tokenizer, max_tokens=1024)
    print(f"FP32 Baseline:  PPL = {fp_ppl:.2f}  ({n_tok} tokens)\n", flush=True)

    ppl_configs = [
        ("Binary (sign*scale)",      "binary"),
        ("NEW HVQ K=64",             "new_hvq"),
        ("LowRank r=8 INT8",        "lr_int8"),
        ("Mixed (L0/11 prot.)",      "mixed"),
    ]

    for config_name, config_key in ppl_configs:
        print(f"  Quantizing: {config_name} … ", end="", flush=True)
        t0 = time.time()
        qmodel = copy.deepcopy(model)
        total_params = 0
        total_bpp_w = 0.0

        for li in range(12):
            block = qmodel.transformer.h[li]
            targets = [
                (block.mlp.c_fc,    model.transformer.h[li].mlp.c_fc),
                (block.mlp.c_proj,  model.transformer.h[li].mlp.c_proj),
                (block.attn.c_attn, model.transformer.h[li].attn.c_attn),
                (block.attn.c_proj, model.transformer.h[li].attn.c_proj),
            ]
            for q_sub, orig_sub in targets:
                W = orig_sub.weight.detach().cpu().numpy().T
                d_out, d_in = W.shape
                H_diag = np.ones(d_in, dtype=np.float32)

                if config_key == "binary":
                    W_q, bpp = binary_baseline(W, H_diag)
                elif config_key == "new_hvq":
                    W_q, bpp = new_hvq_k64(W, H_diag)
                elif config_key == "lr_int8":
                    W_q, bpp = lr_r8_int8(W, H_diag)
                elif config_key == "mixed":
                    is_sensitive = (li == 0 or li == 11)
                    is_mlp = (q_sub in [block.mlp.c_fc, block.mlp.c_proj])
                    if is_sensitive:
                        rank = 12 if is_mlp else 8
                        from onebit.quantization.lowrank_residual import LowRankResidual
                        lr = LowRankResidual(rank=rank, factor_bits=8)
                        W_q = lr.quantize(W, H_diag)
                        bpp = lr.effective_bpp()
                    elif is_mlp:
                        from onebit.quantization.lowrank_residual import LowRankResidual
                        lr = LowRankResidual(rank=6, factor_bits=8)
                        W_q = lr.quantize(W, H_diag)
                        bpp = lr.effective_bpp()
                    else:
                        W_q, bpp = new_hvq_k64(W, H_diag)
                else:
                    W_q, bpp = binary_baseline(W, H_diag)

                n = W.size
                total_params += n
                total_bpp_w += bpp * n
                q_sub.weight.data = torch.tensor(W_q.T, dtype=torch.float32)

        avg_bpp = total_bpp_w / total_params
        ppl, _ = evaluate_perplexity(qmodel, tokenizer, max_tokens=1024)
        ratio = ppl / fp_ppl
        elapsed = time.time() - t0
        print(f"PPL={ppl:>12.2f}  ratio={ratio:>8.1f}x  bpp={avg_bpp:.3f}  ({elapsed:.0f}s)", flush=True)
        del qmodel

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results_improvements.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("SALOMI Quantization Improvements — Test Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Method':<35} {'Avg Corr':>9} {'Min Corr':>9} {'BPP':>7}\n")
        f.write("-" * 70 + "\n")
        for name, _ in METHODS:
            avg_c = np.mean(all_results[name]["corrs"])
            min_c = np.min(all_results[name]["corrs"])
            avg_bpp = np.mean(all_results[name]["bpps"])
            f.write(f"{name:<35} {avg_c:>9.5f} {min_c:>9.5f} {avg_bpp:>7.2f}\n")

    print(f"\nResults saved to {outpath}")
    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
