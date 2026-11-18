#!/usr/bin/env python3
"""CPU proof-of-concept: OneBit-style ~1.03 bpp GPT-2 via SVID init + knowledge distillation.

W (in,out) ~= diag(h) . sign(W_lat) . diag(g)   (signs 1 bit; h,g FP16 vectors)
 - Init: W_lat = W, [h,g] = rank-1 SVD of |W|  (SVID, OneBit arXiv:2402.11295)
 - Train: STE through sign, KL(teacher || student) on logits, WikiText-2 *train* stream.
 - Eval: WikiText-2 *test*, non-overlapping 1024-token windows.
Embeddings / LayerNorm / biases stay FP (standard convention); bpp reported over block weights.

PRELIMINARY / NOT A VALIDATED RESULT. Known limitations of the default settings:
  * trains on WikiText-2 *train* (same domain as the test set) -- use a general corpus
    (FineWeb / OpenWebText) for real runs;
  * evaluates on the first ``--eval_windows`` test windows only (use 0 for the full set);
  * single seed. See docs/PHASE0_AUDIT.md section 4.

    python scripts/onebit_kd_poc.py --minutes 40 --bs 2 --seq 256 --eval_every 100
"""
import math, time, argparse, copy, json, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onebit.eval.harness import load_eval_ids, perplexity as harness_perplexity, REPO_ROOT  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.manual_seed(0)


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        s = torch.sign(w)
        s[s == 0] = 1
        return s

    @staticmethod
    def backward(ctx, g):
        return g


class OneBitConv1D(nn.Module):
    """Drop-in for HF Conv1D (y = x @ W + b, W shape (in, out))."""

    def __init__(self, conv, train_signs=True):
        super().__init__()
        W = conv.weight.data.float()  # (in, out)
        U, S, Vh = torch.linalg.svd(W.abs(), full_matrices=False)
        h = (U[:, 0] * S[0].sqrt()).abs()
        g = (Vh[0, :] * S[0].sqrt()).abs()
        self.h = nn.Parameter(h)
        self.g = nn.Parameter(g)
        self.w_lat = nn.Parameter(W.clone(), requires_grad=train_signs)
        self.bias = nn.Parameter(conv.bias.data.float().clone())
        self.nin, self.nout = W.shape

    def forward(self, x):
        s = SignSTE.apply(self.w_lat)
        y = ((x * self.h) @ s) * self.g
        return y + self.bias

    def bits(self):
        return self.nin * self.nout + 16 * (self.nin + self.nout)


def build_student(teacher, train_signs):
    st = copy.deepcopy(teacher)
    n_w, n_bits = 0, 0
    for blk in st.transformer.h:
        for parent, name in [(blk.attn, "c_attn"), (blk.attn, "c_proj"), (blk.mlp, "c_fc"), (blk.mlp, "c_proj")]:
            m = OneBitConv1D(getattr(parent, name), train_signs)
            setattr(parent, name, m)
            n_w += m.nin * m.nout
            n_bits += m.bits()
    return st, n_bits / n_w


def wt_ids(tok, split):
    return load_eval_ids(tok, "wikitext2", split)


def ppl(model, ids, windows, seqlen=1024):
    return harness_perplexity(model, ids, seqlen=seqlen, max_windows=windows)["ppl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2, help="2 fits in ~5 GB RAM on CPU")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_windows", type=int, default=20, help="0 = full WikiText-2 test set")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--scale_only", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "experiments", "onebit_kd_poc.json"))
    a = ap.parse_args()
    a.eval_windows = a.eval_windows or None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    test_ids, train_ids = wt_ids(tok, "test"), wt_ids(tok, "train")

    student, bpp = build_student(teacher, train_signs=not a.scale_only)
    # Keep embeddings frozen (FP, tied with lm_head); train LN + OneBit params.
    student.transformer.wte.weight.requires_grad_(False)
    student.transformer.wpe.weight.requires_grad_(False)
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"bpp(block weights)={bpp:.4f}  trainable={sum(p.numel() for p in params)/1e6:.2f}M", flush=True)

    log = {"bpp": bpp, "args": vars(a), "evals": []}
    fp = ppl(teacher, test_ids, a.eval_windows)
    p0 = ppl(student, test_ids, a.eval_windows)
    print(f"teacher FP32 ppl={fp:.2f}   student@init(SVID, no training) ppl={p0:.2f}", flush=True)
    log["teacher_ppl"], log["evals"] = fp, [{"step": 0, "tokens": 0, "ppl": p0}]

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    t0, step, toks = time.time(), 0, 0
    budget = a.minutes * 60
    while time.time() - t0 < budget:
        starts = torch.randint(0, train_ids.numel() - a.seq - 1, (a.bs,))
        x = torch.stack([train_ids[s:s + a.seq] for s in starts])
        with torch.no_grad():
            tl = teacher(x).logits
        sl = student(x).logits
        loss = F.kl_div(F.log_softmax(sl, -1).flatten(0, 1), F.log_softmax(tl, -1).flatten(0, 1),
                        log_target=True, reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        # cosine decay over the time budget
        frac = min(1.0, (time.time() - t0) / budget)
        for gp in opt.param_groups:
            gp["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * frac))
        opt.step()
        step += 1
        toks += x.numel()
        if step % 20 == 0:
            print(f"step {step} tokens {toks} kl={loss.item():.3f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0:
            p = ppl(student, test_ids, a.eval_windows)
            print(f"  EVAL step {step} tokens {toks}: ppl={p:.2f}", flush=True)
            log["evals"].append({"step": step, "tokens": toks, "ppl": p})
            json.dump(log, open(a.out, "w"), indent=1)
    p = ppl(student, test_ids, a.eval_windows)
    print(f"FINAL step {step} tokens {toks}: ppl={p:.2f} (teacher {fp:.2f}, init {p0:.2f}, bpp {bpp:.4f})", flush=True)
    log["evals"].append({"step": step, "tokens": toks, "ppl": p, "final": True})
    json.dump(log, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
