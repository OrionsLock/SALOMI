"""
Full 48-Matrix Test: Optimal 1.00 BPP vs Ternary

Tests HessianVQ-128 (targeting ~1.00 bpp) against Ternary (1.58 bpp)
across all 48 weight matrices in GPT-2.
"""
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')


def hessian_kmeans(X, H, k, max_iter=15):
    """Hessian-weighted K-means clustering."""
    np.random.seed(42)
    C = X[np.random.choice(len(X), k, replace=False)].copy()
    for _ in range(max_iter):
        d = np.sum((X[:, None] - C[None])**2, axis=2)
        a = np.argmin(d, axis=1)
        for i in range(k):
            m = (a == i)
            if m.sum() > 0:
                C[i] = np.sum(X[m] * H[m], axis=0) / (np.sum(H[m], axis=0) + 1e-8)
    return C, a


def quantize_1bpp(W, H_diag, K=128):
    """Quantize weight matrix to ~1.00 bpp using Hessian-weighted VQ."""
    h, w = W.shape
    bs = 4

    # Extract signs
    S = np.sign(W)
    S[S == 0] = 1.0
    M = np.abs(W)

    # Pad to block size
    ph, pw = (bs - h % bs) % bs, (bs - w % bs) % bs
    Mp = np.pad(M, ((0, ph), (0, pw)))
    Hm = np.tile(H_diag, (h, 1))
    Hp = np.pad(Hm, ((0, ph), (0, pw)), constant_values=1e-6)

    # Extract blocks
    hp, wp = Mp.shape
    blk = Mp.reshape(hp//bs, bs, wp//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)
    wts = Hp.reshape(hp//bs, bs, wp//bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs*bs)

    # Cluster
    C, a = hessian_kmeans(blk, wts, K)

    # Reconstruct
    rec = C[a].reshape(hp//bs, wp//bs, bs, bs).transpose(0, 2, 1, 3).reshape(hp, wp)
    W_q = S * rec[:h, :w]

    # Compute BPP
    cnt = np.bincount(a, minlength=K)
    p = cnt / cnt.sum()
    p = p[p > 0]
    ent = -np.sum(p * np.log2(p))
    bpp = 0.5 + len(blk) * ent / (h * w) + K * 16 * 16 / (h * w)

    return W_q, bpp


def main():
    print("=" * 70)
    print("FULL 48-MATRIX TEST: OPTIMAL 1.00 BPP vs TERNARY")
    print("=" * 70)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "The quick brown fox jumps over the lazy dog. " * 30
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    results = {'tern': [], 'vq': [], 'bpp': []}
    mod_names = ['c_attn', 'c_proj', 'c_fc', 'c_proj2']
    mod_results = {n: {'tern': [], 'vq': []} for n in mod_names}

    print("\nProcessing 48 weight matrices with Hessian weighting...")

    for li in range(12):
        layer = model.transformer.h[li]
        mods = [
            ('c_attn', layer.attn.c_attn),
            ('c_proj', layer.attn.c_proj),
            ('c_fc', layer.mlp.c_fc),
            ('c_proj2', layer.mlp.c_proj)
        ]

        for name, mod in mods:
            W = mod.weight.detach().cpu().numpy().T
            din = W.shape[1]

            # Get Hessian via activation capture
            acts = []
            def hook(m, i, o):
                acts.append(i[0].detach().cpu().numpy())
            h = mod.register_forward_hook(hook)
            model(inputs.input_ids)
            h.remove()

            X = acts[0].reshape(-1, din)
            H_diag = np.mean(X**2, axis=0)

            # Test set
            Xt = X[:200]
            Yt = Xt @ W.T

            # Ternary baseline
            S = np.sign(W)
            S[S == 0] = 1
            thr = np.percentile(np.abs(W), 30)
            sc = np.mean(np.abs(W[np.abs(W) > thr])) if (np.abs(W) > thr).any() else 1.0
            Wt = S * (np.abs(W) > thr) * sc
            ct = np.corrcoef((Xt @ Wt.T).ravel(), Yt.ravel())[0, 1]

            # VQ @ ~1.00 bpp
            Wq, bpp = quantize_1bpp(W, H_diag, K=128)
            cq = np.corrcoef((Xt @ Wq.T).ravel(), Yt.ravel())[0, 1]

            results['tern'].append(ct)
            results['vq'].append(cq)
            results['bpp'].append(bpp)
            mod_results[name]['tern'].append(ct)
            mod_results[name]['vq'].append(cq)

        tern_avg = np.mean(results['tern'][-4:])
        vq_avg = np.mean(results['vq'][-4:])
        print(f"  Layer {li:2d}: Tern={tern_avg:.4f}  VQ={vq_avg:.4f}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (48 matrices)")
    print("=" * 70)

    tm, ts = np.mean(results['tern']), np.std(results['tern'])
    vm, vs = np.mean(results['vq']), np.std(results['vq'])
    bm = np.mean(results['bpp'])
    imp = (vm - tm) / tm * 100

    print(f"\n{'Method':<20} {'Mean Corr':>12} {'Std':>10} {'BPP':>8}")
    print("-" * 52)
    print(f"{'Ternary':<20} {tm:>12.4f} {ts:>10.4f} {'1.58':>8}")
    print(f"{'HessianVQ-128':<20} {vm:>12.4f} {vs:>10.4f} {bm:>8.2f}")
    print(f"\n*** IMPROVEMENT: {imp:+.2f}% correlation with {(1-bm/1.58)*100:.0f}% fewer bits ***")

    print("\n" + "-" * 70)
    print("By module type:")
    for n in mod_names:
        t = np.mean(mod_results[n]['tern'])
        v = np.mean(mod_results[n]['vq'])
        d = (v - t) / t * 100
        print(f"  {n:<10}: Tern={t:.4f}  VQ={v:.4f}  ({d:+.1f}%)")


if __name__ == "__main__":
    main()

