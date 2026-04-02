from __future__ import annotations

import numpy as np
import pytest

from onebit.runtime.controller_e2e import E2EConfig, infer_one_token_e2e
from onebit.runtime.ctg_grammar import CTG, CTGRule, CTGState
from onebit.core.packbits import pack_input_signs


def _has_opencl() -> bool:
    try:
        from onebit.backends.opencl import host_opencl
    except Exception:
        return False
    # host_opencl exposes a module-level `cl` that is None when pyopencl
    # is not installed. We treat that as "no OpenCL" for this test.
    return getattr(host_opencl, "cl", None) is not None


@pytest.mark.skipif(not _has_opencl(), reason="OpenCL backend not available")
def test_e2e_cpu_vs_opencl_parity_with_ctg_enabled() -> None:
    """CPU and OpenCL paths must agree when CTG grammar is enabled.

    We only exercise a tiny synthetic problem; the goal is byte-level parity on
    the high-level outputs that are exposed to users and telemetry.
    """
    rng = np.random.default_rng(123)

    n_ctx = 8
    d_attn = 32
    d_model = 32
    d_kv = 16
    vocab_size = 32

    # Synthetic packed bits
    K_attn = rng.standard_normal((n_ctx, d_attn), dtype=np.float32)
    K_attn_bits = np.array([pack_input_signs(K_attn[i]) for i in range(n_ctx)])

    K_kv = rng.standard_normal((n_ctx, d_kv), dtype=np.float32)
    V_kv = rng.standard_normal((n_ctx, d_kv), dtype=np.float32)
    K_kv_bits = np.array([pack_input_signs(K_kv[i]) for i in range(n_ctx)])
    V_kv_bits = np.array([pack_input_signs(V_kv[i]) for i in range(n_ctx)])

    vocab_ids = np.arange(vocab_size, dtype=np.int32)

    # Simple CTG: inhibit odd token IDs, no inversion
    inhibit_ids = np.arange(1, vocab_size, 2, dtype=np.int32)
    ctg = CTG(
        rules=[CTGRule(op="INHIBIT", ids=inhibit_ids, period=8, prob_num=1, prob_den=1)],
        vocab_size=vocab_size,
        phase_period=8,
    )
    ctg_state = CTGState(phase=0, mask_digest=0)

    cfg_cpu = E2EConfig(
        kA=4,
        k_max_attn=8,
        d_kv=d_kv,
        backend="cpu",
        ctg=ctg,
        ctg_state=ctg_state,
    )

    cfg_ocl = E2EConfig(
        kA=4,
        k_max_attn=8,
        d_kv=d_kv,
        backend="opencl",
        ctg=ctg,
        ctg_state=ctg_state,
    )

    seed = 999

    Q_attn = rng.standard_normal(d_attn, dtype=np.float32)
    Q_attn_bits = pack_input_signs(Q_attn)
    Q_logits = rng.standard_normal(d_model, dtype=np.float32)
    Q_logits_bits = pack_input_signs(Q_logits)

    out_cpu = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits,
        K_kv_bits, V_kv_bits,
        Q_logits_bits, vocab_ids,
        cfg=cfg_cpu,
        prf_seed=seed,
        d_attn=d_attn,
        d_model=d_model,
        token_idx=0,
        ctg=ctg,
        ctg_state=ctg_state,
    )

    out_ocl = infer_one_token_e2e(
        Q_attn_bits, K_attn_bits,
        K_kv_bits, V_kv_bits,
        Q_logits_bits, vocab_ids,
        cfg=cfg_ocl,
        prf_seed=seed,
        d_attn=d_attn,
        d_model=d_model,
        token_idx=0,
        ctg=ctg,
        ctg_state=ctg_state,
    )

    assert out_cpu["logits_top1"] == out_ocl["logits_top1"]
    assert out_cpu["k_logits_used"] == out_ocl["k_logits_used"]
    assert out_cpu.get("ctg_mask_digest") == out_ocl.get("ctg_mask_digest")

