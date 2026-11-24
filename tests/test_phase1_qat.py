"""Phase 1 KD-QAT unit tests (fast, synthetic models/tensors, no Hub downloads)."""
from __future__ import annotations

import copy
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from onebit.eval.harness import BitReport, unique_param_count
from onebit.quantization import baselines as B
from onebit.qat import (
    BinaryLinear,
    DistillationLoss,
    PackedStreamingBatchIterator,
    QATConfig,
    QuantizedEmbedding,
    SyntheticBatchIterator,
    TernaryLinear,
    TiedQuantizedLMHead,
    TrainConfig,
    build_bit_report,
    build_cosine_scheduler,
    build_train_dataloader,
    clamp_round_ste,
    convert_model_to_qat,
    copy_qat_weights_to_fp_model,
    load_checkpoint,
    lsq_grad_scale,
    parameter_groups,
    round_ste,
    run_kd_qat_training,
    save_checkpoint,
    sign_ste,
)


# ---------------------------------------------------------------------------
# Minimal GPT-2-like model (with HF-style Conv1D & tied wte/lm_head)
# ---------------------------------------------------------------------------


class Conv1D(nn.Module):
    """Minimal replica of transformers.pytorch_utils.Conv1D (weight shape (in, out))."""

    def __init__(self, nf: int, nx: int) -> None:
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.randn(nx, nf) * 0.05)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class TinyBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = nn.Module()
        self.attn.c_attn = Conv1D(3 * d, d)
        self.attn.c_proj = Conv1D(d, d)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Module()
        self.mlp.c_fc = Conv1D(4 * d, d)
        self.mlp.c_proj = Conv1D(d, 4 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        qkv = self.attn.c_attn(h)
        d = h.size(-1)
        q, k, v = qkv.split(d, dim=-1)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn.c_proj(att)
        x = x + self.mlp.c_proj(F.gelu(self.mlp.c_fc(self.ln_2(x))))
        return x


class TinyGPT2(nn.Module):
    """Tiny GPT-2-compatible causal LM with Conv1D blocks and tied wte/lm_head."""

    def __init__(self, vocab: int = 64, d: int = 32, n_layer: int = 2, n_ctx: int = 32, tied: bool = True) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, d)
        self.transformer.wpe = nn.Embedding(n_ctx, d)
        self.transformer.h = nn.ModuleList([TinyBlock(d) for _ in range(n_layer)])
        self.transformer.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        if tied:
            self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        hidden_states = [x] if output_hidden_states else None
        for blk in self.transformer.h:
            x = blk(x)
            if hidden_states is not None:
                hidden_states.append(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits, hidden_states=tuple(hidden_states) if hidden_states is not None else None)


# ---------------------------------------------------------------------------
# 1. STE backward gradients
# ---------------------------------------------------------------------------


def test_sign_ste_forward_and_gradients():
    x = torch.tensor([-2.5, -0.3, 0.0, 0.4, 3.0], requires_grad=True)
    y = sign_ste(x)
    assert torch.equal(y.detach(), torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))
    grad_out = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y.backward(grad_out)
    assert torch.allclose(x.grad, grad_out)

    # Clipped STE zeroes gradients strictly outside |x| <= 1.0.
    x2 = torch.tensor([-2.5, -0.3, 0.0, 1.0, 3.0], requires_grad=True)
    y2 = sign_ste(x2, grad_clip=1.0)
    assert torch.equal(y2.detach(), torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))
    y2.backward(grad_out)
    assert torch.allclose(x2.grad, torch.tensor([0.0, 2.0, 3.0, 4.0, 0.0]))


def test_round_and_clamp_round_ste_gradients():
    x = torch.tensor([-2.4, -0.6, 0.2, 0.9, 2.5], requires_grad=True)
    y = clamp_round_ste(x, -1.0, 1.0)
    assert torch.equal(y.detach(), torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]))
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))

    x3 = torch.tensor([-1.7, 0.3, 2.8], requires_grad=True)
    r = round_ste(x3)
    assert torch.equal(r.detach(), torch.tensor([-2.0, 0.0, 3.0]))
    r.sum().backward()
    assert torch.allclose(x3.grad, torch.ones_like(x3))


def test_lsq_grad_scale():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = lsq_grad_scale(x, 0.25)
    assert torch.allclose(y.detach(), x.detach())
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 0.25))


def test_qat_linear_modules_backward_gradients():
    torch.manual_seed(0)
    x = torch.randn(4, 16, requires_grad=True)

    # Ternary with learnable scale + shift
    lin = nn.Linear(16, 8)
    t_mod = TernaryLinear.from_module(lin, group_size=8, use_shift=True, lsq_scale_grad=True)
    loss = t_mod(x).pow(2).mean()
    loss.backward()
    assert t_mod.w_lat.grad is not None and t_mod.w_lat.grad.abs().sum() > 0
    assert t_mod.scale.grad is not None and t_mod.scale.grad.abs().sum() > 0
    assert t_mod.shift.grad is not None and t_mod.shift.grad.abs().sum() > 0
    assert t_mod.bias.grad is not None and t_mod.bias.grad.abs().sum() > 0

    # Binary OneBit with shift
    x2 = torch.randn(4, 16, requires_grad=True)
    b_mod = BinaryLinear.from_module(lin, mode="onebit", use_shift=True)
    b_mod(x2).pow(2).mean().backward()
    assert b_mod.w_lat.grad is not None and b_mod.w_lat.grad.abs().sum() > 0
    assert b_mod.row_scale.grad is not None and b_mod.row_scale.grad.abs().sum() > 0
    assert b_mod.col_scale.grad is not None and b_mod.col_scale.grad.abs().sum() > 0
    assert b_mod.shift.grad is not None and b_mod.shift.grad.abs().sum() > 0

    # Binary ParetoQ with shift
    x3 = torch.randn(4, 16, requires_grad=True)
    p_mod = BinaryLinear.from_module(lin, mode="paretoq", group_size=8, use_shift=True)
    p_mod(x3).pow(2).mean().backward()
    assert p_mod.w_lat.grad is not None and p_mod.w_lat.grad.abs().sum() > 0
    assert p_mod.scale.grad is not None and p_mod.scale.grad.abs().sum() > 0
    assert p_mod.shift.grad is not None and p_mod.shift.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 2. SVID and Ternary init + Conv1D parity
# ---------------------------------------------------------------------------


def test_svid_init_matches_phase0_baseline_and_factored_forward():
    torch.manual_seed(1)
    lin = nn.Linear(64, 32)
    b_mod = BinaryLinear.from_module(lin, mode="onebit", use_shift=False)

    w_ref, bits_ref = B.binary_svid(lin.weight.detach())
    assert torch.allclose(b_mod.dequantized_weight(), w_ref, atol=1e-5, rtol=1e-5)
    assert b_mod.storage_bits() == pytest.approx(bits_ref)

    # Factored forward must match explicit F.linear(x, quantized_weight, bias) both with and without shift.
    x = torch.randn(3, 5, 64)
    for use_shift in (False, True):
        bm = BinaryLinear.from_module(lin, mode="onebit", use_shift=use_shift)
        y_fast = bm(x)
        y_ref = F.linear(x, bm.quantized_weight(), bm.bias)
        assert torch.allclose(y_fast, y_ref, atol=1e-5, rtol=1e-5)


def test_ternary_twn_init_beats_absmean_and_has_at_most_3_levels():
    torch.manual_seed(2)
    lin = nn.Linear(128, 32)
    t_twn = TernaryLinear.from_module(lin, group_size=None, init_method="twn")
    t_abs = TernaryLinear.from_module(lin, group_size=None, init_method="absmean")

    w = lin.weight.detach()
    wq_twn = t_twn.dequantized_weight()
    wq_abs = t_abs.dequantized_weight()

    assert all(torch.unique(row).numel() <= 3 for row in wq_twn)
    mse_twn = float(((w - wq_twn) ** 2).mean())
    mse_abs = float(((w - wq_abs) ** 2).mean())
    assert mse_twn < mse_abs

    _, bits_ref = B.ternary_twn(w, group=None)
    assert t_twn.storage_bits() == pytest.approx(bits_ref)


def test_conv1d_to_qat_weight_orientation():
    torch.manual_seed(3)
    conv = Conv1D(nf=24, nx=16)  # weight shape (16, 24)
    t_mod = TernaryLinear.from_module(conv, init_method="twn")
    assert t_mod.w_lat.shape == (24, 16)
    assert torch.allclose(t_mod.w_lat.detach(), conv.weight.detach().t())
    assert torch.allclose(t_mod.bias.detach(), conv.bias.detach())


# ---------------------------------------------------------------------------
# 3. Tied embedding quantization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits,symmetric", [(8, True), (4, True), (4, False)])
def test_tied_embedding_quantization_consistency_and_gradients(bits: int, symmetric: bool):
    torch.manual_seed(4)
    model = TinyGPT2(vocab=48, d=32, n_layer=2, n_ctx=16, tied=True)
    base_params = unique_param_count(model)

    cfg = QATConfig(mode="ternary", embed_bits=bits, embed_symmetric=symmetric, quantize_pos_embeddings=True)
    model, report = convert_model_to_qat(model, cfg)

    assert isinstance(model.transformer.wte, QuantizedEmbedding)
    assert isinstance(model.transformer.wpe, QuantizedEmbedding)
    assert isinstance(model.lm_head, TiedQuantizedLMHead)

    # TiedQuantizedLMHead must NOT duplicate wte weight in state_dict.
    sd_keys = list(model.state_dict().keys())
    assert "transformer.wte.weight" in sd_keys
    assert not any(k.startswith("lm_head.weight") or "_tied_embedding" in k for k in sd_keys)

    # Row-selected embedding lookup must equal indexing the full quantized_weight().
    ids = torch.randint(0, 48, (2, 10))
    emb_fast = model.transformer.wte(ids)
    emb_full = F.embedding(ids, model.transformer.wte.quantized_weight())
    assert torch.allclose(emb_fast, emb_full, atol=1e-6)

    # LM head must use the exact same quantized weight.
    h = torch.randn(2, 10, 32)
    assert torch.allclose(model.lm_head(h), F.linear(h, model.transformer.wte.quantized_weight()), atol=1e-6)

    # Gradients flow through both input lookup and output LM head into wte.weight.
    out = model(ids).logits
    out.pow(2).mean().backward()
    assert model.transformer.wte.weight.grad is not None
    assert model.transformer.wte.weight.grad.abs().sum() > 0

    # Exact BitReport accounting: total_params matches base FP model; tied LM head counted once.
    assert report.total_params == base_params
    expected_emb_params = 48 * 32 + 16 * 32
    assert report.embed_params == expected_emb_params
    meta_per_row = 16 if symmetric else 32
    expected_emb_bits = expected_emb_params * bits + (48 + 16) * meta_per_row
    assert report.embed_bits == pytest.approx(expected_emb_bits)


def test_untied_lm_head_is_quantized_and_counted_separately():
    torch.manual_seed(5)
    model = TinyGPT2(vocab=48, d=32, n_layer=1, n_ctx=16, tied=False)
    base_params = unique_param_count(model)
    cfg = QATConfig(mode="binary", embed_bits=8, quantize_pos_embeddings=False)
    model, report = convert_model_to_qat(model, cfg)
    assert report.total_params == base_params
    # Both wte (48*32) and untied lm_head (48*32) counted separately.
    assert report.embed_params == 2 * 48 * 32


# ---------------------------------------------------------------------------
# 4. Exact BitReport math & equal-byte architecture geometry
# ---------------------------------------------------------------------------


def test_bitreport_exact_bpp_across_modes_and_embedding_widths():
    torch.manual_seed(6)
    base = TinyGPT2(vocab=64, d=32, n_layer=2, n_ctx=16, tied=True)
    base_total = unique_param_count(base)
    # 2 layers * 12 * 32^2 = 24,576 linear params
    expected_linear = 2 * 12 * 32 * 32

    # Case A: Ternary perchan, FP16 embeddings
    m_t_fp = copy.deepcopy(base)
    m_t_fp, rep_t_fp = convert_model_to_qat(m_t_fp, QATConfig(mode="ternary", embed_bits=0))
    assert rep_t_fp.total_params == base_total
    assert rep_t_fp.linear_params == expected_linear
    assert rep_t_fp.embed_params == 0
    # Per block out_features sum = 3d + d + 4d + d = 9d -> 2 * 9 * 32 = 576 scales
    expected_t_bits = expected_linear * 1.6 + (2 * 9 * 32) * 16
    assert rep_t_fp.linear_bits == pytest.approx(expected_t_bits)
    assert rep_t_fp.bpp_whole_model == pytest.approx(
        (expected_t_bits + 16.0 * (base_total - expected_linear)) / base_total
    )

    # Case B: Ternary perchan, INT8 embeddings -> same bpp_linear, much lower bpp_whole_model
    m_t_e8 = copy.deepcopy(base)
    m_t_e8, rep_t_e8 = convert_model_to_qat(m_t_e8, QATConfig(mode="ternary", embed_bits=8))
    assert rep_t_e8.bpp_linear == pytest.approx(rep_t_fp.bpp_linear)
    assert rep_t_e8.bpp_whole_model < rep_t_fp.bpp_whole_model
    # Recomputing build_bit_report without passing base_total_params must match identically.
    rep_recomputed = build_bit_report(m_t_e8)
    assert rep_recomputed.total_params == base_total
    assert rep_recomputed.bpp_linear == pytest.approx(rep_t_e8.bpp_linear)
    assert rep_recomputed.bpp_whole_model == pytest.approx(rep_t_e8.bpp_whole_model)

    # Case C: Binary OneBit (no shift vs with shift)
    m_b = copy.deepcopy(base)
    m_b, rep_b = convert_model_to_qat(m_b, QATConfig(mode="binary", use_shift=False))
    # Per block (out + in) sum = (3d+d) + (d+d) + (4d+d) + (d+4d) = 16d -> 2 * 16 * 32 = 1024 fp16 scales
    expected_b_bits = expected_linear * 1.0 + (2 * 16 * 32) * 16
    assert rep_b.linear_bits == pytest.approx(expected_b_bits)

    m_bs = copy.deepcopy(base)
    m_bs, rep_bs = convert_model_to_qat(m_bs, QATConfig(mode="binary", use_shift=True))
    # Shift adds out_features per layer = 2 * 9 * 32 fp16 values
    assert rep_bs.linear_bits == pytest.approx(expected_b_bits + (2 * 9 * 32) * 16)


def test_equal_byte_gpt2_geometry_matches_phase0_json():
    from scripts.compare_equal_bytes import GPT2_FAMILY

    g_small = GPT2_FAMILY[0]
    assert g_small.total_params == 124_439_808
    assert g_small.linear_params == 84_934_656
    assert g_small.storage_summary("binary_onebit", 16)["bpp_linear"] == pytest.approx(1.0277777777777777)
    assert g_small.storage_summary("ternary_perchan", 16)["bpp_linear"] == pytest.approx(1.615625)
    assert g_small.storage_summary("gptq_int4_g128", 16)["bpp_linear"] == pytest.approx(4.25)


# ---------------------------------------------------------------------------
# 5. Parameter grouping & staged ternary -> binary transfer
# ---------------------------------------------------------------------------


def test_parameter_groups_partition_and_staged_weight_transfer():
    torch.manual_seed(7)
    base = TinyGPT2(vocab=48, d=32, n_layer=2, n_ctx=16, tied=True)
    m_tern, _ = convert_model_to_qat(copy.deepcopy(base), QATConfig(mode="ternary", embed_bits=0))

    groups = parameter_groups(m_tern, lr_weights=3e-4, lr_scales=1e-3, lr_other=1e-4, weight_decay=0.01)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name.keys()) == {"latent_weights", "scales", "other"}
    assert by_name["latent_weights"]["lr"] == 3e-4
    assert by_name["scales"]["lr"] == 1e-3
    assert by_name["other"]["lr"] == 1e-4

    # Every trainable unique parameter appears in exactly one group.
    grouped_ids = [id(p) for g in groups for p in g["params"]]
    expected_ids = []
    seen = set()
    for p in m_tern.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            expected_ids.append(id(p))
    assert len(grouped_ids) == len(set(grouped_ids)) == len(expected_ids)
    assert set(grouped_ids) == set(expected_ids)

    # Perturb ternary latent weights, then transfer to a fresh FP model -> Binary SVID.
    with torch.no_grad():
        m_tern.transformer.h[0].mlp.c_fc.w_lat.add_(0.5)
    fp_target = copy.deepcopy(base)
    copy_qat_weights_to_fp_model(m_tern, fp_target, source="latent")
    m_bin, rep_bin = convert_model_to_qat(fp_target, QATConfig(mode="binary"))
    assert torch.allclose(
        m_bin.transformer.h[0].mlp.c_fc.w_lat.detach(),
        m_tern.transformer.h[0].mlp.c_fc.w_lat.detach(),
    )
    assert rep_bin.total_params == unique_param_count(base)


# ---------------------------------------------------------------------------
# 6. Teacher-student distillation loss sanity & data hygiene
# ---------------------------------------------------------------------------


def test_distillation_loss_sanity():
    torch.manual_seed(8)
    ids = torch.randint(0, 32, (2, 8))
    t_logits = torch.randn(2, 8, 32)
    s_logits = t_logits.clone().requires_grad_(True)
    t_hs = [torch.randn(2, 8, 16) for _ in range(3)]
    s_hs = [h.clone().requires_grad_(True) for h in t_hs]

    # Identical logits + hidden states -> KL = 0, hidden_mse = 0.
    crit = DistillationLoss(kl_weight=1.0, ce_weight=0.0, temperature=2.0, hidden_mse_weight=0.5)
    loss, m = crit(s_logits, t_logits, ids, student_hidden_states=s_hs, teacher_hidden_states=t_hs)
    assert m["kl_loss"] == pytest.approx(0.0, abs=1e-6)
    assert m["hidden_mse"] == pytest.approx(0.0, abs=1e-6)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    # Distinct logits -> positive KL + CE + hidden MSE and finite student grads.
    s_logits2 = (t_logits + 0.5 * torch.randn_like(t_logits)).requires_grad_(True)
    s_hs2 = [(h + 0.2 * torch.randn_like(h)).requires_grad_(True) for h in t_hs]
    crit2 = DistillationLoss(kl_weight=1.0, ce_weight=0.2, temperature=1.5, hidden_mse_weight=0.1)
    loss2, m2 = crit2(s_logits2, t_logits, ids, student_hidden_states=s_hs2, teacher_hidden_states=t_hs)
    assert m2["kl_loss"] > 0 and m2["ce_loss"] > 0 and m2["hidden_mse"] > 0
    loss2.backward()
    assert torch.isfinite(s_logits2.grad).all() and s_logits2.grad.abs().sum() > 0
    assert all(torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0 for h in s_hs2[1:])


def test_data_loader_refuses_test_split_and_packs_stream():
    for bad in ("test", "validation", "val", "eval"):
        with pytest.raises(ValueError, match="Refusing"):
            build_train_dataloader("synthetic", tokenizer=None, batch_size=2, seqlen=8, split=bad, vocab_size=32)

    class DummyTok:
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[ord(c) % 30 + 1 for c in text])

    docs = iter(["hello world", "salomi phase one", "ternary and binary qat"] * 5)
    loader = PackedStreamingBatchIterator(docs, DummyTok(), batch_size=2, seqlen=8)
    b1 = next(loader)
    b2 = next(loader)
    assert b1.shape == (2, 8) and b1.dtype == torch.long
    assert b2.shape == (2, 8) and (b1 == 0).any()  # EOS separators present


# ---------------------------------------------------------------------------
# 7. Checkpoint save/resume determinism & mini end-to-end training
# ---------------------------------------------------------------------------


def test_checkpoint_save_resume_exact_determinism(tmp_path):
    torch.manual_seed(42)
    teacher = TinyGPT2(vocab=48, d=32, n_layer=2, n_ctx=16, tied=True).eval()
    qat_cfg = QATConfig(mode="ternary", init_method="twn", embed_bits=8)

    # Run A: uninterrupted 6 steps
    torch.manual_seed(100)
    student_a, report_a = convert_model_to_qat(copy.deepcopy(teacher), qat_cfg)
    dl_a = SyntheticBatchIterator(vocab_size=48, batch_size=2, seqlen=16, seed=7)
    cfg_a = TrainConfig(
        model="tiny-gpt2",
        dataset="synthetic",
        seqlen=16,
        batch_size=2,
        grad_accum=2,
        max_steps=6,
        lr=5e-3,
        scale_lr=1e-2,
        warmup_steps=2,
        dtype="fp32",
        eval_every=0,
        log_every=1,
        save_every=0,
        run_name="det_full",
        work_dir=str(tmp_path / "ckpt_a"),
        results_dir=str(tmp_path / "res_a"),
    )
    out_a = run_kd_qat_training(
        student=student_a,
        teacher=copy.deepcopy(teacher),
        report=report_a,
        qat_cfg=qat_cfg,
        train_cfg=cfg_a,
        dataloader=dl_a,
        eval_ids=None,
        device=torch.device("cpu"),
        write_final_harness_json=False,
    )

    # Run B: 3 steps -> save -> fresh model/optimizer/scheduler/dataloader -> resume -> 6 steps
    torch.manual_seed(100)
    student_b1, report_b1 = convert_model_to_qat(copy.deepcopy(teacher), qat_cfg)
    dl_b1 = SyntheticBatchIterator(vocab_size=48, batch_size=2, seqlen=16, seed=7)
    cfg_b1 = copy.deepcopy(cfg_a)
    cfg_b1.max_steps = 3
    cfg_b1.run_name = "det_split"
    cfg_b1.work_dir = str(tmp_path / "ckpt_b")
    cfg_b1.results_dir = str(tmp_path / "res_b")
    # Important: use the same cosine schedule total_steps=6! We test resume via save/load + 6 total steps:
    # Build matching total_steps by running with a 6-step schedule and stopping at 3 via save_every=3?
    # Even cleaner: verify save_checkpoint/load_checkpoint directly with the exact same total_steps=6!
    opt_b1 = torch.optim.AdamW(parameter_groups(student_b1, 5e-3, 1e-2), betas=(0.9, 0.95))
    sched_b1 = build_cosine_scheduler(opt_b1, total_steps=6, warmup_steps=2, min_lr_ratio=0.1)
    crit = DistillationLoss(kl_weight=1.0, ce_weight=0.0, temperature=1.0)
    all_p = [p for g in opt_b1.param_groups for p in g["params"]]
    losses_b = []
    tokens_b = 0
    for step in range(1, 4):
        opt_b1.zero_grad(set_to_none=True)
        al = 0.0
        for _ in range(2):
            batch = next(dl_b1)
            with torch.no_grad():
                tl = teacher(batch).logits
            sl = student_b1(batch).logits
            loss, m = crit(sl, tl, batch)
            (loss / 2.0).backward()
            al += m["loss"] / 2.0
            tokens_b += batch.numel()
        torch.nn.utils.clip_grad_norm_(all_p, 1.0)
        opt_b1.step()
        sched_b1.step()
        losses_b.append(al)

    ckpt_path = str(tmp_path / "ckpt_b" / "step3.pt")
    save_checkpoint(
        ckpt_path,
        student_b1,
        opt_b1,
        sched_b1,
        step=3,
        tokens_seen=tokens_b,
        best_ppl=float("inf"),
        qat_cfg=qat_cfg,
        train_cfg=cfg_a,
        report=report_b1,
        dataloader=dl_b1,
        micro_batches_seen=6,
    )

    # Resume from step 3 -> step 6 using run_kd_qat_training!
    student_b2, report_b2 = convert_model_to_qat(copy.deepcopy(teacher), qat_cfg)
    dl_b2 = SyntheticBatchIterator(vocab_size=48, batch_size=2, seqlen=16, seed=999)  # wrong seed; must be overwritten by ckpt!
    cfg_b2 = copy.deepcopy(cfg_a)
    cfg_b2.run_name = "det_resumed"
    cfg_b2.work_dir = str(tmp_path / "ckpt_b2")
    cfg_b2.results_dir = str(tmp_path / "res_b2")
    cfg_b2.resume = ckpt_path
    out_b2 = run_kd_qat_training(
        student=student_b2,
        teacher=copy.deepcopy(teacher),
        report=report_b2,
        qat_cfg=qat_cfg,
        train_cfg=cfg_b2,
        dataloader=dl_b2,
        eval_ids=None,
        device=torch.device("cpu"),
        write_final_harness_json=False,
    )

    full_losses_b = losses_b + out_b2["loss_history"]
    assert out_b2["step"] == out_a["step"] == 6
    assert out_b2["tokens_seen"] == out_a["tokens_seen"]
    assert len(full_losses_b) == len(out_a["loss_history"]) == 6
    for la, lb in zip(out_a["loss_history"], full_losses_b):
        assert lb == pytest.approx(la, rel=1e-6, abs=1e-6)

    for (ka, va), (kb, vb) in zip(student_a.state_dict().items(), student_b2.state_dict().items()):
        assert ka == kb
        assert torch.allclose(va, vb, atol=1e-6, rtol=1e-6)


def test_mini_training_reduces_loss_and_writes_canonical_harness_json(tmp_path):
    torch.manual_seed(11)
    teacher = TinyGPT2(vocab=48, d=32, n_layer=2, n_ctx=16, tied=True).eval()
    qat_cfg = QATConfig(mode="binary", use_shift=True, embed_bits=8)
    student, report = convert_model_to_qat(copy.deepcopy(teacher), qat_cfg)

    dl = SyntheticBatchIterator(vocab_size=48, batch_size=4, seqlen=16, seed=0)
    eval_ids = torch.randint(0, 48, (64,), generator=torch.Generator().manual_seed(99))

    train_cfg = TrainConfig(
        model="tiny-gpt2",
        dataset="synthetic",
        seqlen=16,
        batch_size=4,
        grad_accum=1,
        max_steps=20,
        lr=1e-2,
        scale_lr=2e-2,
        warmup_steps=3,
        kl_weight=1.0,
        ce_weight=0.1,
        dtype="fp32",
        eval_every=10,
        eval_windows=2,
        final_eval_windows=2,
        eval_seqlen=16,
        log_every=5,
        save_every=10,
        run_name="tiny_binary_smoke",
        work_dir=str(tmp_path / "ckpt"),
        results_dir=str(tmp_path / "results"),
    )
    out = run_kd_qat_training(
        student=student,
        teacher=teacher,
        report=report,
        qat_cfg=qat_cfg,
        train_cfg=train_cfg,
        dataloader=dl,
        eval_ids=eval_ids,
        device=torch.device("cpu"),
        write_final_harness_json=True,
    )

    assert out["loss_history"][-1] < out["loss_history"][0]
    assert os.path.exists(out["latest_checkpoint"])
    assert os.path.exists(out["log_path"])

    json_path = os.path.join(str(tmp_path / "results"), "tiny_binary_smoke.json")
    assert os.path.exists(json_path)
    with open(json_path, encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["run_name"] == "tiny_binary_smoke"
    assert rec["protocol"]["full_test_set"] is False
    assert "bpp_linear" in rec["result"] and "bpp_whole_model" in rec["result"]
    assert "bpp_embed" in rec["result"] and rec["result"]["embed_params"] > 0
    assert math.isfinite(rec["result"]["ppl"]) and rec["result"]["ppl"] > 0
    assert rec["training"]["tokens_seen"] == 20 * 4 * 16
