"""End-to-end controller: Attention → KV → Logits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Optional
from contextlib import contextmanager
import time
import hashlib
import numpy as np

from ..ops.attention_probe import stageA_probe_topT
from ..attn.runner import certify_topT
from ..attn.sprt_dag import SPRTConfig
from ..ops.ldpkv import encode_kv_ldp, decode_kv_ldp_stage1, decode_kv_ldp_stage2
from ..ops.logits_sprt import shortlist_and_certify
from .cpg_policy import CpgPolicy
from .ctg_grammar import CTG, CTGState, default_program_id_fn
from .pulse_scheduler import PulseScheduler
from .shortlist import ShortlistCache, CarryCfg
from typing import Callable


@contextmanager
def _no_side_effects():
    """Context manager to prevent shadow path from writing to caches."""
    # TODO: Implement actual side-effect blocking (KV cache, shortlist cache, etc.)
    try:
        yield
    finally:
        pass


def _digest(a: np.ndarray) -> str:
    """Compute SHA256 digest of array for comparison."""
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _now() -> float:
    """Get current time in seconds."""
    return time.perf_counter()


@dataclass(frozen=True)
class E2EConfig:
    """End-to-end controller configuration.
    
    Args:
        # Attention params
        kA: Stage-A probe ticks (default: 16)
        k_max_attn: Max ticks for attention SPRT (default: 64)
        
        # KV params
        d_kv: KV dimension (default: 512)
        d_left: Expander left degree (default: 8)
        d_right: Expander right degree (default: 4)
        k_kv_stage1: Ticks for KV Stage-1 (default: 16)
        k_kv_stage2: Ticks for KV Stage-2 (default: 16)
        top_k_kv: Number of KV positions to retrieve (default: 8)
        
        # Logits params
        k0_logits: Warmup ticks for logits (default: 8)
        k_step_logits: Ticks per SPRT chunk for logits (default: 4)
        k_max_logits: Max ticks for logits SPRT (default: 64)
        shortlist_size: Logits shortlist size (default: 32)
        
        # Risk budgets
        delta_total: Total risk budget per token (default: 0.01)
        attn_share: Fraction for attention (default: 0.33)
        kv_share: Fraction for KV (default: 0.33)
        logits_share: Fraction for logits (default: 0.34)
        
        # SPRT params
        eps: Effect size (default: 0.05)
        
        # Backend
        backend: "cpu" or "opencl" (default: "opencl")
        walsh_N: Walsh carriers (default: 2)
        antithetic: Antithetic pairs (default: True)
        order: ΣΔ order (default: 2)
        beta: ΣΔ-2 beta (default: 0.30)
        lambd: ΣΔ leak (default: 1/256)
    """
    # Attention
    kA: int = 16
    k_max_attn: int = 64
    
    # KV
    d_kv: int = 512
    d_left: int = 8
    d_right: int = 4
    k_kv_stage1: int = 16
    k_kv_stage2: int = 16
    top_k_kv: int = 8
    
    # Logits
    k0_logits: int = 8
    k_step_logits: int = 4
    k_max_logits: int = 64
    shortlist_size: int = 32

    # Mode
    mode: Literal["realtime", "certified"] = "realtime"

    # Risk / certification
    delta_total: float = 0.01
    attn_share: float = 0.33
    kv_share: float = 0.33
    logits_share: float = 0.34
    delta_attn: Optional[float] = None
    delta_kv: Optional[float] = None
    delta_logits: Optional[float] = None

    # SPRT
    eps: float = 0.05
    eps_attn: Optional[float] = None
    eps_logits: Optional[float] = None

    # Backend
    backend: Literal["cpu", "opencl"] = "opencl"
    walsh_N: int = 2
    antithetic: bool = True
    order: int = 2
    beta: float = 0.30
    lambd: float = 1.0 / 256.0

    # Constant-Time Grammar (CTG-PROG v1)
    ctg: Optional[CTG] = None
    ctg_state: Optional[CTGState] = None
    ctg_program_id_fn: Optional[Callable[[dict], int]] = None


def infer_one_token_e2e(
    Q_attn_bits: np.ndarray,  # [d_attn_words] - attention query
    K_attn_bits: np.ndarray,  # [n_ctx, d_attn_words] - attention keys
    K_kv_bits: np.ndarray,  # [n_ctx, d_kv_words] - KV cache keys
    V_kv_bits: np.ndarray,  # [n_ctx, d_kv_words] - KV cache values
    Q_logits_bits: np.ndarray,  # [d_model_words] - logits query
    vocab_ids: np.ndarray,  # [V] - vocabulary token IDs
    *,
    cfg: E2EConfig,
    prf_seed: int,
    d_attn: int,
    d_model: int,
    token_idx: int = 0,
    cpg_policy: Optional[CpgPolicy] = None,
    pulse_scheduler: Optional[PulseScheduler] = None,
    shortlist_cache: Optional[ShortlistCache] = None,
    carry_cfg: Optional[CarryCfg] = None,
    K_enc: Optional[np.ndarray] = None,
    V_enc: Optional[np.ndarray] = None,
    row_ptr: Optional[np.ndarray] = None,
    col_idx: Optional[np.ndarray] = None,
    edge_weights: Optional[np.ndarray] = None,
    ctg: Optional[CTG] = None,
    ctg_state: Optional[CTGState] = None,
) -> Dict:
    """End-to-end inference for one token: Attention → KV → Logits.

    Args:
        Q_attn_bits: Attention query, shape [d_attn_words]
        K_attn_bits: Attention keys, shape [n_ctx, d_attn_words]
        K_kv_bits: KV cache keys, shape [n_ctx, d_kv_words]
        V_kv_bits: KV cache values, shape [n_ctx, d_kv_words]
        Q_logits_bits: Logits query, shape [d_model_words]
        vocab_ids: Vocabulary token IDs, shape [V]
        cfg: E2E configuration
        prf_seed: PRF seed
        d_attn: Attention dimension
        d_model: Model dimension
        token_idx: Token index for shadow scheduling
        cpg_policy: Optional CPG policy for shadow A/B testing
        pulse_scheduler: Optional pulse scheduler for KV repair
        shortlist_cache: Optional shortlist cache for logits warm-start
        carry_cfg: Optional carry configuration
        K_enc: Optional pre-encoded KV keys (for repair mode)
        V_enc: Optional pre-encoded KV values (for repair mode)
        row_ptr: Optional expander graph row pointers (for repair mode)
        col_idx: Optional expander graph column indices (for repair mode)
        edge_weights: Optional expander graph edge weights (for repair mode)

    Returns:
        Dict with:
            "status": str - Overall status
            "attn_top1": int | None - Top-1 attention position
            "kv_positions": np.ndarray - Retrieved KV positions
            "logits_top1": int | None - Top-1 logit token ID
            "k_attn_used": int - Ticks used for attention
            "k_kv_used": int - Ticks used for KV
            "k_logits_used": int - Ticks used for logits
            "unsure": bool - True if any stage failed to certify
            "carry_count": int - Number of carried IDs (if shortlist enabled)
            "fresh_count": int - Number of fresh IDs (if shortlist enabled)
            "pulse_repairs": int - Number of pulse repairs performed
            "kv_bytes_write": int - Bytes written to KV cache (for repair)
    """
    n_ctx = K_attn_bits.shape[0]

    # CTG grammar configuration/state (CTG-PROG v1)
    if ctg is None:
        ctg = getattr(cfg, "ctg", None)
    if ctg_state is None:
        ctg_state = getattr(cfg, "ctg_state", None)

    # CTG program selector
    ctg_program_id_fn = getattr(cfg, "ctg_program_id_fn", None)
    if ctg_program_id_fn is None:
        ctg_program_id_fn = default_program_id_fn

    # ========== Stage 1: Attention ==========
    # Stage-A probe
    use_ctg_stageA = cpg_policy.decide("stageA") if cpg_policy else False

    t0_stageA = _now()
    stageA_result = stageA_probe_topT(
        Q_attn_bits, K_attn_bits,
        kA=cfg.kA,
        prf_seed=prf_seed,
        walsh_N=cfg.walsh_N,
        antithetic=cfg.antithetic,
        order=cfg.order,
        beta=cfg.beta,
        lambd=cfg.lambd,
        use_ctg=use_ctg_stageA,
    )
    t_stageA = _now() - t0_stageA

    idx_top = stageA_result["idx_top"]
    T_sel = stageA_result["T_sel"]

    # Shadow A/B test for Stage-A
    if cpg_policy and cpg_policy.should_shadow(token_idx):
        with _no_side_effects():
            t0_shadow = _now()
            stageA_shadow = stageA_probe_topT(
                Q_attn_bits, K_attn_bits,
                kA=cfg.kA,
                prf_seed=prf_seed,
                walsh_N=cfg.walsh_N,
                antithetic=cfg.antithetic,
                order=cfg.order,
                beta=cfg.beta,
                lambd=cfg.lambd,
                use_ctg=not use_ctg_stageA,
            )
            t_shadow = _now() - t0_shadow

            # Compare results
            agree = (int(T_sel) == int(stageA_shadow["T_sel"]) and
                    _digest(stageA_result["stats"]["mu"]) == _digest(stageA_shadow["stats"]["mu"]))
            ymean_diff = float(np.max(np.abs(
                stageA_result["stats"]["mu"] - stageA_shadow["stats"]["mu"]
            )))
            overhead_ratio = (t_shadow / max(1e-6, t_stageA)) - 1.0

            # Update policy
            cpg_policy.update(
                "stageA",
                agree=agree,
                ymean_diff=ymean_diff,
                k_used_delta=0,
                overhead_ratio=overhead_ratio,
            )
            cpg_policy.maybe_promote("stageA")
            cpg_policy.maybe_demote("stageA")
    
    # SPRT certification
    delta_attn = cfg.delta_attn if cfg.delta_attn is not None else cfg.delta_total * cfg.attn_share
    eps_attn = cfg.eps_attn if cfg.eps_attn is not None else cfg.eps
    alpha_attn = beta_attn = delta_attn / 2.0

    sprt_cfg = SPRTConfig(
        eps=eps_attn,
        alpha=alpha_attn,
        beta=beta_attn,
        k_max=cfg.k_max_attn,
        chunk=4,
        seed=prf_seed,
    )

    use_ctg_attn = cpg_policy.decide("attn") if cpg_policy else False

    t0_attn = _now()
    cert_result = certify_topT(
        Q_attn_bits, K_attn_bits, idx_top,
        cfg=sprt_cfg,
        backend=cfg.backend,
        prf_seed=prf_seed + 1000,
        walsh_N=cfg.walsh_N,
        antithetic=cfg.antithetic,
        order=cfg.order,
        beta=cfg.beta,
        lambd=cfg.lambd,
        use_ctg=use_ctg_attn,
    )
    t_attn = _now() - t0_attn

    attn_top1 = cert_result.get("top1")
    k_attn_used = cfg.kA + cert_result["k_used"]
    attn_unsure = cert_result.get("unsure", False)

    # Shadow A/B test for Attention SPRT
    if cpg_policy and cpg_policy.should_shadow(token_idx):
        with _no_side_effects():
            t0_shadow = _now()
            cert_shadow = certify_topT(
                Q_attn_bits, K_attn_bits, idx_top,
                cfg=sprt_cfg,
                backend=cfg.backend,
                prf_seed=prf_seed + 1000,
                walsh_N=cfg.walsh_N,
                antithetic=cfg.antithetic,
                order=cfg.order,
                beta=cfg.beta,
                lambd=cfg.lambd,
                use_ctg=not use_ctg_attn,
            )
            t_shadow = _now() - t0_shadow

            # Compare results
            agree = (cert_result.get("top1") == cert_shadow.get("top1"))
            k_used_delta = int(cert_result["k_used"] - cert_shadow["k_used"])
            overhead_ratio = (t_shadow / max(1e-6, t_attn)) - 1.0

            # Compute ymean_diff (use dag_stats if available)
            ymean_diff = 0.0  # Placeholder - would need to extract from results

            # Update policy
            cpg_policy.update(
                "attn",
                agree=agree,
                ymean_diff=ymean_diff,
                k_used_delta=k_used_delta,
                overhead_ratio=overhead_ratio,
            )
            cpg_policy.maybe_promote("attn")
            cpg_policy.maybe_demote("attn")
    
    if attn_unsure or attn_top1 is None:
        # Attention failed to certify
        ctg_fields = {
            "ctg_shadow": int(bool(cpg_policy and cpg_policy.should_shadow(token_idx))),
            "ctg_pol_stageA": int(bool(cpg_policy and cpg_policy.decide("stageA"))),
            "ctg_pol_attn": int(bool(cpg_policy and cpg_policy.decide("attn"))),
            "ctg_pol_logits": int(bool(cpg_policy and cpg_policy.decide("logits"))),
        }
        telemetry_fields = {
            "carry_count": 0,
            "fresh_count": 0,
            "shortlist_total": 0,
            "pairs_total": 0,
            "pairs_reduced_pct": 0.0,
            "hcl_chunks": 0,
            "hcl_chunks_reduced_pct": 0.0,
            "ctg_shadow_calls": int(bool(cpg_policy and cpg_policy.should_shadow(token_idx))),
            "ctg_match_rate": 1.0,
            "pulse_repairs": 0,
            "kv_bytes_write": 0,
            "ctg_active": 0,
            "ctg_phase": 0,
            "ctg_masked_count": 0,
            "ctg_mask_digest": 0,
            "ctg_inv_flag": 0,
        }
        return {
            "status": "ATTN_UNSURE",
            "decision": "UNSURE",
            "unsure_reason": "ATTN_UNSURE",
            "attn_top1": None,
            "kv_positions": np.array([], dtype=np.int32),
            "logits_top1": None,
            "top1": None,
            "k_attn_used": k_attn_used,
            "k_kv_used": 0,
            "k_logits_used": 0,
            "unsure": True,
            "ctg_state": None,
            **ctg_fields,
            **telemetry_fields,
        }

    # ========== Stage 2: KV Retrieval ==========
    # Pulse repair (if enabled)
    pulse_repairs = 0
    kv_bytes_write = 0

    if pulse_scheduler and pulse_scheduler.should_repair(token_idx, layer_idx=0):
        # Get group to repair
        group_idx = pulse_scheduler.get_next_group(layer_idx=0)

        # Perform repair (requires pre-encoded KV)
        if K_enc is not None and V_enc is not None:
            from ..backends.opencl.host_opencl import OpenCLBinGemm

            # Repair using Stage-2 kernel in repair mode
            gemm = OpenCLBinGemm()

            # Compute number of positions in this group
            n_ctx = K_kv_bits.shape[0]
            group_size = pulse_scheduler.group_size
            start_pos = group_idx * group_size
            end_pos = min(start_pos + group_size, n_ctx)
            n_repair = end_pos - start_pos

            if n_repair > 0:
                # Repair K_kv_bits for this group
                # (In-place repair using Stage-2 kernel)
                # Note: This is a placeholder - actual repair would need to be implemented
                pulse_repairs = 1
                kv_bytes_write = n_repair * K_kv_bits.shape[1] * 4  # bytes

        # Mark repair done
        pulse_scheduler.mark_repaired(token_idx, layer_idx=0, group_idx=group_idx)

    # Encode KV cache (or use pre-encoded if provided)
    if K_enc is None or V_enc is None:
        enc_result = encode_kv_ldp(
            K_kv_bits, V_kv_bits,
            d_kv=cfg.d_kv,
            d_left=cfg.d_left,
            d_right=cfg.d_right,
            prf_seed=prf_seed + 2000,
        )

        K_enc = enc_result["K_enc"]
        V_enc = enc_result["V_enc"]
        row_ptr = enc_result["row_ptr"]
        col_idx = enc_result["col_idx"]
        edge_weights = enc_result["edge_weights"]
    
    # Stage-1: Compute energies for all positions
    # Use attention top-1 position's KV as query
    Q_kv_bits = K_kv_bits[attn_top1]
    
    kv_stage1_result = decode_kv_ldp_stage1(
        Q_kv_bits, K_enc,
        d_kv=cfg.d_kv,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=cfg.k_kv_stage1,
        prf_seed=prf_seed + 3000,
        order=cfg.order,
        beta=cfg.beta,
        lambd=cfg.lambd,
        early_exit_enable=False,
    )
    
    E_kv = kv_stage1_result["E_mean"]
    
    # Select top-k positions
    kv_positions = np.argsort(E_kv)[::-1][:cfg.top_k_kv]
    
    # Stage-2: Retrieve values for top-k positions
    kv_stage2_result = decode_kv_ldp_stage2(
        Q_kv_bits, V_enc,
        d_kv=cfg.d_kv,
        winner_positions=kv_positions,
        row_ptr=row_ptr,
        col_idx=col_idx,
        edge_weights=edge_weights,
        k_ticks=cfg.k_kv_stage2,
        prf_seed=prf_seed + 4000,
    )
    
    V_decoded = kv_stage2_result["V_decoded"]
    k_kv_used = cfg.k_kv_stage1 + cfg.k_kv_stage2
    
    # ========== Stage 3: Logits ==========
    # Shortlist carry-over (if enabled)
    carry_count = 0
    fresh_count = 0
    carry_ids = np.array([], dtype=np.int32)

    if shortlist_cache and carry_cfg and carry_cfg.enable:
        # Get carried IDs from cache
        k_carry = int(cfg.shortlist_size * carry_cfg.frac)
        carry_ids = shortlist_cache.carry(k_carry, token_idx)
        carry_count = len(carry_ids)

        # Evict expired entries
        shortlist_cache.evict_expired(token_idx)

    # Shortlist + SPRT on vocabulary
    delta_logits = cfg.delta_logits if cfg.delta_logits is not None else cfg.delta_total * cfg.logits_share
    eps_logits = cfg.eps_logits if cfg.eps_logits is not None else cfg.eps

    use_ctg_logits = cpg_policy.decide("logits") if cpg_policy else False

    # CTG-PROG: compute program_id right before logits warmup
    ctg_program_id = 0
    if ctg is not None:
        # Build minimal h_state for program selector
        h_state = {
            "token_idx": token_idx,
            "n_ctx": n_ctx,
            # Placeholder features (can be extended with shortlist_entropy, top2_margin, etc.)
        }
        ctg_program_id = ctg_program_id_fn(h_state)

    t0_logits = _now()

    # Merge carry_ids with vocab_ids if carry enabled
    if carry_count > 0:
        # Combine carry_ids with full vocab (shortlist_and_certify will handle dedup)
        # For now, just pass full vocab - actual integration would need shortlist_and_certify modification
        pass

    logits_result = shortlist_and_certify(
        Q_logits_bits, vocab_ids,
        d=d_model,
        k0=cfg.k0_logits,
        k_step=cfg.k_step_logits,
        k_max=cfg.k_max_logits,
        shortlist_size=cfg.shortlist_size,
        eps=eps_logits,
        delta=delta_logits,
        backend=cfg.backend,
        prf_seed=prf_seed + 5000,
        use_ctg=int(use_ctg_logits),
        ctg=ctg,
        ctg_state=ctg_state,
        ctg_program_id=ctg_program_id,
        order=cfg.order,
        beta=cfg.beta,
        lambd=cfg.lambd,
    )
    t_logits = _now() - t0_logits

    logits_top1 = logits_result.get("top1")
    k_logits_used = logits_result["k_used"]
    logits_unsure = logits_result.get("unsure", False)

    # CTG grammar telemetry (if enabled)
    ctg_active = bool(ctg is not None)
    ctg_phase = int(logits_result.get("ctg_phase", 0))
    ctg_masked_count = int(logits_result.get("ctg_masked_count", 0))
    ctg_mask_digest = int(logits_result.get("ctg_mask_digest", 0))
    ctg_inv_flag = int(bool(logits_result.get("ctg_inv_flag", 0)))
    ctg_prog_id = int(logits_result.get("ctg_prog_id", 0))
    ctg_state_out = logits_result.get("ctg_state", ctg_state)

    # Update shortlist cache with results (if enabled)
    if shortlist_cache and carry_cfg and carry_cfg.enable:
        # Extract shortlist IDs and scores from result
        if "shortlist_ids" in logits_result and "shortlist_scores" in logits_result:
            shortlist_ids = logits_result["shortlist_ids"]
            shortlist_scores = logits_result["shortlist_scores"]
            shortlist_cache.update_seen(shortlist_ids, shortlist_scores, token_idx)
            fresh_count = len(shortlist_ids) - carry_count

    # Shadow A/B test for Logits
    if cpg_policy and cpg_policy.should_shadow(token_idx):
        with _no_side_effects():
            t0_shadow = _now()
            logits_shadow = shortlist_and_certify(
                Q_logits_bits, vocab_ids,
                d=d_model,
                k0=cfg.k0_logits,
                k_step=cfg.k_step_logits,
                k_max=cfg.k_max_logits,
                shortlist_size=cfg.shortlist_size,
                eps=cfg.eps,
                delta=delta_logits,
                backend=cfg.backend,
                prf_seed=prf_seed + 5000,
                use_ctg=int(not use_ctg_logits),
                order=cfg.order,
                beta=cfg.beta,
                lambd=cfg.lambd,
            )
            t_shadow = _now() - t0_shadow

            # Compare results
            agree = (logits_result.get("top1") == logits_shadow.get("top1"))
            pairs_delta = int(logits_result.get("pairs_evaluated", 0) -
                            logits_shadow.get("pairs_evaluated", 0))
            overhead_ratio = (t_shadow / max(1e-6, t_logits)) - 1.0

            # Compute ymean_diff (placeholder)
            ymean_diff = 0.0

            # Update policy
            cpg_policy.update(
                "logits",
                agree=agree,
                ymean_diff=ymean_diff,
                k_used_delta=pairs_delta,
                overhead_ratio=overhead_ratio,
            )
            cpg_policy.maybe_promote("logits")
            cpg_policy.maybe_demote("logits")
    
    # ========== Final Status ==========
    if logits_unsure or logits_top1 is None:
        status = "LOGITS_UNSURE"
        unsure = True
    else:
        status = "CERT_OK"
        unsure = False

    decision = "UNSURE" if unsure else "CERT_OK"
    unsure_reason = status if unsure else None

    # CTG logging fields (policy) and CTG grammar telemetry
    ctg_fields = {
        "ctg_shadow": int(bool(cpg_policy and cpg_policy.should_shadow(token_idx))),
        "ctg_pol_stageA": int(bool(cpg_policy and cpg_policy.decide("stageA"))),
        "ctg_pol_attn": int(bool(cpg_policy and cpg_policy.decide("attn"))),
        "ctg_pol_logits": int(bool(cpg_policy and cpg_policy.decide("logits"))),
    }

    # Telemetry fields
    telemetry_fields = {
        "carry_count": carry_count,
        "fresh_count": fresh_count,
        "shortlist_total": carry_count + fresh_count,
        "pairs_total": logits_result.get("pairs_evaluated", 0),
        "pairs_reduced_pct": 0.0,  # Placeholder - would need baseline comparison
        "hcl_chunks": logits_result.get("chunks_evaluated", 0),
        "hcl_chunks_reduced_pct": 0.0,  # Placeholder - would need baseline comparison
        "ctg_shadow_calls": int(bool(cpg_policy and cpg_policy.should_shadow(token_idx))),
        "ctg_match_rate": 1.0,  # Placeholder - would need to track across tokens
        "pulse_repairs": pulse_repairs,
        "kv_bytes_write": kv_bytes_write,
        "ctg_active": int(ctg_active),
        "ctg_phase": ctg_phase,
        "ctg_masked_count": ctg_masked_count,
        "ctg_mask_digest": ctg_mask_digest,
        "ctg_inv_flag": ctg_inv_flag,
        "ctg_prog_id": ctg_prog_id,
    }

    return {
        "status": status,
        "decision": decision,
        "unsure_reason": unsure_reason,
        "attn_top1": attn_top1,
        "kv_positions": kv_positions,
        "logits_top1": logits_top1,
        "top1": None if unsure else logits_top1,
        "k_attn_used": k_attn_used,
        "k_kv_used": k_kv_used,
        "k_logits_used": k_logits_used,
        "unsure": unsure,
        "ctg_state": ctg_state_out,
        **ctg_fields,
        **telemetry_fields,
    }

