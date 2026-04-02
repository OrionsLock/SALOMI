"""Logits SPRT: shortlist + certification for HCL logits."""
from __future__ import annotations

from typing import Literal
import numpy as np

from ..attn.sprt_dag import SPRTDAG, SPRTConfig
from ..ops.hcl import hcl_energy_cpu


def shortlist_and_certify(
    q_bits: np.ndarray,
    v_ids: np.ndarray,
    *,
    d: int,
    k0: int,
    k_step: int,
    k_max: int,
    shortlist_size: int,
    eps: float,
    delta: float,
    backend: Literal["cpu", "opencl"] = "cpu",
    prf_seed: int = 0,
    use_ctg: int = 0,
    ctg=None,
    ctg_state=None,
    ctg_program_id: int = 0,
    order: int = 2,
    beta: float = 0.30,
    lambd: float = 1.0 / 256.0,
) -> dict:
    """Shortlist and certify Top-1 logit using SPRT-DAG.
    
    Process:
        1. Warmup k0 ticks with early_exit_enable=False
        2. Rank by E_mean, keep shortlist_size candidates
        3. Run chunked SPRT-DAG with k_step batches on shortlist
        4. Stop when Top-1 certified or k_max reached
    
    Args:
        q_bits: Query vector (packed bits), shape [d_words]
        v_ids: Candidate token IDs, shape [V]
        d: Dimension
        k0: Warmup ticks
        k_step: Ticks per SPRT chunk
        k_max: Max ticks for SPRT
        shortlist_size: Number of candidates to keep after warmup
        eps: SPRT effect size
        delta: SPRT delta (risk budget)
        backend: "cpu" or "opencl"
        prf_seed: PRF seed
        use_ctg: Enable CTG (0 or 1)
        order: SD modulator order
        beta: SD-2 beta parameter
        lambd: SD leak parameter
    
    Returns:
        Dict with:
            "top1": int | None - Top-1 token ID if certified
            "k_used": int - Total ticks used
            "shortlist": np.ndarray[int] - Shortlist token IDs
            "unsure": bool - True if exhausted k_max without certification
            "decisions": list - Decided edges from SPRT-DAG
    """
    V = len(v_ids)
    
    # Warmup: k0 ticks with early_exit_enable=False
    if backend == "cpu":
        warmup_result = hcl_energy_cpu(
            q_bits, v_ids,
            d=d, k=k0,
            use_ctg=use_ctg,
            prf_seed=prf_seed,
            early_exit_enable=False,
            eps=0.0,
            delta=1e-3,
            order=order,
            beta=beta,
            lambd=lambd,
        )
    else:  # opencl
        from ..backends.opencl.host_opencl import OpenCLBinGemm
        gemm = OpenCLBinGemm()

        # Kernel selection: use tiled for warmup (no early-exit) if shapes are large
        use_tiled_warmup = (V >= 128 or d >= 2048) and k0 >= 8

        if use_tiled_warmup:
            warmup_result = gemm.run_hcl_tiled(
                q_bits, v_ids,
                d=d, T=k0,
                use_ctg=(use_ctg == 1),
                prf_seed=prf_seed,
                early_exit_enable=False,
                eps=0.0,
                delta=1e-3,
                order=order,
                beta=beta,
                lambd=lambd,
            )
        else:
            warmup_result = gemm.run_hcl_naive(
                q_bits, v_ids,
                d=d, T=k0,
                use_ctg=(use_ctg == 1),
                prf_seed=prf_seed,
                early_exit_enable=False,
                eps=0.0,
                delta=1e-3,
                order=order,
                beta=beta,
                lambd=lambd,
            )
    
    E_warmup = warmup_result["E_mean"]

    # Rank and select shortlist
    # Higher energy = better (more positive)
    idx_sorted = np.argsort(E_warmup)[::-1]  # Descending order
    shortlist_idx = idx_sorted[:shortlist_size]
    shortlist_v_ids = v_ids[shortlist_idx]

    # CTG-PROG v1: apply Constant-Time Grammar on shortlist (if enabled)
    ctg_active = False
    ctg_phase = 0
    ctg_masked_count = 0
    ctg_mask_digest = 0
    ctg_inv_flag = 0
    ctg_prog_id = int(ctg_program_id)
    ctg_state_out = ctg_state

    if ctg is not None:
        if ctg_state is None:
            raise ValueError("ctg_state must be provided when ctg is enabled")
        ctg_state_out, mask, invert_flag = ctg.apply(ctg_state, shortlist_v_ids, program_id=ctg_prog_id)
        mask = mask.astype(bool)
        original_count = int(shortlist_v_ids.size)
        kept_count = int(mask.sum())
        ctg_masked_count = original_count - kept_count
        shortlist_v_ids = shortlist_v_ids[mask]
        ctg_active = True
        ctg_phase = int(ctg_state_out.phase)
        ctg_mask_digest = int(ctg_state_out.mask_digest)
        ctg_inv_flag = int(bool(invert_flag))

        # If CTG masks out all candidates, return UNSURE early
        if shortlist_v_ids.size == 0:
            pairs_eval = int(len(v_ids) * warmup_result.get("k_used", k0))
            return {
                "top1": None,
                "k_used": k0,
                "shortlist": shortlist_v_ids,
                "unsure": True,
                "decisions": [],
                "pairs_evaluated": pairs_eval,
                "ctg_state": ctg_state_out,
                "ctg_active": ctg_active,
                "ctg_phase": ctg_phase,
                "ctg_masked_count": ctg_masked_count,
                "ctg_mask_digest": ctg_mask_digest,
                "ctg_inv_flag": ctg_inv_flag,
                "ctg_prog_id": ctg_prog_id,
            }

    # SPRT-DAG on shortlist
    T_short = len(shortlist_v_ids)

    # Allocate risk budget
    alpha = beta_sprt = delta / 2.0
    
    sprt_cfg = SPRTConfig(
        eps=eps,
        alpha=alpha,
        beta=beta_sprt,
        k_max=k_max - k0,  # Remaining budget
        chunk=k_step,
        seed=prf_seed,
    )
    
    dag = SPRTDAG(T_short, sprt_cfg)

    # Track work: warmup pairs + SPRT pairs
    k_sprt_used = 0
    pairs_evaluated = int(len(v_ids) * warmup_result.get("k_used", k0))
    sign = -1.0 if ctg_inv_flag else 1.0

    # Chunked SPRT loop
    while k_sprt_used < sprt_cfg.k_max:
        chunk_size = min(k_step, sprt_cfg.k_max - k_sprt_used)

        # Run chunk
        if backend == "cpu":
            chunk_result = hcl_energy_cpu(
                q_bits, shortlist_v_ids,
                d=d, k=chunk_size,
                use_ctg=use_ctg,
                prf_seed=prf_seed + k0 + k_sprt_used,
                early_exit_enable=True,
                eps=eps,
                delta=delta,
                order=order,
                beta=beta,
                lambd=lambd,
            )
        else:  # opencl
            chunk_result = gemm.run_hcl_naive(
                q_bits, shortlist_v_ids,
                d=d, T=chunk_size,
                use_ctg=(use_ctg == 1),
                prf_seed=prf_seed + k0 + k_sprt_used,
                early_exit_enable=True,
                eps=eps,
                delta=delta,
                order=order,
                beta=beta,
                lambd=lambd,
            )

        # Effective ticks used in this chunk (fallback to requested size)
        k_chunk_used = int(chunk_result.get("k_used", chunk_size))
        if k_chunk_used <= 0:
            k_chunk_used = chunk_size

        E_chunk = chunk_result["E_mean"] * sign

        # Update DAG with chunk means (treat as single tick observation)
        dag.update_pairs_from_tick(E_chunk)

        k_sprt_used += k_chunk_used
        pairs_evaluated += int(len(shortlist_v_ids) * k_chunk_used)

        # Check for Top-1 certification
        top1_local = dag.top1_if_certified()
        if top1_local is not None:
            # Certified!
            top1_v_id = shortlist_v_ids[top1_local]
            return {
                "top1": int(top1_v_id),
                "k_used": k0 + k_sprt_used,
                "shortlist": shortlist_v_ids,
                "unsure": False,
                "decisions": dag.decided_edges(),
                "pairs_evaluated": pairs_evaluated,
                "ctg_state": ctg_state_out,
                "ctg_active": ctg_active,
                "ctg_phase": ctg_phase,
                "ctg_masked_count": ctg_masked_count,
                "ctg_mask_digest": ctg_mask_digest,
                "ctg_inv_flag": ctg_inv_flag,
                "ctg_prog_id": ctg_prog_id,
            }

        # Check if all pairs decided
        if dag.all_pairs_decided():
            break

    # Exhausted k_max without certification
    return {
        "top1": None,
        "k_used": k0 + k_sprt_used,
        "shortlist": shortlist_v_ids,
        "unsure": True,
        "decisions": dag.decided_edges(),
        "pairs_evaluated": pairs_evaluated,
        "ctg_state": ctg_state_out,
        "ctg_active": ctg_active,
        "ctg_phase": ctg_phase,
        "ctg_masked_count": ctg_masked_count,
        "ctg_mask_digest": ctg_mask_digest,
        "ctg_inv_flag": ctg_inv_flag,
        "ctg_prog_id": ctg_prog_id,
    }

