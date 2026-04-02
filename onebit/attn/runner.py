"""Runner for SPRT-based Top-T certification using BSDM-W backend."""
from __future__ import annotations

from typing import Literal, Dict
import numpy as np

from .sprt_dag import SPRTDAG, SPRTConfig
from ..core.prf import derive_seed


def certify_topT(
    Q_bits: np.ndarray,
    K_bits: np.ndarray,
    idx_top: np.ndarray,
    *,
    cfg: SPRTConfig,
    backend: Literal["cpu", "opencl"] = "opencl",
    prf_seed: int,
    walsh_N: int = 2,
    antithetic: bool = True,
    order: int = 2,
    beta: float = 0.30,
    lambd: float = 1.0 / 256.0,
    use_ctg: bool = False,
) -> Dict:
    """Certify Top-T ordering using SPRT-DAG with chunked BSDM-W calls.
    
    Loop chunks of ticks (cfg.chunk) with early-exit ON.
    At each tick:
      - Request BSDM-W normalized means y_i,t for i in idx_top
      - Call dag.update_pairs_from_tick(y_t)
      - If dag.top1_if_certified(): stop
      - Stop if all pairs decided or ticks == k_max
    
    Args:
        Q_bits: Query vector (packed bits)
        K_bits: Key matrix [K, Kw] (packed bits) for all keys
        idx_top: Indices of Top-T candidates from Stage-A
        cfg: SPRT configuration
        backend: "cpu" or "opencl"
        prf_seed: PRF seed for deterministic BSDM-W streams
        walsh_N: Walsh carriers per tick (default: 2)
        antithetic: Use antithetic pairs (default: True)
        order: ΣΔ modulator order (1 or 2)
        beta: ΣΔ-2 beta parameter
        lambd: ΣΔ leak parameter
        use_ctg: enable CTG (carrier toggle guard)
    
    Returns:
        Dict with:
            "decided": [(i,j,sign), ...] - decided edges
            "undecided": [(i,j), ...] - undecided pairs
            "top1": Optional[int] - Top-1 candidate if certified
            "k_used": int - total ticks used
            "pairs_evaluated": int - total pair observations
            "dag_stats": dict - DAG statistics
    """
    T = len(idx_top)
    dag = SPRTDAG(T, cfg)
    
    # Select Top-T keys
    K_top = K_bits[idx_top]  # [T, Kw]
    
    # Derive per-pair delta from cfg.alpha (symmetric alpha=beta)
    # For now, use cfg.alpha directly as per-pair budget
    delta_per_pair = cfg.alpha
    
    k_used = 0
    pairs_evaluated = 0
    
    # Import backend
    if backend == "opencl":
        from ..backends.opencl.host_opencl import OpenCLBinGemm
        gemm = OpenCLBinGemm()
    else:
        # CPU backend uses bsdm_w_dot directly
        from ..ops.bsdm_w import bsdm_w_dot, SDConfig
        from ..core.packbits import pack_input_signs
    
    # Chunked loop
    while k_used < cfg.k_max:
        # Determine chunk size (last chunk may be smaller)
        chunk_size = min(cfg.chunk, cfg.k_max - k_used)
        
        if backend == "opencl":
            # Prepare X_bits for this chunk (replicate Q_bits for each tick)
            X_bits = np.tile(Q_bits[None, :], (chunk_size, 1))  # [chunk_size, Kw]
            
            # Run BSDM-W on Top-T keys
            out = gemm.run_bsdm_w_naive_norm(
                K_top, X_bits, T=chunk_size,
                eps=cfg.eps, delta=delta_per_pair,
                order=order, beta=beta, lambd=lambd,
                walsh_N=walsh_N, antithetic=antithetic,
                use_ctg=use_ctg,
                prf_seed=prf_seed + k_used,  # Unique seed per chunk
                early_exit_enable=True,  # Early-exit ON for SPRT
                local_size=256,
                want_y_pack=False, want_pc32=False,
            )
            
            # Extract means for each tick
            # out["Y"] is [T] with final means, but we need per-tick means
            # For now, use final means as proxy (TODO: need per-tick output)
            # WORKAROUND: Run tick-by-tick for now
            y_means = np.zeros((chunk_size, T), dtype=np.float32)
            for t in range(chunk_size):
                X_t = Q_bits[None, :]  # [1, Kw]
                out_t = gemm.run_bsdm_w_naive_norm(
                    K_top, X_t, T=1,
                    eps=cfg.eps, delta=delta_per_pair,
                    order=order, beta=beta, lambd=lambd,
                    walsh_N=walsh_N, antithetic=antithetic,
                    use_ctg=use_ctg,
                    prf_seed=prf_seed + k_used + t,
                    early_exit_enable=True,
                    local_size=256,
                    want_y_pack=False, want_pc32=False,
                )
                y_means[t] = out_t["Y"]
        else:
            # CPU backend: run tick-by-tick
            y_means = np.zeros((chunk_size, T), dtype=np.float32)
            sd_cfg = SDConfig(order=order, beta=beta, lambd=lambd, walsh_N=walsh_N, antithetic=antithetic)
            
            for t in range(chunk_size):
                for i in range(T):
                    # Derive unique seed for this (key, tick)
                    seed_it = prf_seed + k_used + t + i * 1000
                    est, _ = bsdm_w_dot(
                        K_top[i], Q_bits, k=1,
                        cfg=sd_cfg, seed=seed_it,
                        eps=cfg.eps, delta=delta_per_pair,
                        early_exit_enable=True,
                        use_ctg=use_ctg,
                    )
                    y_means[t, i] = est
        
        # Update DAG with each tick
        for t in range(chunk_size):
            dag.update_pairs_from_tick(y_means[t])
            k_used += 1
            
            # Check stopping conditions
            top1 = dag.top1_if_certified()
            if top1 is not None:
                # Top-1 certified, stop early
                break
            
            if dag.all_pairs_decided():
                # All pairs decided, stop
                break
        
        # Check if we should stop
        top1 = dag.top1_if_certified()
        if top1 is not None or dag.all_pairs_decided():
            break
    
    # Collect results
    decided = dag.decided_edges()
    undecided = dag.undecided_pairs()
    top1 = dag.top1_if_certified()
    dag_stats = dag.stats()
    pairs_evaluated = dag_stats["total_observations"]
    
    return {
        "decided": decided,
        "undecided": undecided,
        "top1": top1,
        "k_used": k_used,
        "pairs_evaluated": pairs_evaluated,
        "dag_stats": dag_stats,
    }

