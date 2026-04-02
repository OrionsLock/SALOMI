"""Stage-A probe for attention: fixed kA ticks, no early-exit, dynamic T selection.

Computes mean scores for all keys using BSDM-W, then selects Top-T via elbow detection.
"""
from __future__ import annotations
from typing import Literal
import numpy as np

from onebit.ops.bsdm_w import bsdm_w_dot, SDConfig
from onebit.core.elbow import compute_elbow


def stageA_probe_topT(
    Q_bits: np.ndarray,  # [Kw] uint32 - single query vector (bit-packed)
    K_bits: np.ndarray,  # [L, Kw] uint32 - all key vectors (bit-packed)
    *,
    kA: int = 16,
    T_set: tuple[int, ...] = (8, 12, 16),
    prf_seed: int,
    walsh_N: int = 2,
    antithetic: bool = True,
    order: int = 2,
    beta: float = 0.30,
    lambd: float = 1.0/256.0,
    use_ctg: bool = False,
) -> dict:
    """Stage-A probe: compute means for all keys with fixed kA ticks, select Top-T.

    No early-exit during probe (eps=0 forces full kA ticks).
    Deterministic via prf_seed.

    Args:
        Q_bits: query vector [Kw] uint32
        K_bits: key matrix [L, Kw] uint32
        kA: fixed number of ticks (no early-exit)
        T_set: allowed T values for selection
        prf_seed: PRF seed for determinism
        walsh_N: Walsh carriers per tick
        antithetic: use antithetic pairs
        order: ΣΔ order (1 or 2)
        beta: ΣΔ-2 beta parameter
        lambd: ΣΔ leak parameter
        use_ctg: enable CTG (carrier toggle guard)
        
    Returns:
        dict with keys:
            T_sel: int - selected T from T_set
            idx_top: np.ndarray[int] - indices of top T_sel keys (into K_bits)
            stats: dict with:
                mu: np.ndarray[float32] - means for all L keys
                gap12: float - elbow gap (mu[j*] - mu[j*+1])
                elbow_T_raw: float - raw elbow position before mapping
                teff: int - effective ticks used (should be kA)
    """
    Q_bits = np.asarray(Q_bits, dtype=np.uint32)
    K_bits = np.asarray(K_bits, dtype=np.uint32)
    
    if Q_bits.ndim != 1:
        raise ValueError("Q_bits must be 1D [Kw]")
    if K_bits.ndim != 2:
        raise ValueError("K_bits must be 2D [L, Kw]")
    if K_bits.shape[1] != Q_bits.shape[0]:
        raise ValueError("K_bits second dimension must match Q_bits length")
    
    L = K_bits.shape[0]
    Kw = K_bits.shape[1]
    
    # BSDM-W config
    cfg = SDConfig(
        order=order,
        beta=beta,
        lambd=lambd,
        walsh_N=walsh_N,
        antithetic=antithetic,
    )
    
    # Compute mean for each key with fixed kA ticks (no early-exit)
    # Use eps=0, delta=1e-9 to disable early-exit (will run full kA ticks)
    mu = np.zeros(L, dtype=np.float32)
    teff_arr = np.zeros(L, dtype=np.int32)
    
    for i in range(L):
        # Derive per-key seed from prf_seed and key index
        key_seed = (prf_seed + i) & 0xFFFFFFFFFFFFFFFF
        
        # Run BSDM-W with early_exit_enable=False to force full kA ticks
        est, diags = bsdm_w_dot(
            Q_bits, K_bits[i], kA, cfg,
            seed=key_seed,
            want_pc32=False,
            eps=0.0,
            delta=1e-9,
            early_exit_enable=False,  # force full kA ticks
            use_ctg=use_ctg,
        )
        mu[i] = est
        teff_arr[i] = diags["k_used"]
    
    # Verify no early-exit occurred
    assert np.all(teff_arr == kA), f"Early-exit occurred despite eps=0: teff={teff_arr}"
    
    # Elbow detection
    T_sel, gap = compute_elbow(mu, T_set)
    
    # Get top T_sel indices
    idx_sorted = np.argsort(mu)[::-1]  # descending order
    idx_top = idx_sorted[:T_sel]
    
    # Compute raw elbow position for logging
    mu_sorted = np.sort(mu)[::-1]
    if len(mu_sorted) > 1:
        diffs = mu_sorted[:-1] - mu_sorted[1:]
        n_top = min(64, len(diffs) - 1)
        if n_top > 1:
            second_diffs = diffs[:n_top-1] - diffs[1:n_top]
            j_star = int(np.argmax(second_diffs))
            elbow_T_raw = float(j_star + 1)
        else:
            elbow_T_raw = 1.0
    else:
        elbow_T_raw = 1.0
    
    return {
        "T_sel": T_sel,
        "idx_top": idx_top,
        "stats": {
            "mu": mu,
            "gap12": gap,
            "elbow_T_raw": elbow_T_raw,
            "teff": kA,
        },
    }

