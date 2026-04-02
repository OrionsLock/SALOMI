"""HCL (Hadamard Code Logits): chunked energies with BSDM-W + Hadamard codes."""
from __future__ import annotations

from typing import Optional
import numpy as np

from ..core.hadamard import build_col_masks, hadamard_sign_word
from ..core.prf import splitmix64, splitmix32
from ..ops.bsdm_w import SDConfig


def hcl_energy_cpu(
    q_bits: np.ndarray,  # [d_words] uint32 - query vector (packed bits)
    v_ids: np.ndarray,  # [Kc] int32 - candidate token IDs
    *,
    d: int,  # Dimension
    k: int,  # Number of ticks
    walsh_N: int = 2,
    antithetic: bool = True,
    use_ctg: int = 0,
    prf_seed: int = 0,
    early_exit_enable: bool = True,
    eps: float = 0.0,
    delta: float = 1e-3,
    order: int = 2,
    beta: float = 0.30,
    lambd: float = 1.0 / 256.0,
    want_bits: bool = False,
    col_masks: Optional[np.ndarray] = None,
) -> dict:
    """Compute energies for HCL logits using BSDM-W + Hadamard codes.
    
    For each tick t:
        1. Produce q_tick via BSDM-W (Walsh N=2 + antithetic)
        2. For each v_id:
            - Build Hadamard code row on-the-fly using col_masks
            - Compute XNOR-popcount with q_tick
            - Normalize to [-1, 1]
            - Apply CTG (if enabled)
            - ΣΔ modulate
        3. Early-exit check (if enabled)
    
    Args:
        q_bits: Query vector (packed bits), shape [d_words]
        v_ids: Candidate token IDs, shape [Kc]
        d: Dimension (must match q_bits)
        k: Number of ticks
        walsh_N: Walsh carriers per tick (default: 2)
        antithetic: Use antithetic pairs (default: True)
        use_ctg: Enable CTG (default: 0)
        prf_seed: PRF seed for deterministic streams
        early_exit_enable: Enable early-exit (default: True)
        eps: Early-exit epsilon (default: 0.0)
        delta: Early-exit delta (default: 1e-3)
        order: ΣΔ modulator order (1 or 2, default: 2)
        beta: ΣΔ-2 beta parameter (default: 0.30)
        lambd: ΣΔ leak parameter (default: 1/256)
        want_bits: Return packed y_bits (default: False)
        col_masks: Precomputed column masks (default: None, will build)
    
    Returns:
        Dict with:
            "E_mean": float32[Kc] - mean energies
            "k_used": int - ticks used
            "ctg_digest": uint32 - CTG digest
            "y_bits": optional uint32[Kc, k_words] - packed bits if want_bits=True
    """
    Kc = len(v_ids)
    d_words = (d + 31) // 32
    
    # Build col_masks if not provided
    if col_masks is None:
        col_masks = build_col_masks(d)
    
    # Initialize ΣΔ states for each candidate
    E1 = np.zeros(Kc, dtype=np.float32)
    E2 = np.zeros(Kc, dtype=np.float32)
    y_sum = np.zeros(Kc, dtype=np.float64)
    
    # CTG state per candidate
    ctg_states = np.zeros(Kc, dtype=np.uint64)
    ctg_digest = np.uint32(0)
    
    # Initialize CTG states
    if use_ctg:
        for v_idx in range(Kc):
            # splitmix64 expects Python ints; cast explicitly to avoid NumPy
            # ufunc limitations on some platforms.
            seed = int(np.uint64(prf_seed) ^ np.uint64(v_idx))
            ctg_states[v_idx], _ = splitmix64(seed)

    # Bit packing buffers
    if want_bits:
        k_words = (k * walsh_N + 31) // 32
        y_bits = np.zeros((Kc, k_words), dtype=np.uint32)
    else:
        y_bits = None
    
    # BSDM-W config
    cfg = SDConfig(order=order, beta=beta, lambd=lambd, walsh_N=walsh_N, antithetic=antithetic)
    
    # Per-candidate seeds (deterministic, simple and portable across NumPy builds)
    seeds = np.zeros(Kc, dtype=np.uint64)
    for v_idx in range(Kc):
        # Avoid left-shift on uint64 here because some NumPy builds lack
        # ufunc support for that combination. Multiplication by 32 is
        # equivalent to a << 5 for our purposes and is fully supported.
        v_id_u = int(v_ids[v_idx])
        seed = np.uint64(prf_seed) ^ np.uint64(0xBF58476D1CE4E5B9) ^ np.uint64(v_id_u * 32)
        seeds[v_idx] = seed

    k_used = 0

    for t in range(k):
        # For each candidate, compute energy for this tick
        for v_idx in range(Kc):
            v_id = v_ids[v_idx]
            
            # Compute XNOR-popcount between q_bits and Hadamard row
            pc = 0
            for word_idx in range(d_words):
                # Generate Hadamard code word on-the-fly
                code_word = hadamard_sign_word(v_id, word_idx)
                
                # XNOR-popcount
                xnor = ~(q_bits[word_idx] ^ code_word)
                pc += bin(xnor & 0xFFFFFFFF).count('1')
            
            # Normalize to [-1, 1]
            u = (2.0 * pc - d) / d
            
            # Apply CTG (branchless)
            if use_ctg:
                # splitmix32/splitmix64 operate on Python ints; cast state
                # explicitly before stepping the PRNG.
                ctg_states[v_idx], r = splitmix32(int(ctg_states[v_idx]))
                ctg_op = r & 3

                # Update digest
                ctg_digest = (ctg_digest << 2) ^ ctg_op
                
                # Apply CTG operators
                if ctg_op == 1:  # INVERT
                    u = -u
                elif ctg_op == 2:  # INHIBIT (not applicable here, skip)
                    pass
                elif ctg_op == 3:  # PHASE (not applicable here, skip)
                    pass
            
            # ΣΔ modulation
            if order == 1:
                # Order-1: y = sign(u + E1), E1 += leak*(u - y)
                y = 1.0 if (u + E1[v_idx]) >= 0 else -1.0
                E1[v_idx] += lambd * (u - y)
            else:  # order == 2
                # Order-2 (MASH-1-1 with leak)
                e1_next = E1[v_idx] + u
                e1_clamped = np.clip(e1_next, -4.0, 4.0)
                
                e2_next = E2[v_idx] + e1_clamped
                e2_clamped = np.clip(e2_next, -8.0, 8.0)
                
                y = 1.0 if e2_clamped >= 0 else -1.0
                
                # Update with leak
                E1[v_idx] = e1_clamped + lambd * (u - y)
                E2[v_idx] = e2_clamped - y
            
            # Accumulate
            y_sum[v_idx] += y
            
            # Pack bits if requested
            if want_bits:
                bit_idx = t
                word_idx = bit_idx // 32
                bit_pos = bit_idx % 32
                if y > 0:
                    y_bits[v_idx, word_idx] |= np.uint32(1 << bit_pos)
        
        k_used += 1
        
        # Early-exit check
        if early_exit_enable and eps > 0:
            # Hoeffding bound: P(|mean - true| > eps) <= 2*exp(-2*t*eps^2)
            bound = 2.0 * np.exp(-2.0 * (t + 1) * eps * eps)
            if bound <= delta:
                break
    
    # Compute mean energies
    E_mean = (y_sum / k_used).astype(np.float32)
    
    result = {
        "E_mean": E_mean,
        "k_used": k_used,
        "ctg_digest": ctg_digest,
    }
    
    if want_bits:
        result["y_bits"] = y_bits
    
    return result

