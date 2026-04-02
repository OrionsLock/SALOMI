"""LDP-KV: Locally Differentially Private Key-Value cache with expander graph encoding."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from ..core.prf import splitmix64, splitmix32
from ..core.packbits import pack_input_signs


def build_expander_csr(
    n_pos: int,
    d_left: int,
    d_right: int,
    prf_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build expander graph in CSR format for encoding.
    
    Each left node (position) connects to d_left right nodes (dimensions).
    Each right node connects to d_right left nodes.
    
    Args:
        n_pos: Number of positions (left nodes)
        d_left: Degree of left nodes (edges per position)
        d_right: Degree of right nodes (edges per dimension)
        prf_seed: PRF seed for deterministic graph generation
    
    Returns:
        Tuple of (row_ptr, col_idx, edge_weights):
            row_ptr: [n_pos + 1] - CSR row pointers
            col_idx: [n_pos * d_left] - Column indices (dimension IDs)
            edge_weights: [n_pos * d_left] - Edge weights (±1)
    """
    # Total edges
    n_edges = n_pos * d_left
    
    # Compute dimension count from degree constraint
    # n_pos * d_left = d_dim * d_right
    d_dim = (n_pos * d_left) // d_right
    
    # CSR arrays
    row_ptr = np.arange(n_pos + 1, dtype=np.int32) * d_left
    col_idx = np.zeros(n_edges, dtype=np.int32)
    edge_weights = np.zeros(n_edges, dtype=np.int8)
    
    # Track degree of each right node
    right_degree = np.zeros(d_dim, dtype=np.int32)
    
    # Generate edges deterministically
    rng_state = np.uint64(prf_seed)
    
    for pos in range(n_pos):
        edge_start = pos * d_left
        
        for e in range(d_left):
            # Find a dimension with available degree
            attempts = 0
            while attempts < 1000:
                rng_state, r = splitmix64(rng_state)
                dim_id = int(r % d_dim)
                
                if right_degree[dim_id] < d_right:
                    # Assign edge
                    col_idx[edge_start + e] = dim_id
                    
                    # Random sign
                    rng_state, r_sign = splitmix64(rng_state)
                    edge_weights[edge_start + e] = 1 if (r_sign & 1) == 0 else -1
                    
                    right_degree[dim_id] += 1
                    break
                
                attempts += 1
            
            if attempts >= 1000:
                raise RuntimeError(f"Failed to find available dimension for pos={pos}, edge={e}")
    
    return row_ptr, col_idx, edge_weights


def encode_kv_ldp(
    K_bits: np.ndarray,  # [n_pos, d_kv_words] - packed key bits
    V_bits: np.ndarray,  # [n_pos, d_kv_words] - packed value bits
    *,
    d_kv: int,  # Dimension of K/V
    d_left: int,  # Expander left degree
    d_right: int,  # Expander right degree
    prf_seed: int,
) -> dict:
    """Encode K/V cache using expander graph.
    
    Args:
        K_bits: Key bits, shape [n_pos, d_kv_words]
        V_bits: Value bits, shape [n_pos, d_kv_words]
        d_kv: Dimension of K/V
        d_left: Expander left degree
        d_right: Expander right degree
        prf_seed: PRF seed
    
    Returns:
        Dict with:
            "K_enc": [d_dim] - Encoded keys (float32)
            "V_enc": [d_dim] - Encoded values (float32)
            "row_ptr": [n_pos + 1] - CSR row pointers
            "col_idx": [n_pos * d_left] - Column indices
            "edge_weights": [n_pos * d_left] - Edge weights
            "d_dim": int - Encoded dimension
    """
    n_pos = K_bits.shape[0]
    d_kv_words = (d_kv + 31) // 32
    
    # Build expander graph
    row_ptr, col_idx, edge_weights = build_expander_csr(n_pos, d_left, d_right, prf_seed)
    
    d_dim = (n_pos * d_left) // d_right
    
    # Encode K and V
    K_enc = np.zeros(d_dim, dtype=np.float32)
    V_enc = np.zeros(d_dim, dtype=np.float32)
    
    for pos in range(n_pos):
        # Unpack K/V bits to ±1
        k_vec = np.zeros(d_kv, dtype=np.int8)
        v_vec = np.zeros(d_kv, dtype=np.int8)
        
        for j in range(d_kv):
            word_idx = j // 32
            bit_idx = j % 32
            
            k_bit = (K_bits[pos, word_idx] >> bit_idx) & 1
            v_bit = (V_bits[pos, word_idx] >> bit_idx) & 1
            
            k_vec[j] = 1 if k_bit == 1 else -1
            v_vec[j] = 1 if v_bit == 1 else -1
        
        # Aggregate over edges
        edge_start = row_ptr[pos]
        edge_end = row_ptr[pos + 1]
        
        for e in range(edge_start, edge_end):
            dim_id = col_idx[e]
            weight = edge_weights[e]
            
            # Simple aggregation: sum over all d_kv dimensions
            k_contrib = weight * np.sum(k_vec) / d_kv
            v_contrib = weight * np.sum(v_vec) / d_kv
            
            K_enc[dim_id] += k_contrib
            V_enc[dim_id] += v_contrib
    
    return {
        "K_enc": K_enc,
        "V_enc": V_enc,
        "row_ptr": row_ptr,
        "col_idx": col_idx,
        "edge_weights": edge_weights,
        "d_dim": d_dim,
    }


def decode_kv_ldp_stage1(
    Q_bits: np.ndarray,  # [d_kv_words] - query bits
    K_enc: np.ndarray,  # [d_dim] - encoded keys
    *,
    d_kv: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weights: np.ndarray,
    k_ticks: int,
    prf_seed: int,
    use_ctg: int = 0,
    order: int = 2,
    beta: float = 0.30,
    lambd: float = 1.0 / 256.0,
    early_exit_enable: bool = False,
) -> dict:
    """Stage-1 decode: compute energies for all positions.
    
    Args:
        Q_bits: Query bits, shape [d_kv_words]
        K_enc: Encoded keys, shape [d_dim]
        d_kv: Dimension of K/V
        row_ptr: CSR row pointers
        col_idx: Column indices
        edge_weights: Edge weights
        k_ticks: Number of ticks
        prf_seed: PRF seed
        use_ctg: Enable CTG
        order: ΣΔ order
        beta: ΣΔ-2 beta
        lambd: ΣΔ leak
        early_exit_enable: Enable early-exit
    
    Returns:
        Dict with:
            "E_mean": [n_pos] - Mean energies
            "k_used": int - Ticks used
    """
    n_pos = len(row_ptr) - 1
    d_kv_words = (d_kv + 31) // 32
    
    # Unpack Q to ±1
    q_vec = np.zeros(d_kv, dtype=np.int8)
    for j in range(d_kv):
        word_idx = j // 32
        bit_idx = j % 32
        bit_val = (Q_bits[word_idx] >> bit_idx) & 1
        q_vec[j] = 1 if bit_val == 1 else -1
    
    # Initialize ΣΔ states
    E1 = np.zeros(n_pos, dtype=np.float32)
    E2 = np.zeros(n_pos, dtype=np.float32)
    y_sum = np.zeros(n_pos, dtype=np.float64)
    
    k_used = 0
    
    for t in range(k_ticks):
        for pos in range(n_pos):
            # Compute energy via expander graph
            edge_start = row_ptr[pos]
            edge_end = row_ptr[pos + 1]
            
            u = 0.0
            for e in range(edge_start, edge_end):
                dim_id = col_idx[e]
                weight = edge_weights[e]
                
                # Contribution from this edge
                u += weight * K_enc[dim_id]
            
            # Normalize
            u = u / (edge_end - edge_start)
            
            # ΣΔ modulation
            if order == 1:
                y = 1.0 if (u + E1[pos]) >= 0 else -1.0
                E1[pos] += lambd * (u - y)
            else:  # order == 2
                e1_next = E1[pos] + u
                e1_clamped = np.clip(e1_next, -4.0, 4.0)
                
                e2_next = E2[pos] + e1_clamped
                e2_clamped = np.clip(e2_next, -8.0, 8.0)
                
                y = 1.0 if e2_clamped >= 0 else -1.0
                
                E1[pos] = e1_clamped + lambd * (u - y)
                E2[pos] = e2_clamped - y
            
            y_sum[pos] += y
        
        k_used += 1
        
        # Early-exit (simplified, no Hoeffding check for now)
        if early_exit_enable and t >= 8:
            break
    
    E_mean = (y_sum / k_used).astype(np.float32)
    
    return {
        "E_mean": E_mean,
        "k_used": k_used,
    }


def decode_kv_ldp_stage2(
    Q_bits: np.ndarray,
    V_enc: np.ndarray,
    *,
    d_kv: int,
    winner_positions: np.ndarray,  # [n_winners] - positions to decode
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weights: np.ndarray,
    k_ticks: int,
    prf_seed: int,
    repair_pass: bool = False,
    group_idx: Optional[int] = None,
    K_enc: Optional[np.ndarray] = None,
    K_bits_inout: Optional[np.ndarray] = None,
    group_size: int = 64,
) -> dict:
    """Stage-2 decode: retrieve values for winner positions.

    Args:
        Q_bits: Query bits (not used in this simple version)
        V_enc: Encoded values, shape [d_dim]
        d_kv: Dimension of K/V
        winner_positions: Positions to decode
        row_ptr: CSR row pointers
        col_idx: Column indices
        edge_weights: Edge weights
        k_ticks: Number of ticks
        prf_seed: PRF seed
        repair_pass: If True, perform in-place repair of KV group (PR-4.0)
        group_idx: Group index to repair (required if repair_pass=True)
        K_enc: Encoded keys (required if repair_pass=True)
        K_bits_inout: KV cache bits to repair in-place (required if repair_pass=True)
        group_size: Number of positions per group (default: 64)

    Returns:
        Dict with:
            "V_decoded": [n_winners, d_kv_words] - Decoded value bits
            "repaired": bool - True if repair was performed
    """
    n_winners = len(winner_positions)
    d_kv_words = (d_kv + 31) // 32

    V_decoded = np.zeros((n_winners, d_kv_words), dtype=np.uint32)
    repaired = False

    # PR-4.0: Repair pass - recompute winners for scheduled group
    if repair_pass:
        if group_idx is None or K_enc is None or K_bits_inout is None:
            raise ValueError("repair_pass requires group_idx, K_enc, and K_bits_inout")

        # Compute group boundaries
        n_pos = len(row_ptr) - 1
        n_groups = (n_pos + group_size - 1) // group_size

        if group_idx < 0 or group_idx >= n_groups:
            raise ValueError(f"group_idx {group_idx} out of range [0, {n_groups})")

        group_start = group_idx * group_size
        group_end = min(group_start + group_size, n_pos)

        # Recompute energies for this group using fresh measurements
        # (Same expander config, but fresh ΣΔ states)
        for pos in range(group_start, group_end):
            edge_start = row_ptr[pos]
            edge_end = row_ptr[pos + 1]

            # Compute energy via expander graph
            k_sum = 0.0
            for e in range(edge_start, edge_end):
                dim_id = col_idx[e]
                weight = edge_weights[e]
                k_sum += weight * K_enc[dim_id]

            # Normalize and threshold
            k_mean = k_sum / (edge_end - edge_start)

            # Overwrite K bits in-place (deterministic repair)
            for j in range(d_kv):
                word_idx = j // 32
                bit_idx = j % 32

                # Clear old bit
                K_bits_inout[pos, word_idx] &= ~np.uint32(1 << bit_idx)

                # Set new bit based on sign
                if k_mean >= 0:
                    K_bits_inout[pos, word_idx] |= np.uint32(1 << bit_idx)

        repaired = True

    # Standard Stage-2: Decode values for winner positions
    for i, pos in enumerate(winner_positions):
        # Decode value via expander graph
        edge_start = row_ptr[pos]
        edge_end = row_ptr[pos + 1]

        v_sum = 0.0
        for e in range(edge_start, edge_end):
            dim_id = col_idx[e]
            weight = edge_weights[e]

            v_sum += weight * V_enc[dim_id]

        # Normalize and threshold
        v_mean = v_sum / (edge_end - edge_start)

        # Simple thresholding: sign determines bit
        # (In practice, would use BSDM-W here too)
        for j in range(d_kv):
            word_idx = j // 32
            bit_idx = j % 32

            # For now, just use sign of v_mean
            if v_mean >= 0:
                V_decoded[i, word_idx] |= np.uint32(1 << bit_idx)

    return {
        "V_decoded": V_decoded,
        "repaired": repaired,
    }

