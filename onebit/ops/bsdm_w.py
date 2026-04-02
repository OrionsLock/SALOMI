from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Union

import numpy as np

from onebit.core.packbits import pack_input_signs
from .walsh import walsh_carrier_bit


@dataclass
class SDConfig:
    order: int = 2           # 1 or 2
    beta: float = 0.30       # only for order=2 (default per PR-1)
    lambd: float = 1.0/256.0 # leak (lambda)
    walsh_N: int = 2         # carriers per tick
    antithetic: bool = True  # if True, include sign-flipped carriers


def _xnor_popcount_dot(a_bits: np.ndarray, b_bits: np.ndarray) -> int:
    """Integer-only XNOR-POPCNT dot for packed +/-1 vectors.

    a_bits, b_bits: uint32 arrays of shape [Kw].
    Returns integer dot in the +/-1 domain: sum_i a_i*b_i in [-K, K].
    """
    if a_bits.dtype != np.uint32 or b_bits.dtype != np.uint32:
        raise TypeError("a_bits and b_bits must be uint32")
    if a_bits.ndim != 1 or b_bits.ndim != 1 or a_bits.shape != b_bits.shape:
        raise ValueError("a_bits and b_bits must be 1D and same shape [Kw]")
    xnor = np.bitwise_not(np.bitwise_xor(a_bits, b_bits))
    # Use uint64 accumulator to avoid overflow on large K
    pc = int(np.uint64(0))
    # NumPy popcount: use vectorized bit_count where available; fall back to loop
    # Here loop across words with Python int.bit_count for portability.
    for word in xnor.tolist():
        pc += int(int(word).bit_count())
    Kbits = int(a_bits.shape[0]) * 32
    return (pc << 1) - Kbits


def _ensure_bits(arr: np.ndarray) -> np.ndarray:
    """Ensure a 1D uint32 packed bit vector from +/-1 float/int array or already-packed."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint32 and arr.ndim == 1:
        return arr
    # Accept 1D floats/ints; pack by sign
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return pack_input_signs(arr.astype(np.float32))


def _sd1_tick(u: float, e: float, lambd: float, E1: float) -> Tuple[int, float]:
    """Sigma-Delta order-1 with leak and clamp in normalized domain.

    Update: y = sign(u+e); e <- clamp(e + u - y - lambd*e, [-E1, E1])
    """
    v = e + float(u)
    y = 1 if v >= 0.0 else -1
    e_next = e + float(u) - float(y) - float(lambd) * e
    if e_next > E1:
        e_next = E1
    elif e_next < -E1:
        e_next = -E1
    return y, float(e_next)


def _sd2_tick(u: float, e1: float, e2: float, beta: float, lambd: float, E1: float, E2: float) -> Tuple[int, float, float]:
    """Sigma-Delta order-2 (MASH-1-1) with leak and clamps.

    Tick t:
      v  = u + e1 + e2
      y  = sign(v)
      e1 = clamp(e1 + u - y - lambd*e1, [-E1, E1])
      e1 = max(-E1, min(E1, e1))
      e2 = e2 + e1 - float(lambd) * e2 + float(beta) * (e1 - const_e1_prev)
      e2 = max(-E2, min(E2, e2))
    """
    float_u = float(u)
    v = e2 + e1 + float_u
    y = 1 if v >= 0.0 else -1
    const_e1_prev = e1
    # update e1 with leak and clamp
    e1 = e1 + float_u - float(y) - float(lambd) * e1
    if e1 > E1:
        e1 = E1
    elif e1 < -E1:
        e1 = -E1
    # update e2 with leak, MASH correction and clamp
    e2 = e2 + e1 - float(lambd) * e2 + float(beta) * (e1 - const_e1_prev)
    if e2 > E2:
        e2 = E2
    elif e2 < -E2:
        e2 = -E2
    return y, float(e1), float(e2)


def bsdm_w_dot(a_bits: np.ndarray, b_bits: np.ndarray, k: int, cfg: SDConfig, seed: int | None,
               want_pc32: bool = False, eps: float = 0.05, delta: float = 0.001,
               early_exit_enable: bool = True, use_ctg: bool = False, instr_on: bool = False,
               want_y_pack: bool = False) -> Tuple[float, Dict]:
    """BSDM-W dot-product estimator using Walsh carriers and ΣΔ modulators in normalized domain.
    
    [Same as before, skipping body for brevity as I only need to append bsdm_w_matmul]
    """
    # ... [Full body from previous read] ...
    # Wait, I cannot skip body in 'write'. I must provide full content.
    # I will paste the original bsdm_w_dot body here.
    
    if k <= 0:
        raise ValueError("k must be > 0")

    a_bits = _ensure_bits(a_bits)
    b_bits = _ensure_bits(b_bits)
    if a_bits.shape != b_bits.shape:
        raise ValueError("a_bits and b_bits shapes must match [Kw]")

    N = int(cfg.walsh_N)
    use_anti = bool(cfg.antithetic)

    # States per carrier: separate main and twin channels
    # Use float32 for CPU/OpenCL parity (OpenCL uses float)
    if cfg.order == 1:
        e_main = np.zeros((N,), dtype=np.float32)
        e_twin = np.zeros((N,), dtype=np.float32) if use_anti else None
    elif cfg.order == 2:
        e1_main = np.zeros((N,), dtype=np.float32)
        e2_main = np.zeros((N,), dtype=np.float32)
        e1_twin = np.zeros((N,), dtype=np.float32) if use_anti else None
        e2_twin = np.zeros((N,), dtype=np.float32) if use_anti else None
    else:
        raise ValueError("SDConfig.order must be 1 or 2")

    # Seeded dithering via randomized initial states (stabilizes variance tests)
    # Use SplitMix64-based uniform_half for CPU/OpenCL parity
    if seed is not None:
        from onebit.core.prf import uniform_half
        init_state = int(seed) & 0xFFFFFFFFFFFFFFFF
        if cfg.order == 1:
            for i in range(N):
                init_state, val = uniform_half(init_state)
                e_main[i] += val
            if use_anti:
                for i in range(N):
                    init_state, val = uniform_half(init_state)
                    e_twin[i] += val
        else:
            for i in range(N):
                init_state, val = uniform_half(init_state)
                e1_main[i] += val
            for i in range(N):
                init_state, val = uniform_half(init_state)
                e2_main[i] += val
            if use_anti:
                for i in range(N):
                    init_state, val = uniform_half(init_state)
                    e1_twin[i] += val
                for i in range(N):
                    init_state, val = uniform_half(init_state)
                    e2_twin[i] += val

    # CTG state (SplitMix64 for determinism)
    ctg_state = int(seed) & 0xFFFFFFFFFFFFFFFF if (use_ctg and seed is not None) else 0
    ctg_digest = np.uint32(0)  # Rolling hash of CTG ops

    # Allocate bit arrays: k ticks × N carriers per channel
    samples_per_channel = k * N
    num_words_per_channel = (samples_per_channel + 31) // 32
    y_bits_main = np.zeros(num_words_per_channel, dtype=np.uint32)
    y_bits_twin = np.zeros(num_words_per_channel, dtype=np.uint32) if use_anti else None

    # Optional pc32 arrays
    pc32_main_list = [] if want_pc32 else None
    pc32_twin_list = [] if want_pc32 else None

    # Energy instrumentation (PR-3.7)
    if instr_on:
        energy_toggles_y_main = np.uint64(0)
        energy_toggles_y_twin = np.uint64(0)
        energy_ones_pc = np.uint64(0)
        energy_zeros_pc = np.uint64(0)
        energy_xnor_ops = np.uint64(0)
        energy_popcnt_ops = np.uint64(0)
        energy_bytes_W = np.uint64(0)
        energy_bytes_X = np.uint64(0)
        prev_y_main = [0] * N  # Previous y bits per carrier
        prev_y_twin = [0] * N if use_anti else None

    # Accumulate y_bar for early-exit (mean of all samples per tick)
    acc_ybar = 0.0
    k_used = 0

    # Normalize by total bit-length so u in [-1, 1]
    Kbits = int(a_bits.shape[0]) * 32

    for t in range(k):
        # Base integer dot (no carrier), computed once per tick
        base_dot = _xnor_popcount_dot(a_bits, b_bits)
        base_u = float(base_dot) / float(Kbits)

        if want_pc32:
            # base_dot = 2*pc - Kbits, so pc = (base_dot + Kbits) // 2
            pc = (base_dot + Kbits) // 2
            pc32_main_list.append(pc)
            if use_anti:
                pc32_twin_list.append(pc)

        # Energy: accumulate popcount stats (PR-3.7)
        if instr_on:
            pc = (base_dot + Kbits) // 2
            energy_ones_pc += np.uint64(pc)
            energy_zeros_pc += np.uint64(Kbits - pc)
            Kw = int(a_bits.shape[0])
            energy_xnor_ops += np.uint64(Kw)
            energy_popcnt_ops += np.uint64(Kw)
            energy_bytes_W += np.uint64(4 * Kw)
            energy_bytes_X += np.uint64(4 * Kw)

        # CTG (Constant-Time Grammar) - deterministic procedural transforms
        # Applied per-tick, after carrier modulation, before ΣΔ updates
        # Ops: PASS(00), INVERT(01), INHIBIT(10), PHASE(11)
        ctg_op = 0
        if use_ctg:
            from onebit.core.prf import splitmix32
            ctg_state, r = splitmix32(ctg_state)
            ctg_op = r & 3
            ctg_digest = np.uint32((ctg_digest << 2) ^ ctg_op)

        # Accumulate samples for this tick
        tick_sum = 0.0
        tick_count = 0

        for n in range(N):
            c = walsh_carrier_bit(n, t)

            # Carrier-modulated values (before CTG)
            u_main = base_u if c > 0 else -base_u
            u_twin = -u_main if use_anti else u_main

            # Apply CTG transforms (branchless)
            if use_ctg:
                # PHASE (11): swap main↔twin
                if ctg_op == 3:
                    u_main, u_twin = u_twin, u_main

                # INVERT (01): flip signs
                if ctg_op == 1:
                    u_main = -u_main
                    u_twin = -u_twin

                # INHIBIT (10): create +v/-v pairing to cancel within tick
                if ctg_op == 2:
                    v = 0.5 * (u_main + u_twin)
                    u_main = v
                    u_twin = -v

            # Use u_main for main channel ΣΔ
            u = u_main

            # Main channel
            if cfg.order == 1:
                y_main, e_main[n] = _sd1_tick(u, float(e_main[n]), float(cfg.lambd), 4.0)
            else:
                y_main, e1_main[n], e2_main[n] = _sd2_tick(u, float(e1_main[n]), float(e2_main[n]),
                                                             float(cfg.beta), float(cfg.lambd), 4.0, 8.0)

            # Pack main bit: +1 -> 1, -1 -> 0
            bit_idx = t * N + n
            yb_main = 1 if y_main > 0 else 0
            if yb_main:
                y_bits_main[bit_idx >> 5] |= (np.uint32(1) << (bit_idx & 31))

            # Energy: track toggles (PR-3.7)
            if instr_on:
                energy_toggles_y_main += np.uint64(yb_main ^ prev_y_main[n])
                prev_y_main[n] = yb_main

            tick_sum += float(y_main)
            tick_count += 1

            if use_anti:
                # u_twin already computed above with CTG applied
                if cfg.order == 1:
                    y_twin, e_twin[n] = _sd1_tick(u_twin, float(e_twin[n]), float(cfg.lambd), 4.0)
                else:
                    y_twin, e1_twin[n], e2_twin[n] = _sd2_tick(u_twin, float(e1_twin[n]), float(e2_twin[n]),
                                                                 float(cfg.beta), float(cfg.lambd), 4.0, 8.0)

                # Pack twin bit
                yb_twin = 1 if y_twin > 0 else 0
                if yb_twin:
                    y_bits_twin[bit_idx >> 5] |= (np.uint32(1) << (bit_idx & 31))

                # Energy: track toggles (PR-3.7)
                if instr_on:
                    energy_toggles_y_twin += np.uint64(yb_twin ^ prev_y_twin[n])
                    prev_y_twin[n] = yb_twin
                
                # Correction: Subtract y_twin because u_twin = -u_main
                tick_sum -= float(y_twin)
                tick_count += 1

        # Compute y_bar for this tick
        y_bar_t = tick_sum / float(tick_count)
        acc_ybar += y_bar_t
        k_used = t + 1

        # Early-exit check (Hoeffding bound on mean of y_bar sequence) - only if enabled
        if early_exit_enable:
            mean_ybar = acc_ybar / float(k_used)
            thr = np.sqrt(0.5 * np.log(2.0 / max(delta, 1e-20)) / float(k_used))
            if abs(mean_ybar) <= (eps + thr):
                break

    # Final estimate: mean of y_bar sequence
    # Multiply by N to compensate for Walsh carrier averaging (only 1 carrier has DC signal)
    est = (acc_ybar / float(k_used)) * float(N) if k_used > 0 else 0.0

    diags = {
        "k_used": int(k_used),
        "y_bits_main": y_bits_main if want_y_pack else None,
        "y_bits_twin": y_bits_twin if want_y_pack else None,
        "pc32_main": pc32_main_list,
        "pc32_twin": pc32_twin_list,
        "ctg_digest": int(ctg_digest),
    }

    # Energy instrumentation (PR-3.7)
    if instr_on:
        # Compute bytes_out
        samples_per_channel = k_used * N
        bytes_bits = (samples_per_channel + 7) // 8
        bytes_pc32 = 4 * k_used if want_pc32 else 0
        energy_bytes_out = np.uint64(bytes_bits + bytes_pc32)

        diags["energy"] = {
            "toggles_y_main": np.array([energy_toggles_y_main], dtype=np.uint64),
            "toggles_y_twin": np.array([energy_toggles_y_twin], dtype=np.uint64),
            "ones_pc": np.array([energy_ones_pc], dtype=np.uint64),
            "zeros_pc": np.array([energy_zeros_pc], dtype=np.uint64),
            "xnor_ops": np.array([energy_xnor_ops], dtype=np.uint64),
            "popcnt_ops": np.array([energy_popcnt_ops], dtype=np.uint64),
            "bytes_W": np.array([energy_bytes_W], dtype=np.uint64),
            "bytes_X": np.array([energy_bytes_X], dtype=np.uint64),
            "bytes_out": np.array([energy_bytes_out], dtype=np.uint64),
        }

    return float(est), diags


def popcount_numpy(x: np.ndarray) -> np.ndarray:
    """Vectorized population count for uint32 array."""
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003f


def bsdm_w_matmul(
    W_bits: np.ndarray,  # [d_out, Kw]
    x_bits: np.ndarray,  # [Kw] OR [k, Kw] OR [B, Kw] OR [B, k, Kw]
    k: int,
    cfg: SDConfig,
    seed: int,  # Base seed
    scale: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """Vectorized BSDM-W matrix multiplication with Batch support.
    
    Computes y = scale * (W @ x) for all rows in parallel.
    Supports batch dimension B.
    
    Args:
        W_bits: Packed weight matrix [d_out, Kw]
        x_bits: Packed input. 
                Static: [Kw] or [B, Kw]
                Stream: [k, Kw] or [B, k, Kw]
        k: Number of ticks
        ...
        
    Returns:
        y: [d_out] or [B, d_out]
    """
    d_out = W_bits.shape[0]
    Kw = W_bits.shape[1]
    Kbits = Kw * 32
    
    # 1. Normalize Input Shape to [B, k, Kw] (Stream) or [B, Kw] (Static)
    x_mode = "static"
    B = 1
    
    # Ensure x_bits is numpy
    if not isinstance(x_bits, np.ndarray):
        x_bits = np.asarray(x_bits)
    
    if x_bits.ndim == 1: # [Kw]
        x_bits = x_bits[None, :] # [1, Kw]
        x_mode = "static"
        B = 1
    elif x_bits.ndim == 2: # [B, Kw] or [k, Kw]
        if x_bits.shape[1] != Kw:
             # Padding logic
             if x_bits.shape[1] < Kw:
                 new_x = np.zeros((x_bits.shape[0], Kw), dtype=x_bits.dtype)
                 new_x[:, :x_bits.shape[1]] = x_bits
                 x_bits = new_x
             elif x_bits.shape[1] > Kw:
                 x_bits = x_bits[:, :Kw]
        
        if x_bits.shape[0] == k and B != k:
            # Ambiguous if B==k. But usually batch size (512) != T (64).
            # Assume stream if matches k
            x_mode = "stream"
            B = 1
            x_bits = x_bits[None, :, :] # [1, k, Kw]
        else:
            x_mode = "static"
            B = x_bits.shape[0]
            
    elif x_bits.ndim == 3: # [B, k, Kw]
        if x_bits.shape[2] != Kw:
             # Padding fix
             if x_bits.shape[2] < Kw:
                 new_x = np.zeros((x_bits.shape[0], x_bits.shape[1], Kw), dtype=x_bits.dtype)
                 new_x[:, :, :x_bits.shape[2]] = x_bits
                 x_bits = new_x
             elif x_bits.shape[2] > Kw:
                 x_bits = x_bits[:, :, :Kw]
        
        x_mode = "stream"
        B = x_bits.shape[0]
        if x_bits.shape[1] != k:
            raise ValueError(f"x_bits stream length {x_bits.shape[1]} != k={k}")
    else:
        raise ValueError(f"Invalid x_bits shape {x_bits.shape}")

    N = int(cfg.walsh_N)
    use_anti = bool(cfg.antithetic)
    if cfg.order != 2:
        raise ValueError("Only order=2 supported")

    rng = np.random.default_rng(seed)
    
    # Generate base states [d_out, N]
    e1_main_base = (rng.random((d_out, N), dtype=np.float32) - 0.5)
    e2_main_base = (rng.random((d_out, N), dtype=np.float32) - 0.5)
    e1_twin_base = (rng.random((d_out, N), dtype=np.float32) - 0.5) if use_anti else None
    e2_twin_base = (rng.random((d_out, N), dtype=np.float32) - 0.5) if use_anti else None
    
    # Broadcast to [B, d_out, N]
    e1_main = np.broadcast_to(e1_main_base, (B, d_out, N)).copy()
    e2_main = np.broadcast_to(e2_main_base, (B, d_out, N)).copy()
    e1_twin = np.broadcast_to(e1_twin_base, (B, d_out, N)).copy() if use_anti else None
    e2_twin = np.broadcast_to(e2_twin_base, (B, d_out, N)).copy() if use_anti else None
    
    acc_ybar = np.zeros((B, d_out), dtype=np.float32)
    
    lambd = float(cfg.lambd)
    beta = float(cfg.beta)
    
    # Pre-expand W_bits for broadcasting: [1, d_out, Kw]
    W_bits_bc = W_bits[None, :, :] 
    
    # Pre-expand x_bits if static: [B, 1, Kw]
    if x_mode == "static":
        x_bits_bc = x_bits[:, None, :] # [B, 1, Kw]
    
    for t in range(k):
        # Fetch xt: [B, 1, Kw]
        if x_mode == "static":
            xt = x_bits_bc
        else:
            xt = x_bits[:, t, :][:, None, :] # [B, 1, Kw]
            
        # XNOR: [B, d_out, Kw]
        xnor = np.bitwise_not(np.bitwise_xor(W_bits_bc, xt))
        
        # Popcount
        pc_words = popcount_numpy(xnor) # [B, d_out, Kw]
        pc_rows = np.sum(pc_words, axis=2) # [B, d_out]
        
        # Base dot
        base_dot = (pc_rows.astype(np.float32) * 2.0) - float(Kbits)
        base_u = base_dot / float(Kbits) # [B, d_out]
        
        # Expand for N carriers: [B, d_out, 1]
        base_u = base_u[:, :, None] 
        
        tick_sum = np.zeros((B, d_out), dtype=np.float32)
        tick_count = 0
        
        for n in range(N):
            c = walsh_carrier_bit(n, t)
            
            # Select u: [B, d_out]
            if c > 0:
                u_main = base_u[:, :, 0]
            else:
                u_main = -base_u[:, :, 0]
                
            u_twin = -u_main if use_anti else u_main
            
            # Sigma-Delta Update (Main)
            e1_mn = e1_main[:, :, n]
            e2_mn = e2_main[:, :, n]
            
            v = e2_mn + e1_mn + u_main
            y = np.where(v >= 0.0, 1.0, -1.0)
            
            e1_prev = e1_mn.copy()
            
            e1_next = e1_mn + u_main - y - lambd * e1_mn
            e1_main[:, :, n] = np.clip(e1_next, -4.0, 4.0)
            
            e2_next = e2_mn + e1_main[:, :, n] - lambd * e2_mn + beta * (e1_main[:, :, n] - e1_prev)
            e2_main[:, :, n] = np.clip(e2_next, -8.0, 8.0)
            
            tick_sum += y
            tick_count += 1
            
            if use_anti:
                e1_tn = e1_twin[:, :, n]
                e2_tn = e2_twin[:, :, n]
                
                v = e2_tn + e1_tn + u_twin
                y = np.where(v >= 0.0, 1.0, -1.0)
                
                e1_prev = e1_tn.copy()
                
                e1_next = e1_tn + u_twin - y - lambd * e1_tn
                e1_twin[:, :, n] = np.clip(e1_next, -4.0, 4.0)
                
                e2_next = e2_tn + e1_tn - lambd * e2_tn + beta * (e1_tn - e1_prev)
                e2_twin[:, :, n] = np.clip(e2_next, -8.0, 8.0)
                
                tick_sum -= y
                tick_count += 1
                
        # Accumulate
        y_bar_t = tick_sum / float(tick_count)
        acc_ybar += y_bar_t
        
    # Final estimate
    est = (acc_ybar / float(k)) * float(N) # [B, d_out]
    
    # Apply Scale
    est = est * scale
    
    # Remove Batch dim if input was B=1 and we inferred it
    # Logic: if input was 2D [k, Kw] (Stream), we assumed B=1. Return [d_out].
    # if input was 3D [B, k, Kw] (Stream), we return [B, d_out].
    # if input was 1D [Kw] (Static), we assumed B=1. Return [d_out].
    # if input was 2D [B, Kw] (Static), we return [B, d_out].
    
    # To detect: if original x_bits ndim was 1 or (ndim=2 and mode=stream)
    # But we modified x_bits.
    # Let's look at B again.
    if B == 1:
         # Check if originally intended to be scalar batch.
         # We can't know easily.
         # Standard behavior: squeeze batch if it's 1?
         # RuntimeTransformer expects [d_out] for single token.
         # It will expect [Seq, d_out] for batch.
         # If we return [1, d_out], it might break things expecting 1D.
         pass
         
    # Actually, let's NOT squeeze automatically to avoid ambiguity.
    # But RuntimeTransformer expects 1D for single token.
    # So if B=1, we squeeze.
    if B == 1:
        return est[0] # [d_out]
         
    return est
