"""PRF seeding utilities for deterministic BSDM-W runs.

Uses SplitMix64 for deterministic seed derivation from (layer, row, token, run_id).
"""
from __future__ import annotations


def splitmix64(state: int) -> tuple[int, int]:
    """SplitMix64 PRNG step.
    
    Args:
        state: 64-bit state
        
    Returns:
        (next_state, output) both as 64-bit unsigned integers
    """
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    return state, z


def splitmix32(state: int) -> tuple[int, int]:
    """SplitMix32 PRNG step (returns 32-bit output from SplitMix64).

    Args:
        state: 64-bit state

    Returns:
        (next_state, output) where output is 32-bit unsigned integer
    """
    state, z = splitmix64(state)
    return state, (z >> 32) & 0xFFFFFFFF


def uniform_half(state: int) -> tuple[int, float]:
    """Generate uniform random in [-0.5, 0.5) using SplitMix64.

    Args:
        state: 64-bit state

    Returns:
        (next_state, uniform_value)
    """
    state, z = splitmix64(state)
    # Map upper 32 bits to [0, 1) then shift to [-0.5, 0.5)
    u01 = float(z >> 32) / 4294967296.0
    return state, u01 - 0.5


def derive_seed(layer: int, row: int, token: int, run_id: int) -> int:
    """Derive a 64-bit PRF seed from (layer, row, token, run_id) using SplitMix64.

    Deterministic mapping ensures CPU and OpenCL produce identical sequences.

    Args:
        layer: layer index (0-based)
        row: row index within layer
        token: token index in sequence
        run_id: run identifier (e.g., from uuid4 hash)

    Returns:
        64-bit unsigned seed
    """
    # Mix inputs into initial state
    state = (run_id & 0xFFFFFFFFFFFFFFFF)
    state ^= (layer & 0xFFFF) << 48
    state ^= (row & 0xFFFFFFFF) << 16
    state ^= (token & 0xFFFF)
    state &= 0xFFFFFFFFFFFFFFFF

    # Advance SplitMix64 three times to mix thoroughly
    state, _ = splitmix64(state)
    state, _ = splitmix64(state)
    state, out = splitmix64(state)

    return out

