from __future__ import annotations

import numpy as np


def walsh_carrier_bit(row_idx: int, t: int) -> int:
    """Return ±1 Walsh carrier value for the given row and time index.

    Uses the Hadamard (Walsh) definition with sequency-style rows: the sign is
    (-1)^{popcount(row_idx & t)}. Returns +1 for even parity, -1 for odd parity.
    """
    v = row_idx & t
    # Compute parity of v (number of set bits mod 2)
    # Kernighan's bit counting for portability without Python 3.10 int.bit_count
    parity = 0
    while v:
        v &= v - 1
        parity ^= 1
    return 1 if parity == 0 else -1


def walsh_row_vector(row_idx: int, length: int) -> np.ndarray:
    """Generate a full Walsh row of given length as ±1 int vector.

    Primarily for testing or vectorized use; not used in the inner loop.
    """
    out = np.empty((length,), dtype=np.int8)
    for t in range(length):
        out[t] = 1 if walsh_carrier_bit(row_idx, t) > 0 else -1
    return out

