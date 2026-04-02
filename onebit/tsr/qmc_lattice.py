from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, Union

UInt = np.uint64
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
INV_2_64 = 1.0 / float(1 << 64)


def _to_uint64(x: Union[int, bytes]) -> np.uint64:
    if isinstance(x, bytes):
        # Interpret as big-endian integer
        val = int.from_bytes(x, byteorder="big", signed=False)
    else:
        val = int(x)
    return np.uint64(val & 0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.uint64) -> np.uint64:
    """Deterministic 64-bit mixer. Returns a new 64-bit value.

    This is a standard SplitMix64 mix function producing a pseudo-random 64-bit
    value from the input (used here for seeding per-dimension params).
    """
    z = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & MASK
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & MASK
    z = z ^ (z >> np.uint64(31))
    return z


def derive_seed64(master_seed: Union[int, bytes], layer: int = 0, stream: int = 0) -> np.uint64:
    """Derive a per-stream 64-bit seed from a master seed and indices.

    This provides deterministic mapping master->layer->stream.
    """
    s = _to_uint64(master_seed)
    s = splitmix64(s ^ np.uint64(layer))
    s = splitmix64(s ^ np.uint64(stream * 0x9E3779B1))
    return s


def _stream_params(master_seed: Union[int, bytes], K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct lattice parameters (a_k, o_k) for each dimension k.

    We use a rank-1 lattice with per-dimension odd multipliers a_k and offsets o_k.
    """
    a = np.empty((K,), dtype=UInt)
    o = np.empty((K,), dtype=UInt)
    s = _to_uint64(master_seed)
    cur = s
    for k in range(K):
        cur = splitmix64(cur ^ np.uint64(k + 1))
        ak = splitmix64(cur)
        ok = splitmix64(ak)
        a[k] = (ak | np.uint64(1))  # force odd
        o[k] = ok
    return a, o


def lattice_uniforms(master_seed: Union[int, bytes], K: int, T: int) -> np.ndarray:
    """Generate a [T, K] array of quasi-random uniforms in [0,1).

    Uses a rank-1 lattice with Cranley-Patterson rotation:
      u[t, k] = ((o_k + t * a_k) mod 2^64) / 2^64
    """
    if T <= 0 or K <= 0:
        return np.zeros((max(T, 0), max(K, 0)), dtype=np.float32)
    a, o = _stream_params(master_seed, K)
    t_idx = np.arange(T, dtype=UInt)[:, None]
    # Broadcasted multiply-add modulo 2^64
    vals = (o[None, :] + t_idx * a[None, :]) & MASK
    u = vals.astype(np.float64) * INV_2_64
    return u.astype(np.float32)

