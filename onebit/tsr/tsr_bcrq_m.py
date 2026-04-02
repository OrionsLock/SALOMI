from __future__ import annotations

from typing import Union

import numpy as np

from onebit.core import pack_signs_rowmajor
from .qmc_lattice import lattice_uniforms, derive_seed64


def tsr_pack_input_bits(
    x_f: np.ndarray,
    T: int,
    master_seed: Union[int, bytes] = 0,
    layer: int = 0,
    stream: int = 0,
    method: str = "iid",  # "iid" or "lattice"
) -> np.ndarray:
    """TSR(BCRQ-M): sample ±1 for inputs using uniforms and pack to bits.

    - x_f: shape [K], expected in [-1, 1]. Values are clipped into [-1, 1].
    - T: number of passes to sample.
    - master_seed/layer/stream: deterministic seeding.
    - method: "iid" (independent RNG per (t,k)) or "lattice" (rank-1 QMC).

    Returns: uint32 array of shape [T, ceil(K/32)] compatible with OpenCL kernel.
    """
    x = np.asarray(x_f, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("x_f must be 1-D [K]")
    K = x.shape[0]
    if T <= 0:
        return np.zeros((0, (K + 31) // 32), dtype=np.uint32)
    x = np.clip(x, -1.0, 1.0)

    # Uniforms in [0,1)
    seed64 = int(derive_seed64(master_seed, layer=layer, stream=stream))
    if method == "lattice":
        U = lattice_uniforms(seed64, K=K, T=T).astype(np.float32)
    else:
        rng = np.random.default_rng(seed64)
        U = rng.random((T, K), dtype=np.float32)

    P = (x[None, :] + 1.0) * 0.5  # in [0,1]
    signs = np.where(U < P, 1.0, -1.0).astype(np.float32)  # +1 or -1

    # Reuse the packer: rows = T, columns = K
    X_bits = pack_signs_rowmajor(signs)
    return X_bits

