import math

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any


class HessianVQ:
    """Hessian-weighted vector quantization on magnitude blocks.

    Splits weights into sign + magnitude, groups magnitude values into
    fixed-size blocks, and runs Hessian-weighted K-means so that
    codebook capacity concentrates on loss-sensitive regions.

    Distance used for both assignment and k-means++ seeding (per block ``b``,
    centroid ``c``)::

        dist[b, c] = sum_j w[b, j] * (x[b, j] - C[c, j]) ** 2

    where ``w[b, j]`` is the Hessian-diagonal entry of the input column that
    element ``j`` of block ``b`` belongs to (all ones when
    ``use_hessian_weight=False``).  The centroid update is the matching
    per-dimension weighted mean, so assignment and update minimise the same
    objective.  Padding elements (when a dimension is not a multiple of
    ``block_size``) get weight 0 and therefore never influence the result.

    Phase-0 correctness notes:
      * Earlier versions averaged the weights over each block before computing
        distances, which made the "Hessian weighting" a no-op for assignment.
      * Earlier versions applied a global mean shift
        ``W_recon - W_recon.mean() + W.mean()``.  That needs an extra stored
        float and is not part of the VQ objective, so it has been removed.
      * ``gptq_refine`` was not GPTQ (its loop result was multiplied by 0.0 and
        only an uncounted per-row offset survived).  It now raises
        ``NotImplementedError``; use ``onebit/quantization/gptq.py`` instead.
      * Randomness (k-means++ seeding) uses a local ``np.random.Generator``
        seeded from ``seed`` so results are reproducible.
    """

    # Number of blocks processed per chunk when computing the (n_blocks, k)
    # distance matrix; bounds peak memory to ~chunk * k * 8 bytes per matrix.
    _CHUNK = 16384

    def __init__(
        self,
        n_codes: int = 32,
        block_size: int = 4,
        max_iter: int = 25,
        use_hessian_weight: bool = True,
        gptq_refine: bool = False,
        *,
        seed: int = 0,
    ):
        if gptq_refine:
            raise NotImplementedError(
                "HessianVQ(gptq_refine=True) has been removed: the previous "
                "'_gptq_refine' was not GPTQ (its error-feedback result was multiplied "
                "by 0.0 and only an uncounted per-row offset survived). For real GPTQ "
                "use onebit/quantization/gptq.py (gptq_layer / gptq_quantize_model)."
            )
        self.n_codes = n_codes
        self.block_size = block_size
        self.max_iter = max_iter
        self.use_hessian_weight = use_hessian_weight
        self.gptq_refine = gptq_refine
        self.seed = seed

        self.indices = None
        self.codebook = None
        self.signs = None
        self._index_entropy = 0.0

    # ------------------------------------------------------------------
    # Weighted distances
    # ------------------------------------------------------------------
    @classmethod
    def _weighted_dists(
        cls,
        blocks: np.ndarray,
        weights: np.ndarray,
        centroids: np.ndarray,
        include_const: bool = True,
    ) -> np.ndarray:
        """Return ``dist[b, c] = sum_j w[b,j] * (x[b,j] - C[c,j])**2``.

        Computed via the expansion ``sum(w x^2) - 2 (w x) . C + w . C^2`` with
        two ``(chunk, d) @ (d, k)`` matmuls per chunk of blocks.  When
        ``include_const`` is False the per-row constant ``sum(w x^2)`` is
        dropped (sufficient for argmin).
        """
        n = blocks.shape[0]
        k = centroids.shape[0]
        out = np.empty((n, k), dtype=np.float64)
        Ct = centroids.T
        C2t = (centroids ** 2).T
        for s in range(0, n, cls._CHUNK):
            e = min(s + cls._CHUNK, n)
            x = blocks[s:e]
            w = weights[s:e]
            wx = w * x
            d = w @ C2t - 2.0 * (wx @ Ct)
            if include_const:
                d += np.sum(wx * x, axis=1, keepdims=True)
            out[s:e] = d
        if include_const:
            np.maximum(out, 0.0, out=out)
        return out

    @classmethod
    def _assign(cls, blocks: np.ndarray, weights: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        n = blocks.shape[0]
        assignments = np.empty(n, dtype=np.int64)
        Ct = centroids.T
        C2t = (centroids ** 2).T
        for s in range(0, n, cls._CHUNK):
            e = min(s + cls._CHUNK, n)
            w = weights[s:e]
            d = w @ C2t - 2.0 * ((w * blocks[s:e]) @ Ct)
            assignments[s:e] = np.argmin(d, axis=1)
        return assignments

    # ------------------------------------------------------------------
    # K-means++ initialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _kmeans_pp_init(
        blocks: np.ndarray,
        k: int,
        weights: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """K-means++ seeding with per-element weighted distances.

        ``weights=None`` means unweighted (all ones).  Min-distances are
        updated incrementally (O(n k d) instead of O(n k^2 d)).
        """
        if rng is None:
            rng = np.random.default_rng(0)
        n = len(blocks)
        if n <= k:
            return blocks.copy()
        if weights is None:
            weights = np.ones_like(blocks)

        centroids = np.empty((k, blocks.shape[1]), dtype=blocks.dtype)
        idx = int(rng.integers(n))
        centroids[0] = blocks[idx]
        min_dists = np.sum(weights * (blocks - centroids[0]) ** 2, axis=1)

        for c in range(1, k):
            total = min_dists.sum()
            if total <= 0:
                idx = int(rng.integers(n))
            else:
                idx = int(rng.choice(n, p=min_dists / total))
            centroids[c] = blocks[idx]
            d_new = np.sum(weights * (blocks - centroids[c]) ** 2, axis=1)
            np.minimum(min_dists, d_new, out=min_dists)

        return centroids

    # ------------------------------------------------------------------
    # Hessian-weighted K-means
    # ------------------------------------------------------------------
    def _weighted_kmeans(
        self,
        blocks: np.ndarray,
        weights: np.ndarray,
        k: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """K-means with per-element weighted distance and weighted centroid updates.

        ``weights`` must already encode the choice of weighting (ones for
        unweighted, zero for padding).
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)
        centroids = self._kmeans_pp_init(blocks, k, weights, rng)
        k_eff, d = centroids.shape

        wx = weights * blocks
        prev_assignments = None
        assignments = self._assign(blocks, weights, centroids)

        for iteration in range(self.max_iter):
            if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
                break
            prev_assignments = assignments

            num = np.empty((k_eff, d), dtype=np.float64)
            den = np.empty((k_eff, d), dtype=np.float64)
            for j in range(d):
                num[:, j] = np.bincount(assignments, weights=wx[:, j], minlength=k_eff)
                den[:, j] = np.bincount(assignments, weights=weights[:, j], minlength=k_eff)
            safe = den > 0
            centroids = np.where(safe, num / np.where(safe, den, 1.0), centroids)

            assignments = self._assign(blocks, weights, centroids)

        return centroids, assignments

    # ------------------------------------------------------------------
    # BPP accounting
    # ------------------------------------------------------------------
    def _codebook_bits(self) -> float:
        # Conservative: every codebook value stored as fp32 (fp16 would halve this).
        return float(self.n_codes * (self.block_size ** 2) * 32)

    def effective_bpp(self, n_weights: int) -> float:
        """BPP with *ideal* entropy-coded indices.

        = (1 sign bit per weight
           + n_blocks * H(indices)       # ideal zero-order entropy, no coder overhead
           + codebook at 32 bits/value) / n_weights

        The index term is a lower bound for any real entropy coder (no table,
        no per-symbol overhead).  See :meth:`effective_bpp_fixed` for simple
        fixed-width index storage.
        """
        if self.indices is None:
            return 0.0
        sign_bits = n_weights  # 1 bit per weight
        n_blocks = len(self.indices)
        counts = np.bincount(self.indices, minlength=self.n_codes)
        probs = counts / (counts.sum() + 1e-30)
        probs = probs[probs > 0]
        self._index_entropy = float(-np.sum(probs * np.log2(probs)))
        vq_bits = n_blocks * self._index_entropy
        codebook_bits = self._codebook_bits()
        return (sign_bits + vq_bits + codebook_bits) / n_weights

    def effective_bpp_fixed(self, n_weights: int) -> float:
        """BPP with fixed-width indices: ceil(log2(n_codes)) bits per block index.

        = (1 sign bit per weight + n_blocks * ceil(log2 K) + codebook at 32 bits/value)
          / n_weights
        """
        if self.indices is None:
            return 0.0
        n_blocks = len(self.indices)
        idx_bits = math.ceil(math.log2(max(self.n_codes, 2)))
        return (n_weights + n_blocks * idx_bits + self._codebook_bits()) / n_weights

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def quantize(self, W: np.ndarray, H_diag: np.ndarray) -> np.ndarray:
        """Quantize weight matrix using Hessian-weighted block VQ.

        Args:
            W: (d_out, d_in) weight matrix.
            H_diag: (d_in,) Hessian diagonal (e.g. mean(X**2, axis=0)).

        Returns:
            W_recon: (d_out, d_in) reconstructed weight matrix ``sign(W) * M_vq``
            (no global mean shift; see class docstring).
        """
        W = np.asarray(W, dtype=np.float64)
        d_out, d_in = W.shape
        bs = self.block_size

        S = np.sign(W)
        S[S == 0] = 1.0
        M = np.abs(W)

        pad_h = (bs - d_out % bs) % bs
        pad_w = (bs - d_in % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)))

        if self.use_hessian_weight:
            h = np.asarray(H_diag, dtype=np.float64).reshape(-1)
            if h.shape[0] != d_in:
                raise ValueError(f"H_diag has length {h.shape[0]}, expected d_in={d_in}")
            h = np.clip(h, 0.0, None)
            hmax = float(h.max()) if h.size else 0.0
            if hmax <= 0:
                h = np.ones(d_in, dtype=np.float64)
            else:
                # Scale-invariant; floor keeps every real element minimally weighted.
                h = np.maximum(h / h.mean(), 1e-12)
        else:
            h = np.ones(d_in, dtype=np.float64)
        H_mat = np.broadcast_to(h, (d_out, d_in))
        # Padding elements get weight 0 so they never affect assignment/centroids.
        H_pad = np.pad(H_mat, ((0, pad_h), (0, pad_w)), constant_values=0.0)

        h_p, w_p = M_pad.shape
        n_bh = h_p // bs
        n_bw = w_p // bs

        blocks = M_pad.reshape(n_bh, bs, n_bw, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)
        weights = H_pad.reshape(n_bh, bs, n_bw, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)
        blocks = np.ascontiguousarray(blocks)
        weights = np.ascontiguousarray(weights)

        rng = np.random.default_rng(self.seed)
        centroids, assignments = self._weighted_kmeans(blocks, weights, self.n_codes, rng)

        recon_blocks = centroids[assignments]
        M_recon = (
            recon_blocks.reshape(n_bh, n_bw, bs, bs)
            .transpose(0, 2, 1, 3)
            .reshape(h_p, w_p)[:d_out, :d_in]
        )

        W_recon = S * M_recon

        self.indices = assignments
        self.codebook = centroids
        self.signs = S

        return W_recon
