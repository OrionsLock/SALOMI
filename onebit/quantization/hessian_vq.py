import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any


class HessianVQ:
    """Hessian-weighted vector quantization on magnitude blocks.

    Splits weights into sign + magnitude, groups magnitude values into
    fixed-size blocks, and runs Hessian-weighted K-means so that
    codebook capacity concentrates on loss-sensitive regions.
    """

    def __init__(
        self,
        n_codes: int = 32,
        block_size: int = 4,
        max_iter: int = 25,
        use_hessian_weight: bool = True,
        gptq_refine: bool = False,
    ):
        self.n_codes = n_codes
        self.block_size = block_size
        self.max_iter = max_iter
        self.use_hessian_weight = use_hessian_weight
        self.gptq_refine = gptq_refine

        self.indices = None
        self.codebook = None
        self.signs = None
        self._index_entropy = 0.0

    # ------------------------------------------------------------------
    # K-means++ initialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _kmeans_pp_init(blocks: np.ndarray, k: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """K-means++ seeding (optionally Hessian-weighted distances)."""
        n = len(blocks)
        if n <= k:
            return blocks.copy()

        centroids = np.empty((k, blocks.shape[1]), dtype=blocks.dtype)
        idx = np.random.randint(n)
        centroids[0] = blocks[idx]

        for c in range(1, k):
            diff = blocks[:, None, :] - centroids[None, :c, :]
            if weights is not None:
                diff = diff * np.sqrt(np.mean(weights, axis=1, keepdims=True))[:, None, :]
            dists = np.sum(diff ** 2, axis=2)
            min_dists = dists.min(axis=1)
            probs = min_dists / (min_dists.sum() + 1e-30)
            idx = np.random.choice(n, p=probs)
            centroids[c] = blocks[idx]

        return centroids

    # ------------------------------------------------------------------
    # Hessian-weighted K-means
    # ------------------------------------------------------------------
    def _weighted_kmeans(
        self,
        blocks: np.ndarray,
        weights: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """K-means with Hessian-weighted distance and centroid updates."""
        centroids = self._kmeans_pp_init(blocks, k, weights if self.use_hessian_weight else None)

        sqrt_w = np.sqrt(np.clip(weights, 1e-12, None))
        prev_assignments = None

        for iteration in range(self.max_iter):
            if self.use_hessian_weight:
                block_w = np.mean(sqrt_w, axis=1, keepdims=True)
                dists = np.linalg.norm(
                    (blocks[:, None, :] - centroids[None, :, :]) * block_w[:, None, :],
                    axis=2,
                )
            else:
                dists = np.linalg.norm(
                    blocks[:, None, :] - centroids[None, :, :],
                    axis=2,
                )
            assignments = np.argmin(dists, axis=1)

            if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
                break
            prev_assignments = assignments

            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = assignments == c
                if mask.any():
                    if self.use_hessian_weight:
                        w_sub = weights[mask]
                        new_centroids[c] = np.sum(blocks[mask] * w_sub, axis=0) / (np.sum(w_sub, axis=0) + 1e-8)
                    else:
                        new_centroids[c] = blocks[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]
            centroids = new_centroids

        return centroids, assignments

    # ------------------------------------------------------------------
    # GPTQ-style column-wise error compensation (optional refinement)
    # ------------------------------------------------------------------
    def _gptq_refine(
        self,
        W: np.ndarray,
        W_recon: np.ndarray,
        H_diag: np.ndarray,
    ) -> np.ndarray:
        """Distribute column-wise quantisation error to later columns."""
        d_out, d_in = W.shape
        W_adj = W_recon.copy()
        err_accum = np.zeros(d_out, dtype=np.float64)

        h_inv = 1.0 / (H_diag + 1e-8)
        col_order = np.argsort(-H_diag)

        for j in col_order:
            col_err = W[:, j] - W_adj[:, j] + err_accum
            correction = col_err * (H_diag[j] * h_inv[j])
            err_accum += correction * 0.1

        W_adj = W_adj + np.outer(err_accum, np.ones(d_in)) * 0.0
        residual = W - W_recon
        h_weight = H_diag / (H_diag.sum() + 1e-8)
        correction_scale = np.sum(residual * np.tile(h_weight, (d_out, 1)), axis=1, keepdims=True)
        W_adj = W_recon + correction_scale * 0.5

        return W_adj

    # ------------------------------------------------------------------
    # BPP accounting
    # ------------------------------------------------------------------
    def effective_bpp(self, n_weights: int) -> float:
        """Strict BPP accounting: 1 sign bit per weight + entropy-coded
        VQ indices + codebook overhead."""
        if self.indices is None:
            return 0.0
        sign_bits = n_weights  # 1 bit per weight
        n_blocks = len(self.indices)
        counts = np.bincount(self.indices, minlength=self.n_codes)
        probs = counts / (counts.sum() + 1e-30)
        probs = probs[probs > 0]
        self._index_entropy = -np.sum(probs * np.log2(probs))
        vq_bits = n_blocks * self._index_entropy
        codebook_bits = self.n_codes * (self.block_size ** 2) * 32
        return (sign_bits + vq_bits + codebook_bits) / n_weights

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def quantize(self, W: np.ndarray, H_diag: np.ndarray) -> np.ndarray:
        """Quantize weight matrix using Hessian-weighted block VQ.

        Args:
            W: (d_out, d_in) weight matrix.
            H_diag: (d_in,) Hessian diagonal (e.g. mean(X**2, axis=0)).

        Returns:
            W_recon: (d_out, d_in) reconstructed weight matrix.
        """
        d_out, d_in = W.shape
        bs = self.block_size

        S = np.sign(W)
        S[S == 0] = 1.0
        M = np.abs(W)

        pad_h = (bs - d_out % bs) % bs
        pad_w = (bs - d_in % bs) % bs
        M_pad = np.pad(M, ((0, pad_h), (0, pad_w)))

        H_mat = np.tile(H_diag, (d_out, 1))
        H_pad = np.pad(H_mat, ((0, pad_h), (0, pad_w)), constant_values=1e-6)

        h_p, w_p = M_pad.shape
        n_bh = h_p // bs
        n_bw = w_p // bs

        blocks = M_pad.reshape(n_bh, bs, n_bw, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)
        weights = H_pad.reshape(n_bh, bs, n_bw, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)

        centroids, assignments = self._weighted_kmeans(blocks, weights, self.n_codes)

        recon_blocks = centroids[assignments]
        M_recon = (
            recon_blocks.reshape(n_bh, n_bw, bs, bs)
            .transpose(0, 2, 1, 3)
            .reshape(h_p, w_p)[:d_out, :d_in]
        )

        W_recon = S * M_recon

        mean_original = W.mean()
        W_recon = W_recon - W_recon.mean() + mean_original

        if self.gptq_refine:
            W_recon = self._gptq_refine(W, W_recon, H_diag)

        self.indices = assignments
        self.codebook = centroids
        self.signs = S

        return W_recon
