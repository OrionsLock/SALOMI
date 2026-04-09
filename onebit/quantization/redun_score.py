"""Redundancy Score probe for Proxy-SR-VQ.

Computes a per-block redundancy score that predicts how well a weight
block will compress under low-bit quantization.  Higher R means the block
is *more* redundant (easier to compress); lower R means it is sensitive
and needs a gentler quantiser.

    R_block = alpha * tr(H_block)
            + beta  * (sigma_mag / mu_mag)
            + gamma * Var(act_block)

The three terms capture:
    - Hessian trace: how flat the loss landscape is around this block
    - Magnitude CV: how uniform the weight magnitudes are (uniform = compressible)
    - Activation variance: how noisy the inputs are (low noise = compressible)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class RedunResult:
    """Per-component redundancy measurement."""
    redun_score: float
    hessian_trace: float
    mag_cv: float
    act_var: float
    n_weights: int
    component: str = ""
    layer_idx: int = -1


class RedunScoreComputer:
    """Compute the composite Redun Score for weight blocks.

    Parameters
    ----------
    alpha, beta, gamma : float
        Mixing coefficients.  Defaults are sensible for GPT-2 124M and
        can be tuned with ``fit_coefficients``.
    block_size : int
        Spatial block size matching HessianVQ (default 4).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        block_size: int = 4,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.block_size = block_size

    # ------------------------------------------------------------------
    # Single matrix
    # ------------------------------------------------------------------
    def compute_layer_redun(
        self,
        W: np.ndarray,
        H_diag: np.ndarray,
        activations: Optional[np.ndarray] = None,
    ) -> RedunResult:
        """Compute Redun Score for one weight matrix.

        Args:
            W: (d_out, d_in) weight matrix.
            H_diag: (d_in,) Hessian diagonal (e.g. mean(X**2, axis=0)).
            activations: (n_samples, d_in) optional raw activations for
                         variance term.  If *None* the gamma term is zero.

        Returns:
            RedunResult with the composite score and individual terms.
        """
        h_trace = float(np.sum(H_diag))

        mag = np.abs(W)
        mu = mag.mean() + 1e-10
        sigma = mag.std()
        mag_cv = float(sigma / mu)

        if activations is not None:
            act_var = float(np.mean(np.var(activations, axis=0)))
        else:
            act_var = 0.0

        h_norm = h_trace / (W.size + 1e-10)
        score = self.alpha * h_norm + self.beta * mag_cv + self.gamma * act_var

        if np.isnan(score) or np.isinf(score):
            score = self.beta * mag_cv

        return RedunResult(
            redun_score=score,
            hessian_trace=h_trace,
            mag_cv=mag_cv,
            act_var=act_var,
            n_weights=W.size,
        )

    # ------------------------------------------------------------------
    # Whole model (hooks into HuggingFace GPT-2)
    # ------------------------------------------------------------------
    def compute_model_redun(
        self,
        model,
        tokenizer,
        calib_texts: List[str],
        n_samples: int = 64,
        max_len: int = 64,
        adapter=None,
    ) -> Dict[str, Dict[str, RedunResult]]:
        """Probe every weight matrix in a transformer model.

        Registers hooks on ALL target submodules at once, then runs
        forward passes only ``n_samples`` times total (not per-component).

        Args:
            adapter: optional ``ModelAdapter`` from model_factory.  If
                     *None*, a new one is created from *model*.

        Returns ``{f"h.{i}": {"mlp_fc": RedunResult, ...}}`` keyed by
        transformer block and sub-component.
        """
        import torch

        if adapter is None:
            from ..proxy.model_factory import ModelAdapter
            adapter = ModelAdapter(model)

        n_layers = adapter.n_layers

        all_targets: Dict[str, Dict[str, Any]] = {}
        act_stores: Dict[str, Dict[str, list]] = {}
        handles = []

        for layer_idx in range(n_layers):
            layer_key = f"h.{layer_idx}"
            targets = {name: mod for name, mod in adapter.get_layer_targets(layer_idx)}
            all_targets[layer_key] = targets
            act_stores[layer_key] = {k: [] for k in targets}

            for comp_name, submod in targets.items():
                store = act_stores[layer_key][comp_name]

                def _make_hook(_store):
                    def _hook(mod, inp, out):
                        x = inp[0].detach().cpu().float().numpy().reshape(-1, inp[0].shape[-1])
                        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                        _store.append(x)
                    return _hook

                handles.append(submod.register_forward_hook(_make_hook(store)))

        device = next(model.parameters()).device
        with torch.no_grad():
            for text in calib_texts[:n_samples]:
                ids = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=max_len,
                )
                ids = {k: v.to(device) for k, v in ids.items()}
                model(**ids)

        for h in handles:
            h.remove()

        results: Dict[str, Dict[str, RedunResult]] = {}
        for layer_idx in range(n_layers):
            layer_key = f"h.{layer_idx}"
            layer_results: Dict[str, RedunResult] = {}

            for comp_name, submod in all_targets[layer_key].items():
                acts = act_stores[layer_key][comp_name]
                W = adapter.get_weight_numpy(submod)
                W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
                X = np.concatenate(acts, axis=0) if acts else np.zeros((1, W.shape[1]))
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                H_diag = np.mean(X ** 2, axis=0).astype(np.float32)
                H_diag = np.nan_to_num(H_diag, nan=0.0, posinf=0.0, neginf=0.0)

                r = self.compute_layer_redun(W, H_diag, X)
                r.component = comp_name
                r.layer_idx = layer_idx
                layer_results[comp_name] = r

            results[layer_key] = layer_results

        return results

    # ------------------------------------------------------------------
    # Coefficient tuning
    # ------------------------------------------------------------------
    def fit_coefficients(
        self,
        weight_list: List[np.ndarray],
        hdiag_list: List[np.ndarray],
        quality_list: List[float],
        act_list: Optional[List[np.ndarray]] = None,
        grid_steps: int = 5,
    ) -> Tuple[float, float, float]:
        """Tune (alpha, beta, gamma) to maximise rank-correlation between
        Redun Score and observed post-quantisation quality.

        Args:
            weight_list: list of weight matrices.
            hdiag_list: matching Hessian diagonals.
            quality_list: post-quant quality metric (higher = better).
            act_list: optional matching activations.
            grid_steps: resolution of the grid search.

        Returns:
            Best (alpha, beta, gamma) tuple.
        """
        grid = np.linspace(0.1, 2.0, grid_steps)
        best_corr = -2.0
        best_abc = (self.alpha, self.beta, self.gamma)

        for a in grid:
            for b in grid:
                for g in grid:
                    self.alpha, self.beta, self.gamma = a, b, g
                    scores = []
                    for i, (W, H) in enumerate(zip(weight_list, hdiag_list)):
                        act = act_list[i] if act_list else None
                        r = self.compute_layer_redun(W, H, act)
                        scores.append(r.redun_score)

                    if np.std(scores) < 1e-12 or np.std(quality_list) < 1e-12:
                        continue
                    corr = float(np.corrcoef(scores, quality_list)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        best_abc = (a, b, g)

        self.alpha, self.beta, self.gamma = best_abc
        return best_abc
