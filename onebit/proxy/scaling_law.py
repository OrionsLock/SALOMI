"""Scaling law fitting and extrapolation for Proxy-SR-VQ.

Fits the relationship  R ~ a * log(N) + b  per component type from
proxy sweep results, then predicts Redun Scores and allocation policies
for arbitrarily large models.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class FittedParams:
    """Least-squares fit for one component type."""
    component: str
    a: float  # slope
    b: float  # intercept
    r_squared: float
    n_points: int
    log_n_values: List[float] = field(default_factory=list)
    r_values: List[float] = field(default_factory=list)


class ScalingLawFitter:
    """Fit and extrapolate Redun Score scaling laws.

    The core hypothesis: R_component ~ a * log(N_params) + b,
    meaning redundancy grows logarithmically with model size.
    If validated on GPT-2 proxies, we can predict the optimal
    quantisation allocation for any larger model.
    """

    def __init__(self):
        self.fitted: Dict[str, FittedParams] = {}

    def fit_redun_scaling(
        self,
        proxy_results: Dict[str, Dict[str, Dict[str, float]]],
        model_sizes: Dict[str, int],
    ) -> Dict[str, FittedParams]:
        """Fit R ~ a*log(N) + b from proxy sweep data.

        Args:
            proxy_results: ``{model_label: {"h.i": {"comp": redun_score}}}``
            model_sizes: ``{model_label: n_params}``

        Returns:
            Dict mapping component type to FittedParams.
        """
        comp_data: Dict[str, List[Tuple[float, float]]] = {}

        for label, layer_results in proxy_results.items():
            n_params = model_sizes[label]
            log_n = np.log(n_params)

            for _layer_key, comps in layer_results.items():
                for comp_name, score in comps.items():
                    ctype = self._component_type(comp_name)
                    comp_data.setdefault(ctype, []).append((log_n, score))

        self.fitted = {}
        for ctype, pairs in comp_data.items():
            xs = np.array([p[0] for p in pairs])
            ys = np.array([p[1] for p in pairs])

            if len(xs) < 2:
                self.fitted[ctype] = FittedParams(
                    component=ctype, a=0.0, b=float(ys.mean()),
                    r_squared=0.0, n_points=len(xs),
                    log_n_values=xs.tolist(), r_values=ys.tolist(),
                )
                continue

            a, b = np.polyfit(xs, ys, 1)
            y_pred = a * xs + b
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - ys.mean()) ** 2) + 1e-10
            r_sq = float(1 - ss_res / ss_tot)

            self.fitted[ctype] = FittedParams(
                component=ctype, a=float(a), b=float(b),
                r_squared=r_sq, n_points=len(xs),
                log_n_values=xs.tolist(), r_values=ys.tolist(),
            )

        return self.fitted

    def predict_redun(
        self,
        target_n_params: int,
        component: Optional[str] = None,
    ) -> Dict[str, float]:
        """Extrapolate Redun Score for a target model size.

        Args:
            target_n_params: parameter count of the target model.
            component: if given, predict only this type.

        Returns:
            ``{component_type: predicted_redun_score}``.
        """
        log_n = np.log(target_n_params)
        preds: Dict[str, float] = {}
        for ctype, fp in self.fitted.items():
            if component is not None and ctype != component:
                continue
            preds[ctype] = fp.a * log_n + fp.b
        return preds

    def predict_allocation(
        self,
        target_n_params: int,
        allocator,
        n_layers: int,
        target_bpp: float = 1.2,
    ):
        """Generate a MixedPrecisionConfig for a target model via extrapolation.

        Synthesizes fake RedunResult objects from the scaling law and
        feeds them to the DynamicAllocator.
        """
        from ..quantization.redun_score import RedunResult

        predicted = self.predict_redun(target_n_params)
        attn_r = predicted.get("attn", 0.5)
        mlp_r = predicted.get("mlp", 0.5)

        synthetic_scores: Dict[str, Dict[str, "RedunResult"]] = {}
        for i in range(n_layers):
            synthetic_scores[f"h.{i}"] = {
                "attn_qkv": RedunResult(redun_score=attn_r, hessian_trace=0, mag_cv=0, act_var=0, n_weights=0, component="attn_qkv", layer_idx=i),
                "attn_proj": RedunResult(redun_score=attn_r, hessian_trace=0, mag_cv=0, act_var=0, n_weights=0, component="attn_proj", layer_idx=i),
                "mlp_fc": RedunResult(redun_score=mlp_r, hessian_trace=0, mag_cv=0, act_var=0, n_weights=0, component="mlp_fc", layer_idx=i),
                "mlp_proj": RedunResult(redun_score=mlp_r, hessian_trace=0, mag_cv=0, act_var=0, n_weights=0, component="mlp_proj", layer_idx=i),
            }

        return allocator.allocate(synthetic_scores, n_layers=n_layers, target_bpp=target_bpp)

    def plot_scaling_curve(
        self,
        output_path: Optional[str] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> Optional[Any]:
        """Matplotlib visualization of the scaling hypothesis.

        Returns the figure object (caller can ``plt.show()`` or save).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot.")
            return None

        n_components = len(self.fitted)
        if n_components == 0:
            return None

        fig, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 5))
        if n_components == 1:
            axes = [axes]

        for ax, (ctype, fp) in zip(axes, self.fitted.items()):
            xs = np.array(fp.log_n_values)
            ys = np.array(fp.r_values)
            ax.scatter(xs, ys, label="proxy data", zorder=5)

            x_line = np.linspace(xs.min() - 0.5, xs.max() + 2.0, 100)
            y_line = fp.a * x_line + fp.b
            ax.plot(x_line, y_line, "--", color="red",
                    label=f"R={fp.a:.3f}*ln(N)+{fp.b:.3f}\nR²={fp.r_squared:.3f}")

            if target_sizes:
                for ts in target_sizes:
                    lx = np.log(ts)
                    ax.axvline(lx, color="gray", alpha=0.4, ls=":")
                    ax.annotate(f"{ts/1e9:.0f}B", (lx, ax.get_ylim()[1]),
                                fontsize=8, ha="center")

            ax.set_xlabel("ln(N_params)")
            ax.set_ylabel("Redun Score")
            ax.set_title(ctype)
            ax.legend(fontsize=8)

        fig.suptitle("Proxy-SR-VQ Scaling Law", fontsize=14)
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved scaling curve to {output_path}")

        return fig

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted parameters for JSON export."""
        return {
            ctype: {
                "a": fp.a, "b": fp.b,
                "r_squared": fp.r_squared,
                "n_points": fp.n_points,
            }
            for ctype, fp in self.fitted.items()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScalingLawFitter":
        """Restore from JSON dict."""
        fitter = cls()
        for ctype, vals in d.items():
            fitter.fitted[ctype] = FittedParams(
                component=ctype,
                a=vals["a"], b=vals["b"],
                r_squared=vals.get("r_squared", 0.0),
                n_points=vals.get("n_points", 0),
            )
        return fitter

    @staticmethod
    def _component_type(comp_name: str) -> str:
        if comp_name.startswith("attn"):
            return "attn"
        if comp_name.startswith("mlp"):
            return "mlp"
        return comp_name
