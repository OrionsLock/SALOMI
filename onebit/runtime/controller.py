"""Controller for per-token inference with Stage-A + SPRT certification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Optional
import numpy as np
import os

from ..ops.attention_probe import stageA_probe_topT
from ..attn.runner import certify_topT
from ..attn.sprt_dag import SPRTConfig


@dataclass(frozen=True)
class BudgetCfg:
    """Budget smoothing configuration (PR-4.1).

    Args:
        alpha_T: EMA decay for Stage-A T (default: 0.20)
        alpha_k: EMA decay for attn k_used (default: 0.10)
        beta_save: Target % savings vs EMA k (default: 0.10)
        warmup_tokens: No smoothing until this many tokens (default: 16)
        k_floor: Minimum k budget (default: 8)
        k_cap: Maximum k budget (default: 64)
        T_lo_hi: Hysteresis thresholds for T quantization (default: (10, 14))
    """
    alpha_T: float = 0.20
    alpha_k: float = 0.10
    beta_save: float = 0.10
    warmup_tokens: int = 16
    k_floor: int = 8
    k_cap: int = 64
    T_lo_hi: tuple[int, int] = (10, 14)

    def __post_init__(self):
        # Allow env overrides
        if val := os.getenv("ONEBIT_ALPHA_T"):
            object.__setattr__(self, "alpha_T", float(val))
        if val := os.getenv("ONEBIT_ALPHA_K"):
            object.__setattr__(self, "alpha_k", float(val))
        if val := os.getenv("ONEBIT_BETA_SAVE"):
            object.__setattr__(self, "beta_save", float(val))


@dataclass
class BudgetState:
    """Budget smoothing state (PR-4.1).

    Args:
        tok_seen: Number of tokens processed
        ema_T: EMA of Stage-A T per (layer, head), shape [L, H]
        ema_k: EMA of attn k_used per (layer, head), shape [L, H]
    """
    tok_seen: int
    ema_T: np.ndarray  # [L, H], float32
    ema_k: np.ndarray  # [L, H], float32


@dataclass(frozen=True)
class CtrlConfig:
    """Controller configuration for per-token inference.

    Args:
        kA: Stage-A probe ticks (default: 16)
        k_max: Max ticks for SPRT certification (default: 64)
        chunk: Ticks per backend call (default: 4)
        delta_total: Total risk budget per token (default: 0.01)
        attn_share: Fraction of delta_total for attention (default: 0.5)
        eps: Effect size for SPRT (default: 0.05)
        backend: "cpu" or "opencl" (default: "opencl")
        walsh_N: Walsh carriers per tick (default: 2)
        antithetic: Use antithetic pairs (default: True)
        order: ΣΔ modulator order (default: 2)
        beta: ΣΔ-2 beta parameter (default: 0.30)
        lambd: ΣΔ leak parameter (default: 1/256)
    """
    kA: int = 16
    k_max: int = 64
    chunk: int = 4
    delta_total: float = 0.01
    attn_share: float = 0.5
    delta_attn: Optional[float] = None
    eps: float = 0.05
    eps_attn: Optional[float] = None
    backend: Literal["cpu", "opencl"] = "opencl"
    walsh_N: int = 2
    antithetic: bool = True
    order: int = 2
    beta: float = 0.30
    lambd: float = 1.0 / 256.0


# ========== PR-4.1: Budget Smoothing Helpers ==========

def _ema(prev: float, x: float, a: float) -> float:
    """Exponential moving average."""
    return (1.0 - a) * prev + a * x


def _quantize_T(x: float, lo: int, hi: int) -> int:
    """Quantize T to buckets {8, 12, 16} with hysteresis.

    Args:
        x: Raw T value
        lo: Lower threshold (default: 10)
        hi: Upper threshold (default: 14)

    Returns:
        8 if x <= lo, 16 if x >= hi, else 12
    """
    if x <= lo:
        return 8
    elif x >= hi:
        return 16
    else:
        return 12


class BudgetSmoother:
    """Per-head budget smoother with EMA (PR-4.1).

    Maintains EMA of Stage-A T and attn k_used per (layer, head).
    Provides smoothed T and k budgets with warmup and hysteresis.
    """

    def __init__(self, cfg: BudgetCfg, st: BudgetState):
        self.cfg = cfg
        self.st = st

    def next_T(self, l: int, h: int, T_raw: int) -> int:
        """Compute smoothed T for Stage-A.

        Args:
            l: Layer index
            h: Head index
            T_raw: Raw T from elbow (in {8, 12, 16})

        Returns:
            Smoothed T (quantized to {8, 12, 16})
        """
        # Warmup: passthrough
        if self.st.tok_seen < self.cfg.warmup_tokens:
            return T_raw

        # Update EMA
        v = _ema(self.st.ema_T[l, h], float(T_raw), self.cfg.alpha_T)
        self.st.ema_T[l, h] = v

        # Quantize with hysteresis
        return _quantize_T(v, *self.cfg.T_lo_hi)

    def next_k_budget(self, l: int, h: int, k_used_prev: int, unsure_prev: bool) -> int:
        """Compute k budget for SPRT.

        Args:
            l: Layer index
            h: Head index
            k_used_prev: k_used from previous token
            unsure_prev: True if previous token was UNSURE

        Returns:
            k budget (clamped to [k_floor, k_cap])
        """
        # Update EMA on realized k (clip to cap)
        k_obs = min(max(k_used_prev, self.cfg.k_floor), self.cfg.k_cap)
        a = self.cfg.alpha_k
        self.st.ema_k[l, h] = _ema(self.st.ema_k[l, h], float(k_obs), a)

        # Derive target budget with savings
        target = int(round(self.st.ema_k[l, h] * (1.0 - self.cfg.beta_save)))

        # Penalty bump if last token was UNSURE
        if unsure_prev:
            target = min(self.cfg.k_cap, target + 8)

        return max(self.cfg.k_floor, min(self.cfg.k_cap, target))

    def tick(self):
        """Increment token counter."""
        self.st.tok_seen += 1


# ========== End PR-4.1 ==========


def infer_one_token(
    Q_bits: np.ndarray,
    K_bits: np.ndarray,
    *,
    cfg: CtrlConfig,
    prf_seed: int,
) -> Dict:
    """Orchestrate Stage-A + SPRT for one token.
    
    Risk budget allocation:
        alpha = beta = (delta_total * attn_share) / 2
    
    Status codes:
        ATTN_CERT_OK: Top-1 certified
        UNSURE: Exhausted k_max without certification
    
    Args:
        Q_bits: Query vector [Kw] (packed bits)
        K_bits: Key matrix [L, Kw] (packed bits)
        cfg: Controller configuration
        prf_seed: PRF seed for deterministic streams
    
    Returns:
        Certificate bundle with:
            "status": "ATTN_CERT_OK" or "UNSURE"
            "top1": Optional[int] - Top-1 index if certified
            "idx_top": Top-T indices from Stage-A
            "T_sel": Selected T from elbow
            "kA": Stage-A ticks used
            "k_attn_used": SPRT ticks used
            "pairs_evaluated": Total pair observations
            "alpha": Type I error per pair
            "beta": Type II error per pair
            "delta_total": Total risk budget
            "backend": Backend used
            "prf_seed": PRF seed
            "decided": Decided edges
            "undecided": Undecided pairs
            "dag_stats": DAG statistics
    """
    # Stage-A: probe all keys with kA ticks
    stageA_result = stageA_probe_topT(
        Q_bits, K_bits,
        kA=cfg.kA,
        prf_seed=prf_seed,
        walsh_N=cfg.walsh_N,
        antithetic=cfg.antithetic,
        order=cfg.order,
        beta=cfg.beta,
        lambd=cfg.lambd,
    )
    
    T_sel = stageA_result["T_sel"]
    idx_top = stageA_result["idx_top"]
    
    # Allocate risk budget
    # alpha = beta = (delta_total * attn_share) / 2
    delta_attn = cfg.delta_attn if cfg.delta_attn is not None else cfg.delta_total * cfg.attn_share
    eps_attn = cfg.eps_attn if cfg.eps_attn is not None else cfg.eps
    alpha = beta = delta_attn / 2.0

    # SPRT certification
    sprt_cfg = SPRTConfig(
        eps=eps_attn,
        alpha=alpha,
        beta=beta,
        k_max=cfg.k_max,
        chunk=cfg.chunk,
        seed=prf_seed,
    )

    sprt_result = certify_topT(
        Q_bits, K_bits, idx_top,
        cfg=sprt_cfg,
        backend=cfg.backend,
        prf_seed=prf_seed + 1000,  # Different seed for SPRT
        walsh_N=cfg.walsh_N,
        antithetic=cfg.antithetic,
        order=cfg.order,
        beta=cfg.beta,
        lambd=cfg.lambd,
    )
    
    # Determine status
    top1 = sprt_result["top1"]
    if top1 is not None:
        status = "ATTN_CERT_OK"
    else:
        status = "UNSURE"
    
    # Build certificate
    cert = {
        "status": status,
        "top1": top1,
        "idx_top": idx_top,
        "T_sel": T_sel,
        "kA": cfg.kA,
        "k_attn_used": sprt_result["k_used"],
        "pairs_evaluated": sprt_result["pairs_evaluated"],
        "alpha": alpha,
        "beta": beta,
        "delta_total": cfg.delta_total,
        "backend": cfg.backend,
        "prf_seed": prf_seed,
        "decided": sprt_result["decided"],
        "undecided": sprt_result["undecided"],
        "dag_stats": sprt_result["dag_stats"],
    }
    
    return cert

