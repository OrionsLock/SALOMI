"""SPRT-DAG for pairwise certification of Top-T candidates.

Uses Wald's Sequential Probability Ratio Test (SPRT) to certify ordering
of candidates with early stopping. DAG pruning reduces redundant comparisons
via transitive closure.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Tuple


@dataclass(frozen=True)
class SPRTConfig:
    """Configuration for SPRT-based certification.
    
    Args:
        eps: Effect size for H1 (e.g., 0.05 means p1 = 0.5 + eps/2)
        alpha: Type I error rate per tested pair
        beta: Type II error rate per tested pair
        k_max: Maximum ticks budget for attention certification
        chunk: Ticks per backend call (default: 4)
        seed: PRF seed for deterministic BSDM-W streams
    """
    eps: float
    alpha: float
    beta: float
    k_max: int
    chunk: int = 4
    seed: int = 0


class SPRTDAG:
    """Maintains pair-wise SPRT states over candidates {0..T-1}.
    
    DAG pruning: test only edges on the frontier; skip edges implied
    by decided chains (topological partial order).
    
    SPRT thresholds:
        A = log((1-beta)/alpha)  # accept H1 (i > j)
        B = log(beta/(1-alpha))  # accept H0 (i <= j, or j > i)
    
    LLR update per observation X_ij = 1[y_i > y_j]:
        p0 = 0.5 (null: equal means)
        p1 = 0.5 + eps/2 (alternative: i better by eps)
        LLR += X*log(p1/p0) + (1-X)*log((1-p1)/(1-p0))
    """
    
    def __init__(self, T: int, cfg: SPRTConfig):
        """Initialize SPRT-DAG for T candidates.
        
        Args:
            T: Number of candidates (typically 8, 12, or 16)
            cfg: SPRT configuration
        """
        self.T = T
        self.cfg = cfg
        
        # SPRT thresholds
        self.A = np.log((1 - cfg.beta) / cfg.alpha)
        self.B = np.log(cfg.beta / (1 - cfg.alpha))
        
        # Per-pair (i, j) with i < j:
        self.llr = np.zeros((T, T), dtype=np.float64)  # Log-likelihood ratio
        self.n = np.zeros((T, T), dtype=np.int32)      # Number of observations
        self.state = np.zeros((T, T), dtype=np.int8)   # 0=undecided, +1=i>j, -1=j>i
        
        # Adjacency matrix: adj[i,j]=1 if decided i>j
        self.adj = np.zeros((T, T), dtype=np.int8)
        
        # Reachability matrix (transitive closure of adj)
        self.reach = np.zeros((T, T), dtype=np.int8)
        
        # Precompute log-likelihood ratio increments
        p0 = 0.5
        p1 = 0.5 + cfg.eps / 2.0
        self.llr_inc_1 = np.log(p1 / p0)  # When X=1 (i > j)
        self.llr_inc_0 = np.log((1 - p1) / (1 - p0))  # When X=0 (i <= j)
    
    def _update_reachability(self):
        """Update transitive closure using Floyd-Warshall.
        
        After each new decision, recompute which pairs are transitively implied.
        """
        self.reach = self.adj.copy()
        
        # Floyd-Warshall: reach[i,j] = 1 if path i->j exists
        for k in range(self.T):
            for i in range(self.T):
                for j in range(self.T):
                    if self.reach[i, k] and self.reach[k, j]:
                        self.reach[i, j] = 1
    
    def undecided_pairs(self) -> List[Tuple[int, int]]:
        """Return frontier pairs not already implied by transitive closure.
        
        A pair (i, j) is on the frontier if:
        - Not yet decided (state[i,j] == 0)
        - Not transitively implied (no path i->j or j->i in reach)
        
        Returns:
            List of (i, j) pairs with i < j
        """
        pairs = []
        for i in range(self.T):
            for j in range(i + 1, self.T):
                if self.state[i, j] == 0:
                    # Check if transitively implied
                    if not (self.reach[i, j] or self.reach[j, i]):
                        pairs.append((i, j))
        return pairs
    
    def update_pairs_from_tick(self, y_tick: np.ndarray):
        """Update LLRs from one tick of normalized scores.
        
        Args:
            y_tick: float[T] normalized mean scores for T candidates at this tick
        
        For each undecided pair (i, j):
            - Compute X_ij = 1 if y_i > y_j, else 0
            - Update LLR[i,j] += X*llr_inc_1 + (1-X)*llr_inc_0
            - Check thresholds: if LLR >= A, decide i>j; if LLR <= B, decide j>i
            - Update adjacency and reachability
        """
        frontier = self.undecided_pairs()
        
        for i, j in frontier:
            # Bernoulli observation: X = 1[y_i > y_j]
            # Deterministic tie-breaking: if y_i == y_j, treat as X=0 (bias to H0)
            if y_tick[i] > y_tick[j]:
                X = 1
                self.llr[i, j] += self.llr_inc_1
            else:
                X = 0
                self.llr[i, j] += self.llr_inc_0
            
            self.n[i, j] += 1
            
            # Check thresholds
            if self.llr[i, j] >= self.A:
                # Accept H1: i > j
                self.state[i, j] = 1
                self.adj[i, j] = 1
                self._update_reachability()
            elif self.llr[i, j] <= self.B:
                # Accept H0: j > i
                self.state[i, j] = -1
                self.adj[j, i] = 1
                self._update_reachability()
    
    def decided_edges(self) -> List[Tuple[int, int, int]]:
        """Return list of decided edges.
        
        Returns:
            List of (i, j, sign) where sign=+1 means i>j, sign=-1 means j>i
        """
        edges = []
        for i in range(self.T):
            for j in range(i + 1, self.T):
                if self.state[i, j] != 0:
                    edges.append((i, j, int(self.state[i, j])))
        return edges
    
    def top1_if_certified(self) -> Optional[int]:
        """Return Top-1 candidate if certified (dominates all others).
        
        A candidate k is Top-1 if there exists a path k->j for all j != k
        in the adjacency graph (i.e., k dominates all others).
        
        Returns:
            Index of Top-1 candidate if certified, else None
        """
        for k in range(self.T):
            # Check if k has path to all others
            dominates_all = True
            for j in range(self.T):
                if j != k and not self.reach[k, j]:
                    dominates_all = False
                    break
            
            if dominates_all:
                return k
        
        return None
    
    def all_pairs_decided(self) -> bool:
        """Check if all pairs have been decided.
        
        Returns:
            True if no undecided pairs remain
        """
        return len(self.undecided_pairs()) == 0
    
    def stats(self) -> dict:
        """Return statistics about current state.
        
        Returns:
            Dict with counts of decided/undecided pairs, observations, etc.
        """
        decided = np.sum(self.state != 0)
        undecided = np.sum(self.state == 0)
        total_obs = np.sum(self.n)
        
        return {
            "decided_pairs": int(decided),
            "undecided_pairs": int(undecided),
            "total_observations": int(total_obs),
            "frontier_size": len(self.undecided_pairs()),
        }

