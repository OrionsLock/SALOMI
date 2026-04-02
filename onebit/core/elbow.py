"""Elbow detection for Stage-A probe T selection.

Two-line least squares break detector for selecting T from {8,12,16}.
"""
from __future__ import annotations
import numpy as np


def compute_elbow(mu: np.ndarray, T_set: tuple[int, ...] = (8, 12, 16)) -> tuple[int, float]:
    """Detect elbow in sorted means and map to nearest T in T_set.
    
    Algorithm:
    1. Sort mu descending
    2. Compute differences d_i = mu_i - mu_{i+1}
    3. Find elbow j* = argmax_i(d_i - d_{i+1}) on top 64 keys (or len(mu) if smaller)
    4. Map j* to nearest T in T_set with min absolute error
    5. If ambiguous (tie), prefer smaller T
    
    Args:
        mu: array of means (any order)
        T_set: allowed T values
        
    Returns:
        (T_sel, gap) where:
            T_sel: selected T from T_set
            gap: mu[j*] - mu[j*+1] (the elbow gap)
    """
    mu = np.asarray(mu, dtype=np.float32)
    L = len(mu)
    if L == 0:
        return T_set[0], 0.0
    
    # Sort descending
    mu_sorted = np.sort(mu)[::-1]
    
    # Compute differences
    if L == 1:
        return T_set[0], 0.0
    
    diffs = mu_sorted[:-1] - mu_sorted[1:]  # d_i = mu_i - mu_{i+1}
    
    # Find elbow on top 64 keys (or all if fewer)
    n_top = min(64, L - 1)
    if n_top <= 1:
        j_star = 0
    else:
        # Compute second differences: d_i - d_{i+1}
        second_diffs = diffs[:n_top-1] - diffs[1:n_top]
        j_star = int(np.argmax(second_diffs))
    
    # Gap at elbow
    gap = float(diffs[j_star]) if j_star < len(diffs) else 0.0
    
    # Map j* to nearest T in T_set
    # j* is the index of the elbow, so we want T ≈ j* + 1 (since j* is 0-indexed)
    elbow_T_raw = j_star + 1
    
    # Find nearest T in T_set
    T_arr = np.array(T_set, dtype=np.int32)
    distances = np.abs(T_arr - elbow_T_raw)
    min_dist = np.min(distances)
    candidates = T_arr[distances == min_dist]
    
    # If tie, prefer smaller T
    T_sel = int(np.min(candidates))
    
    return T_sel, gap

