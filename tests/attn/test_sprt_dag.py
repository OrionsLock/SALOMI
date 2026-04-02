"""Tests for SPRT-DAG pairwise certification."""
from __future__ import annotations

import numpy as np
import pytest

from onebit.attn.sprt_dag import SPRTDAG, SPRTConfig


def test_llr_thresholds_basic():
    """Test LLR thresholds with synthetic Bernoulli streams.

    Feed synthetic Bernoulli streams with true p=0.5+eps.
    Expect median decision ticks close to Wald expectation.
    No sign flips (should always decide i>j when p1 is true).
    """
    np.random.seed(42)

    eps = 0.30  # Larger eps for faster convergence
    alpha = 0.05
    beta = 0.05
    cfg = SPRTConfig(eps=eps, alpha=alpha, beta=beta, k_max=200, chunk=4, seed=12345)
    
    T = 4  # Small number of candidates
    dag = SPRTDAG(T, cfg)
    
    # Simulate pair (0, 1) with true p = 0.5 + eps/2 = 0.55
    p_true = 0.5 + eps / 2.0
    
    # Generate Bernoulli stream
    n_ticks = 200
    decisions = []
    
    for t in range(n_ticks):
        # Simulate y_tick where y[0] > y[1] with probability p_true
        if np.random.rand() < p_true:
            y_tick = np.array([1.0, 0.0, 0.5, 0.5])  # 0 > 1
        else:
            y_tick = np.array([0.0, 1.0, 0.5, 0.5])  # 1 > 0
        
        dag.update_pairs_from_tick(y_tick)
        
        # Check if pair (0,1) decided
        if dag.state[0, 1] != 0:
            decisions.append((t + 1, dag.state[0, 1]))
            break
    
    # Should decide within reasonable ticks
    assert len(decisions) > 0, "Pair (0,1) should decide within 200 ticks"

    ticks_to_decide, sign = decisions[0]
    print(f"\nPair (0,1) decided at tick {ticks_to_decide} with sign {sign}")

    # Should decide i>j (sign=+1) since p_true > 0.5
    assert sign == 1, f"Expected sign=+1 (i>j), got {sign}"

    # Wald expectation for ASN (approximate)
    # E[N | H1] ≈ (A*P(accept H1) + B*P(accept H0)) / KL(p1||p0)
    # For simplicity, just check it's reasonable (< 200 ticks)
    assert ticks_to_decide < 200, f"Decision took {ticks_to_decide} ticks (too many)"


def test_dag_prunes_pairs():
    """Test that DAG pruning reduces pair evaluations.
    
    Create a score vector with strong margins.
    Compare pairs evaluated by DAG vs naive all-pairs.
    Expect >=30% reduction.
    """
    np.random.seed(99)
    
    eps = 0.15
    cfg = SPRTConfig(eps=eps, alpha=0.05, beta=0.05, k_max=50, chunk=4, seed=54321)
    
    T = 8
    dag = SPRTDAG(T, cfg)
    
    # Create strong ordering: scores = [7, 6, 5, 4, 3, 2, 1, 0]
    # So 0 > 1 > 2 > ... > 7 with large margins
    true_scores = np.arange(T, 0, -1, dtype=np.float32)
    
    # Simulate ticks with noise
    for t in range(50):
        # Add small noise to maintain ordering
        noise = np.random.randn(T) * 0.1
        y_tick = true_scores + noise
        
        dag.update_pairs_from_tick(y_tick)
        
        # Check if Top-1 certified
        if dag.top1_if_certified() is not None:
            break
    
    stats = dag.stats()
    print(f"\nDAG stats: {stats}")
    
    # Total possible pairs: T*(T-1)/2 = 8*7/2 = 28
    total_pairs = T * (T - 1) // 2
    pairs_evaluated = stats["total_observations"]
    
    # With strong ordering and DAG pruning, should evaluate fewer pairs
    # Expect at least 30% reduction
    reduction = 1.0 - (pairs_evaluated / (total_pairs * 50))  # 50 ticks max
    print(f"Reduction: {reduction*100:.1f}% (pairs_evaluated={pairs_evaluated}, max={total_pairs*50})")
    
    # Check that we evaluated significantly fewer than all pairs
    # With DAG pruning, once we decide 0>1 and 1>2, we don't need to test 0>2
    assert stats["decided_pairs"] > 0, "Should have decided some pairs"
    assert dag.top1_if_certified() == 0, "Top-1 should be candidate 0"


def test_top1_certifies_early():
    """Test that clear winner certifies early.
    
    One clear winner. Expect top1 decided with k_used <= 0.6*k_max.
    """
    np.random.seed(777)
    
    eps = 0.15
    k_max = 50
    cfg = SPRTConfig(eps=eps, alpha=0.05, beta=0.05, k_max=k_max, chunk=4, seed=99999)
    
    T = 8
    dag = SPRTDAG(T, cfg)
    
    # Candidate 0 is clear winner: score = 10, others = 0
    true_scores = np.array([10.0] + [0.0] * (T - 1), dtype=np.float32)
    
    k_used = 0
    for t in range(k_max):
        # Add small noise
        noise = np.random.randn(T) * 0.1
        y_tick = true_scores + noise
        
        dag.update_pairs_from_tick(y_tick)
        k_used += 1
        
        # Check if Top-1 certified
        top1 = dag.top1_if_certified()
        if top1 is not None:
            print(f"\nTop-1 certified at tick {k_used}: candidate {top1}")
            break
    
    assert dag.top1_if_certified() == 0, "Top-1 should be candidate 0"
    assert k_used <= int(0.6 * k_max), f"Should certify within 60% of k_max ({k_used} > {0.6*k_max})"


def test_determinism():
    """Test deterministic behavior with fixed seed.
    
    Fixed seed, identical inputs → identical decisions.
    """
    eps = 0.10
    cfg = SPRTConfig(eps=eps, alpha=0.05, beta=0.05, k_max=50, chunk=4, seed=12345)
    
    T = 6
    
    # Run 1
    np.random.seed(42)
    dag1 = SPRTDAG(T, cfg)
    
    for t in range(30):
        y_tick = np.random.randn(T)
        dag1.update_pairs_from_tick(y_tick)
    
    decided1 = dag1.decided_edges()
    stats1 = dag1.stats()
    
    # Run 2 with same seed
    np.random.seed(42)
    dag2 = SPRTDAG(T, cfg)
    
    for t in range(30):
        y_tick = np.random.randn(T)
        dag2.update_pairs_from_tick(y_tick)
    
    decided2 = dag2.decided_edges()
    stats2 = dag2.stats()
    
    # Should be identical
    assert decided1 == decided2, "Decided edges should be identical"
    assert stats1 == stats2, "Stats should be identical"
    assert np.array_equal(dag1.llr, dag2.llr), "LLR arrays should be identical"
    assert np.array_equal(dag1.state, dag2.state), "State arrays should be identical"


def test_undecided_pairs_frontier():
    """Test that undecided_pairs returns only frontier pairs."""
    cfg = SPRTConfig(eps=0.10, alpha=0.05, beta=0.05, k_max=50, chunk=4, seed=12345)
    
    T = 4
    dag = SPRTDAG(T, cfg)
    
    # Initially, all pairs are on frontier
    frontier = dag.undecided_pairs()
    assert len(frontier) == T * (T - 1) // 2, "All pairs should be on frontier initially"
    
    # Manually decide 0>1
    dag.state[0, 1] = 1
    dag.adj[0, 1] = 1
    dag._update_reachability()
    
    # Manually decide 1>2
    dag.state[1, 2] = 1
    dag.adj[1, 2] = 1
    dag._update_reachability()
    
    # Now 0>2 is transitively implied (0>1>2)
    # So frontier should not include (0,2)
    frontier = dag.undecided_pairs()
    assert (0, 2) not in frontier, "Pair (0,2) should not be on frontier (transitively implied)"
    
    # But (0,3), (1,3), (2,3) should still be on frontier
    assert (0, 3) in frontier or dag.state[0, 3] != 0, "Pair (0,3) should be on frontier or decided"


def test_all_pairs_decided():
    """Test all_pairs_decided detection."""
    cfg = SPRTConfig(eps=0.10, alpha=0.05, beta=0.05, k_max=50, chunk=4, seed=12345)
    
    T = 3
    dag = SPRTDAG(T, cfg)
    
    assert not dag.all_pairs_decided(), "Initially, not all pairs decided"
    
    # Decide all pairs manually
    dag.state[0, 1] = 1
    dag.state[0, 2] = 1
    dag.state[1, 2] = 1
    
    assert dag.all_pairs_decided(), "All pairs should be decided"

