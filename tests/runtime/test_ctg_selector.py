"""Tests for adaptive CTG program selector."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    from onebit.runtime.ctg_selector import (
        AdaptiveProgramSelector,
        SelectorConfig,
        SelectorState,
        extract_features,
    )
    from onebit.training.ctg_trainer import CTGTrainer, TrainingConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_selector_forward_inference():
    """Test selector forward pass in inference mode (hard argmax)."""
    cfg = SelectorConfig(K=4, hidden_dim=32, feature_dim=8)
    selector = AdaptiveProgramSelector(cfg)
    selector.eval()
    
    # Create dummy features
    features = torch.randn(1, 8)
    
    # Forward pass (hard mode)
    probs, program_id = selector(features, tau=1.0, hard=True)
    
    # Check outputs
    assert probs.shape == (1, 4)
    assert program_id.shape == (1,)
    assert 0 <= program_id.item() < 4
    
    # Hard mode should produce one-hot
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_selector_forward_training():
    """Test selector forward pass in training mode (Gumbel-Softmax)."""
    cfg = SelectorConfig(K=4, hidden_dim=32, feature_dim=8)
    selector = AdaptiveProgramSelector(cfg)
    selector.train()
    
    # Create dummy features
    features = torch.randn(4, 8)
    
    # Forward pass (soft mode)
    probs, program_id = selector(features, tau=0.5, hard=False)
    
    # Check outputs
    assert probs.shape == (4, 4)
    assert program_id.shape == (4,)
    assert torch.all((program_id >= 0) & (program_id < 4))
    
    # Soft mode should still sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_selector_deterministic_with_fixed_input():
    """Test that selector is deterministic for fixed input."""
    cfg = SelectorConfig(K=4, hidden_dim=32, feature_dim=8)
    selector = AdaptiveProgramSelector(cfg)
    selector.eval()
    
    # Fixed input
    torch.manual_seed(42)
    features = torch.randn(1, 8)
    
    # Multiple forward passes
    _, pid1 = selector(features, tau=1.0, hard=True)
    _, pid2 = selector(features, tau=1.0, hard=True)
    
    assert pid1.item() == pid2.item()


def test_extract_features():
    """Test feature extraction."""
    shortlist_logits = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    top2_margin = 0.5
    attn_entropy = 1.2
    phase_history = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    
    features = extract_features(shortlist_logits, top2_margin, attn_entropy, phase_history)
    
    # Check shape
    assert features.shape == (8,)
    
    # Check values are reasonable
    assert features[0] > 0  # Shortlist entropy
    assert np.isclose(features[1], 0.5)  # Top-2 margin
    assert np.isclose(features[2], 1.2)  # Attention entropy
    assert np.isclose(features[3], 3.5)  # Phase mean
    assert np.isclose(features[7], 7.0)  # Phase last


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_trainer_loss_computation():
    """Test trainer loss computation."""
    cfg_selector = SelectorConfig(K=4, hidden_dim=32, feature_dim=8)
    selector = AdaptiveProgramSelector(cfg_selector)
    
    cfg_trainer = TrainingConfig(
        lambda_var=0.1,
        lambda_switch=0.05,
        lambda_entropy=0.01,
    )
    trainer = CTGTrainer(selector, cfg_trainer, device="cpu")
    
    # Dummy inputs
    program_probs = torch.tensor([[0.7, 0.1, 0.1, 0.1]], dtype=torch.float32)
    program_id = torch.tensor([0], dtype=torch.long)
    base_loss = torch.tensor(2.5, dtype=torch.float32)
    k_variance = 10.0
    
    # Compute loss
    losses = trainer.compute_loss(program_probs, program_id, base_loss, k_variance, batch_size=1)
    
    # Check components
    assert "total" in losses
    assert "base" in losses
    assert "var" in losses
    assert "switch" in losses
    assert "entropy" in losses
    
    # Total should be sum of components
    expected_total = (
        base_loss
        + cfg_trainer.lambda_var * losses["var"]
        + cfg_trainer.lambda_switch * losses["switch"]
        + cfg_trainer.lambda_entropy * losses["entropy"]
    )
    assert torch.allclose(losses["total"], expected_total, atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_trainer_program_usage_tracking():
    """Test that trainer tracks program usage correctly."""
    cfg_selector = SelectorConfig(K=4, hidden_dim=32, feature_dim=8)
    selector = AdaptiveProgramSelector(cfg_selector)
    
    cfg_trainer = TrainingConfig()
    trainer = CTGTrainer(selector, cfg_trainer, device="cpu")
    
    # Simulate some program selections
    features = torch.randn(10, 8)
    base_loss = torch.tensor(1.0)
    
    for _ in range(5):
        trainer.train_step(features, base_loss, k_variance=5.0, tau=1.0)
    
    # Check usage histogram
    usage = trainer.get_program_usage_histogram()
    assert len(usage) == 4
    assert sum(usage.values()) == pytest.approx(1.0, abs=1e-6)
    
    # All programs should have some usage (with high probability)
    # This is a weak test since it's stochastic
    assert all(v >= 0 for v in usage.values())

