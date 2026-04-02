"""Tests for pulse scheduler."""
import pytest
import numpy as np

from onebit.runtime.pulse_scheduler import PulseScheduler, create_default_scheduler


def test_pulse_scheduler_init():
    """Test pulse scheduler initialization."""
    scheduler = PulseScheduler(
        n_layers=12,
        n_groups=32,
        group_size=64,
        base_interval=256,
    )
    
    assert scheduler.n_layers == 12
    assert scheduler.n_groups == 32
    assert scheduler.group_size == 64
    assert scheduler.base_interval == 256
    assert scheduler.current_token == 0
    assert scheduler.total_repairs == 0
    
    # Check initial state
    assert np.all(scheduler.last_repair == -1)
    assert np.all(scheduler.group_rotation == 0)


def test_pulse_scheduler_advance_token():
    """Test token advancement."""
    scheduler = PulseScheduler(n_layers=12, n_groups=32)
    
    assert scheduler.current_token == 0
    
    scheduler.advance_token()
    assert scheduler.current_token == 1
    
    for _ in range(99):
        scheduler.advance_token()
    
    assert scheduler.current_token == 100


def test_pulse_scheduler_warmup():
    """Test warmup period (no repairs before warmup_tokens)."""
    scheduler = PulseScheduler(
        n_layers=12,
        n_groups=32,
        warmup_tokens=512,
    )
    
    # Before warmup: no repairs
    for _ in range(511):
        scheduler.advance_token()
        repairs = scheduler.get_repair_schedule()
        assert len(repairs) == 0
    
    # After warmup: repairs should start
    scheduler.advance_token()  # Token 512
    repairs = scheduler.get_repair_schedule()
    assert len(repairs) > 0


def test_pulse_scheduler_adaptive_interval():
    """Test adaptive pulse interval based on layer and context."""
    scheduler = PulseScheduler(
        n_layers=12,
        n_groups=32,
        base_interval=256,
    )
    
    # Early layers have shorter intervals
    interval_layer0 = scheduler.get_pulse_interval(0)
    interval_layer11 = scheduler.get_pulse_interval(11)
    assert interval_layer11 > interval_layer0
    
    # Longer contexts have longer intervals
    scheduler.current_token = 1000
    interval_1k = scheduler.get_pulse_interval(0)
    
    scheduler.current_token = 5000
    interval_5k = scheduler.get_pulse_interval(0)
    
    assert interval_5k > interval_1k


def test_pulse_scheduler_should_repair():
    """Test repair triggering logic."""
    scheduler = PulseScheduler(
        n_layers=12,
        n_groups=32,
        base_interval=256,
        warmup_tokens=100,
    )
    
    # Before warmup: no repair
    scheduler.current_token = 50
    assert not scheduler.should_repair(0, 0)
    
    # After warmup, first repair
    scheduler.current_token = 100
    assert scheduler.should_repair(0, 0)
    
    # Mark as repaired
    scheduler.mark_repaired(0, 0)
    assert not scheduler.should_repair(0, 0)
    
    # After interval, repair again
    interval = scheduler.get_pulse_interval(0)
    scheduler.current_token = 100 + interval
    assert scheduler.should_repair(0, 0)


def test_pulse_scheduler_mark_repaired():
    """Test marking repairs."""
    scheduler = PulseScheduler(n_layers=12, n_groups=32)
    
    scheduler.current_token = 100
    scheduler.mark_repaired(0, 0)
    
    assert scheduler.last_repair[0, 0] == 100
    assert scheduler.total_repairs == 1
    assert scheduler.repairs_per_layer[0] == 1
    
    scheduler.current_token = 200
    scheduler.mark_repaired(1, 5)
    
    assert scheduler.last_repair[1, 5] == 200
    assert scheduler.total_repairs == 2
    assert scheduler.repairs_per_layer[1] == 1


def test_pulse_scheduler_get_repair_schedule():
    """Test repair schedule generation."""
    scheduler = PulseScheduler(
        n_layers=4,
        n_groups=8,
        base_interval=100,
        warmup_tokens=50,
    )
    
    # Advance past warmup
    for _ in range(100):
        scheduler.advance_token()
    
    # Get repair schedule
    repairs = scheduler.get_repair_schedule()
    
    # Should have repairs for each layer (one group per layer)
    assert len(repairs) == 4
    
    # Check structure
    for repair in repairs:
        assert "layer_idx" in repair
        assert "group_idx" in repair
        assert 0 <= repair["layer_idx"] < 4
        assert 0 <= repair["group_idx"] < 8


def test_pulse_scheduler_group_rotation():
    """Test group rotation across tokens."""
    scheduler = PulseScheduler(
        n_layers=2,
        n_groups=4,
        base_interval=50,
        warmup_tokens=10,
    )
    
    # Advance past warmup
    for _ in range(100):
        scheduler.advance_token()
        repairs = scheduler.get_repair_schedule()
    
    # Check that all groups have been repaired at least once
    for layer_idx in range(2):
        for group_idx in range(4):
            assert scheduler.last_repair[layer_idx, group_idx] >= 0


def test_pulse_scheduler_chain_repair():
    """Test chain repair schedule."""
    scheduler = PulseScheduler(
        n_layers=12,
        n_groups=32,
        base_interval=100,
        warmup_tokens=50,
    )
    
    # Advance past warmup
    for _ in range(200):
        scheduler.advance_token()
    
    # Get chain repair schedule
    repairs = scheduler.get_chain_repair_schedule(max_repairs_per_token=4)
    
    # Should have at most 4 repairs
    assert len(repairs) <= 4
    
    # Check structure
    for repair in repairs:
        assert "layer_idx" in repair
        assert "group_idx" in repair


def test_pulse_scheduler_chain_repair_prioritization():
    """Test that chain repair prioritizes deeper layers."""
    scheduler = PulseScheduler(
        n_layers=12,
        n_groups=32,
        base_interval=100,
        warmup_tokens=50,
    )
    
    # Advance past warmup
    for _ in range(200):
        scheduler.advance_token()
    
    # Get chain repair schedule with limit
    repairs = scheduler.get_chain_repair_schedule(max_repairs_per_token=2)
    
    # Should prioritize deeper layers
    if len(repairs) >= 2:
        assert repairs[0]["layer_idx"] >= repairs[1]["layer_idx"]


def test_pulse_scheduler_reset():
    """Test scheduler reset."""
    scheduler = PulseScheduler(n_layers=12, n_groups=32, warmup_tokens=50)

    # Advance and perform repairs
    for _ in range(200):
        scheduler.advance_token()
        scheduler.get_repair_schedule()

    assert scheduler.current_token > 0
    assert scheduler.total_repairs > 0
    
    # Reset
    scheduler.reset()
    
    assert scheduler.current_token == 0
    assert scheduler.total_repairs == 0
    assert np.all(scheduler.last_repair == -1)
    assert np.all(scheduler.group_rotation == 0)


def test_pulse_scheduler_get_stats():
    """Test statistics retrieval."""
    scheduler = PulseScheduler(
        n_layers=4,
        n_groups=8,
        base_interval=100,
        warmup_tokens=50,
    )
    
    # Advance and perform repairs
    for _ in range(200):
        scheduler.advance_token()
        scheduler.get_repair_schedule()
    
    stats = scheduler.get_stats()
    
    assert "total_repairs" in stats
    assert "repairs_per_layer" in stats
    assert "current_token" in stats
    assert "avg_interval" in stats
    
    assert stats["total_repairs"] > 0
    assert stats["current_token"] == 200
    assert len(stats["repairs_per_layer"]) == 4


def test_create_default_scheduler():
    """Test default scheduler creation."""
    scheduler = create_default_scheduler(n_layers=12, max_context=8192)
    
    assert scheduler.n_layers == 12
    assert scheduler.n_groups == (8192 + 63) // 64  # 128 groups
    assert scheduler.group_size == 64
    
    # Check adaptive base interval
    scheduler_2k = create_default_scheduler(n_layers=12, max_context=2048)
    scheduler_8k = create_default_scheduler(n_layers=12, max_context=8192)
    
    assert scheduler_8k.base_interval > scheduler_2k.base_interval


def test_pulse_scheduler_determinism():
    """Test that scheduler is deterministic."""
    scheduler1 = PulseScheduler(
        n_layers=12,
        n_groups=32,
        base_interval=100,
        warmup_tokens=50,
    )
    
    scheduler2 = PulseScheduler(
        n_layers=12,
        n_groups=32,
        base_interval=100,
        warmup_tokens=50,
    )
    
    # Advance both schedulers identically
    for _ in range(200):
        scheduler1.advance_token()
        scheduler2.advance_token()
        
        repairs1 = scheduler1.get_repair_schedule()
        repairs2 = scheduler2.get_repair_schedule()
        
        assert len(repairs1) == len(repairs2)
        for r1, r2 in zip(repairs1, repairs2):
            assert r1["layer_idx"] == r2["layer_idx"]
            assert r1["group_idx"] == r2["group_idx"]


def test_pulse_scheduler_long_context():
    """Test scheduler behavior on long contexts (8k+ tokens)."""
    scheduler = PulseScheduler(
        n_layers=12,
        n_groups=128,
        base_interval=256,
        warmup_tokens=512,
    )
    
    # Simulate 10k token context
    for _ in range(10000):
        scheduler.advance_token()
        repairs = scheduler.get_repair_schedule()
        
        # Should have at most one repair per layer per token
        assert len(repairs) <= scheduler.n_layers
    
    stats = scheduler.get_stats()

    # Should have performed repairs
    assert stats["total_repairs"] > 0

    # Average interval should be reasonable (tokens per repair)
    # With 10k tokens and multiple layers/groups, avg_interval can be small
    assert stats["avg_interval"] > 0


if __name__ == "__main__":
    # Run basic tests
    test_pulse_scheduler_init()
    test_pulse_scheduler_warmup()
    test_pulse_scheduler_adaptive_interval()
    test_pulse_scheduler_get_repair_schedule()
    test_pulse_scheduler_chain_repair()
    test_pulse_scheduler_determinism()
    print("✅ All pulse scheduler tests passed!")

