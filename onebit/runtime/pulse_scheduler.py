"""Pulse Scheduler for adaptive KV repair scheduling.

PR-4.6: Pulse Scheduler + Chain Repair

The pulse scheduler determines when to trigger KV repair operations
based on context length and layer depth. It uses an adaptive schedule
that balances repair overhead with signal quality.

Key concepts:
- **Pulse interval**: Number of tokens between repair operations
- **Chain repair**: Repair multiple layers in sequence
- **Adaptive scheduling**: Adjust interval based on context length
- **Group rotation**: Rotate through KV groups to amortize repair cost

Invariants:
- Export stays exactly 1.00 bpp (repair uses transient compute only)
- Deterministic for fixed seeds
- No extra resident buffers
"""

import numpy as np
from typing import Optional, Dict, List


class PulseScheduler:
    """Adaptive scheduler for KV repair operations.
    
    The scheduler maintains a repair schedule for each layer and group,
    triggering repairs at adaptive intervals based on context length.
    
    Args:
        n_layers: Number of transformer layers
        n_groups: Number of KV groups per layer
        group_size: Positions per group (default: 64)
        base_interval: Base pulse interval in tokens (default: 256)
        max_interval: Maximum pulse interval (default: 1024)
        min_interval: Minimum pulse interval (default: 64)
        warmup_tokens: Tokens before first repair (default: 512)
    """
    
    def __init__(
        self,
        n_layers: int,
        n_groups: int,
        group_size: int = 64,
        base_interval: int = 256,
        max_interval: int = 1024,
        min_interval: int = 64,
        warmup_tokens: int = 512,
    ):
        self.n_layers = n_layers
        self.n_groups = n_groups
        self.group_size = group_size
        self.base_interval = base_interval
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.warmup_tokens = warmup_tokens
        
        # State: last repair token for each (layer, group)
        self.last_repair = np.full((n_layers, n_groups), -1, dtype=np.int32)
        
        # State: current token position
        self.current_token = 0
        
        # State: group rotation index for each layer
        self.group_rotation = np.zeros(n_layers, dtype=np.int32)
        
        # Stats: total repairs performed
        self.total_repairs = 0
        
        # Stats: repairs per layer
        self.repairs_per_layer = np.zeros(n_layers, dtype=np.int32)
    
    def reset(self):
        """Reset scheduler state."""
        self.last_repair.fill(-1)
        self.current_token = 0
        self.group_rotation.fill(0)
        self.total_repairs = 0
        self.repairs_per_layer.fill(0)
    
    def advance_token(self):
        """Advance to next token position."""
        self.current_token += 1
    
    def get_pulse_interval(self, layer_idx: int) -> int:
        """Compute adaptive pulse interval for layer.
        
        Deeper layers use longer intervals (less frequent repairs).
        Interval increases with context length.
        
        Args:
            layer_idx: Layer index (0-based)
        
        Returns:
            Pulse interval in tokens
        """
        # Deeper layers → longer intervals
        layer_factor = 1.0 + 0.5 * (layer_idx / max(1, self.n_layers - 1))
        
        # Longer contexts → longer intervals
        context_factor = 1.0
        if self.current_token > 2048:
            context_factor = 1.5
        if self.current_token > 4096:
            context_factor = 2.0
        if self.current_token > 8192:
            context_factor = 2.5
        
        interval = int(self.base_interval * layer_factor * context_factor)
        return np.clip(interval, self.min_interval, self.max_interval)
    
    def should_repair(self, layer_idx: int, group_idx: int) -> bool:
        """Check if repair is needed for (layer, group).
        
        Args:
            layer_idx: Layer index
            group_idx: Group index
        
        Returns:
            True if repair should be triggered
        """
        # Skip warmup period
        if self.current_token < self.warmup_tokens:
            return False
        
        # Check if enough tokens have passed since last repair
        last_repair_token = self.last_repair[layer_idx, group_idx]
        if last_repair_token < 0:
            # Never repaired, trigger first repair
            return True
        
        interval = self.get_pulse_interval(layer_idx)
        tokens_since_repair = self.current_token - last_repair_token
        
        return tokens_since_repair >= interval
    
    def mark_repaired(self, layer_idx: int, group_idx: int):
        """Mark (layer, group) as repaired at current token.
        
        Args:
            layer_idx: Layer index
            group_idx: Group index
        """
        self.last_repair[layer_idx, group_idx] = self.current_token
        self.total_repairs += 1
        self.repairs_per_layer[layer_idx] += 1
    
    def get_repair_schedule(self) -> List[Dict]:
        """Get list of repairs to perform at current token.
        
        Uses group rotation to amortize repair cost across tokens.
        
        Returns:
            List of dicts with keys: layer_idx, group_idx
        """
        repairs = []
        
        # For each layer, check if current rotation group needs repair
        for layer_idx in range(self.n_layers):
            group_idx = self.group_rotation[layer_idx]
            
            if self.should_repair(layer_idx, group_idx):
                repairs.append({
                    "layer_idx": layer_idx,
                    "group_idx": group_idx,
                })
                
                # Mark as repaired
                self.mark_repaired(layer_idx, group_idx)
            
            # Rotate to next group
            self.group_rotation[layer_idx] = (group_idx + 1) % self.n_groups
        
        return repairs
    
    def get_chain_repair_schedule(self, max_repairs_per_token: int = 4) -> List[Dict]:
        """Get chain repair schedule (multiple layers in sequence).
        
        Chain repair processes multiple layers in a single token to
        maintain signal quality across the full transformer stack.
        
        Args:
            max_repairs_per_token: Maximum repairs per token (default: 4)
        
        Returns:
            List of dicts with keys: layer_idx, group_idx
        """
        repairs = []
        
        # Collect all pending repairs
        for layer_idx in range(self.n_layers):
            for group_idx in range(self.n_groups):
                if self.should_repair(layer_idx, group_idx):
                    repairs.append({
                        "layer_idx": layer_idx,
                        "group_idx": group_idx,
                    })
        
        # Limit to max_repairs_per_token
        if len(repairs) > max_repairs_per_token:
            # Prioritize deeper layers (more critical)
            repairs.sort(key=lambda r: -r["layer_idx"])
            repairs = repairs[:max_repairs_per_token]
        
        # Mark all as repaired
        for repair in repairs:
            self.mark_repaired(repair["layer_idx"], repair["group_idx"])
        
        return repairs
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics.
        
        Returns:
            Dict with keys:
                - total_repairs: Total repairs performed
                - repairs_per_layer: Repairs per layer
                - current_token: Current token position
                - avg_interval: Average repair interval
        """
        avg_interval = 0.0
        if self.total_repairs > 0:
            avg_interval = self.current_token / self.total_repairs
        
        return {
            "total_repairs": int(self.total_repairs),
            "repairs_per_layer": self.repairs_per_layer.tolist(),
            "current_token": int(self.current_token),
            "avg_interval": float(avg_interval),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PulseScheduler(n_layers={self.n_layers}, n_groups={self.n_groups}, "
            f"current_token={stats['current_token']}, "
            f"total_repairs={stats['total_repairs']}, "
            f"avg_interval={stats['avg_interval']:.1f})"
        )


def create_default_scheduler(n_layers: int, max_context: int = 8192) -> PulseScheduler:
    """Create default pulse scheduler for given configuration.
    
    Args:
        n_layers: Number of transformer layers
        max_context: Maximum context length (default: 8192)
    
    Returns:
        PulseScheduler instance with default parameters
    """
    # Compute number of groups based on max context
    group_size = 64
    n_groups = (max_context + group_size - 1) // group_size
    
    # Adaptive base interval based on context length
    if max_context <= 2048:
        base_interval = 128
    elif max_context <= 4096:
        base_interval = 256
    else:
        base_interval = 512
    
    return PulseScheduler(
        n_layers=n_layers,
        n_groups=n_groups,
        group_size=group_size,
        base_interval=base_interval,
        max_interval=1024,
        min_interval=64,
        warmup_tokens=512,
    )

