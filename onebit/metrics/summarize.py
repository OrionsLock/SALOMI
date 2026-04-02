"""Summarize token-level metrics from CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np


def summarize_tokens(csv_path: str | Path) -> Dict[str, Any]:
    """Summarize token-level metrics from CSV.
    
    Reads CSV with columns:
        token_idx, time_ms, k_attn_used, k_logits_used, Teff_qk, status, unsure
    
    Returns:
        Dict with:
            P50_ms: float - Median latency in ms
            P95_ms: float - 95th percentile latency in ms
            mean_k: float - Mean k across attention and logits
            median_k: float - Median k
            P95_k: int - 95th percentile k
            unsure_rate: float - Fraction of tokens with unsure=True
            total_tokens: int - Number of tokens processed
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV
    times = []
    k_values = []
    unsure_count = 0
    total_tokens = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_tokens += 1
            
            # Parse values
            time_ms = float(row['time_ms'])
            k_attn = int(row['k_attn_used'])
            k_logits = int(row['k_logits_used'])
            unsure = row['unsure'].strip().lower() in ('true', '1', 'yes')
            
            times.append(time_ms)
            # Total k = k_attn + k_logits
            k_total = k_attn + k_logits
            k_values.append(k_total)
            
            if unsure:
                unsure_count += 1
    
    if total_tokens == 0:
        raise ValueError("CSV file is empty or has no data rows")
    
    # Compute statistics
    times_arr = np.array(times)
    k_arr = np.array(k_values)
    
    summary = {
        "P50_ms": float(np.percentile(times_arr, 50)),
        "P95_ms": float(np.percentile(times_arr, 95)),
        "mean_k": float(np.mean(k_arr)),
        "median_k": float(np.median(k_arr)),
        "P95_k": int(np.percentile(k_arr, 95)),
        "unsure_rate": unsure_count / total_tokens,
        "total_tokens": total_tokens,
    }
    
    return summary


def save_summary_json(summary: Dict[str, Any], json_path: str | Path) -> None:
    """Save summary dict to JSON file."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

