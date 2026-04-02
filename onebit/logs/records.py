"""Structured logging for controller records."""
from __future__ import annotations

from typing import Dict
import json


def controller_record(cert: Dict, token_idx: int = 0, time_ms: float = 0.0) -> Dict:
    """Format controller certificate as structured log record.
    
    Args:
        cert: Certificate bundle from infer_one_token()
        token_idx: Token index (default: 0)
        time_ms: Inference time in milliseconds (default: 0.0)
    
    Returns:
        Structured record with all required fields:
            - token_idx: Token index
            - status: "ATTN_CERT_OK" or "UNSURE"
            - top1: Top-1 index if certified, else None
            - T_sel: Selected T from elbow
            - kA: Stage-A ticks used
            - k_attn_used: SPRT ticks used
            - pairs_evaluated: Total pair observations
            - alpha: Type I error per pair
            - beta: Type II error per pair
            - delta_total: Total risk budget
            - backend: Backend used
            - prf_seed: PRF seed
            - time_ms: Inference time in milliseconds
            - decided_count: Number of decided edges
            - undecided_count: Number of undecided pairs
    """
    record = {
        "token_idx": token_idx,
        "status": cert["status"],
        "top1": cert["top1"] if cert["top1"] is not None else -1,  # Use -1 for None
        "T_sel": cert["T_sel"],
        "kA": cert["kA"],
        "k_attn_used": cert["k_attn_used"],
        "pairs_evaluated": cert["pairs_evaluated"],
        "alpha": cert["alpha"],
        "beta": cert["beta"],
        "delta_total": cert["delta_total"],
        "backend": cert["backend"],
        "prf_seed": cert["prf_seed"],
        "time_ms": time_ms,
        "decided_count": len(cert["decided"]),
        "undecided_count": len(cert["undecided"]),
    }
    
    return record


def record_to_csv_row(record: Dict) -> str:
    """Convert record to CSV row.
    
    Args:
        record: Structured record from controller_record()
    
    Returns:
        CSV row string (no newline)
    """
    fields = [
        record["token_idx"],
        record["status"],
        record["top1"],
        record["T_sel"],
        record["kA"],
        record["k_attn_used"],
        record["pairs_evaluated"],
        f"{record['alpha']:.6f}",
        f"{record['beta']:.6f}",
        f"{record['delta_total']:.6f}",
        record["backend"],
        record["prf_seed"],
        f"{record['time_ms']:.2f}",
        record["decided_count"],
        record["undecided_count"],
    ]
    
    return ",".join(str(f) for f in fields)


def csv_header() -> str:
    """Return CSV header row.
    
    Returns:
        CSV header string (no newline)
    """
    return "token_idx,status,top1,T_sel,kA,k_attn_used,pairs_evaluated,alpha,beta,delta_total,backend,prf_seed,time_ms,decided_count,undecided_count"


def record_to_json(record: Dict) -> str:
    """Convert record to JSON string.
    
    Args:
        record: Structured record from controller_record()
    
    Returns:
        JSON string
    """
    return json.dumps(record)

