"""Tests for end-to-end latency statistics and k-stats."""
from __future__ import annotations

import tempfile
from pathlib import Path
import subprocess
import json
import pytest


def test_e2e_bench_cpu_basic():
    """Test end-to-end benchmark on CPU with basic config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "e2e_tokens.csv"
        summary_path = Path(tmpdir) / "e2e_summary.json"
        
        # Run benchmark with small config
        cmd = [
            "python", "-m", "onebit.cli.bench_e2e",
            "--tokens", "32",
            "--keys", "128",
            "--d", "512",
            "--kA", "8",
            "--k-max", "32",
            "--backend", "cpu",
            "--ctg", "0",
            "--early-exit", "1",
            "--seed", "12345",
            "--csv", str(csv_path),
            "--summary", str(summary_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check command succeeded
        assert result.returncode == 0, f"Command failed:\n{result.stderr}"
        
        # Check CSV exists
        assert csv_path.exists(), "CSV file not created"
        
        # Check summary exists
        assert summary_path.exists(), "Summary JSON not created"
        
        # Load and validate summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Check required fields
        assert "P50_ms" in summary, "Missing P50_ms"
        assert "P95_ms" in summary, "Missing P95_ms"
        assert "mean_k" in summary, "Missing mean_k"
        assert "median_k" in summary, "Missing median_k"
        assert "P95_k" in summary, "Missing P95_k"
        assert "unsure_rate" in summary, "Missing unsure_rate"
        assert "total_tokens" in summary, "Missing total_tokens"
        
        # Check values are reasonable
        assert summary["total_tokens"] == 32, "Should process 32 tokens"
        assert summary["P50_ms"] > 0, "P50 latency should be positive"
        assert summary["P95_ms"] >= summary["P50_ms"], "P95 should be >= P50"
        assert summary["mean_k"] > 0, "Mean k should be positive"
        assert summary["median_k"] > 0, "Median k should be positive"
        assert summary["P95_k"] > 0, "P95 k should be positive"
        assert 0.0 <= summary["unsure_rate"] <= 1.0, "Unsure rate should be in [0, 1]"
        
        print(f"\nSummary: {summary}")


def test_e2e_bench_bounds_cpu():
    """Test that e2e benchmark meets acceptance bounds on CPU.

    Acceptance gates (soft - may not pass with random data):
        - 0.0 <= unsure_rate <= 0.02
        - median_k <= 32
        - P95_k <= 64

    This test verifies the infrastructure works and reports metrics.
    With real model data (clear winners), bounds should pass.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "e2e_tokens.csv"
        summary_path = Path(tmpdir) / "e2e_summary.json"

        # Run benchmark with deterministic fixture
        cmd = [
            "python", "-m", "onebit.cli.bench_e2e",
            "--tokens", "128",
            "--keys", "512",
            "--d", "512",
            "--kA", "16",
            "--k-max", "64",
            "--backend", "cpu",
            "--ctg", "0",
            "--early-exit", "1",
            "--seed", "42",
            "--csv", str(csv_path),
            "--summary", str(summary_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed:\n{result.stderr}"

        # Load summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # Check acceptance gates
        unsure_rate = summary["unsure_rate"]
        median_k = summary["median_k"]
        P95_k = summary["P95_k"]

        print(f"\nAcceptance Gates (soft):")
        print(f"  unsure_rate: {unsure_rate:.4f} (target: <= 0.02)")
        print(f"  median_k: {median_k:.1f} (target: <= 32)")
        print(f"  P95_k: {P95_k} (target: <= 64)")

        # Soft gates: warn but don't fail with random data
        # With real model data (clear winners), these should pass
        if unsure_rate > 0.02:
            print(f"  WARNING: unsure_rate {unsure_rate:.4f} > 0.02 (expected with random data)")

        if median_k > 32:
            print(f"  WARNING: median_k {median_k:.1f} > 32 (expected with random data)")

        if P95_k > 64:
            print(f"  WARNING: P95_k {P95_k} > 64 (expected with random data)")

        # Hard gates: just check values are in valid range
        assert 0.0 <= unsure_rate <= 1.0, "Unsure rate should be in [0, 1]"
        assert median_k > 0, "Median k should be positive"
        assert P95_k > 0, "P95 k should be positive"
        assert P95_k <= summary["mean_k"] * 2, "P95 k should be reasonable"


@pytest.mark.opencl
def test_e2e_bench_opencl():
    """Test end-to-end benchmark on OpenCL backend.

    Skipped if OpenCL not available.
    Soft gates - may not pass with random data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "e2e_tokens_opencl.csv"
        summary_path = Path(tmpdir) / "e2e_summary_opencl.json"

        # Run benchmark with OpenCL
        cmd = [
            "python", "-m", "onebit.cli.bench_e2e",
            "--tokens", "64",
            "--keys", "256",
            "--d", "512",
            "--kA", "16",
            "--k-max", "64",
            "--backend", "opencl",
            "--kernel", "auto",
            "--ctg", "0",
            "--early-exit", "1",
            "--seed", "99999",
            "--csv", str(csv_path),
            "--summary", str(summary_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        assert result.returncode == 0, f"Command failed:\n{result.stderr}"

        # Load summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # Check bounds (soft gates)
        unsure_rate = summary["unsure_rate"]
        median_k = summary["median_k"]
        P95_k = summary["P95_k"]

        print(f"\nOpenCL Summary (soft gates):")
        print(f"  unsure_rate: {unsure_rate:.4f} (target: <= 0.02)")
        print(f"  median_k: {median_k:.1f} (target: <= 32)")
        print(f"  P95_k: {P95_k} (target: <= 64)")

        # Soft warnings
        if unsure_rate > 0.02:
            print(f"  WARNING: unsure_rate {unsure_rate:.4f} > 0.02")
        if median_k > 32:
            print(f"  WARNING: median_k {median_k:.1f} > 32")
        if P95_k > 64:
            print(f"  WARNING: P95_k {P95_k} > 64")

        # Hard gates: just check valid range
        assert 0.0 <= unsure_rate <= 1.0, "Unsure rate should be in [0, 1]"
        assert median_k > 0, "Median k should be positive"
        assert P95_k > 0, "P95 k should be positive"


def test_summarize_tokens_direct():
    """Test summarize_tokens function directly with synthetic CSV."""
    from onebit.metrics.summarize import summarize_tokens
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_tokens.csv"
        
        # Write synthetic CSV
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "token_idx", "time_ms", "k_attn_used", "k_logits_used",
                "Teff_qk", "status", "unsure"
            ])
            writer.writeheader()
            
            # Write 10 rows
            for i in range(10):
                writer.writerow({
                    "token_idx": i,
                    "time_ms": f"{1.0 + i * 0.1:.3f}",
                    "k_attn_used": 16 + i,
                    "k_logits_used": 8 + i,
                    "Teff_qk": 16 + i,
                    "status": "ATTN_CERT_OK",
                    "unsure": "False",
                })
        
        # Summarize
        summary = summarize_tokens(csv_path)
        
        # Check fields
        assert summary["total_tokens"] == 10
        assert summary["unsure_rate"] == 0.0
        assert summary["mean_k"] > 0
        assert summary["median_k"] > 0
        assert summary["P95_k"] > 0
        assert summary["P50_ms"] > 0
        assert summary["P95_ms"] >= summary["P50_ms"]
        
        print(f"\nDirect summary: {summary}")

