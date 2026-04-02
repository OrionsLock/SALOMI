"""Tests for golden matrix checker."""
from __future__ import annotations

import os
import subprocess
import json
from pathlib import Path
import tempfile
import pytest


def test_golden_matrix_minimal_cpu():
    """Test golden matrix checker with first case on CPU only.
    
    This test always runs (no GPU required).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "golden_minimal.jsonl"
        
        # Run checker with first case only, CPU only
        cmd = [
            "python", "-m", "onebit.cli.check_golden_matrix",
            "--out", str(out_path),
            "--cases", "1",
            "--skip-opencl",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check command succeeded
        assert result.returncode == 0, f"Command failed:\n{result.stdout}\n{result.stderr}"
        
        # Check output file exists
        assert out_path.exists(), "Output JSONL not created"
        
        # Parse JSONL
        records = []
        with open(out_path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        
        # Should have 1 record (CPU only)
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"
        
        # Check record structure
        record = records[0]
        assert "case" in record, "Missing 'case' field"
        assert "backend" in record, "Missing 'backend' field"
        assert "kernel" in record, "Missing 'kernel' field"
        assert "digests" in record, "Missing 'digests' field"
        assert "T_eff" in record, "Missing 'T_eff' field"
        
        # Check backend is CPU
        assert record["backend"] == "cpu", "Expected CPU backend"
        
        # Check digests
        digests = record["digests"]
        assert "y_main" in digests, "Missing y_main digest"
        assert "y_twin" in digests, "Missing y_twin digest"
        assert "pc32_main" in digests, "Missing pc32_main digest"
        assert "pc32_twin" in digests, "Missing pc32_twin digest"
        assert "y_mean" in digests, "Missing y_mean digest"
        assert "ctg" in digests, "Missing ctg digest"
        
        # Check CTG digest is zero (CTG off)
        assert digests["ctg"] == "0000000000000000", "CTG digest should be zero"
        
        # Check T_eff matches k
        case = record["case"]
        assert record["T_eff"] == case["k"], f"T_eff {record['T_eff']} != k {case['k']}"
        
        print(f"\n[PASS] Minimal CPU test passed")
        print(f"   Case: seed={case['seed']}, M={case['M']}, Kw={case['Kw']}, k={case['k']}, order={case['order']}")
        print(f"   Digests: y_main={digests['y_main']}, y_twin={digests['y_twin']}")


@pytest.mark.opencl
def test_golden_matrix_minimal_opencl():
    """Test golden matrix checker with first case on CPU + OpenCL.
    
    Skipped if OpenCL not available.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "golden_opencl.jsonl"
        
        # Run checker with first case only, CPU + OpenCL
        cmd = [
            "python", "-m", "onebit.cli.check_golden_matrix",
            "--out", str(out_path),
            "--cases", "1",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check command succeeded
        assert result.returncode == 0, f"Command failed:\n{result.stdout}\n{result.stderr}"
        
        # Check output file exists
        assert out_path.exists(), "Output JSONL not created"
        
        # Parse JSONL
        records = []
        with open(out_path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        
        # Should have 3 records (CPU + OpenCL naive + OpenCL tiled)
        assert len(records) == 3, f"Expected 3 records, got {len(records)}"
        
        # Extract digests
        cpu_digests = None
        opencl_naive_digests = None
        opencl_tiled_digests = None
        
        for record in records:
            if record["backend"] == "cpu":
                cpu_digests = record["digests"]
            elif record["backend"] == "opencl" and record["kernel"] == "naive":
                opencl_naive_digests = record["digests"]
            elif record["backend"] == "opencl" and record["kernel"] == "tiled":
                opencl_tiled_digests = record["digests"]
        
        assert cpu_digests is not None, "Missing CPU record"
        assert opencl_naive_digests is not None, "Missing OpenCL naive record"
        assert opencl_tiled_digests is not None, "Missing OpenCL tiled record"
        
        # Primary check: OpenCL naive vs tiled (hard requirement)
        for field in ["y_main", "y_twin", "pc32_main", "pc32_twin", "y_mean"]:
            assert opencl_naive_digests[field] == opencl_tiled_digests[field], \
                f"OpenCL naive vs tiled mismatch on {field}: {opencl_naive_digests[field]} != {opencl_tiled_digests[field]}"

        # Secondary check: CPU vs OpenCL - only y_mean must match (estimates)
        # y_bits may differ due to float32 rounding in Sigma-Delta modulator
        assert cpu_digests["y_mean"] == opencl_naive_digests["y_mean"], \
            f"CPU vs OpenCL y_mean mismatch: {cpu_digests['y_mean']} != {opencl_naive_digests['y_mean']}"

        # pc32 should match (raw popcounts, no floating point)
        assert cpu_digests["pc32_main"] == opencl_naive_digests["pc32_main"], \
            f"CPU vs OpenCL pc32_main mismatch: {cpu_digests['pc32_main']} != {opencl_naive_digests['pc32_main']}"
        assert cpu_digests["pc32_twin"] == opencl_naive_digests["pc32_twin"], \
            f"CPU vs OpenCL pc32_twin mismatch: {cpu_digests['pc32_twin']} != {opencl_naive_digests['pc32_twin']}"

        print(f"\n[PASS] OpenCL test passed")
        print(f"   OpenCL naive == tiled: y_main={opencl_naive_digests['y_main']}, y_twin={opencl_naive_digests['y_twin']}")
        print(f"   CPU y_mean == OpenCL y_mean: {cpu_digests['y_mean']}")
        print(f"   Note: CPU y_bits differ from OpenCL (expected due to float32 rounding)")


def test_golden_matrix_full_optional():
    """Test full golden matrix (all cases, all backends).
    
    Only runs if ONEBIT_FULL_MATRIX=1 is set.
    """
    if os.getenv("ONEBIT_FULL_MATRIX") != "1":
        pytest.skip("set ONEBIT_FULL_MATRIX=1 to run full matrix")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "golden_full.jsonl"
        
        # Run full matrix
        cmd = [
            "python", "-m", "onebit.cli.check_golden_matrix",
            "--out", str(out_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check command succeeded
        assert result.returncode == 0, f"Command failed:\n{result.stdout}\n{result.stderr}"
        
        # Check output file exists
        assert out_path.exists(), "Output JSONL not created"
        
        # Parse JSONL
        records = []
        with open(out_path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        
        # Should have at least 6 records (3 cases × 2 backends minimum)
        # If OpenCL available: 3 cases × 3 runs = 9 records
        assert len(records) >= 6, f"Expected at least 6 records, got {len(records)}"
        
        print(f"\n[PASS] Full matrix test passed")
        print(f"   Total records: {len(records)}")


def test_golden_matrix_determinism():
    """Test that CPU runs are deterministic (same seed → same digests)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path1 = Path(tmpdir) / "golden_run1.jsonl"
        out_path2 = Path(tmpdir) / "golden_run2.jsonl"
        
        # Run 1
        cmd = [
            "python", "-m", "onebit.cli.check_golden_matrix",
            "--out", str(out_path1),
            "--cases", "1",
            "--skip-opencl",
        ]
        result1 = subprocess.run(cmd, capture_output=True, text=True)
        assert result1.returncode == 0, f"Run 1 failed:\n{result1.stderr}"
        
        # Run 2
        cmd = [
            "python", "-m", "onebit.cli.check_golden_matrix",
            "--out", str(out_path2),
            "--cases", "1",
            "--skip-opencl",
        ]
        result2 = subprocess.run(cmd, capture_output=True, text=True)
        assert result2.returncode == 0, f"Run 2 failed:\n{result2.stderr}"
        
        # Parse both runs
        with open(out_path1, 'r') as f:
            record1 = json.loads(f.readline())
        
        with open(out_path2, 'r') as f:
            record2 = json.loads(f.readline())
        
        # Check digests match
        digests1 = record1["digests"]
        digests2 = record2["digests"]
        
        for field in ["y_main", "y_twin", "pc32_main", "pc32_twin", "y_mean"]:
            assert digests1[field] == digests2[field], \
                f"Determinism failed on {field}: {digests1[field]} != {digests2[field]}"
        
        print(f"\n[PASS] Determinism test passed - two runs produce identical digests")


@pytest.mark.opencl
def test_golden_matrix_logits():
    """Test golden matrix for logits operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "golden_logits.jsonl"

        # Run with 1 case for speed
        cmd = [
            "python", "-m", "onebit.cli.check_golden_matrix",
            "--op", "logits",
            "--out", str(out_path),
            "--cases", "1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        assert result.returncode == 0, f"Command failed:\n{result.stdout}\n{result.stderr}"

        # Check output file exists
        assert out_path.exists(), "Output JSONL not created"

        # Parse JSONL
        records = []
        with open(out_path, 'r') as f:
            for line in f:
                records.append(json.loads(line))

        # Should have 2 records (OpenCL naive + tiled)
        assert len(records) == 2, f"Expected 2 records, got {len(records)}"

        # Extract digests
        naive_digests = None
        tiled_digests = None

        for record in records:
            assert record["op"] == "logits", "Expected logits operation"
            if record["kernel"] == "naive":
                naive_digests = record["digests"]
            elif record["kernel"] == "tiled":
                tiled_digests = record["digests"]

        assert naive_digests is not None, "Missing naive record"
        assert tiled_digests is not None, "Missing tiled record"

        # Check naive vs tiled (hard requirement)
        for field in ["e_mean", "t_eff"]:
            assert naive_digests[field] == tiled_digests[field], \
                f"Logits naive vs tiled mismatch on {field}: {naive_digests[field]} != {tiled_digests[field]}"

        print(f"\n[PASS] Logits golden matrix test passed")
        print(f"   Naive == Tiled: e_mean={naive_digests['e_mean']}, t_eff={naive_digests['t_eff']}")


@pytest.mark.opencl
@pytest.mark.parametrize("seed,V,d,T", [
    (42, 256, 768, 16),
    (123, 512, 1024, 16),
])
def test_logits_naive_tiled_parity(seed, V, d, T):
    """Test that logits naive and tiled kernels produce identical digests."""
    import numpy as np
    from onebit.cli.check_golden_matrix import run_logits_opencl, compute_logits_digests

    d_words = (d + 31) // 32

    # Generate test data
    np.random.seed(seed)
    q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
    v_ids = np.arange(min(V, 256), dtype=np.int32)

    # Run naive
    result_naive = run_logits_opencl(q_bits, v_ids, d, T, seed, "naive")
    digests_naive = compute_logits_digests(result_naive)

    # Run tiled
    result_tiled = run_logits_opencl(q_bits, v_ids, d, T, seed, "tiled")
    digests_tiled = compute_logits_digests(result_tiled)

    # Compare digests
    assert digests_naive["e_mean"] == digests_tiled["e_mean"], \
        f"E_mean digest mismatch: {digests_naive['e_mean']} != {digests_tiled['e_mean']}"

    assert digests_naive["t_eff"] == digests_tiled["t_eff"], \
        f"T_eff digest mismatch: {digests_naive['t_eff']} != {digests_tiled['t_eff']}"

    # Check T_eff values
    assert result_naive["T_eff"][0] == T, f"Naive T_eff={result_naive['T_eff'][0]} != T={T}"
    assert result_tiled["T_eff"][0] == T, f"Tiled T_eff={result_tiled['T_eff'][0]} != T={T}"


@pytest.mark.opencl
def test_logits_determinism():
    """Test that logits kernels are deterministic across runs."""
    import numpy as np
    from onebit.cli.check_golden_matrix import run_logits_opencl, compute_logits_digests

    seed = 42
    V = 256
    d = 768
    T = 16
    d_words = (d + 31) // 32

    # Generate test data
    np.random.seed(seed)
    q_bits = np.random.randint(0, 2**32, size=d_words, dtype=np.uint32)
    v_ids = np.arange(min(V, 256), dtype=np.int32)

    # Run twice
    result1 = run_logits_opencl(q_bits, v_ids, d, T, seed, "tiled")
    digests1 = compute_logits_digests(result1)

    result2 = run_logits_opencl(q_bits, v_ids, d, T, seed, "tiled")
    digests2 = compute_logits_digests(result2)

    # Compare digests
    assert digests1 == digests2, "Logits kernel is not deterministic"

