"""
Triage bundle packer for failing stress test cases.

Writes a self-contained directory with:
- meta.json: full args + device info + kernel hash
- W_bits.rle.json: RLE-encoded weight bits
- X_bits.rle.json: RLE-encoded input bits
- y_main.hex, y_twin.hex: output bit streams
- pc32_main.json, pc32_twin.json: popcount arrays
- controller_trace.json: token-level trace (if available)
- energy.json: energy counters (if instr_on=1)
- opencl_build_log.txt: build log (if build fails)
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 and return first 16 hex chars."""
    return hashlib.sha256(data).hexdigest()[:16]


def rle_encode_bits(bits: np.ndarray) -> Dict[str, Any]:
    """
    RLE encode a uint32 bit array.
    
    Returns dict with:
    - shape: original shape
    - dtype: original dtype
    - runs: list of (value, count) pairs
    """
    if bits.size == 0:
        return {"shape": list(bits.shape), "dtype": str(bits.dtype), "runs": []}
    
    flat = bits.flatten()
    runs = []
    current_val = int(flat[0])
    current_count = 1
    
    for i in range(1, len(flat)):
        val = int(flat[i])
        if val == current_val:
            current_count += 1
        else:
            runs.append([current_val, current_count])
            current_val = val
            current_count = 1
    
    runs.append([current_val, current_count])
    
    return {
        "shape": list(bits.shape),
        "dtype": str(bits.dtype),
        "runs": runs
    }


def save_triage_bundle(
    triage_dir: Path,
    case_id: str,
    meta: Dict[str, Any],
    W_bits: Optional[np.ndarray] = None,
    X_bits: Optional[np.ndarray] = None,
    y_main: Optional[np.ndarray] = None,
    y_twin: Optional[np.ndarray] = None,
    pc32_main: Optional[np.ndarray] = None,
    pc32_twin: Optional[np.ndarray] = None,
    controller_trace: Optional[Dict[str, Any]] = None,
    energy: Optional[Dict[str, np.ndarray]] = None,
    opencl_build_log: Optional[str] = None,
) -> Path:
    """
    Save a triage bundle for a failing test case.
    
    Args:
        triage_dir: Base directory for triage bundles
        case_id: Unique case identifier
        meta: Metadata dict (args, device info, kernel hash, etc.)
        W_bits: Weight bits (uint32 array)
        X_bits: Input bits (uint32 array)
        y_main: Main channel output bits (uint32 array)
        y_twin: Twin channel output bits (uint32 array)
        pc32_main: Main channel popcounts (int32 array)
        pc32_twin: Twin channel popcounts (int32 array)
        controller_trace: Token-level trace dict
        energy: Energy counters dict
        opencl_build_log: OpenCL build log (if build failed)
    
    Returns:
        Path to the created triage directory
    """
    # Create case directory
    case_dir = triage_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(case_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Save W_bits (RLE encoded)
    if W_bits is not None:
        with open(case_dir / "W_bits.rle.json", "w") as f:
            json.dump(rle_encode_bits(W_bits), f, indent=2)
    
    # Save X_bits (RLE encoded)
    if X_bits is not None:
        with open(case_dir / "X_bits.rle.json", "w") as f:
            json.dump(rle_encode_bits(X_bits), f, indent=2)
    
    # Save y_main (hex)
    if y_main is not None:
        with open(case_dir / "y_main.hex", "w") as f:
            f.write(y_main.tobytes().hex())
    
    # Save y_twin (hex)
    if y_twin is not None:
        with open(case_dir / "y_twin.hex", "w") as f:
            f.write(y_twin.tobytes().hex())
    
    # Save pc32_main (JSON)
    if pc32_main is not None:
        with open(case_dir / "pc32_main.json", "w") as f:
            json.dump(pc32_main.tolist(), f)
    
    # Save pc32_twin (JSON)
    if pc32_twin is not None:
        with open(case_dir / "pc32_twin.json", "w") as f:
            json.dump(pc32_twin.tolist(), f)
    
    # Save controller trace
    if controller_trace is not None:
        with open(case_dir / "controller_trace.json", "w") as f:
            json.dump(controller_trace, f, indent=2)
    
    # Save energy counters
    if energy is not None:
        energy_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in energy.items()
        }
        with open(case_dir / "energy.json", "w") as f:
            json.dump(energy_serializable, f, indent=2)
    
    # Save OpenCL build log
    if opencl_build_log is not None:
        with open(case_dir / "opencl_build_log.txt", "w") as f:
            f.write(opencl_build_log)
    
    return case_dir


def load_triage_bundle(case_dir: Path) -> Dict[str, Any]:
    """
    Load a triage bundle from disk.
    
    Returns dict with all available fields.
    """
    bundle = {}
    
    # Load metadata
    meta_path = case_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            bundle["meta"] = json.load(f)
    
    # Load W_bits
    w_path = case_dir / "W_bits.rle.json"
    if w_path.exists():
        with open(w_path) as f:
            rle_data = json.load(f)
            # Decode RLE
            runs = rle_data["runs"]
            flat = []
            for val, count in runs:
                flat.extend([val] * count)
            arr = np.array(flat, dtype=np.uint32)
            bundle["W_bits"] = arr.reshape(rle_data["shape"])
    
    # Load X_bits
    x_path = case_dir / "X_bits.rle.json"
    if x_path.exists():
        with open(x_path) as f:
            rle_data = json.load(f)
            runs = rle_data["runs"]
            flat = []
            for val, count in runs:
                flat.extend([val] * count)
            arr = np.array(flat, dtype=np.uint32)
            bundle["X_bits"] = arr.reshape(rle_data["shape"])
    
    # Load y_main
    y_main_path = case_dir / "y_main.hex"
    if y_main_path.exists():
        with open(y_main_path) as f:
            hex_data = f.read()
            bundle["y_main"] = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint32)
    
    # Load y_twin
    y_twin_path = case_dir / "y_twin.hex"
    if y_twin_path.exists():
        with open(y_twin_path) as f:
            hex_data = f.read()
            bundle["y_twin"] = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint32)
    
    # Load pc32_main
    pc_main_path = case_dir / "pc32_main.json"
    if pc_main_path.exists():
        with open(pc_main_path) as f:
            bundle["pc32_main"] = np.array(json.load(f), dtype=np.int32)
    
    # Load pc32_twin
    pc_twin_path = case_dir / "pc32_twin.json"
    if pc_twin_path.exists():
        with open(pc_twin_path) as f:
            bundle["pc32_twin"] = np.array(json.load(f), dtype=np.int32)
    
    # Load controller trace
    trace_path = case_dir / "controller_trace.json"
    if trace_path.exists():
        with open(trace_path) as f:
            bundle["controller_trace"] = json.load(f)
    
    # Load energy
    energy_path = case_dir / "energy.json"
    if energy_path.exists():
        with open(energy_path) as f:
            energy_data = json.load(f)
            bundle["energy"] = {
                k: np.array(v, dtype=np.uint64) if isinstance(v, list) else v
                for k, v in energy_data.items()
            }
    
    # Load OpenCL build log
    log_path = case_dir / "opencl_build_log.txt"
    if log_path.exists():
        with open(log_path) as f:
            bundle["opencl_build_log"] = f.read()
    
    return bundle


def get_device_fingerprint() -> Dict[str, str]:
    """
    Get device fingerprint for triage metadata.
    
    Returns dict with vendor, name, driver, kernel_hash.
    """
    try:
        from onebit.backends.opencl.host_opencl import OpenCLBinGemm
        backend = OpenCLBinGemm()
        
        # Get device info
        device = backend.device
        vendor = device.vendor.strip()
        name = device.name.strip()
        driver = device.driver_version.strip()
        
        # Get kernel hash
        kernel_source = backend.prog.get_info(backend.prog.BUILD_INFO)
        kernel_hash = sha256_hex(str(kernel_source).encode())
        
        return {
            "vendor": vendor,
            "name": name,
            "driver": driver,
            "kernel_hash": kernel_hash
        }
    except Exception as e:
        return {
            "vendor": "unknown",
            "name": "unknown",
            "driver": "unknown",
            "kernel_hash": "unknown",
            "error": str(e)
        }

