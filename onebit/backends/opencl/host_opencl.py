from __future__ import annotations

import os
import math

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover - optional dep
    cl = None  # type: ignore


class OpenCLBinGemm:
    """Host launcher for fused binary GEMM OpenCL kernel.

    Usage:
        gemm = OpenCLBinGemm()
        Y = gemm.run(W_bits, X_bits, cv=np.array([1.0, 0.0], dtype=np.float32))
    """

    def __init__(
        self,
        context: Optional["cl.Context"] = None,
        queue: Optional["cl.CommandQueue"] = None,
        program: Optional["cl.Program"] = None,
        kernel_path: Optional[str] = None,
    ) -> None:
        if cl is None:
            raise ImportError("pyopencl is required for OpenCL backend")

        self.ctx = context or self._create_default_context()
        self.queue = queue or cl.CommandQueue(self.ctx)

        if program is not None:
            self.prog = program
        else:
            src_main = self._load_kernel_source(kernel_path)
            # Try to load tiled kernel and concatenate, if present
            try:
                base_path = Path(kernel_path) if kernel_path is not None else Path(__file__).with_name("fused_bin_gemm.cl")
                tiled_path = base_path.with_name("fused_bin_gemm_tiled.cl")
                with open(tiled_path, "r", encoding="utf-8") as f:
                    src_tiled = f.read()
            except Exception:
                src_tiled = ""
            # Try to load BSDM-W naive normalized kernel and concatenate, if present
            try:
                base_path2 = Path(kernel_path) if kernel_path is not None else Path(__file__).with_name("fused_bin_gemm.cl")
                bsdm_path = base_path2.with_name("bsdm_w_naive_norm.cl")
                with open(bsdm_path, "r", encoding="utf-8") as f:
                    src_bsdm = f.read()
            except Exception:
                src_bsdm = ""
            # Try to load BSDM-W tiled kernel and concatenate, if present
            try:
                base_path3 = Path(kernel_path) if kernel_path is not None else Path(__file__).with_name("fused_bin_gemm.cl")
                bsdm_tiled_path = base_path3.with_name("bsdm_w_tiled.cl")
                with open(bsdm_tiled_path, "r", encoding="utf-8") as f:
                    src_bsdm_tiled = f.read()
            except Exception:
                src_bsdm_tiled = ""
            src = src_main + "\n\n" + src_tiled + "\n\n" + src_bsdm + "\n\n" + src_bsdm_tiled
            self.kernel_source = src  # Store for hash computation
            try:
                self.prog = cl.Program(self.ctx, src).build()
            except Exception as e:
                # Surface build log for easier debugging
                try:
                    build_log = self.prog.get_build_info(self.ctx.devices[0], cl.program_build_info.LOG)
                except Exception:
                    build_log = str(e)
                raise RuntimeError(f"OpenCL program build failed. Log:\n{build_log}") from e

        # Initialize auto-tune cache
        self._autotune_config = None
        self._device_key = None

    def _create_default_context(self) -> "cl.Context":
        plats = cl.get_platforms()
        if not plats:
            raise RuntimeError("No OpenCL platforms found")
        # Prefer GPU, else CPU
        for plat in plats:
            gpus = plat.get_devices(device_type=cl.device_type.GPU)
            if gpus:
                return cl.Context(devices=[gpus[0]])
        for plat in plats:
            cpus = plat.get_devices(device_type=cl.device_type.CPU)
            if cpus:
                return cl.Context(devices=[cpus[0]])
        # Fallback: any device
        return cl.Context(devices=[plats[0].get_devices()[0]])

    def _load_kernel_source(self, kernel_path: Optional[str]) -> str:
        if kernel_path is None:
            # default to the kernel next to this file
            kernel_path = Path(__file__).with_name("fused_bin_gemm.cl")
        else:
            kernel_path = Path(kernel_path)
        with open(kernel_path, "r", encoding="utf-8") as f:
            return f.read()

    def run(
        self,
        W_bits: np.ndarray,  # [M, Kw], uint32
        X_bits: np.ndarray,  # [>=T, Kw], uint32
        cv: np.ndarray,      # [2], float32
        T: int,
        eps_margin: float = 0.0,
        delta: float = 0.05,
        return_teff: str = "none",  # "none" | "scalar" | "per_row"
        local_size: Optional[int] = None,
        kernel: str = "auto",  # "auto" | "naive" | "tiled"
    ) -> dict:
        if W_bits.dtype != np.uint32 or X_bits.dtype != np.uint32:
            raise TypeError("W_bits and X_bits must be dtype uint32")
        if cv is None:
            cv = np.array([1.0, 0.0], dtype=np.float32)
        cv = np.asarray(cv, dtype=np.float32)
        if cv.shape != (2,):
            raise ValueError("cv must be shape [2]")
        M, Kw = int(W_bits.shape[0]), int(W_bits.shape[1])
        # choose kernel variant
        if kernel not in ("auto", "naive", "tiled"):
            raise ValueError("kernel must be 'auto' | 'naive' | 'tiled'")
        if kernel == "auto":
            # Use tiled only when eps_margin == 0 (no early-exit) and problem is large enough
            use_tiled = (eps_margin <= 0.0) and (M >= 128) and (Kw >= 64) and (T >= 2)
        else:
            use_tiled = (kernel == "tiled")

        if int(X_bits.shape[1]) != Kw:
            raise ValueError("X_bits second dimension must match W_bits")
        if T <= 0:
            raise ValueError("T must be > 0")
        if int(X_bits.shape[0]) < T:
            raise ValueError("X_bits must have at least T rows")

        X_slice = X_bits[:T, :]

        Y = np.empty((M,), dtype=np.float32)
        mf = cl.mem_flags
        bufW = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_bits)
        bufX = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X_slice)
        bufCV = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cv)
        bufY = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=Y.nbytes)
        # Always allocate TEFF buffer to satisfy kernel signature
        teff_arr = np.empty((M,), dtype=np.int32)
        bufTEFF = cl.Buffer(self.ctx, mf.READ_WRITE, size=teff_arr.nbytes)

        # Launch kernel
        global_size = (M,)
        local = None if local_size is None else (max(1, min(local_size, M)),)

        # For non-accumulating run, pass bufY as dummy ACC_LO/ACC_HI and accum_mode=0
        if use_tiled and hasattr(self.prog, "fused_bin_gemm_tiled"):
            evt = self.prog.fused_bin_gemm_tiled(
                self.queue, global_size, local,
                bufW, bufX, bufCV,
                np.int32(Kw), np.int32(T), np.float32(delta), np.float32(eps_margin),
                bufY, bufTEFF, bufY, bufY, np.int32(0)
            )
        else:
            evt = self.prog.fused_bin_gemm(
                self.queue, global_size, local,
                bufW, bufX, bufCV,
                np.int32(Kw), np.int32(T), np.float32(delta), np.float32(eps_margin),
                bufY, bufTEFF, bufY, bufY, np.int32(0)
            )
        evt.wait()
        cl.enqueue_copy(self.queue, Y, bufY).wait()

    def run_bsdm_w_naive_norm(
        self,
        W_bits: np.ndarray,   # [M, Kw] uint32
        X_bits: np.ndarray,   # [T, Kw] uint32 (prepacked per tick)
        T: int,
        eps: float,
        delta: float,
        order: int,
        beta: float = 0.30,
        lambd: float = 1.0/256.0,
        walsh_N: int = 2,
        antithetic: bool = True,
        use_ctg: bool = False,
        prf_seed: int = 0,
        early_exit_enable: bool = True,
        local_size: int | None = 256,
        want_y_pack: bool = False,
        want_pc32: bool = False,
        kernel: str = "auto",
        instr_on: bool = False,  # PR-3.7: energy instrumentation
    ) -> dict:
        """Run BSDM-W naive normalized kernel with Walsh N=2 + antithetic.

        Args:
            kernel: "auto" | "naive" | "tiled"
                "auto": use tiled if early_exit_enable==0 and shapes exceed thresholds
                "naive": always use naive kernel
                "tiled": always use tiled kernel (requires early_exit_enable==0)

        Returns dict with keys:
          Y (float32 [M]): mean of y_bar sequence
          T_eff (int32 [M]): effective ticks used
          y_bits_main (optional uint32 [M, ceil(T*N/32)]): main channel bits
          y_bits_twin (optional uint32 [M, ceil(T*N/32)]): twin channel bits
          pc32_main (optional int32 [M, T]): popcount per tick (main)
          pc32_twin (optional int32 [M, T]): popcount per tick (twin)
        """
        W_bits = np.asarray(W_bits, dtype=np.uint32)
        X_bits = np.asarray(X_bits, dtype=np.uint32)
        M, Kw = int(W_bits.shape[0]), int(W_bits.shape[1])
        if int(X_bits.shape[1]) != Kw:
            raise ValueError("X_bits second dimension must match W_bits")
        if int(X_bits.shape[0]) < T:
            raise ValueError("X_bits must have at least T rows")

        # Kernel selection logic
        if kernel == "auto":
            # Use tiled only if early_exit_enable==0 and shapes exceed thresholds
            use_tiled = (not early_exit_enable) and (M >= 128 or Kw >= 64) and (T >= 2)
            selected_kernel = "tiled" if use_tiled else "naive"
        else:
            selected_kernel = kernel

        # Validate kernel selection
        if selected_kernel == "tiled":
            if early_exit_enable:
                raise ValueError("Tiled kernel requires early_exit_enable=False")
            if not hasattr(self.prog, "bsdm_w_tiled"):
                raise RuntimeError("OpenCL program does not contain 'bsdm_w_tiled' kernel")
        elif selected_kernel == "naive":
            if not hasattr(self.prog, "bsdm_w_naive_norm"):
                raise RuntimeError("OpenCL program does not contain 'bsdm_w_naive_norm' kernel")
        else:
            raise ValueError(f"Invalid kernel selection: {selected_kernel}")

        # Get auto-tuned configuration for tiled kernel
        tile_kw_words = 64  # Default
        local_size_override = None

        if selected_kernel == "tiled":
            from onebit.autotune.tuner import get_best_config, get_device_key, get_kernel_hash

            if self._device_key is None:
                self._device_key = get_device_key(self.ctx)

            kernel_hash = get_kernel_hash(self.kernel_source)
            tile_kw_words, local_size_override = get_best_config(
                self._device_key, kernel_hash, default_tile_kw=64, default_local_size=256
            )

        X_slice = X_bits[:T, :]
        Y = np.empty((M,), dtype=np.float32)
        teff = np.empty((M,), dtype=np.int32)

        # Allocate bit arrays: T ticks × N carriers per channel
        samples_per_channel = T * walsh_N
        words_per_channel = (samples_per_channel + 31) // 32
        y_main_packed = np.zeros((M, words_per_channel), dtype=np.uint32) if want_y_pack else None
        y_twin_packed = np.zeros((M, words_per_channel), dtype=np.uint32) if (want_y_pack and antithetic) else None

        # Allocate pc32 arrays
        pc32_main_arr = np.zeros((M, T), dtype=np.int32) if want_pc32 else None
        pc32_twin_arr = np.zeros((M, T), dtype=np.int32) if (want_pc32 and antithetic) else None

        # Allocate CTG digest array
        ctg_digest_arr = np.zeros((M,), dtype=np.uint32)

        mf = cl.mem_flags
        bufW = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_bits)
        bufX = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X_slice)
        bufY = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=Y.nbytes)
        bufTEFF = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=teff.nbytes)

        # Create buffers for y_bits (zero-initialized via COPY_HOST_PTR)
        if y_main_packed is not None:
            bufYM = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y_main_packed)
        else:
            bufYM = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=4)

        if y_twin_packed is not None:
            bufYT = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y_twin_packed)
        else:
            bufYT = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=4)

        # Create buffers for pc32 (zero-initialized via COPY_HOST_PTR)
        if pc32_main_arr is not None:
            bufPC_M = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pc32_main_arr)
        else:
            bufPC_M = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=4)

        if pc32_twin_arr is not None:
            bufPC_T = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pc32_twin_arr)
        else:
            bufPC_T = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=4)

        # CTG digest buffer
        bufCTG = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ctg_digest_arr)

        # Energy instrumentation buffers (PR-3.7)
        if instr_on:
            energy_toggles_y_main = np.zeros((M,), dtype=np.uint64)
            energy_toggles_y_twin = np.zeros((M,), dtype=np.uint64)
            energy_ones_pc = np.zeros((M,), dtype=np.uint64)
            energy_zeros_pc = np.zeros((M,), dtype=np.uint64)
            energy_xnor_ops = np.zeros((M,), dtype=np.uint64)
            energy_popcnt_ops = np.zeros((M,), dtype=np.uint64)
            energy_bytes_W = np.zeros((M,), dtype=np.uint64)
            energy_bytes_X = np.zeros((M,), dtype=np.uint64)
            energy_bytes_out = np.zeros((M,), dtype=np.uint64)

            bufE_tog_m = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_toggles_y_main)
            bufE_tog_t = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_toggles_y_twin)
            bufE_ones = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_ones_pc)
            bufE_zeros = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_zeros_pc)
            bufE_xnor = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_xnor_ops)
            bufE_pop = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_popcnt_ops)
            bufE_bW = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_bytes_W)
            bufE_bX = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_bytes_X)
            bufE_bOut = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energy_bytes_out)
        else:
            # Create dummy buffers (4 bytes each) when instrumentation is off
            dummy = np.zeros((1,), dtype=np.uint64)
            bufE_tog_m = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)
            bufE_tog_t = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)
            bufE_ones = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)
            bufE_zeros = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)
            bufE_xnor = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)
            bufE_pop = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)
            bufE_bW = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)
            bufE_bX = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)
            bufE_bOut = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=8)

        global_size = (M,)

        # Use auto-tuned local size for tiled kernel, or user-provided local_size
        if selected_kernel == "tiled" and local_size_override is not None and local_size is None:
            local = (max(1, min(local_size_override, M)),)
        else:
            local = None if local_size is None else (max(1, min(local_size, M)),)

        # Select and invoke kernel
        if selected_kernel == "naive":
            evt = self.prog.bsdm_w_naive_norm(
                self.queue, global_size, local,
                bufW, bufX,
                np.int32(Kw), np.int32(T), np.float32(eps), np.float32(delta),
                np.ubyte(order), np.float32(beta), np.float32(lambd),
                np.ubyte(1 if use_ctg else 0), np.uint64(np.uint64(prf_seed)),
                np.ubyte(walsh_N), np.ubyte(1 if antithetic else 0),
                np.ubyte(1 if early_exit_enable else 0),
                bufY, bufTEFF, bufYM, bufYT, bufPC_M, bufPC_T, bufCTG,
                # Energy instrumentation (PR-3.7)
                np.ubyte(1 if instr_on else 0),
                bufE_tog_m, bufE_tog_t, bufE_ones, bufE_zeros,
                bufE_xnor, bufE_pop, bufE_bW, bufE_bX, bufE_bOut
            )
        else:  # tiled
            evt = self.prog.bsdm_w_tiled(
                self.queue, global_size, local,
                bufW, bufX,
                np.int32(Kw), np.int32(T), np.float32(eps), np.float32(delta),
                np.ubyte(order), np.float32(beta), np.float32(lambd),
                np.ubyte(1 if use_ctg else 0), np.uint64(np.uint64(prf_seed)),
                np.ubyte(0),  # early_exit_enable must be 0 for tiled
                bufY, bufTEFF, bufYM, bufYT, bufPC_M, bufPC_T, bufCTG,
                # Energy instrumentation (PR-3.7)
                np.ubyte(1 if instr_on else 0),
                bufE_tog_m, bufE_tog_t, bufE_ones, bufE_zeros,
                bufE_xnor, bufE_pop, bufE_bW, bufE_bX, bufE_bOut
            )
        evt.wait()
        cl.enqueue_copy(self.queue, Y, bufY).wait()
        cl.enqueue_copy(self.queue, teff, bufTEFF).wait()
        cl.enqueue_copy(self.queue, ctg_digest_arr, bufCTG).wait()

        result = {"Y": Y, "T_eff": teff, "ctg_digest": ctg_digest_arr}

        if want_y_pack and y_main_packed is not None:
            cl.enqueue_copy(self.queue, y_main_packed, bufYM).wait()
            result["y_bits_main"] = y_main_packed
        if want_y_pack and y_twin_packed is not None:
            cl.enqueue_copy(self.queue, y_twin_packed, bufYT).wait()
            result["y_bits_twin"] = y_twin_packed

        if want_pc32 and pc32_main_arr is not None:
            cl.enqueue_copy(self.queue, pc32_main_arr, bufPC_M).wait()
            result["pc32_main"] = pc32_main_arr
        if want_pc32 and pc32_twin_arr is not None:
            cl.enqueue_copy(self.queue, pc32_twin_arr, bufPC_T).wait()
            result["pc32_twin"] = pc32_twin_arr

        # Energy instrumentation (PR-3.7)
        if instr_on:
            cl.enqueue_copy(self.queue, energy_toggles_y_main, bufE_tog_m).wait()
            cl.enqueue_copy(self.queue, energy_toggles_y_twin, bufE_tog_t).wait()
            cl.enqueue_copy(self.queue, energy_ones_pc, bufE_ones).wait()
            cl.enqueue_copy(self.queue, energy_zeros_pc, bufE_zeros).wait()
            cl.enqueue_copy(self.queue, energy_xnor_ops, bufE_xnor).wait()
            cl.enqueue_copy(self.queue, energy_popcnt_ops, bufE_pop).wait()
            cl.enqueue_copy(self.queue, energy_bytes_W, bufE_bW).wait()
            cl.enqueue_copy(self.queue, energy_bytes_X, bufE_bX).wait()
            cl.enqueue_copy(self.queue, energy_bytes_out, bufE_bOut).wait()

            result["energy"] = {
                "toggles_y_main": energy_toggles_y_main,
                "toggles_y_twin": energy_toggles_y_twin,
                "ones_pc": energy_ones_pc,
                "zeros_pc": energy_zeros_pc,
                "xnor_ops": energy_xnor_ops,
                "popcnt_ops": energy_popcnt_ops,
                "bytes_W": energy_bytes_W,
                "bytes_X": energy_bytes_X,
                "bytes_out": energy_bytes_out,
            }

        return result

    def stageA_probe_topT_opencl(
        self,
        Q_bits: np.ndarray,  # [Kw] uint32
        K_bits: np.ndarray,  # [L, Kw] uint32
        *,
        kA: int = 16,
        T_set: tuple[int, ...] = (8, 12, 16),
        prf_seed: int,
        walsh_N: int = 2,
        antithetic: bool = True,
        order: int = 2,
        beta: float = 0.30,
        lambd: float = 1.0/256.0,
        local_size: int | None = 256,
    ) -> dict:
        """Stage-A probe using OpenCL: compute means for all keys, select Top-T.

        Mirrors CPU stageA_probe_topT with identical outputs.
        No early-exit (early_exit_enable=False).

        Returns:
            dict with keys:
                T_sel: int
                idx_top: np.ndarray[int]
                stats: dict with mu, gap12, elbow_T_raw, teff
        """
        from onebit.core.elbow import compute_elbow

        Q_bits = np.asarray(Q_bits, dtype=np.uint32)
        K_bits = np.asarray(K_bits, dtype=np.uint32)

        if Q_bits.ndim != 1:
            raise ValueError("Q_bits must be 1D [Kw]")
        if K_bits.ndim != 2:
            raise ValueError("K_bits must be 2D [L, Kw]")
        if K_bits.shape[1] != Q_bits.shape[0]:
            raise ValueError("K_bits second dimension must match Q_bits length")

        L = K_bits.shape[0]
        Kw = K_bits.shape[1]

        # Prepare X_bits: tile Q_bits kA times
        X_ticks = np.tile(Q_bits[None, :], (kA, 1))  # [kA, Kw]

        # Run BSDM-W on all keys with early_exit_enable=False
        out = self.run_bsdm_w_naive_norm(
            K_bits, X_ticks, T=kA,
            eps=0.0, delta=1e-9,  # ignored when early_exit_enable=False
            order=order, beta=beta, lambd=lambd,
            walsh_N=walsh_N, antithetic=antithetic,
            use_ctg=False, prf_seed=prf_seed,
            early_exit_enable=False,  # force full kA ticks
            local_size=local_size,
            want_y_pack=False, want_pc32=False,
        )

        mu = out["Y"]  # [L] float32
        teff_arr = out["T_eff"]  # [L] int32

        # Verify no early-exit
        assert np.all(teff_arr == kA), f"Early-exit occurred: teff={teff_arr}"

        # Elbow detection
        T_sel, gap = compute_elbow(mu, T_set)

        # Get top T_sel indices
        idx_sorted = np.argsort(mu)[::-1]
        idx_top = idx_sorted[:T_sel]

        # Compute raw elbow position
        mu_sorted = np.sort(mu)[::-1]
        if len(mu_sorted) > 1:
            diffs = mu_sorted[:-1] - mu_sorted[1:]
            n_top = min(64, len(diffs) - 1)
            if n_top > 1:
                second_diffs = diffs[:n_top-1] - diffs[1:n_top]
                j_star = int(np.argmax(second_diffs))
                elbow_T_raw = float(j_star + 1)
            else:
                elbow_T_raw = 1.0
        else:
            elbow_T_raw = 1.0

        return {
            "T_sel": T_sel,
            "idx_top": idx_top,
            "stats": {
                "mu": mu,
                "gap12": gap,
                "elbow_T_raw": elbow_T_raw,
                "teff": kA,
            },
        }

    def run_hcl_naive(
        self,
        Q_bits: np.ndarray,
        v_ids: np.ndarray,
        d: int,
        T: int,
        eps: float = 0.0,
        delta: float = 1e-3,
        order: int = 2,
        beta: float = 0.30,
        lambd: float = 1.0 / 256.0,
        use_ctg: bool = False,
        prf_seed: int = 0,
        early_exit_enable: bool = True,
        want_bits: bool = False,
        local_size: int = 256,
    ) -> dict:
        """Run HCL naive kernel for logits energy computation.

        Args:
            Q_bits: Query vector (packed bits), shape [d_words]
            v_ids: Candidate token IDs, shape [Kc]
            d: Dimension
            T: Number of ticks
            eps: Early-exit epsilon
            delta: Early-exit delta
            order: SD modulator order (1 or 2)
            beta: SD-2 beta parameter
            lambd: SD leak parameter
            use_ctg: Enable CTG
            prf_seed: PRF seed
            early_exit_enable: Enable early-exit
            want_bits: Return packed y_bits
            local_size: Work-group size

        Returns:
            Dict with E_mean, T_eff, ctg_digest, optionally y_bits
        """
        Kc = len(v_ids)
        d_words = (d + 31) // 32

        # Ensure HCL kernel is loaded
        if not hasattr(self, '_hcl_kernel_loaded'):
            hcl_path = Path(__file__).with_name("hcl_naive.cl")
            with open(hcl_path, "r", encoding="utf-8") as f:
                hcl_src = f.read()

            # Rebuild program with HCL kernel
            self.prog = cl.Program(self.ctx, hcl_src).build()
            self._hcl_kernel_loaded = True

        # Create buffers
        mf = cl.mem_flags

        buf_Q = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q_bits)
        buf_V = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v_ids.astype(np.int32))

        buf_E_mean = cl.Buffer(self.ctx, mf.WRITE_ONLY, Kc * 4)
        buf_T_eff = cl.Buffer(self.ctx, mf.WRITE_ONLY, Kc * 4)
        buf_ctg_digest = cl.Buffer(self.ctx, mf.WRITE_ONLY, Kc * 4)

        # Optional y_bits buffer
        if want_bits:
            k_words = (T + 31) // 32
            y_bits_host = np.zeros((Kc, k_words), dtype=np.uint32)
            buf_y_bits = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y_bits_host)
        else:
            buf_y_bits = cl.Buffer(self.ctx, mf.READ_WRITE, 4)  # Dummy buffer

        # Launch kernel
        global_size = ((Kc + local_size - 1) // local_size) * local_size

        evt = self.prog.hcl_naive(
            self.queue,
            (global_size,),
            (local_size,),
            buf_Q,
            buf_V,
            np.int32(Kc),
            np.int32(d),
            np.int32(d_words),
            np.int32(T),
            np.float32(eps),
            np.float32(delta),
            np.uint8(order),
            np.float32(beta),
            np.float32(lambd),
            np.uint8(1 if use_ctg else 0),
            np.uint64(prf_seed),
            np.uint8(1 if early_exit_enable else 0),
            buf_E_mean,
            buf_T_eff,
            buf_y_bits if want_bits else cl.Buffer(self.ctx, mf.READ_WRITE, 4),
            buf_ctg_digest,
        )
        evt.wait()

        # Read results
        E_mean = np.empty(Kc, dtype=np.float32)
        T_eff = np.empty(Kc, dtype=np.int32)
        ctg_digest = np.empty(Kc, dtype=np.uint32)

        cl.enqueue_copy(self.queue, E_mean, buf_E_mean).wait()
        cl.enqueue_copy(self.queue, T_eff, buf_T_eff).wait()
        cl.enqueue_copy(self.queue, ctg_digest, buf_ctg_digest).wait()

        result = {
            "E_mean": E_mean,
            "T_eff": T_eff,
            "ctg_digest": ctg_digest,
        }

        if want_bits:
            y_bits = np.empty((Kc, k_words), dtype=np.uint32)
            cl.enqueue_copy(self.queue, y_bits, buf_y_bits).wait()
            result["y_bits"] = y_bits

        return result

    def run_hcl_tiled(
        self,
        Q_bits: np.ndarray,
        v_ids: np.ndarray,
        d: int,
        T: int,
        eps: float = 0.0,
        delta: float = 1e-3,
        order: int = 2,
        beta: float = 0.30,
        lambd: float = 1.0 / 256.0,
        use_ctg: bool = False,
        prf_seed: int = 0,
        early_exit_enable: bool = False,
        want_bits: bool = False,
        local_size: int = 256,
        tile_d_words: int = 64,
    ) -> dict:
        """Run HCL tiled kernel for logits energy computation.

        Args:
            Q_bits: Query vector (packed bits), shape [d_words]
            v_ids: Candidate token IDs, shape [Kc]
            d: Dimension
            T: Number of ticks
            eps: Early-exit epsilon (unused in tiled)
            delta: Early-exit delta (unused in tiled)
            order: SD modulator order (1 or 2)
            beta: SD-2 beta parameter
            lambd: SD leak parameter
            use_ctg: Enable CTG
            prf_seed: PRF seed
            early_exit_enable: Must be False for tiled
            want_bits: Return packed y_bits
            local_size: Work-group size
            tile_d_words: Tile size for dimension

        Returns:
            Dict with E_mean, T_eff, ctg_digest, optionally y_bits
        """
        if early_exit_enable:
            raise ValueError("HCL tiled kernel does not support early-exit")

        Kc = len(v_ids)
        d_words = (d + 31) // 32

        # Ensure HCL tiled kernel is loaded
        if not hasattr(self, '_hcl_tiled_kernel_loaded'):
            hcl_tiled_path = Path(__file__).with_name("hcl_tiled.cl")
            with open(hcl_tiled_path, "r", encoding="utf-8") as f:
                hcl_tiled_src = f.read()

            # Replace TILE_D_WORDS with actual value
            hcl_tiled_src = hcl_tiled_src.replace(
                "#define TILE_D_WORDS 64",
                f"#define TILE_D_WORDS {tile_d_words}"
            )

            # Rebuild program with HCL tiled kernel
            self.prog = cl.Program(self.ctx, hcl_tiled_src).build()
            self._hcl_tiled_kernel_loaded = True

        # Create buffers
        mf = cl.mem_flags

        buf_Q = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q_bits)
        buf_V = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v_ids.astype(np.int32))

        buf_E_mean = cl.Buffer(self.ctx, mf.WRITE_ONLY, Kc * 4)
        buf_T_eff = cl.Buffer(self.ctx, mf.WRITE_ONLY, Kc * 4)
        buf_ctg_digest = cl.Buffer(self.ctx, mf.WRITE_ONLY, Kc * 4)

        # Optional y_bits buffer
        if want_bits:
            k_words = (T + 31) // 32
            y_bits_host = np.zeros((Kc, k_words), dtype=np.uint32)
            buf_y_bits = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y_bits_host)
        else:
            buf_y_bits = cl.Buffer(self.ctx, mf.READ_WRITE, 4)  # Dummy buffer

        # Launch kernel
        global_size = ((Kc + local_size - 1) // local_size) * local_size

        evt = self.prog.hcl_tiled(
            self.queue,
            (global_size,),
            (local_size,),
            buf_Q,
            buf_V,
            np.int32(Kc),
            np.int32(d),
            np.int32(d_words),
            np.int32(T),
            np.float32(eps),
            np.float32(delta),
            np.uint8(order),
            np.float32(beta),
            np.float32(lambd),
            np.uint8(1 if use_ctg else 0),
            np.uint64(prf_seed),
            np.uint8(0),  # early_exit_enable must be 0
            buf_E_mean,
            buf_T_eff,
            buf_y_bits if want_bits else cl.Buffer(self.ctx, mf.READ_WRITE, 4),
            buf_ctg_digest,
        )
        evt.wait()

        # Read results
        E_mean = np.empty(Kc, dtype=np.float32)
        T_eff = np.empty(Kc, dtype=np.int32)
        ctg_digest = np.empty(Kc, dtype=np.uint32)

        cl.enqueue_copy(self.queue, E_mean, buf_E_mean).wait()
        cl.enqueue_copy(self.queue, T_eff, buf_T_eff).wait()
        cl.enqueue_copy(self.queue, ctg_digest, buf_ctg_digest).wait()

        result = {
            "E_mean": E_mean,
            "T_eff": T_eff,
            "ctg_digest": ctg_digest,
        }

        if want_bits:
            y_bits = np.empty((Kc, k_words), dtype=np.uint32)
            cl.enqueue_copy(self.queue, y_bits, buf_y_bits).wait()
            result["y_bits"] = y_bits

        return result

    def run_ldpkv_decode_stage1(
        self,
        K_enc: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        edge_weights: np.ndarray,
        n_pos: int,
        T: int,
        order: int = 2,
        beta: float = 0.30,
        lambd: float = 1.0 / 256.0,
        early_exit_enable: bool = False,
        local_size: int = 256,
    ) -> dict:
        """Run LDP-KV decode Stage-1 kernel.

        Args:
            K_enc: Encoded keys, shape [d_dim]
            row_ptr: CSR row pointers, shape [n_pos + 1]
            col_idx: Column indices, shape [n_edges]
            edge_weights: Edge weights (±1), shape [n_edges]
            n_pos: Number of positions
            T: Number of ticks
            order: SD modulator order (1 or 2)
            beta: SD-2 beta parameter
            lambd: SD leak parameter
            early_exit_enable: Enable early-exit
            local_size: Work-group size

        Returns:
            Dict with E_mean, T_eff
        """
        # Ensure LDP-KV kernel is loaded
        if not hasattr(self, '_ldpkv_kernel_loaded'):
            ldpkv_path = Path(__file__).with_name("ldpkv_decode.cl")
            with open(ldpkv_path, "r", encoding="utf-8") as f:
                ldpkv_src = f.read()

            # Rebuild program with LDP-KV kernel
            self.prog = cl.Program(self.ctx, ldpkv_src).build()
            self._ldpkv_kernel_loaded = True

        # Create buffers
        mf = cl.mem_flags

        buf_K_enc = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K_enc.astype(np.float32))
        buf_row_ptr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=row_ptr.astype(np.int32))
        buf_col_idx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=col_idx.astype(np.int32))
        buf_edge_weights = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edge_weights.astype(np.int8))

        buf_E_mean = cl.Buffer(self.ctx, mf.WRITE_ONLY, n_pos * 4)
        buf_T_eff = cl.Buffer(self.ctx, mf.WRITE_ONLY, n_pos * 4)

        # Launch kernel
        global_size = ((n_pos + local_size - 1) // local_size) * local_size

        evt = self.prog.ldpkv_decode_stage1(
            self.queue,
            (global_size,),
            (local_size,),
            buf_K_enc,
            buf_row_ptr,
            buf_col_idx,
            buf_edge_weights,
            np.int32(n_pos),
            np.int32(T),
            np.uint8(order),
            np.float32(beta),
            np.float32(lambd),
            np.uint8(1 if early_exit_enable else 0),
            buf_E_mean,
            buf_T_eff,
        )
        evt.wait()

        # Read results
        E_mean = np.empty(n_pos, dtype=np.float32)
        T_eff = np.empty(n_pos, dtype=np.int32)

        cl.enqueue_copy(self.queue, E_mean, buf_E_mean).wait()
        cl.enqueue_copy(self.queue, T_eff, buf_T_eff).wait()

        return {
            "E_mean": E_mean,
            "T_eff": T_eff,
        }

    def run_ldpkv_decode_stage2(
        self,
        V_enc: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        edge_weights: np.ndarray,
        winner_positions: np.ndarray,
        d_kv: int,
        repair_pass: bool = False,
        group_idx: int = 0,
        group_size: int = 64,
        K_enc: Optional[np.ndarray] = None,
        K_bits_inout: Optional[np.ndarray] = None,
        local_size: int = 256,
    ) -> dict:
        """Run LDP-KV decode Stage-2 kernel.

        Args:
            V_enc: Encoded values, shape [d_dim]
            row_ptr: CSR row pointers, shape [n_pos + 1]
            col_idx: Column indices, shape [n_edges]
            edge_weights: Edge weights (±1), shape [n_edges]
            winner_positions: Positions to decode, shape [n_winners]
            d_kv: Dimension of K/V
            repair_pass: If True, perform repair (PR-4.0)
            group_idx: Group index to repair
            group_size: Positions per group
            K_enc: Encoded keys (required if repair_pass=True)
            K_bits_inout: KV cache to repair in-place (required if repair_pass=True)
            local_size: Work-group size

        Returns:
            Dict with V_decoded, repaired
        """
        # Ensure LDP-KV kernel is loaded
        if not hasattr(self, '_ldpkv_kernel_loaded'):
            ldpkv_path = Path(__file__).with_name("ldpkv_decode.cl")
            with open(ldpkv_path, "r", encoding="utf-8") as f:
                ldpkv_src = f.read()

            # Rebuild program with LDP-KV kernel
            self.prog = cl.Program(self.ctx, ldpkv_src).build()
            self._ldpkv_kernel_loaded = True

        n_winners = len(winner_positions)
        d_kv_words = (d_kv + 31) // 32

        # Create buffers
        mf = cl.mem_flags

        buf_V_enc = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V_enc.astype(np.float32))
        buf_row_ptr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=row_ptr.astype(np.int32))
        buf_col_idx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=col_idx.astype(np.int32))
        buf_edge_weights = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edge_weights.astype(np.int8))

        # Handle empty winner_positions array
        if n_winners > 0:
            buf_winner_positions = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=winner_positions.astype(np.int32))
        else:
            # Create dummy buffer for empty array
            buf_winner_positions = cl.Buffer(self.ctx, mf.READ_ONLY, 4)

        # Output buffer for decoded values
        if n_winners > 0:
            V_decoded_host = np.zeros((n_winners, d_kv_words), dtype=np.uint32)
            buf_V_decoded = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=V_decoded_host)
        else:
            # Create dummy buffer for empty array
            buf_V_decoded = cl.Buffer(self.ctx, mf.READ_WRITE, 4)

        # Repair mode buffers
        if repair_pass:
            if K_enc is None or K_bits_inout is None:
                raise ValueError("repair_pass=True requires K_enc and K_bits_inout")

            buf_K_enc = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K_enc.astype(np.float32))
            buf_K_bits_inout = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=K_bits_inout)
        else:
            # Dummy buffers
            buf_K_enc = cl.Buffer(self.ctx, mf.READ_ONLY, 4)
            buf_K_bits_inout = cl.Buffer(self.ctx, mf.READ_WRITE, 4)

        # Launch kernel
        global_size = ((n_winners + local_size - 1) // local_size) * local_size if n_winners > 0 else local_size

        evt = self.prog.ldpkv_decode_stage2(
            self.queue,
            (global_size,),
            (local_size,),
            buf_V_enc,
            buf_row_ptr,
            buf_col_idx,
            buf_edge_weights,
            buf_winner_positions,
            np.int32(n_winners),
            np.int32(d_kv),
            np.int32(d_kv_words),
            buf_V_decoded,
            np.uint8(1 if repair_pass else 0),
            np.int32(group_idx),
            np.int32(group_size),
            buf_K_enc,
            buf_K_bits_inout,
        )
        evt.wait()

        # Read results
        if n_winners > 0:
            V_decoded = np.empty((n_winners, d_kv_words), dtype=np.uint32)
            cl.enqueue_copy(self.queue, V_decoded, buf_V_decoded).wait()
        else:
            V_decoded = np.empty((0, d_kv_words), dtype=np.uint32)

        # If repair mode, read back repaired K_bits
        if repair_pass:
            cl.enqueue_copy(self.queue, K_bits_inout, buf_K_bits_inout).wait()

        return {
            "V_decoded": V_decoded,
            "repaired": repair_pass,
        }


class OpenCLBinGemmIncremental:
    """Incremental API: start/extend/finalize accumulating acc_lo/acc_hi/TEFF.

    Early-exit aggregation across chunks is supported implicitly: TEFF accumulates.
    The kernel always runs with accum_mode=1 in extend().
    """
    def __init__(self, base: Optional[OpenCLBinGemm] = None) -> None:
        self.base = base or OpenCLBinGemm()
        self.ctx = self.base.ctx
        self.queue = self.base.queue
        self.prog = self.base.prog
        self._mf = cl.mem_flags
        self._inited = False

    def start(self, W_bits: np.ndarray, cv: np.ndarray) -> None:
        cv = np.asarray(cv, dtype=np.float32)
        if cv.shape != (2,):
            raise ValueError("cv must be shape [2]")
        self.W_bits = np.asarray(W_bits, dtype=np.uint32)
        self.M, self.Kw = int(self.W_bits.shape[0]), int(self.W_bits.shape[1])
        self.bufW = cl.Buffer(self.ctx, self._mf.READ_ONLY | self._mf.COPY_HOST_PTR, hostbuf=self.W_bits)
        self.bufCV = cl.Buffer(self.ctx, self._mf.READ_ONLY | self._mf.COPY_HOST_PTR, hostbuf=cv)
        self.bufACC_LO = cl.Buffer(self.ctx, self._mf.READ_WRITE, size=self.M * 4)
        self.bufACC_HI = cl.Buffer(self.ctx, self._mf.READ_WRITE, size=self.M * 4)
        self.bufTEFF = cl.Buffer(self.ctx, self._mf.READ_WRITE, size=self.M * 4)
        # zero init
        zero_f = np.zeros((self.M,), dtype=np.float32)
        zero_i = np.zeros((self.M,), dtype=np.int32)
        cl.enqueue_copy(self.queue, self.bufACC_LO, zero_f).wait()
        cl.enqueue_copy(self.queue, self.bufACC_HI, zero_f).wait()
        cl.enqueue_copy(self.queue, self.bufTEFF, zero_i).wait()
        # dummy Y buffer
        self.bufY = cl.Buffer(self.ctx, self._mf.WRITE_ONLY, size=self.M * 4)
        self._inited = True

    def extend(self, X_bits_chunk: np.ndarray, T_chunk: int, eps_margin: float = 0.0, delta: float = 0.05, local_size: Optional[int] = None) -> None:
        if not self._inited:
            raise RuntimeError("call start() first")
        X_bits_chunk = np.asarray(X_bits_chunk, dtype=np.uint32)
        if int(X_bits_chunk.shape[1]) != self.Kw:
            raise ValueError("X_bits_chunk second dimension must match W_bits")
        if int(X_bits_chunk.shape[0]) < T_chunk:
            raise ValueError("X_bits_chunk must have at least T_chunk rows")
        X_slice = X_bits_chunk[:T_chunk, :]
        bufX = cl.Buffer(self.ctx, self._mf.READ_ONLY | self._mf.COPY_HOST_PTR, hostbuf=X_slice)
        global_size = (self.M,)
        local = None if local_size is None else (max(1, min(local_size, self.M)),)
        # accum_mode=1; we pass bufY as dummy Y
        evt = self.prog.fused_bin_gemm(
            self.queue, global_size, local,
            self.bufW, bufX, self.bufCV,
            np.int32(self.Kw), np.int32(T_chunk), np.float32(delta), np.float32(eps_margin),
            self.bufY, self.bufTEFF, self.bufACC_LO, self.bufACC_HI, np.int32(1)
        )
        evt.wait()

    def finalize(self) -> dict:
        if not self._inited:
            raise RuntimeError("call start() first")
        acc_lo = np.empty((self.M,), dtype=np.float32)
        acc_hi = np.empty((self.M,), dtype=np.float32)
        teff = np.empty((self.M,), dtype=np.int32)
        cl.enqueue_copy(self.queue, acc_lo, self.bufACC_LO).wait()
        cl.enqueue_copy(self.queue, acc_hi, self.bufACC_HI).wait()
        cl.enqueue_copy(self.queue, teff, self.bufTEFF).wait()
        denom = np.maximum(teff.astype(np.int32), 1).astype(np.float32)
        Y = (acc_lo + acc_hi) / denom
        return {"Y": Y, "T_eff": teff}

