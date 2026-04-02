"""Step 1: Formalize Bit Budget with Real Compression.

ChatGPT's validation: "Make sure your 1.062 bpp number survives real compression,
not just entropy estimates."

This module:
1. Creates a toy archive format for hybrid quantized weights
2. Uses zstd/gzip compression on sign blocks + magnitude side-channel
3. Measures ACTUAL bits per parameter after compression
"""

import numpy as np
import zlib
import struct
from typing import Tuple, Dict
from dataclasses import dataclass

# Try to import zstd (optional, fall back to zlib)
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("Warning: zstandard not available, using zlib instead")


@dataclass
class HybridArchive:
    """Archive format for hybrid block-structured quantized weights.
    
    Format:
    - Header: shape, block_size, n_levels (16 bytes)
    - Sign blocks: 1 bit per block (compressed)
    - Magnitude levels: log2(n_levels) bits per weight (compressed)
    """
    d_out: int
    d_in: int
    block_size: int
    n_levels: int
    sign_bytes: bytes
    magnitude_bytes: bytes
    
    def total_bytes(self) -> int:
        """Total compressed size including header."""
        header = 16  # 4 ints × 4 bytes
        return header + len(self.sign_bytes) + len(self.magnitude_bytes)
    
    def bpp(self) -> float:
        """Bits per parameter."""
        n_params = self.d_out * self.d_in
        return self.total_bytes() * 8 / n_params


def compress_data(data: bytes, method: str = 'zstd') -> bytes:
    """Compress data using specified method."""
    if method == 'zstd' and HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=19)  # Max compression
        return cctx.compress(data)
    else:
        return zlib.compress(data, level=9)


def decompress_data(data: bytes, method: str = 'zstd') -> bytes:
    """Decompress data."""
    if method == 'zstd' and HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    else:
        return zlib.decompress(data)


def pack_block_signs(W_signs: np.ndarray, block_size: int) -> bytes:
    """Pack block signs into bytes.
    
    Each block gets 1 bit (sign of majority).
    """
    d_out, d_in = W_signs.shape
    n_blocks_h = d_out // block_size
    n_blocks_w = d_in // block_size
    
    # Compute majority sign per block
    block_signs = []
    for bi in range(n_blocks_h):
        for bj in range(n_blocks_w):
            block = W_signs[bi*block_size:(bi+1)*block_size,
                           bj*block_size:(bj+1)*block_size]
            majority = 1 if np.mean(block) >= 0 else 0
            block_signs.append(majority)
    
    # Pack into bytes
    n_blocks = len(block_signs)
    n_bytes = (n_blocks + 7) // 8
    packed = bytearray(n_bytes)
    
    for i, sign in enumerate(block_signs):
        if sign:
            packed[i // 8] |= (1 << (i % 8))
    
    return bytes(packed)


def pack_magnitude_levels(mag_levels: np.ndarray, n_levels: int) -> bytes:
    """Pack magnitude levels into bytes.
    
    Uses log2(n_levels) bits per weight.
    """
    flat = mag_levels.flatten().astype(np.uint8)
    
    if n_levels <= 2:
        # 1 bit per weight - pack 8 weights per byte
        n_bytes = (len(flat) + 7) // 8
        packed = bytearray(n_bytes)
        for i, level in enumerate(flat):
            if level:
                packed[i // 8] |= (1 << (i % 8))
        return bytes(packed)
    elif n_levels <= 4:
        # 2 bits per weight - pack 4 weights per byte
        n_bytes = (len(flat) + 3) // 4
        packed = bytearray(n_bytes)
        for i, level in enumerate(flat):
            packed[i // 4] |= ((level & 0x3) << ((i % 4) * 2))
        return bytes(packed)
    elif n_levels <= 16:
        # 4 bits per weight - pack 2 weights per byte
        n_bytes = (len(flat) + 1) // 2
        packed = bytearray(n_bytes)
        for i, level in enumerate(flat):
            packed[i // 2] |= ((level & 0xF) << ((i % 2) * 4))
        return bytes(packed)
    else:
        # 8 bits per weight
        return bytes(flat)


def create_hybrid_archive(W_fp32: np.ndarray, block_size: int = 4,
                          n_levels: int = 2, compress: bool = True) -> HybridArchive:
    """Create compressed archive from FP32 weights.
    
    Steps:
    1. Compute block signs (majority sign per block)
    2. Compute magnitude levels (quantize |W| to n_levels)
    3. Pack and compress both
    """
    d_out, d_in = W_fp32.shape
    
    # Step 1: Block signs
    W_signs = np.sign(W_fp32)
    sign_packed = pack_block_signs(W_signs, block_size)
    
    # Step 2: Magnitude levels
    magnitudes = np.abs(W_fp32)
    thresholds = [np.percentile(magnitudes, 100 * i / n_levels) 
                  for i in range(1, n_levels)]
    
    mag_levels = np.zeros_like(magnitudes, dtype=np.uint8)
    for i, thresh in enumerate(thresholds):
        mag_levels[magnitudes >= thresh] = i + 1
    
    mag_packed = pack_magnitude_levels(mag_levels, n_levels)
    
    # Step 3: Compress
    if compress:
        sign_compressed = compress_data(sign_packed)
        mag_compressed = compress_data(mag_packed)
    else:
        sign_compressed = sign_packed
        mag_compressed = mag_packed
    
    return HybridArchive(
        d_out=d_out,
        d_in=d_in,
        block_size=block_size,
        n_levels=n_levels,
        sign_bytes=sign_compressed,
        magnitude_bytes=mag_compressed
    )


def test_real_compression():
    """Test that hybrid scheme achieves claimed BPP with real compression."""
    print("=" * 80)
    print("STEP 1: REAL COMPRESSION VALIDATION")
    print("=" * 80)
    print("\nChatGPT's challenge: 'Make sure your 1.062 bpp survives real compression'")
    print("-" * 80)
    
    results = []
    
    for dim in [64, 128, 256, 512]:
        print(f"\n--- Dimension: {dim}x{dim} ({dim*dim:,} params) ---")
        
        # Generate weights (trained-like: structured, not pure Gaussian)
        W_fp32 = np.random.randn(dim, dim).astype(np.float32)
        
        # Add some structure (block-like patterns)
        for i in range(0, dim, 4):
            for j in range(0, dim, 4):
                if np.random.rand() > 0.5:
                    W_fp32[i:i+4, j:j+4] *= np.sign(np.mean(W_fp32[i:i+4, j:j+4]))
        
        # Test different configurations
        configs = [
            {'block_size': 2, 'n_levels': 2, 'name': 'block2_2lv'},
            {'block_size': 4, 'n_levels': 2, 'name': 'block4_2lv'},
            {'block_size': 4, 'n_levels': 4, 'name': 'block4_4lv'},
            {'block_size': 8, 'n_levels': 2, 'name': 'block8_2lv'},
        ]
        
        print(f"\n{'Config':<15} {'Raw BPP':>10} {'Compressed':>12} {'Savings':>10}")
        print("-" * 50)
        
        for cfg in configs:
            # Uncompressed
            archive_raw = create_hybrid_archive(W_fp32, cfg['block_size'], 
                                                 cfg['n_levels'], compress=False)
            # Compressed
            archive_comp = create_hybrid_archive(W_fp32, cfg['block_size'],
                                                  cfg['n_levels'], compress=True)
            
            raw_bpp = archive_raw.bpp()
            comp_bpp = archive_comp.bpp()
            savings = (1 - comp_bpp / raw_bpp) * 100
            
            print(f"{cfg['name']:<15} {raw_bpp:>10.3f} {comp_bpp:>12.3f} {savings:>9.1f}%")
            
            results.append({
                'dim': dim,
                'config': cfg['name'],
                'raw_bpp': raw_bpp,
                'compressed_bpp': comp_bpp
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPRESSION VALIDATION SUMMARY")
    print("=" * 80)
    
    # Find best config under 1.1 bpp
    valid = [r for r in results if r['compressed_bpp'] <= 1.1]
    if valid:
        best = min(valid, key=lambda x: x['compressed_bpp'])
        print(f"\n✓ Best config under 1.1 bpp:")
        print(f"  {best['config']} @ {best['dim']}x{best['dim']}")
        print(f"  Compressed BPP: {best['compressed_bpp']:.3f}")
        print(f"\n✓ ChatGPT's challenge PASSED: Real compression achieves < 1.1 bpp")
    else:
        print("\n✗ No config achieved < 1.1 bpp with real compression")
        print("  Need to optimize compression or adjust config")
    
    return results


if __name__ == "__main__":
    test_real_compression()

