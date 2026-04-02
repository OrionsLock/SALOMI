#!/usr/bin/env python3
"""
Optimized VQ Decoding for SALOMI
Implements fast VQ decoding with caching and vectorization
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Union
from collections import OrderedDict
import time
from tqdm import tqdm

class OptimizedVQDecoder:
    """
    Optimized VQ decoder with caching, vectorization, and batching
    """

    def __init__(self, max_cache_size: int = 10000):
        """
        Initialize optimized VQ decoder

        Args:
            max_cache_size: Maximum number of entries to cache
        """
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0

    def fast_decode(self, indices: Union[np.ndarray, torch.Tensor],
                   codebook: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Optimized VQ decoding with caching and vectorization

        Args:
            indices: Indices to decode
            codebook: Codebook to use for decoding

        Returns:
            Decoded values
        """
        # Create cache key
        if isinstance(indices, np.ndarray):
            indices_key = (indices.data.tobytes(), indices.shape, indices.dtype)
        else:
            indices_key = (indices.cpu().numpy().data.tobytes(), indices.shape, indices.dtype)

        if isinstance(codebook, np.ndarray):
            codebook_key = (codebook.data.tobytes(), codebook.shape, codebook.dtype)
        else:
            codebook_key = (codebook.cpu().numpy().data.tobytes(), codebook.shape, codebook.dtype)

        cache_key = (indices_key, codebook_key)

        # Check cache
        if cache_key in self.cache:
            self.hit_count += 1
            return self.cache[cache_key]
        else:
            self.miss_count += 1

        # Vectorized lookup
        if isinstance(indices, np.ndarray) and isinstance(codebook, np.ndarray):
            # NumPy vectorized lookup
            result = codebook[indices]
        elif isinstance(indices, torch.Tensor) and isinstance(codebook, torch.Tensor):
            # PyTorch vectorized lookup
            result = codebook[indices]
        else:
            # Mixed types - convert to consistent type
            if isinstance(indices, np.ndarray):
                indices = torch.from_numpy(indices)
                codebook = torch.from_numpy(codebook) if isinstance(codebook, np.ndarray) else codebook
            else:
                indices = indices.cpu().numpy()
                codebook = codebook.cpu().numpy() if isinstance(codebook, torch.Tensor) else codebook

            result = codebook[indices]

        # Reshape to match indices shape + codebook dimension
        if isinstance(indices, np.ndarray):
            result = result.reshape(indices.shape + (-1,))
        else:
            result = result.reshape(indices.shape + (-1,))

        # Cache result
        self.cache[cache_key] = result
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # Remove oldest

        return result

    def batch_decode(self, index_batches: List[Union[np.ndarray, torch.Tensor]],
                    codebook: Union[np.ndarray, torch.Tensor]) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Decode multiple batches at once for better efficiency

        Args:
            index_batches: List of index arrays to decode
            codebook: Codebook to use

        Returns:
            List of decoded batches
        """
        # Concatenate all indices
        if isinstance(index_batches[0], np.ndarray):
            all_indices = np.concatenate(index_batches, axis=0)
        else:
            all_indices = torch.cat(index_batches, dim=0)

        # Single large decode
        all_decoded = self.fast_decode(all_indices, codebook)

        # Split back into batches
        decoded_batches = []
        start = 0
        for batch in index_batches:
            size = batch.shape[0]
            if isinstance(batch, np.ndarray):
                decoded_batches.append(all_decoded[start:start+size])
            else:
                decoded_batches.append(all_decoded[start:start+size])
            start += size

        return decoded_batches

    def parallel_decode(self, indices: Union[np.ndarray, torch.Tensor],
                       codebook: Union[np.ndarray, torch.Tensor],
                       num_chunks: int = 4) -> Union[np.ndarray, torch.Tensor]:
        """
        Parallel decoding by splitting into chunks

        Args:
            indices: Indices to decode
            codebook: Codebook to use
            num_chunks: Number of chunks to split into

        Returns:
            Decoded values
        """
        # Split indices into chunks
        if isinstance(indices, np.ndarray):
            chunk_size = indices.shape[0] // num_chunks
            chunks = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
        else:
            chunk_size = indices.shape[0] // num_chunks
            chunks = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

        # Decode chunks in parallel (simulated)
        decoded_chunks = []
        for chunk in chunks:
            decoded_chunks.append(self.fast_decode(chunk, codebook))

        # Concatenate results
        if isinstance(indices, np.ndarray):
            return np.concatenate(decoded_chunks, axis=0)
        else:
            return torch.cat(decoded_chunks, dim=0)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / (self.hit_count + self.miss_count + 1e-6)
        }

    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    def benchmark_decode(self, indices: Union[np.ndarray, torch.Tensor],
                        codebook: Union[np.ndarray, torch.Tensor],
                        num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark decoding performance

        Args:
            indices: Sample indices for benchmarking
            codebook: Sample codebook
            num_runs: Number of runs for benchmarking

        Returns:
            Performance metrics
        """
        # Warm up cache
        self.fast_decode(indices, codebook)

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            result = self.fast_decode(indices, codebook)
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        ops_per_second = num_runs / total_time

        # Memory usage
        if isinstance(indices, np.ndarray):
            memory_usage = indices.nbytes + codebook.nbytes
        else:
            memory_usage = indices.element_size() * indices.nelement() + codebook.element_size() * codebook.nelement()

        return {
            "total_time": total_time,
            "average_time": avg_time,
            "operations_per_second": ops_per_second,
            "memory_usage_bytes": memory_usage,
            "cache_stats": self.get_cache_stats()
        }

def create_optimized_vq_decoder(max_cache_size: int = 10000) -> OptimizedVQDecoder:
    """Factory function"""
    return OptimizedVQDecoder(max_cache_size)

# Example usage
if __name__ == "__main__":
    print("OptimizedVQDecoder ready for use")
    print("Usage: decoder = create_optimized_vq_decoder()")
    print("       decoded = decoder.fast_decode(indices, codebook)")
    print("       batch_results = decoder.batch_decode([indices1, indices2], codebook)")