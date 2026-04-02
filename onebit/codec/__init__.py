"""Codecs for serialization and compression."""

from .rle import pack_runs, unpack_runs

__all__ = ["pack_runs", "unpack_runs"]

