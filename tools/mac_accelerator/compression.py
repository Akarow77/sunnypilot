#!/usr/bin/env python3
"""Zstd codec using a Python wheel on macOS and the 3X system library as fallback."""

from __future__ import annotations

import ctypes
import ctypes.util


class ZstdError(RuntimeError):
  pass


class ZstdCodec:
  name = 'zstd-1'

  def __init__(self, level: int = 1):
    self.level = level
    try:
      import zstandard
    except ImportError:
      self._python_compressor = None
      self._python_decompressor = None
      library_name = ctypes.util.find_library('zstd') or 'libzstd.so.1'
      self._library = ctypes.CDLL(library_name)
      self._library.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
      self._library.ZSTD_compressBound.restype = ctypes.c_size_t
      self._library.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                               ctypes.c_size_t, ctypes.c_int]
      self._library.ZSTD_compress.restype = ctypes.c_size_t
      self._library.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                                ctypes.c_size_t]
      self._library.ZSTD_decompress.restype = ctypes.c_size_t
      self._library.ZSTD_isError.argtypes = [ctypes.c_size_t]
      self._library.ZSTD_isError.restype = ctypes.c_uint
      self._library.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
      self._library.ZSTD_getErrorName.restype = ctypes.c_char_p
      self._compression_buffer = None
      self._decompression_buffer = None
    else:
      self._library = None
      self._python_compressor = zstandard.ZstdCompressor(level=level)
      self._python_decompressor = zstandard.ZstdDecompressor()

  def _check(self, result: int) -> int:
    if self._library is not None and self._library.ZSTD_isError(result):
      error = self._library.ZSTD_getErrorName(result).decode()
      raise ZstdError(error)
    return result

  def compress(self, payload: bytes) -> bytes:
    if self._python_compressor is not None:
      return self._python_compressor.compress(payload)
    assert self._library is not None
    capacity = self._library.ZSTD_compressBound(len(payload))
    if self._compression_buffer is None or ctypes.sizeof(self._compression_buffer) < capacity:
      self._compression_buffer = ctypes.create_string_buffer(capacity)
    source = ctypes.c_char_p(payload)
    written = self._check(self._library.ZSTD_compress(
      self._compression_buffer, capacity, source, len(payload), self.level))
    return self._compression_buffer.raw[:written]

  def decompress(self, payload: bytes, expected_size: int) -> bytes:
    if self._python_decompressor is not None:
      output = self._python_decompressor.decompress(payload, max_output_size=expected_size)
    else:
      assert self._library is not None
      if self._decompression_buffer is None or ctypes.sizeof(self._decompression_buffer) < expected_size:
        self._decompression_buffer = ctypes.create_string_buffer(expected_size)
      source = ctypes.c_char_p(payload)
      written = self._check(self._library.ZSTD_decompress(
        self._decompression_buffer, expected_size, source, len(payload)))
      output = self._decompression_buffer.raw[:written]
    if len(output) != expected_size:
      raise ZstdError(f'decompressed size {len(output)} != {expected_size}')
    return output
