"""Bounded one-way frame mailbox. No worker output enters the local model path."""

from __future__ import annotations

import fcntl
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

from transport import WARPED_BYTES, WARPED_SHAPE

HEADER = struct.Struct('<QQII')
META_SIZE = 4096
SIZE = HEADER.size + META_SIZE + WARPED_BYTES
ERROR_PIXEL_SIZE = 0


class FrameMailbox:
  def __init__(self, path: Path, *, create: bool = False):
    flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
    self.fd = os.open(path, flags, 0o600)
    if create:
      os.ftruncate(self.fd, SIZE)
    if os.fstat(self.fd).st_size != SIZE:
      os.close(self.fd)
      raise ValueError('invalid shadow mailbox size')
    self.memory = mmap.mmap(self.fd, SIZE)
    self.sequence = 0

  def publish(self, pixels, metadata: dict) -> bool:
    # Each process opens independently: duplicated flock FDs share ownership.
    try:
      fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      return False
    try:
      view = memoryview(pixels).cast('B')
      if len(view) != WARPED_BYTES:
        raise ValueError('invalid shadow pixel count')
      meta = json.dumps(metadata, allow_nan=False).encode()
      if len(meta) > META_SIZE:
        raise ValueError('shadow metadata too large')
      # Finish fallible validation before touching the last committed packet.
      self.memory[HEADER.size + META_SIZE:] = view
      self.memory[HEADER.size:HEADER.size + len(meta)] = meta
      self.sequence += 1
      self.memory[:HEADER.size] = HEADER.pack(self.sequence, time.monotonic_ns(), len(meta), WARPED_BYTES)
      return True
    finally:
      fcntl.flock(self.fd, fcntl.LOCK_UN)

  def publish_error(self, reason: str) -> bool:
    """Best-effort producer failure handoff without Params or filesystem I/O."""
    try:
      fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      return False
    try:
      meta = json.dumps({'producer_error': str(reason)[:1024]}, allow_nan=False).encode()
      if len(meta) > META_SIZE:
        return False
      self.memory[HEADER.size:HEADER.size + len(meta)] = meta
      self.sequence += 1
      self.memory[:HEADER.size] = HEADER.pack(
        self.sequence, time.monotonic_ns(), len(meta), ERROR_PIXEL_SIZE)
      return True
    finally:
      fcntl.flock(self.fd, fcntl.LOCK_UN)

  def receive(self, after: int) -> tuple[int, dict, bytes] | None:
    try:
      fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      return None
    try:
      sequence, queued_ns, meta_size, pixel_size = HEADER.unpack_from(self.memory)
      if sequence <= after:
        return None
      if not 0 < meta_size <= META_SIZE or pixel_size not in (ERROR_PIXEL_SIZE, WARPED_BYTES):
        raise ValueError('invalid shadow mailbox header')
      metadata = json.loads(self.memory[HEADER.size:HEADER.size + meta_size])
      if pixel_size == ERROR_PIXEL_SIZE:
        raise RuntimeError(f"producer failure: {metadata.get('producer_error', 'unknown')}")
      metadata['queued_ns'] = queued_ns
      pixels = self.memory[HEADER.size + META_SIZE:]
      return sequence, metadata, pixels
    finally:
      fcntl.flock(self.fd, fcntl.LOCK_UN)

  def close(self) -> None:
    self.memory.close()
    os.close(self.fd)


class ShadowPublisher:
  """Post-publication snapshot plus independent network/logging process.

  Host readback is synchronous and needs measurement on a real 3X. A slow
  readback disables future snapshots; it cannot undo that first delay.
  """
  def __init__(self, config: dict, *, copy_budget_ms: float = 2.0):
    self.config = config
    self.copy_budget_ms = copy_budget_ms
    self.temp_dir = tempfile.TemporaryDirectory(prefix='mac-shadow-')
    self.path = Path(self.temp_dir.name) / 'frames'
    self.mailbox = FrameMailbox(self.path, create=True)
    self.process: subprocess.Popen | None = None
    self.failed_reason = None
    self.busy_drops = 0
    self.closed = False

  def start(self) -> None:
    script = Path(__file__).with_name('shadow_process.py')
    self.process = subprocess.Popen(
      [sys.executable, str(script), '--mailbox', str(self.path), '--parent-pid', str(os.getpid()),
       '--config', json.dumps(self.config)],
      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
      env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
    )

  def fail(self, reason: str) -> None:
    if self.failed_reason is not None:
      return
    self.failed_reason = reason
    self.mailbox.publish_error(reason)
    # No logging, parameter writes or network I/O in the modeld path.

  def capture(self, tensor, **metadata) -> bool:
    if self.failed_reason or self.process is None:
      return False
    if self.process.poll() is not None:
      self.fail('shadow worker exited')
      return False
    try:
      import numpy as np
      started_ns = time.monotonic_ns()
      pixels = tensor.numpy()
      if pixels.shape != WARPED_SHAPE or pixels.dtype != np.uint8 or not pixels.flags.c_contiguous:
        raise ValueError('unsupported live warp shape, dtype or layout')
      readback_ms = (time.monotonic_ns() - started_ns) / 1e6
      if readback_ms > self.copy_budget_ms:
        self.fail(f'host readback exceeded copy budget: {readback_ms:.3f} ms')
        return False
      accepted = self.mailbox.publish(pixels, {**metadata, 'copy_started_ns': started_ns,
                                                'readback_ms': readback_ms, 'producer_busy_drops': self.busy_drops})
      if not accepted:
        self.busy_drops += 1
      if (time.monotonic_ns() - started_ns) / 1e6 > self.copy_budget_ms:
        self.fail('shadow snapshot exceeded copy budget')
      return accepted
    except Exception as error:
      self.fail(f'{type(error).__name__}: {error}')
      return False

  def close(self) -> None:
    if self.closed:
      return
    self.closed = True
    if self.process is not None and self.process.poll() is None:
      self.process.terminate()
      try:
        self.process.wait(timeout=2)
      except subprocess.TimeoutExpired:
        self.process.kill()
        self.process.wait(timeout=2)
    self.mailbox.close()
    self.temp_dir.cleanup()
