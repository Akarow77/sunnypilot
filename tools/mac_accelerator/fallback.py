#!/usr/bin/env python3
"""Chestnut-style, ignition-cycle-latched fallback state for remote inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math


class AcceleratorState(StrEnum):
  LOADING = 'loading'
  ACTIVE = 'active'
  FAILED = 'failed'


@dataclass
class FallbackLatch:
  deadline_ms: float
  state: AcceleratorState = AcceleratorState.LOADING
  failure_frame: int | None = None
  failure_reason: str | None = None

  def activate(self) -> None:
    if self.state is not AcceleratorState.FAILED:
      self.state = AcceleratorState.ACTIVE

  def observe(self, frame_id: int, latency_ms: float, valid: bool = True) -> None:
    if self.state is not AcceleratorState.ACTIVE:
      return
    if not valid:
      self.fail(frame_id, 'invalid output')
    elif not math.isfinite(latency_ms):
      self.fail(frame_id, 'non-finite latency')
    elif latency_ms > self.deadline_ms:
      self.fail(frame_id, f'deadline missed: {latency_ms:.2f} ms > {self.deadline_ms:.2f} ms')

  def fail(self, frame_id: int, reason: str) -> None:
    if self.state is AcceleratorState.FAILED:
      return
    self.state = AcceleratorState.FAILED
    self.failure_frame = frame_id
    self.failure_reason = reason

  def reset(self) -> None:
    self.state = AcceleratorState.LOADING
    self.failure_frame = None
    self.failure_reason = None
