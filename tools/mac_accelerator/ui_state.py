#!/usr/bin/env python3
"""Optional comma parameter publisher for Chestnut-style Mac worker UI state."""

from __future__ import annotations

import sys
import time


PARAM_KEYS = (
  'MacAcceleratorPresent',
  'MacAcceleratorLoading',
  'MacAcceleratorReady',
  'MacAcceleratorActive',
  'MacAcceleratorModelError',
)


class NullUIState:
  def heartbeat(self) -> None: pass
  def loading(self) -> None: pass
  def ready(self) -> None: pass
  def active(self) -> None: pass
  def failed(self) -> None: pass
  def disconnected(self) -> None: pass


class MacAcceleratorUIState:
  def __init__(self):
    from openpilot.common.params import Params
    self.params = Params()
    self.last_heartbeat = 0.0
    for key in PARAM_KEYS:
      self.params.check_key(key)
    self.params.check_key('MacAcceleratorHeartbeat')

  def heartbeat(self) -> None:
    now = time.monotonic()
    if now - self.last_heartbeat >= 0.25:
      self.params.put('MacAcceleratorHeartbeat', now)
      self.last_heartbeat = now

  def _set(self, *, present: bool, loading: bool, ready: bool,
           active: bool, failed: bool) -> None:
    values = dict(zip(PARAM_KEYS, (present, loading, ready, active, failed), strict=True))
    for key, value in values.items():
      self.params.put_bool(key, value)
    self.heartbeat()

  def loading(self) -> None:
    self._set(present=True, loading=True, ready=False, active=False, failed=False)

  def ready(self) -> None:
    self._set(present=True, loading=False, ready=True, active=False, failed=False)

  def active(self) -> None:
    self._set(present=True, loading=False, ready=True, active=True, failed=False)

  def failed(self) -> None:
    self._set(present=True, loading=False, ready=False, active=False, failed=True)

  def disconnected(self) -> None:
    self._set(present=False, loading=False, ready=False, active=False, failed=False)


def make_ui_state(enabled: bool):
  if not enabled:
    return NullUIState()
  try:
    return MacAcceleratorUIState()
  except Exception as error:
    print(f'UI state publishing unavailable on this checkout: {error}', file=sys.stderr)
    return NullUIState()
