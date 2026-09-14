#!/usr/bin/env python3
"""Map Mac worker health onto the existing Chestnut-style UI states."""

from __future__ import annotations


def mac_accelerator_state(*, present: bool, loading: bool, ready: bool,
                          active: bool, failed: bool) -> str | None:
  if not present:
    return None
  if failed:
    return 'failed'
  if loading:
    return 'loading'
  if active:
    return 'active'
  if ready:
    return 'ready'
  return 'loading'
