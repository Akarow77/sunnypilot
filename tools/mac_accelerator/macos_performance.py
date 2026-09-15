#!/usr/bin/env python3
"""Best-effort macOS scheduling setup for the Core ML request thread."""

from __future__ import annotations

import ctypes
import sys


QOS_CLASS_USER_INTERACTIVE = 0x21


def configure_user_interactive_qos() -> bool:
  if sys.platform != 'darwin':
    return False
  library = ctypes.CDLL(None, use_errno=True)
  function = library.pthread_set_qos_class_self_np
  function.argtypes = [ctypes.c_uint, ctypes.c_int]
  function.restype = ctypes.c_int
  result = function(QOS_CLASS_USER_INTERACTIVE, 0)
  if result:
    raise OSError(result, 'pthread_set_qos_class_self_np failed')
  current = library.qos_class_self
  current.argtypes = []
  current.restype = ctypes.c_uint
  if current() != QOS_CLASS_USER_INTERACTIVE:
    raise RuntimeError('macOS did not retain user-interactive QoS')
  return True
