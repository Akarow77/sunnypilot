#!/usr/bin/env python3
"""Drop inherited modeld realtime scheduling before importing the shadow worker."""

import os
import sys

SCHED_OTHER = getattr(os, 'SCHED_OTHER', 0)


def configure_shadow_scheduling() -> str:
  """Require normal timesharing before importing any benchmark worker code."""
  if sys.platform != 'linux':
    return 'default'
  # A child may inherit FIFO scheduling from its launcher. Never continue if
  # demotion fails: CPU affinity alone cannot prevent realtime starvation.
  os.sched_setscheduler(0, SCHED_OTHER, os.sched_param(0))
  cores = list(range(min(4, os.cpu_count() or 1)))
  try:
    os.sched_setaffinity(0, cores)
  except OSError:
    return 'timeshare-default-affinity'
  return 'timeshare-little-cores'


def main():
  configure_shadow_scheduling()
  from live_shadow import main as worker_main
  worker_main()


if __name__ == '__main__':
  main()
