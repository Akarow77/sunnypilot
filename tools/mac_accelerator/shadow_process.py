#!/usr/bin/env python3
"""Drop inherited modeld realtime scheduling before importing the shadow worker."""

import os
import sys


def main():
  if sys.platform == 'linux':
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    if sorted(os.sched_getaffinity(0)) == [7] and os.cpu_count() == 8:
      os.sched_setaffinity(0, [0, 1, 2, 3])
    os.nice(5)
  from live_shadow import main as worker_main
  worker_main()


if __name__ == '__main__':
  main()
