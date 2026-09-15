#!/usr/bin/env python3
"""Drop inherited modeld realtime scheduling before importing the shadow worker."""

import os
import sys

SHADOW_CORE = 2
SHADOW_PRIORITY = 5
SCHED_FIFO = getattr(os, 'SCHED_FIFO', 1)
SCHED_OTHER = getattr(os, 'SCHED_OTHER', 0)


def configure_shadow_scheduling() -> str:
  """Keep transport responsive without competing with modeld or controls."""
  if sys.platform != 'linux':
    return 'default'
  target_core = SHADOW_CORE if (os.cpu_count() or 0) > SHADOW_CORE else 0
  try:
    # modeld is FIFO 54 on core 7. The worker blocks on its mailbox/socket most
    # of the time, so FIFO 5 on an otherwise lightly loaded little core gives it
    # prompt wakeups while control/planner/modeld retain strict precedence.
    os.sched_setscheduler(0, SCHED_FIFO, os.sched_param(SHADOW_PRIORITY))
    os.sched_setaffinity(0, [target_core])
    return f'fifo-{SHADOW_PRIORITY}-core-{target_core}'
  except OSError:
    cores = list(range(min(4, os.cpu_count() or 1)))
    try:
      os.sched_setscheduler(0, SCHED_OTHER, os.sched_param(0))
    except OSError:
      pass
    try:
      os.sched_setaffinity(0, cores)
    except OSError:
      pass
    return 'timeshare-little-cores'


def main():
  configure_shadow_scheduling()
  from live_shadow import main as worker_main
  worker_main()


if __name__ == '__main__':
  main()
