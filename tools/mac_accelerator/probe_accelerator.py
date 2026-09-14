#!/usr/bin/env python3
"""Recognize a Mac worker as a validated sunnypilot accelerator from a comma 3X."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from accelerator_client import AcceleratorClient
from accelerator_protocol import read_auth_key


def main() -> None:
  parser = argparse.ArgumentParser(description='Probe a sunnypilot Mac accelerator')
  parser.add_argument('host')
  parser.add_argument('--port', type=int, default=8066)
  parser.add_argument('--deadline-ms', type=float, default=45.0)
  parser.add_argument('--expected-checkpoint')
  parser.add_argument('--expected-model-sha256')
  parser.add_argument('--expected-output-floats', type=int)
  parser.add_argument('--expected-backend', default='METAL')
  parser.add_argument('--auth-key-file', type=Path)
  args = parser.parse_args()

  client = AcceleratorClient(
    args.host, args.port, deadline_ms=args.deadline_ms,
    auth_key=read_auth_key(args.auth_key_file),
    expected_checkpoint=args.expected_checkpoint,
    expected_model_sha256=args.expected_model_sha256,
    expected_output_floats=args.expected_output_floats,
    expected_backend=args.expected_backend,
  )
  try:
    identity = client.connect()
    print(json.dumps({'present': True, 'ready': True, 'identity': asdict(identity)}, indent=2, sort_keys=True))
  except Exception as error:
    print(json.dumps({
      'present': False,
      'ready': False,
      'error': f'{type(error).__name__}: {error}',
    }, indent=2, sort_keys=True))
    raise SystemExit(1) from error
  finally:
    client.close()


if __name__ == '__main__':
  main()
