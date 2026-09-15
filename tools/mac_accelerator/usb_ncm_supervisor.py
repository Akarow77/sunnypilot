#!/usr/bin/env python3
"""Restart the Mac worker when its scoped USB-NCM endpoint is re-enumerated."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import signal
import socket
import subprocess
import threading


def parse_scoped_endpoint(endpoint: str) -> tuple[str, str]:
  host, separator, interface = endpoint.rpartition('%')
  if not separator or not host or not interface:
    raise ValueError('USB endpoint must be a scoped IPv6 address')
  address = ipaddress.IPv6Address(host)
  if not address.is_link_local:
    raise ValueError('USB endpoint must be IPv6 link-local')
  return str(address), interface


def parse_link_info(output: str) -> tuple[str, str]:
  values = {}
  for line in output.splitlines():
    name, separator, value = line.partition(':')
    if separator and name in ('Mac', 'comma'):
      values[name] = value.strip()
  if set(values) != {'Mac', 'comma'}:
    raise ValueError('USB discovery did not return both endpoints')
  parse_scoped_endpoint(values['Mac'])
  parse_scoped_endpoint(values['comma'])
  return values['Mac'], values['comma']


def endpoint_is_current(endpoint: str, expected_index: int) -> bool:
  host, interface = parse_scoped_endpoint(endpoint)
  try:
    if socket.if_nametoindex(interface) != expected_index:
      return False
    state = subprocess.run(['/sbin/ifconfig', interface], capture_output=True, text=True,
                           timeout=1, check=False)
  except (OSError, subprocess.SubprocessError):
    return False
  return (state.returncode == 0 and 'status: active' in state.stdout
          and f'inet6 {host}%{interface}' in state.stdout)


class Supervisor:
  def __init__(self, setup: Path, server_command: list[str], retry_seconds: float, poll_seconds: float):
    self.setup = setup
    self.server_command = server_command
    self.retry_seconds = retry_seconds
    self.poll_seconds = poll_seconds
    self.stop_event = threading.Event()
    self.server: subprocess.Popen | None = None

  def stop(self, _signum=None, _frame=None) -> None:
    self.stop_event.set()
    if self.server is not None and self.server.poll() is None:
      self.server.terminate()

  def wait_or_stop(self, seconds: float) -> bool:
    return self.stop_event.wait(seconds)

  def discover(self) -> tuple[str, int] | None:
    result = subprocess.run([str(self.setup)], capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0:
      detail = result.stderr.strip().splitlines()
      suffix = f': {detail[-1]}' if detail else ''
      print(f'waiting for USB NCM{suffix}', flush=True)
      return None
    print(result.stdout.rstrip(), flush=True)
    endpoint, _comma = parse_link_info(result.stdout)
    _host, interface = parse_scoped_endpoint(endpoint)
    return endpoint, socket.if_nametoindex(interface)

  def run(self) -> int:
    while not self.stop_event.is_set():
      try:
        discovered = self.discover()
      except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(f'waiting for USB NCM: {error}', flush=True)
        discovered = None
      if discovered is None:
        self.wait_or_stop(self.retry_seconds)
        continue

      endpoint, interface_index = discovered
      self.server = subprocess.Popen([*self.server_command, '--host', endpoint])
      restart_for_link = False
      while self.server.poll() is None and not self.stop_event.is_set():
        if self.wait_or_stop(self.poll_seconds):
          break
        if not endpoint_is_current(endpoint, interface_index):
          print('USB link changed; restarting scoped listener', flush=True)
          restart_for_link = True
          self.server.terminate()
          break

      if self.server.poll() is None:
        try:
          self.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
          self.server.kill()
          self.server.wait(timeout=2)
      return_code = self.server.returncode
      self.server = None
      if self.stop_event.is_set():
        return 0
      if restart_for_link:
        self.wait_or_stop(self.retry_seconds)
        continue
      return return_code if return_code is not None else 1
    return 0


def main() -> None:
  parser = argparse.ArgumentParser(description='supervise a scoped USB-NCM Core ML server')
  parser.add_argument('--setup', type=Path, required=True)
  parser.add_argument('--retry-seconds', type=float, default=1.0)
  parser.add_argument('--poll-seconds', type=float, default=0.5)
  parser.add_argument('server_command', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  command = args.server_command[1:] if args.server_command[:1] == ['--'] else args.server_command
  if (not args.setup.is_file() or not command or args.retry_seconds <= 0
      or args.poll_seconds <= 0):
    parser.error('valid setup, server command, and positive timing values are required')
  supervisor = Supervisor(args.setup, command, args.retry_seconds, args.poll_seconds)
  signal.signal(signal.SIGTERM, supervisor.stop)
  signal.signal(signal.SIGINT, supervisor.stop)
  raise SystemExit(supervisor.run())


if __name__ == '__main__':
  main()
