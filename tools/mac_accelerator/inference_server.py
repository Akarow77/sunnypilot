#!/usr/bin/env python3
"""Single-client, shadow-only Metal inference worker."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np

from openpilot.selfdrive.modeld.helpers import load_oob

from compile_policy import REMOTE_INPUTS, make_remote_input_queues
from transport import (FLAG_RESET, OUTPUT_FLOATS, POLICY_INPUTS, REQUEST, REQUEST_BYTES,
                       RESPONSE, TIMINGS, WARPED_BYTES, WARPED_SHAPE, ProtocolError,
                       recv_message, send_message)


class PolicySession:
  def __init__(self, artifact: dict):
    self.artifact = artifact
    self.queues, self.npy, self.frames = make_remote_input_queues(
      artifact['metadata'], artifact['input_devices']['model'])
    self.output_slices = artifact['metadata']['output_slices']

  def infer(self, payload: bytes) -> bytes:
    if len(payload) != REQUEST_BYTES:
      raise ProtocolError(f"request length {len(payload)} != {REQUEST_BYTES}")
    warped = np.frombuffer(payload, dtype=np.uint8, count=WARPED_BYTES).reshape(WARPED_SHAPE)
    np.copyto(self.frames['warped'], warped)
    *desire, traffic0, traffic1, action0, action1 = POLICY_INPUTS.unpack_from(payload, WARPED_BYTES)
    self.npy['desire'][:] = desire
    self.npy['traffic_convention'][:] = (traffic0, traffic1)
    self.npy['action_t'][:] = (action0, action1)

    output, = self.artifact['run_policy'](**{name: self.queues[name] for name in REMOTE_INPUTS})
    result = output.numpy()[0]
    if result.size != OUTPUT_FLOATS:
      raise ProtocolError(f"model output length {result.size} != {OUTPUT_FLOATS}")
    if not np.all(np.isfinite(result)):
      raise ProtocolError("model produced non-finite output")
    self.npy['prev_feat'][:] = result[self.output_slices['hidden_state']]
    return result.astype('<f4', copy=False).tobytes()


def serve_client(conn: socket.socket, peer, artifact: dict, timeout: float) -> None:
  conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
  conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  conn.settimeout(timeout)
  session: PolicySession | None = None
  active_session = -1
  last_frame = -1
  print(f"client connected: {peer}", flush=True)
  while True:
    flags, session_id, frame_id, capture_ns, payload = recv_message(conn, REQUEST)
    received_ns = time.monotonic_ns()
    if session is None or session_id != active_session or flags & FLAG_RESET:
      session = PolicySession(artifact)
      active_session = session_id
      last_frame = -1
    if frame_id <= last_frame:
      raise ProtocolError(f"non-monotonic frame: {frame_id} <= {last_frame}")
    inference_start_ns = time.monotonic_ns()
    output = session.infer(payload)
    inference_end_ns = time.monotonic_ns()
    response = TIMINGS.pack(received_ns, inference_start_ns, inference_end_ns) + output
    send_message(conn, RESPONSE, flags & FLAG_RESET, session_id, frame_id, capture_ns, response)
    last_frame = frame_id


def main() -> None:
  default_artifact = Path(__file__).resolve().parent / 'artifacts' / 'driving_policy_metal.pkl'
  parser = argparse.ArgumentParser(description="Shadow-only sunnypilot Metal inference server")
  parser.add_argument('--artifact', type=Path, default=default_artifact)
  parser.add_argument('--host', default='::1', help='bind address; use the Mac USB link-local address for a 3X test')
  parser.add_argument('--port', type=int, default=8066)
  parser.add_argument('--timeout', type=float, default=2.0)
  args = parser.parse_args()
  if not args.artifact.is_file():
    parser.error(f"artifact not found: {args.artifact}; run compile_policy.sh first")

  with args.artifact.open('rb') as artifact_file:
    artifact = load_oob(artifact_file)
  with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host, separator, interface = args.host.partition('%')
    scope_id = socket.if_nametoindex(interface) if separator else 0
    listener.bind((host, args.port, 0, scope_id))
    listener.listen(1)
    print(f"listening on [{args.host}]:{args.port}", flush=True)
    while True:
      conn, peer = listener.accept()
      with conn:
        try:
          serve_client(conn, peer, artifact, args.timeout)
        except (EOFError, OSError, ProtocolError) as error:
          print(f"client disconnected: {error}", flush=True)


if __name__ == '__main__':
  main()
