#!/usr/bin/env python3
"""Single-client, shadow-only Metal inference worker."""

from __future__ import annotations

import argparse
import math
import socket
import time
from pathlib import Path

import numpy as np

from openpilot.selfdrive.modeld.helpers import load_oob

from accelerator_protocol import authenticate_payload, read_auth_key, server_handshake, verify_payload
from compile_policy import REMOTE_INPUTS, make_remote_input_queues
from transport import (DEVICE_TYPE, FLAG_RESET, POLICY_INPUTS, REQUEST, REQUEST_BYTES,
                       RESPONSE, TIMINGS, VERSION, WARPED_BYTES, WARPED_SHAPE,
                       ProtocolError, recv_message, send_message)


class PolicySession:
  def __init__(self, artifact: dict):
    self.artifact = artifact
    self.queues, self.npy, self.frames = make_remote_input_queues(
      artifact['metadata'], artifact['input_devices']['model'])
    self.output_slices = artifact['metadata']['output_slices']
    self.output_floats = math.prod(next(iter(artifact['metadata']['output_shapes'].values())))

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
    if result.size != self.output_floats:
      raise ProtocolError(f"model output length {result.size} != {self.output_floats}")
    if not np.all(np.isfinite(result)):
      raise ProtocolError("model produced non-finite output")
    self.npy['prev_feat'][:] = result[self.output_slices['hidden_state']]
    return result.astype('<f4', copy=False).tobytes()


def make_identity(artifact: dict) -> dict:
  metadata = artifact['metadata']
  output_floats = math.prod(next(iter(metadata['output_shapes'].values())))
  return {
    'authentication_required': False,
    'backend': str(artifact['input_devices']['model']),
    'compressions': [],
    'device_type': DEVICE_TYPE,
    'input_shapes': {name: list(shape) for name, shape in metadata['input_shapes'].items()},
    'model_checkpoint': str(metadata.get('model_checkpoint', '')),
    'model_sha256': str(artifact.get('accelerator', {}).get('model_sha256', '')),
    'output_floats': output_floats,
    'output_shapes': {name: list(shape) for name, shape in metadata['output_shapes'].items()},
    'output_slices': {name: [output_slice.start or 0, output_slice.stop]
                      for name, output_slice in metadata['output_slices'].items()},
    'protocol': VERSION,
    'request_bytes': REQUEST_BYTES,
  }


def serve_client(conn: socket.socket, peer, artifact: dict, identity: dict,
                 auth_key: bytes | None, timeout: float) -> None:
  conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
  conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  conn.settimeout(timeout)
  active_session = server_handshake(conn, identity, auth_key)
  session: PolicySession | None = PolicySession(artifact)
  last_frame = -1
  print(f"verified client connected: {peer}", flush=True)
  while True:
    conn.settimeout(timeout)
    flags, session_id, frame_id, capture_ns, payload = recv_message(conn, REQUEST)
    received_ns = time.monotonic_ns()
    if session_id != active_session:
      raise ProtocolError('request session identity mismatch')
    if flags & ~FLAG_RESET:
      raise ProtocolError(f'unsupported request flags: {flags:#x}')
    payload = verify_payload(auth_key, REQUEST, flags, session_id, frame_id, capture_ns, payload)
    if session is None or (flags & FLAG_RESET and last_frame >= 0):
      session = PolicySession(artifact)
      last_frame = -1
    if frame_id <= last_frame:
      raise ProtocolError(f"non-monotonic frame: {frame_id} <= {last_frame}")
    inference_start_ns = time.monotonic_ns()
    output = session.infer(payload)
    inference_end_ns = time.monotonic_ns()
    response = TIMINGS.pack(received_ns, inference_start_ns, inference_end_ns) + output
    response_flags = flags & FLAG_RESET
    response = authenticate_payload(auth_key, RESPONSE, response_flags, session_id,
                                    frame_id, capture_ns, response)
    send_message(conn, RESPONSE, response_flags, session_id, frame_id, capture_ns, response)
    last_frame = frame_id


def main() -> None:
  default_artifact = Path(__file__).resolve().parent / 'artifacts' / 'driving_policy_metal.pkl'
  parser = argparse.ArgumentParser(description="Shadow-only sunnypilot Metal inference server")
  parser.add_argument('--artifact', type=Path, default=default_artifact)
  parser.add_argument('--host', default='::1', help='bind address; use the Mac USB link-local address for a 3X test')
  parser.add_argument('--port', type=int, default=8066)
  parser.add_argument('--timeout', type=float, default=2.0)
  parser.add_argument('--auth-key-file', type=Path,
                      help='shared key; mandatory for any future active-control configuration')
  parser.add_argument('--startup-warmup', type=int, default=3)
  args = parser.parse_args()
  if not args.artifact.is_file():
    parser.error(f"artifact not found: {args.artifact}; run compile_policy.sh first")
  if args.startup_warmup < 1:
    parser.error('--startup-warmup must be at least 1')

  with args.artifact.open('rb') as artifact_file:
    artifact = load_oob(artifact_file)
  identity = make_identity(artifact)
  auth_key = read_auth_key(args.auth_key_file)
  if identity['backend'] != 'METAL':
    parser.error(f"artifact backend must be METAL, got {identity['backend']}")
  warmup = PolicySession(artifact)
  for _ in range(args.startup_warmup):
    warmup.infer(bytes(REQUEST_BYTES))
  del warmup
  ready_message = f"ready: {identity['device_type']} checkpoint={identity['model_checkpoint']}"
  ready_message += f" outputs={identity['output_floats']} authenticated={auth_key is not None}"
  print(ready_message, flush=True)
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
          serve_client(conn, peer, artifact, identity, auth_key, args.timeout)
        except (EOFError, OSError, ProtocolError) as error:
          print(f"client disconnected: {error}", flush=True)


if __name__ == '__main__':
  main()
