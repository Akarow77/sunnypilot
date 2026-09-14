#!/usr/bin/env python3
"""Authenticated Core ML/ANE policy worker for a comma 3X."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
from pathlib import Path
import pickle
import socket
import time

import coremltools as ct
import numpy as np

from accelerator_protocol import authenticate_payload, read_auth_key, server_handshake, verify_payload
from compression import ZstdCodec, ZstdError
from transport import (DEVICE_TYPE, FLAG_RESET, FLAG_ZSTD_REQUEST, POLICY_INPUTS, REQUEST, REQUEST_BYTES,
                       RESPONSE, TIMINGS, VERSION, WARPED_BYTES, WARPED_SHAPE,
                       ProtocolError, recv_message, send_message)


BACKEND = 'COREML_ANE'


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


class CoreMLPolicySession:
  def __init__(self, model: ct.models.MLModel, metadata: dict, output_name: str, frame_skip: int):
    self.model = model
    self.metadata = metadata
    self.output_name = output_name
    self.frame_skip = frame_skip
    shapes = metadata['input_shapes']
    image_frames = shapes['img'][1] // WARPED_SHAPE[1]
    feature_shape = shapes['features_buffer']
    desire_shape = shapes['desire_pulse']
    self.image_q = np.zeros((frame_skip * (image_frames - 1) + 1, *WARPED_SHAPE[1:]), dtype=np.uint8)
    self.big_image_q = np.zeros_like(self.image_q)
    self.feature_q = np.zeros((frame_skip * feature_shape[1], *feature_shape[2:]), dtype=np.float32)
    self.desire_q = np.zeros((frame_skip * desire_shape[1], desire_shape[2]), dtype=np.float32)
    self.hidden_slice = metadata['output_slices']['hidden_state']
    self.output_shape = tuple(metadata['output_shapes']['outputs'])
    self.inputs = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}

  def infer(self, payload: bytes) -> bytes:
    if len(payload) != REQUEST_BYTES:
      raise ProtocolError(f'request length {len(payload)} != {REQUEST_BYTES}')
    warped = np.frombuffer(payload, dtype=np.uint8, count=WARPED_BYTES).reshape(WARPED_SHAPE)
    *desire, traffic0, traffic1, action0, action1 = POLICY_INPUTS.unpack_from(payload, WARPED_BYTES)

    self.image_q[:-1] = self.image_q[1:]
    self.image_q[-1] = warped[0]
    self.big_image_q[:-1] = self.big_image_q[1:]
    self.big_image_q[-1] = warped[1]
    self.desire_q[:-1] = self.desire_q[1:]
    self.desire_q[-1] = desire

    np.copyto(self.inputs['img'], self.image_q[::self.frame_skip].reshape(self.inputs['img'].shape))
    np.copyto(self.inputs['big_img'], self.big_image_q[::self.frame_skip].reshape(self.inputs['big_img'].shape))
    np.copyto(self.inputs['desire_pulse'][0], self.desire_q.reshape(-1, self.frame_skip, len(desire)).max(1))
    self.inputs['traffic_convention'][0] = (traffic0, traffic1)
    self.inputs['action_t'][0] = (action0, action1)
    np.copyto(self.inputs['features_buffer'][0], self.feature_q[::self.frame_skip])
    output = np.asarray(self.model.predict(self.inputs)[self.output_name], dtype=np.float32)
    if output.shape != self.output_shape:
      raise ProtocolError(f'model output shape {output.shape} != {self.output_shape}')
    if not np.all(np.isfinite(output)):
      raise ProtocolError('model produced non-finite output')
    hidden = output[0, self.hidden_slice].reshape(self.feature_q[-1].shape)
    self.feature_q[:-1] = self.feature_q[1:]
    self.feature_q[-1] = hidden
    return output.astype('<f4', copy=False).tobytes()


def make_identity(metadata: dict, model_sha256: str) -> dict:
  return {
    'authentication_required': False,
    'backend': BACKEND,
    'compressions': [ZstdCodec.name],
    'device_type': DEVICE_TYPE,
    'input_shapes': {name: list(shape) for name, shape in metadata['input_shapes'].items()},
    'model_checkpoint': str(metadata.get('model_checkpoint', '')),
    'model_sha256': model_sha256,
    'output_floats': math.prod(metadata['output_shapes']['outputs']),
    'output_shapes': {name: list(shape) for name, shape in metadata['output_shapes'].items()},
    'protocol': VERSION,
    'request_bytes': REQUEST_BYTES,
  }


def serve_client(conn: socket.socket, peer, model: ct.models.MLModel, metadata: dict,
                 output_name: str, frame_skip: int, identity: dict,
                 auth_key: bytes | None, timeout: float) -> None:
  conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
  conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  conn.settimeout(timeout)
  active_session = server_handshake(conn, identity, auth_key)
  session = CoreMLPolicySession(model, metadata, output_name, frame_skip)
  codec = ZstdCodec()
  last_frame = -1
  print(f'verified client connected: {peer}', flush=True)
  while True:
    flags, session_id, frame_id, capture_ns, payload = recv_message(conn, REQUEST)
    received_ns = time.monotonic_ns()
    if session_id != active_session:
      raise ProtocolError('request session identity mismatch')
    payload = verify_payload(auth_key, REQUEST, flags, session_id, frame_id, capture_ns, payload)
    if flags & ~(FLAG_RESET | FLAG_ZSTD_REQUEST):
      raise ProtocolError(f'unsupported request flags: {flags:#x}')
    if flags & FLAG_ZSTD_REQUEST:
      try:
        payload = codec.decompress(payload, REQUEST_BYTES)
      except ZstdError as error:
        raise ProtocolError(f'Zstd request failed: {error}') from error
    if flags & FLAG_RESET and last_frame >= 0:
      session = CoreMLPolicySession(model, metadata, output_name, frame_skip)
      last_frame = -1
    if frame_id <= last_frame:
      raise ProtocolError(f'non-monotonic frame: {frame_id} <= {last_frame}')
    inference_start_ns = time.monotonic_ns()
    output = session.infer(payload)
    inference_end_ns = time.monotonic_ns()
    response_flags = flags & (FLAG_RESET | FLAG_ZSTD_REQUEST)
    response = TIMINGS.pack(received_ns, inference_start_ns, inference_end_ns) + output
    response = authenticate_payload(auth_key, RESPONSE, response_flags, session_id,
                                    frame_id, capture_ns, response)
    send_message(conn, RESPONSE, response_flags, session_id, frame_id, capture_ns, response)
    last_frame = frame_id


def main() -> None:
  parser = argparse.ArgumentParser(description='sunnypilot Core ML/ANE accelerator server')
  parser.add_argument('--model', type=Path, required=True)
  parser.add_argument('--metadata', type=Path, required=True)
  parser.add_argument('--onnx', type=Path, required=True, help='source ONNX used to bind model identity')
  parser.add_argument('--host', default='::1')
  parser.add_argument('--port', type=int, default=8066)
  parser.add_argument('--timeout', type=float, default=2.0)
  parser.add_argument('--auth-key-file', type=Path)
  parser.add_argument('--startup-warmup', type=int, default=3)
  parser.add_argument('--frame-skip', type=int, default=2)
  args = parser.parse_args()
  if args.startup_warmup < 1 or args.frame_skip < 1:
    parser.error('startup warmup and frame skip must be positive')
  for path in (args.model, args.metadata, args.onnx):
    if not path.exists():
      parser.error(f'input not found: {path}')

  with args.metadata.open('rb') as metadata_file:
    metadata = pickle.load(metadata_file)
  model_sha256 = sha256_file(args.onnx)
  identity = make_identity(metadata, model_sha256)
  auth_key = read_auth_key(args.auth_key_file)
  model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE)
  output_name = model.get_spec().description.output[0].name
  warmup = CoreMLPolicySession(model, metadata, output_name, args.frame_skip)
  for _ in range(args.startup_warmup):
    warmup.infer(bytes(REQUEST_BYTES))
  del warmup
  gc.collect()
  gc.disable()
  ready_message = f"ready: {DEVICE_TYPE} backend={BACKEND} checkpoint={identity['model_checkpoint']}"
  ready_message += f" outputs={identity['output_floats']} authenticated={auth_key is not None}"
  print(ready_message, flush=True)

  with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host, separator, interface = args.host.partition('%')
    scope_id = socket.if_nametoindex(interface) if separator else 0
    listener.bind((host, args.port, 0, scope_id))
    listener.listen(1)
    print(f'listening on [{args.host}]:{args.port}', flush=True)
    while True:
      conn, peer = listener.accept()
      with conn:
        try:
          serve_client(conn, peer, model, metadata, output_name, args.frame_skip,
                       identity, auth_key, args.timeout)
        except (EOFError, OSError, ProtocolError) as error:
          print(f'client disconnected: {error}', flush=True)


if __name__ == '__main__':
  main()
