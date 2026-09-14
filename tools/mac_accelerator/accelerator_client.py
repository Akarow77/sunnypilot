#!/usr/bin/env python3
"""Strict comma-side client for recognizing and using the Mac accelerator."""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
import math
import secrets
import socket
import time

from accelerator_protocol import authenticate_payload, canonical_json, make_auth, verify_payload
from compression import ZstdCodec
from fallback import AcceleratorState, FallbackLatch
from transport import (DEVICE_TYPE, FLAG_RESET, FLAG_ZSTD_REQUEST, HELLO, HELLO_RESPONSE, POLICY_INPUTS,
                       REQUEST, RESPONSE, TIMINGS, VERSION, WARPED_BYTES, ProtocolError,
                       recv_message, send_message)

@dataclass(frozen=True)
class AcceleratorIdentity:
  device_type: str
  backend: str
  model_checkpoint: str
  model_sha256: str
  input_shapes: dict[str, list[int]]
  output_shapes: dict[str, list[int]]
  output_floats: int
  request_bytes: int
  compressions: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceResult:
  frame_id: int
  output: bytes
  round_trip_ms: float
  inference_ms: float


class AcceleratorClient:
  def __init__(self, host: str, port: int = 8066, *, deadline_ms: float = 45.0,
               auth_key: bytes | None = None, expected_checkpoint: str | None = None,
               expected_model_sha256: str | None = None, expected_output_floats: int | None = None,
               expected_backend: str = 'METAL', qualification_frames: int = 1,
               qualification_timeout_ms: float = 100.0, compression: str | None = None):
    if deadline_ms <= 0:
      raise ValueError('deadline must be positive')
    if qualification_frames < 1 or qualification_timeout_ms < deadline_ms:
      raise ValueError('qualification frames must be positive and timeout must cover the active deadline')
    self.host = host
    self.port = port
    self.deadline_ms = deadline_ms
    self.auth_key = auth_key
    self.expected_checkpoint = expected_checkpoint
    self.expected_model_sha256 = expected_model_sha256
    self.expected_output_floats = expected_output_floats
    self.expected_backend = expected_backend
    self.qualification_frames = qualification_frames
    self.qualification_timeout_ms = qualification_timeout_ms
    self.qualification_count = 0
    if compression not in (None, ZstdCodec.name):
      raise ValueError(f'unsupported compression: {compression}')
    self.compression = compression
    self.codec = ZstdCodec() if compression == ZstdCodec.name else None
    self.session_id = secrets.randbits(64)
    self.socket: socket.socket | None = None
    self.identity: AcceleratorIdentity | None = None
    self.fallback = FallbackLatch(deadline_ms)
    self.next_frame = 0

  def connect(self, timeout: float = 5.0) -> AcceleratorIdentity:
    self.close()
    conn = socket.create_connection((self.host, self.port), timeout=timeout)
    try:
      conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
      conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
      conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      conn.settimeout(timeout)
      nonce = secrets.token_hex(32)
      hello_body: dict[str, object] = {
        'client': 'comma3x',
        'nonce': nonce,
        'protocol': VERSION,
      }
      if self.auth_key is not None:
        hello_body['auth'] = make_auth(self.auth_key, nonce, {
          'client': hello_body['client'],
          'protocol': hello_body['protocol'],
        })
      send_message(conn, HELLO, 0, self.session_id, 0, time.monotonic_ns(), canonical_json(hello_body))
      _, session_id, _, _, payload = recv_message(conn, HELLO_RESPONSE)
      if session_id != self.session_id:
        raise ProtocolError('hello session identity mismatch')
      response = json.loads(payload)
      response_auth = response.pop('auth', None)
      if self.auth_key is not None:
        expected_auth = make_auth(self.auth_key, nonce, response)
        if not isinstance(response_auth, str) or not hmac.compare_digest(response_auth, expected_auth):
          raise ProtocolError('accelerator authentication failed')
      elif response.get('authentication_required'):
        raise ProtocolError('accelerator requires an authentication key')

      identity = AcceleratorIdentity(
        device_type=str(response['device_type']),
        backend=str(response['backend']),
        model_checkpoint=str(response['model_checkpoint']),
        model_sha256=str(response.get('model_sha256', '')),
        input_shapes={k: list(v) for k, v in response['input_shapes'].items()},
        output_shapes={k: list(v) for k, v in response['output_shapes'].items()},
        output_floats=int(response['output_floats']),
        request_bytes=int(response['request_bytes']),
        compressions=tuple(response.get('compressions', ())),
      )
      self._validate_identity(identity)
      conn.settimeout(self.deadline_ms / 1000.0)
      self.socket = conn
      self.identity = identity
      self.next_frame = 0
      self.qualification_count = 0
      return identity
    except Exception:
      conn.close()
      raise

  def _validate_identity(self, identity: AcceleratorIdentity) -> None:
    if identity.device_type != DEVICE_TYPE:
      raise ProtocolError(f'unsupported accelerator type: {identity.device_type}')
    if identity.backend != self.expected_backend:
      raise ProtocolError(f'accelerator backend mismatch: {identity.backend} != {self.expected_backend}')
    if identity.request_bytes != WARPED_BYTES + POLICY_INPUTS.size:
      raise ProtocolError(f'incompatible request size: {identity.request_bytes}')
    if len(identity.model_sha256) != 64 or any(c not in '0123456789abcdef' for c in identity.model_sha256):
      raise ProtocolError('invalid model SHA-256 identity')
    if self.expected_checkpoint is not None and identity.model_checkpoint != self.expected_checkpoint:
      raise ProtocolError('model checkpoint mismatch')
    if self.expected_model_sha256 is not None and identity.model_sha256 != self.expected_model_sha256:
      raise ProtocolError('model SHA-256 mismatch')
    if self.expected_output_floats is not None and identity.output_floats != self.expected_output_floats:
      raise ProtocolError('model output length mismatch')
    if self.compression is not None and self.compression not in identity.compressions:
      raise ProtocolError(f'accelerator does not support {self.compression}')

  def infer(self, warped: bytes, policy: bytes, *, frame_id: int, capture_ns: int,
            reset: bool = False) -> InferenceResult:
    if self.socket is None or self.identity is None:
      raise RuntimeError('accelerator is not connected')
    if self.fallback.state is AcceleratorState.FAILED:
      raise RuntimeError(f'accelerator failure is latched: {self.fallback.failure_reason}')
    if frame_id != self.next_frame:
      self._fail(frame_id, f'non-sequential client frame: {frame_id} != {self.next_frame}')
    if len(warped) != WARPED_BYTES or len(policy) != POLICY_INPUTS.size:
      self._fail(frame_id, 'invalid local request shape')

    started_ns = time.monotonic_ns()
    flags = FLAG_RESET if reset else 0
    request_payload = warped + policy
    if self.codec is not None:
      request_payload = self.codec.compress(request_payload)
      flags |= FLAG_ZSTD_REQUEST
    request_deadline_ms = (self.qualification_timeout_ms if self.fallback.state is AcceleratorState.LOADING
                           else self.deadline_ms)
    deadline_ns = started_ns + int(request_deadline_ms * 1e6)

    def set_remaining_timeout() -> None:
      remaining = (deadline_ns - time.monotonic_ns()) / 1e9
      if remaining <= 0:
        raise TimeoutError('accelerator deadline expired')
      assert self.socket is not None
      self.socket.settimeout(remaining)

    try:
      set_remaining_timeout()
      request = authenticate_payload(self.auth_key, REQUEST, flags, self.session_id, frame_id,
                                     capture_ns, request_payload)
      send_message(self.socket, REQUEST, flags, self.session_id, frame_id, capture_ns, request)
      set_remaining_timeout()
      response_flags, session_id, response_frame, response_capture, response = recv_message(self.socket, RESPONSE)
      completed_ns = time.monotonic_ns()
      if (session_id, response_frame, response_capture) != (self.session_id, frame_id, capture_ns):
        raise ProtocolError('response identity mismatch')
      if response_flags != flags:
        raise ProtocolError('reset acknowledgement mismatch')
      response = verify_payload(self.auth_key, RESPONSE, response_flags, session_id,
                                response_frame, response_capture, response)
      expected_bytes = TIMINGS.size + self.identity.output_floats * 4
      if len(response) != expected_bytes:
        raise ProtocolError(f'invalid response length: {len(response)} != {expected_bytes}')
      _, inference_start_ns, inference_end_ns = TIMINGS.unpack_from(response)
      output = response[TIMINGS.size:]
      if not all(math.isfinite(value) for value in memoryview(output).cast('f')):
        raise ProtocolError('non-finite model output')
      round_trip_ms = (completed_ns - started_ns) / 1e6
      inference_ms = (inference_end_ns - inference_start_ns) / 1e6
      if self.fallback.state is AcceleratorState.LOADING:
        self.qualification_count = self.qualification_count + 1 if round_trip_ms <= self.deadline_ms else 0
        if self.qualification_count >= self.qualification_frames:
          self.fallback.activate()
      if self.fallback.state is AcceleratorState.ACTIVE:
        self.fallback.observe(frame_id, round_trip_ms, True)
      if self.fallback.state is AcceleratorState.FAILED:
        raise TimeoutError(self.fallback.failure_reason)
      self.next_frame += 1
      return InferenceResult(frame_id, output, round_trip_ms, inference_ms)
    except Exception as error:
      self._fail(frame_id, f'{type(error).__name__}: {error}')

  def _fail(self, frame_id: int, reason: str):
    self.fallback.fail(frame_id, reason)
    self.close()
    raise RuntimeError(reason)

  def close(self) -> None:
    if self.socket is not None:
      self.socket.close()
      self.socket = None

  def reset(self) -> None:
    self.close()
    self.fallback.reset()
    self.session_id = secrets.randbits(64)
    self.identity = None
    self.next_frame = 0
    self.qualification_count = 0

  def __enter__(self):
    self.connect()
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()
