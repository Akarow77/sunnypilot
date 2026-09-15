#!/usr/bin/env python3
"""Strict comma-side client for recognizing and using the Mac accelerator."""

from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import json
import math
import secrets
import socket
import time

try:
  import numpy as np
except ImportError:
  np = None

from accelerator_protocol import authenticate_payload_parts, canonical_json, make_auth, verify_payload
from compression import ZstdCodec
from fallback import AcceleratorState, FallbackLatch
from transport import (DEVICE_TYPE, FLAG_RESET, FLAG_ZSTD_REQUEST, HELLO, HELLO_RESPONSE, POLICY_INPUTS,
                       MAX_PAYLOAD_BYTES, REQUEST, RESPONSE, TIMINGS, VERSION, WARPED_BYTES, ProtocolError,
                       recv_message, send_message, send_message_parts)

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
  output_dtype: str = 'float32'
  output_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
  frame_skip: int | None = None


@dataclass(frozen=True)
class InferenceResult:
  frame_id: int
  output: bytes | memoryview
  round_trip_ms: float
  inference_ms: float
  prepare_ms: float
  send_ms: float
  receive_ms: float
  validate_ms: float


class AcceleratorClient:
  def __init__(self, host: str, port: int = 8066, *, deadline_ms: float = 45.0,
               auth_key: bytes | None = None, expected_checkpoint: str | None = None,
               expected_model_sha256: str | None = None, expected_output_floats: int | None = None,
               expected_backend: str = 'METAL', qualification_frames: int = 1,
               qualification_timeout_ms: float = 100.0, compression: str | None = None):
    if not math.isfinite(deadline_ms) or deadline_ms <= 0:
      raise ValueError('deadline must be positive')
    if qualification_frames < 1 or not math.isfinite(qualification_timeout_ms) or qualification_timeout_ms < deadline_ms:
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
    self._output_buffer = None
    self._finite_buffer = None

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
      if not isinstance(response, dict) or response.get('protocol') != VERSION:
        raise ProtocolError('invalid hello response')
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
        output_dtype=str(response.get('output_dtype', 'float32')),
        output_slices={name: (int(bounds[0]), int(bounds[1]))
                       for name, bounds in response.get('output_slices', {}).items()},
        frame_skip=response.get('frame_skip'),
      )
      self._validate_identity(identity)
      conn.settimeout(self.deadline_ms / 1000.0)
      self.socket = conn
      self.identity = identity
      self.next_frame = 0
      self.qualification_count = 0
      self._prepare_output_buffers(identity)
      return identity
    except Exception:
      conn.close()
      raise

  def _prepare_output_buffers(self, identity: AcceleratorIdentity) -> None:
    if identity.output_dtype == 'float16':
      assert np is not None
      self._output_buffer = np.empty(identity.output_floats, dtype='<f4')
      self._finite_buffer = np.empty(identity.output_floats, dtype=np.bool_)
    else:
      self._output_buffer = None
      self._finite_buffer = None

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
    if identity.output_dtype not in ('float16', 'float32'):
      raise ProtocolError(f'unsupported output dtype: {identity.output_dtype}')
    if identity.output_dtype == 'float16' and np is None:
      raise ProtocolError('float16 output requires NumPy on the client')
    if not 0 < identity.output_floats <= (MAX_PAYLOAD_BYTES - TIMINGS.size - 32) // 4:
      raise ProtocolError('invalid output count')
    for name, bounds in identity.output_slices.items():
      if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] <= identity.output_floats:
        raise ProtocolError(f'invalid output slice {name}: {bounds}')

  def infer(self, warped: bytes, policy: bytes, *, frame_id: int, capture_ns: int,
            reset: bool = False, deadline_ns: int | None = None) -> InferenceResult:
    if self.socket is None or self.identity is None:
      raise RuntimeError('accelerator is not connected')
    if self.fallback.state is AcceleratorState.FAILED:
      raise RuntimeError(f'accelerator failure is latched: {self.fallback.failure_reason}')
    if reset:
      self.fallback.reset()
      self.qualification_count = 0
    if self.identity.output_dtype == 'float16' and self._output_buffer is None:
      self._prepare_output_buffers(self.identity)
    if frame_id != self.next_frame:
      self._fail(frame_id, f'non-sequential client frame: {frame_id} != {self.next_frame}')
    if len(warped) != WARPED_BYTES or len(policy) != POLICY_INPUTS.size:
      self._fail(frame_id, 'invalid local request shape')

    started_ns = time.monotonic_ns()
    stage = 'prepare'
    prepared_ns = sent_ns = received_ns = completed_ns = None
    flags = FLAG_RESET if reset else 0
    request_deadline_ms = (self.qualification_timeout_ms if self.fallback.state is AcceleratorState.LOADING
                           else self.deadline_ms)
    own_deadline_ns = started_ns + int(request_deadline_ms * 1e6)
    deadline_ns = min(deadline_ns, own_deadline_ns) if deadline_ns is not None else own_deadline_ns

    def set_remaining_timeout() -> None:
      remaining = (deadline_ns - time.monotonic_ns()) / 1e9
      if remaining <= 0:
        raise TimeoutError('accelerator deadline expired')
      assert self.socket is not None
      self.socket.settimeout(remaining)

    try:
      request_parts: tuple[bytes | memoryview, ...] = (memoryview(warped), memoryview(policy))
      if self.codec is not None:
        request_parts = (self.codec.compress(warped + policy),)
        flags |= FLAG_ZSTD_REQUEST
      request_parts = authenticate_payload_parts(self.auth_key, REQUEST, flags, self.session_id,
                                                 frame_id, capture_ns, request_parts)
      prepared_ns = time.monotonic_ns()
      stage = 'send'
      set_remaining_timeout()
      send_message_parts(self.socket, REQUEST, flags, self.session_id, frame_id, capture_ns, request_parts)
      sent_ns = time.monotonic_ns()
      stage = 'receive'
      set_remaining_timeout()
      response_flags, session_id, response_frame, response_capture, response = recv_message(self.socket, RESPONSE, deadline_ns=deadline_ns)
      received_ns = time.monotonic_ns()
      stage = 'validate'
      if (session_id, response_frame, response_capture) != (self.session_id, frame_id, capture_ns):
        raise ProtocolError('response identity mismatch')
      if response_flags != flags:
        raise ProtocolError('reset acknowledgement mismatch')
      response = verify_payload(self.auth_key, RESPONSE, response_flags, session_id,
                                response_frame, response_capture, response)
      output_itemsize = 2 if self.identity.output_dtype == 'float16' else 4
      expected_bytes = TIMINGS.size + self.identity.output_floats * output_itemsize
      if len(response) != expected_bytes:
        raise ProtocolError(f'invalid response length: {len(response)} != {expected_bytes}')
      server_receive_ns, inference_start_ns, inference_end_ns = TIMINGS.unpack_from(response)
      if not server_receive_ns <= inference_start_ns <= inference_end_ns:
        raise ProtocolError('invalid server timing order')
      wire_output = response[TIMINGS.size:]
      if np is not None:
        wire_dtype = '<f2' if self.identity.output_dtype == 'float16' else '<f4'
        values = np.frombuffer(wire_output, dtype=wire_dtype)
        finite = self._finite_buffer
        if finite is None:
          finite = np.isfinite(values)
        else:
          np.isfinite(values, out=finite)
        if not np.all(finite):
          raise ProtocolError('non-finite model output')
        if self.identity.output_dtype == 'float16':
          assert self._output_buffer is not None
          np.copyto(self._output_buffer, values, casting='unsafe')
          output = memoryview(self._output_buffer).cast('B')
        else:
          output = wire_output
      else:
        output = wire_output
        if not all(math.isfinite(value) for value in memoryview(output).cast('f')):
          raise ProtocolError('non-finite model output')
      completed_ns = time.monotonic_ns()
      round_trip_ms = (completed_ns - started_ns) / 1e6
      if completed_ns > deadline_ns:
        raise TimeoutError(f'accelerator deadline expired after validation ({round_trip_ms:.2f} ms)')
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
      return InferenceResult(
        frame_id, output, round_trip_ms, inference_ms,
        (prepared_ns - started_ns) / 1e6,
        (sent_ns - prepared_ns) / 1e6,
        (received_ns - sent_ns) / 1e6,
        (completed_ns - received_ns) / 1e6,
      )
    except Exception as error:
      now_ns = time.monotonic_ns()
      elapsed_ms = (now_ns - started_ns) / 1e6
      stage_times = []
      if prepared_ns is not None:
        stage_times.append(f'prepare={((prepared_ns - started_ns) / 1e6):.2f}')
      if sent_ns is not None and prepared_ns is not None:
        stage_times.append(f'send={((sent_ns - prepared_ns) / 1e6):.2f}')
      if received_ns is not None and sent_ns is not None:
        stage_times.append(f'receive={((received_ns - sent_ns) / 1e6):.2f}')
      if completed_ns is not None and received_ns is not None:
        stage_times.append(f'validate={((completed_ns - received_ns) / 1e6):.2f}')
      detail = f' stages[{",".join(stage_times)}]' if stage_times else ''
      self._fail(frame_id, f'{stage} after {elapsed_ms:.2f} ms:{detail} {type(error).__name__}: {error}')

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
    self._output_buffer = None
    self._finite_buffer = None

  def __enter__(self):
    self.connect()
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()
