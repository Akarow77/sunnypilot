#!/usr/bin/env python3
"""Dependency-free authenticated discovery protocol shared by the Mac and comma."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import socket
import struct
import time
from collections.abc import Iterable
from typing import Any

from transport import HELLO, HELLO_RESPONSE, VERSION, ProtocolError, recv_message, send_message


AUTH_TAG_BYTES = hashlib.sha256().digest_size
AUTH_METADATA = struct.Struct('!BBHQQQ')


def canonical_json(value: dict[str, Any]) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def read_auth_key(path: Path | None) -> bytes | None:
  if path is None:
    return None
  key = path.read_bytes()
  if len(key) < 32:
    raise ValueError('accelerator authentication key must contain at least 32 bytes')
  return key


def make_auth(key: bytes, nonce: str, body: dict[str, Any]) -> str:
  return hmac.new(key, nonce.encode() + b'\0' + canonical_json(body), hashlib.sha256).hexdigest()


def authenticate_payload(key: bytes | None, msg_type: int, flags: int, session_id: int,
                         frame_id: int, capture_ns: int, payload: bytes) -> bytes:
  if key is None:
    return payload
  metadata = AUTH_METADATA.pack(VERSION, msg_type, flags, session_id, frame_id, capture_ns)
  tag = hmac.new(key, b'SPMA-PAYLOAD-v2\0' + metadata + payload, hashlib.sha256).digest()
  return payload + tag


def authenticate_payload_parts(key: bytes | None, msg_type: int, flags: int, session_id: int,
                               frame_id: int, capture_ns: int,
                               payload_parts: Iterable[bytes | memoryview]) -> tuple[bytes | memoryview, ...]:
  """Authenticate an iovec request without concatenating its large image buffers."""
  parts = tuple(payload_parts)
  if key is None:
    return parts
  metadata = AUTH_METADATA.pack(VERSION, msg_type, flags, session_id, frame_id, capture_ns)
  digest = hmac.new(key, b'SPMA-PAYLOAD-v2\0' + metadata, hashlib.sha256)
  for part in parts:
    digest.update(part)
  return (*parts, digest.digest())


def verify_payload(key: bytes | None, msg_type: int, flags: int, session_id: int,
                   frame_id: int, capture_ns: int, wire_payload: bytes) -> bytes:
  if key is None:
    return wire_payload
  if len(wire_payload) < AUTH_TAG_BYTES:
    raise ProtocolError('missing payload authentication tag')
  payload, tag = wire_payload[:-AUTH_TAG_BYTES], wire_payload[-AUTH_TAG_BYTES:]
  expected = authenticate_payload(key, msg_type, flags, session_id, frame_id, capture_ns, payload)[-AUTH_TAG_BYTES:]
  if not hmac.compare_digest(tag, expected):
    raise ProtocolError('payload authentication failed')
  return payload


def server_handshake(conn: socket.socket, identity: dict, auth_key: bytes | None) -> int:
  _, session_id, _, _, payload = recv_message(conn, HELLO)
  try:
    hello = json.loads(payload)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ProtocolError(f'invalid hello JSON: {error}') from error
  if not isinstance(hello, dict) or hello.get('client') != 'comma3x' or hello.get('protocol') != VERSION:
    raise ProtocolError('unsupported client hello')
  nonce = hello.get('nonce')
  if not isinstance(nonce, str) or len(nonce) != 64:
    raise ProtocolError('invalid hello nonce')
  try:
    bytes.fromhex(nonce)
  except ValueError as error:
    raise ProtocolError('invalid hello nonce') from error

  client_auth = hello.get('auth')
  if auth_key is not None:
    expected = make_auth(auth_key, nonce, {'client': 'comma3x', 'protocol': VERSION})
    if not isinstance(client_auth, str) or not hmac.compare_digest(client_auth, expected):
      raise ProtocolError('client authentication failed')

  response = dict(identity)
  response['authentication_required'] = auth_key is not None
  if auth_key is not None:
    response['auth'] = make_auth(auth_key, nonce, response)
  send_message(conn, HELLO_RESPONSE, 0, session_id, 0, time.monotonic_ns(), canonical_json(response))
  return session_id
