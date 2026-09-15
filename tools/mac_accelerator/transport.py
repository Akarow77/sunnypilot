#!/usr/bin/env python3
"""Small, dependency-free framing helpers for the Mac accelerator prototype."""

from __future__ import annotations

import socket
import struct
import time
from collections.abc import Iterable
import zlib


MAGIC = b"SPMA"
VERSION = 2
REQUEST = 1
RESPONSE = 2
HELLO = 3
HELLO_RESPONSE = 4
ERROR = 5
FLAG_RESET = 1 << 0
FLAG_ZSTD_REQUEST = 1 << 1

HEADER = struct.Struct("!4sBBHQQQII")
TIMINGS = struct.Struct("!QQQ")
POLICY_INPUTS = struct.Struct("<12f")

WARPED_SHAPE = (2, 6, 128, 256)
WARPED_BYTES = 2 * 6 * 128 * 256
REQUEST_BYTES = WARPED_BYTES + POLICY_INPUTS.size
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
DEVICE_TYPE = "sunnypilot-mac-accelerator-v2"


class ProtocolError(RuntimeError):
  pass


def recv_exact(sock: socket.socket, size: int, deadline_ns: int | None = None) -> bytes:
  buffer = bytearray(size)
  view = memoryview(buffer)
  received = 0
  while received < size:
    if deadline_ns is not None:
      remaining = (deadline_ns - time.monotonic_ns()) / 1e9
      if remaining <= 0:
        raise TimeoutError('receive absolute deadline expired')
      sock.settimeout(remaining)
    count = sock.recv_into(view[received:])
    if not count:
      raise EOFError("peer closed the connection")
    received += count
  return bytes(buffer)


def _send_parts(sock: socket.socket, parts: list[memoryview]) -> None:
  """Send several buffers without first joining the 393 KiB request."""
  pending = [part for part in parts if len(part)]
  while pending:
    if hasattr(sock, 'sendmsg'):
      sent = sock.sendmsg(pending)
    else:
      sent = sock.send(pending[0])
    if sent <= 0:
      raise EOFError('socket closed while sending')
    while pending and sent >= len(pending[0]):
      sent -= len(pending.pop(0))
    if sent:
      pending[0] = pending[0][sent:]


def send_message_parts(sock: socket.socket, msg_type: int, flags: int, session_id: int,
                       frame_id: int, capture_ns: int, payload_parts: Iterable[bytes | memoryview]) -> None:
  parts = [memoryview(part).cast('B') for part in payload_parts]
  payload_len = sum(map(len, parts))
  if payload_len > MAX_PAYLOAD_BYTES:
    raise ProtocolError(f"payload too large: {payload_len}")
  checksum = 0
  for part in parts:
    checksum = zlib.crc32(part, checksum)
  checksum &= 0xFFFFFFFF
  header = HEADER.pack(MAGIC, VERSION, msg_type, flags, session_id, frame_id,
                       capture_ns, payload_len, checksum)
  _send_parts(sock, [memoryview(header), *parts])


def send_message(sock: socket.socket, msg_type: int, flags: int, session_id: int,
                 frame_id: int, capture_ns: int, payload: bytes) -> None:
  send_message_parts(sock, msg_type, flags, session_id, frame_id, capture_ns, (payload,))


def recv_message(sock: socket.socket, expected_type: int, *, deadline_ns: int | None = None) -> tuple[int, int, int, int, bytes]:
  if deadline_ns is None and (timeout := sock.gettimeout()) is not None:
    deadline_ns = time.monotonic_ns() + int(timeout * 1e9)
  raw_header = recv_exact(sock, HEADER.size, deadline_ns)
  magic, version, msg_type, flags, session_id, frame_id, capture_ns, payload_len, checksum = HEADER.unpack(raw_header)
  if magic != MAGIC:
    raise ProtocolError(f"bad magic: {magic!r}")
  if version != VERSION:
    raise ProtocolError(f"unsupported protocol version: {version}")
  if msg_type != expected_type:
    raise ProtocolError(f"unexpected message type: {msg_type}")
  if payload_len > MAX_PAYLOAD_BYTES:
    raise ProtocolError(f"payload too large: {payload_len}")
  payload = recv_exact(sock, payload_len, deadline_ns)
  actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
  if actual_checksum != checksum:
    raise ProtocolError(f"checksum mismatch: {actual_checksum:#x} != {checksum:#x}")
  return flags, session_id, frame_id, capture_ns, payload
