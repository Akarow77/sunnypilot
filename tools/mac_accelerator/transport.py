#!/usr/bin/env python3
"""Small, dependency-free framing helpers for the Mac accelerator prototype."""

from __future__ import annotations

import socket
import struct
import zlib


MAGIC = b"SPMA"
VERSION = 1
REQUEST = 1
RESPONSE = 2
FLAG_RESET = 1 << 0

HEADER = struct.Struct("!4sBBHQQQII")
TIMINGS = struct.Struct("!QQQ")
POLICY_INPUTS = struct.Struct("<12f")

WARPED_SHAPE = (2, 6, 128, 256)
WARPED_BYTES = 2 * 6 * 128 * 256
REQUEST_BYTES = WARPED_BYTES + POLICY_INPUTS.size
OUTPUT_FLOATS = 2576
RESPONSE_BYTES = TIMINGS.size + OUTPUT_FLOATS * 4
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024


class ProtocolError(RuntimeError):
  pass


def recv_exact(sock: socket.socket, size: int) -> bytes:
  buffer = bytearray(size)
  view = memoryview(buffer)
  received = 0
  while received < size:
    count = sock.recv_into(view[received:])
    if not count:
      raise EOFError("peer closed the connection")
    received += count
  return bytes(buffer)


def send_message(sock: socket.socket, msg_type: int, flags: int, session_id: int,
                 frame_id: int, capture_ns: int, payload: bytes) -> None:
  if len(payload) > MAX_PAYLOAD_BYTES:
    raise ProtocolError(f"payload too large: {len(payload)}")
  checksum = zlib.crc32(payload) & 0xFFFFFFFF
  header = HEADER.pack(MAGIC, VERSION, msg_type, flags, session_id, frame_id,
                       capture_ns, len(payload), checksum)
  sock.sendall(header + payload)


def recv_message(sock: socket.socket, expected_type: int) -> tuple[int, int, int, int, bytes]:
  raw_header = recv_exact(sock, HEADER.size)
  magic, version, msg_type, flags, session_id, frame_id, capture_ns, payload_len, checksum = HEADER.unpack(raw_header)
  if magic != MAGIC:
    raise ProtocolError(f"bad magic: {magic!r}")
  if version != VERSION:
    raise ProtocolError(f"unsupported protocol version: {version}")
  if msg_type != expected_type:
    raise ProtocolError(f"unexpected message type: {msg_type}")
  if payload_len > MAX_PAYLOAD_BYTES:
    raise ProtocolError(f"payload too large: {payload_len}")
  payload = recv_exact(sock, payload_len)
  actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
  if actual_checksum != checksum:
    raise ProtocolError(f"checksum mismatch: {actual_checksum:#x} != {checksum:#x}")
  return flags, session_id, frame_id, capture_ns, payload
