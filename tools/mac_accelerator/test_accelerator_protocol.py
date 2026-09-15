#!/usr/bin/env python3

import json
import socket
import struct
import threading
import time
import unittest
from unittest.mock import Mock
from pathlib import Path

from accelerator_client import AcceleratorClient, AcceleratorIdentity
from accelerator_protocol import (authenticate_payload, authenticate_payload_parts, canonical_json, make_auth, read_auth_key,
                                  server_handshake, verify_payload)
from fallback import AcceleratorState
from transport import (DEVICE_TYPE, HELLO, HELLO_RESPONSE, POLICY_INPUTS, REQUEST,
                       RESPONSE, TIMINGS, VERSION, WARPED_BYTES, ProtocolError,
                       recv_message, send_message)


def identity() -> dict:
  return {
    'authentication_required': False,
    'backend': 'METAL',
    'device_type': DEVICE_TYPE,
    'input_shapes': {'img': [1, 12, 128, 256]},
    'model_checkpoint': 'checkpoint',
    'model_sha256': 'a' * 64,
    'output_floats': 2,
    'output_shapes': {'outputs': [1, 2]},
    'protocol': VERSION,
    'request_bytes': WARPED_BYTES + POLICY_INPUTS.size,
  }


class AcceleratorProtocolTest(unittest.TestCase):
  def test_binary_key_preserves_whitespace_bytes(self):
    path = Mock(spec=Path)
    path.read_bytes.return_value = b'\n' + b'k' * 30 + b' '
    self.assertEqual(read_auth_key(path), path.read_bytes.return_value)

  def test_nonfinite_deadline_rejected(self):
    for value in (float('nan'), float('inf')):
      with self.assertRaises(ValueError):
        AcceleratorClient('::1', deadline_ms=value)

  def test_payload_authentication_rejects_tampering(self) -> None:
    key = b'k' * 32
    wire = authenticate_payload(key, REQUEST, 1, 7, 9, 11, b'payload')
    self.assertEqual(verify_payload(key, REQUEST, 1, 7, 9, 11, wire), b'payload')
    tampered = bytearray(wire)
    tampered[0] ^= 1
    with self.assertRaisesRegex(ProtocolError, 'authentication'):
      verify_payload(key, REQUEST, 1, 7, 9, 11, bytes(tampered))

  def test_vectored_payload_authentication_matches_contiguous(self) -> None:
    key = b'k' * 32
    parts = authenticate_payload_parts(key, REQUEST, 1, 7, 9, 11, (b'pay', memoryview(b'load')))
    self.assertEqual(b''.join(parts), authenticate_payload(key, REQUEST, 1, 7, 9, 11, b'payload'))

  def test_authenticated_handshake(self) -> None:
    client, server = socket.socketpair()
    self.addCleanup(client.close)
    self.addCleanup(server.close)
    key = b'k' * 32
    nonce = '12' * 32
    errors = []

    def run_server():
      try:
        server_handshake(server, identity(), key)
      except Exception as error:
        errors.append(error)

    worker = threading.Thread(target=run_server)
    worker.start()
    hello = {'client': 'comma3x', 'nonce': nonce, 'protocol': VERSION}
    hello['auth'] = make_auth(key, nonce, {'client': 'comma3x', 'protocol': VERSION})
    send_message(client, HELLO, 0, 7, 0, 0, canonical_json(hello))
    _, session_id, _, _, payload = recv_message(client, HELLO_RESPONSE)
    response = json.loads(payload)
    response_auth = response.pop('auth')
    worker.join(1)
    self.assertFalse(errors)
    self.assertEqual(session_id, 7)
    self.assertTrue(response['authentication_required'])
    self.assertEqual(response_auth, make_auth(key, nonce, response))

  def test_handshake_rejects_missing_authentication(self) -> None:
    client, server = socket.socketpair()
    self.addCleanup(client.close)
    self.addCleanup(server.close)
    errors = []

    def run_server():
      try:
        server_handshake(server, identity(), b'k' * 32)
      except Exception as error:
        errors.append(error)

    worker = threading.Thread(target=run_server)
    worker.start()
    hello = {'client': 'comma3x', 'nonce': '12' * 32, 'protocol': VERSION}
    send_message(client, HELLO, 0, 7, 0, 0, canonical_json(hello))
    worker.join(1)
    self.assertEqual(len(errors), 1)
    self.assertIsInstance(errors[0], ProtocolError)

  def test_identity_rejects_non_metal_backend(self) -> None:
    client = AcceleratorClient('::1', expected_checkpoint='checkpoint', expected_output_floats=2)
    bad = AcceleratorIdentity(DEVICE_TYPE, 'CPU', 'checkpoint', 'a' * 64,
                              {'img': [1, 12, 128, 256]}, {'outputs': [1, 2]}, 2,
                              WARPED_BYTES + POLICY_INPUTS.size)
    with self.assertRaisesRegex(ProtocolError, 'backend'):
      client._validate_identity(bad)

  def test_identity_rejects_missing_model_hash(self) -> None:
    client = AcceleratorClient('::1')
    bad = AcceleratorIdentity(DEVICE_TYPE, 'METAL', 'checkpoint', '',
                              {'img': [1, 12, 128, 256]}, {'outputs': [1, 2]}, 2,
                              WARPED_BYTES + POLICY_INPUTS.size)
    with self.assertRaisesRegex(ProtocolError, 'SHA-256'):
      client._validate_identity(bad)

  def test_identity_rejects_missing_compression(self) -> None:
    client = AcceleratorClient('::1', compression='zstd-1')
    identity_without_compression = AcceleratorIdentity(
      DEVICE_TYPE, 'METAL', 'checkpoint', 'a' * 64, {}, {'outputs': [1, 2]}, 2,
      WARPED_BYTES + POLICY_INPUTS.size)
    with self.assertRaisesRegex(ProtocolError, 'zstd-1'):
      client._validate_identity(identity_without_compression)

  def test_inference_round_trip(self) -> None:
    client_socket, server_socket = socket.socketpair()
    self.addCleanup(client_socket.close)
    self.addCleanup(server_socket.close)
    client = AcceleratorClient('::1', deadline_ms=50)
    client.socket = client_socket
    client_socket.settimeout(.05)
    client.identity = AcceleratorIdentity(DEVICE_TYPE, 'METAL', 'checkpoint', 'a' * 64,
                                          {}, {'outputs': [1, 2]}, 2,
                                          WARPED_BYTES + POLICY_INPUTS.size)

    def run_server():
      flags, session_id, frame_id, capture_ns, _ = recv_message(server_socket, REQUEST)
      now = time.monotonic_ns()
      payload = TIMINGS.pack(now, now, now + 1_000_000) + b'\0' * 8
      send_message(server_socket, RESPONSE, flags, session_id, frame_id, capture_ns, payload)

    worker = threading.Thread(target=run_server)
    worker.start()
    result = client.infer(bytes(WARPED_BYTES), bytes(POLICY_INPUTS.size), frame_id=0,
                          capture_ns=time.monotonic_ns(), reset=True)
    worker.join(1)
    self.assertEqual(result.frame_id, 0)
    self.assertEqual(result.inference_ms, 1.0)
    self.assertEqual(client.fallback.state, AcceleratorState.ACTIVE)

  def test_float16_wire_output_is_restored_to_float32(self) -> None:
    client_socket, server_socket = socket.socketpair()
    self.addCleanup(client_socket.close)
    self.addCleanup(server_socket.close)
    client = AcceleratorClient('::1', deadline_ms=50)
    client.socket = client_socket
    client_socket.settimeout(.05)
    client.identity = AcceleratorIdentity(DEVICE_TYPE, 'METAL', 'checkpoint', 'a' * 64,
                                          {}, {'outputs': [1, 2]}, 2,
                                          WARPED_BYTES + POLICY_INPUTS.size,
                                          output_dtype='float16')

    def run_server():
      flags, session_id, frame_id, capture_ns, _ = recv_message(server_socket, REQUEST)
      now = time.monotonic_ns()
      payload = TIMINGS.pack(now, now, now) + struct.pack('<ee', 1.5, -2.0)
      send_message(server_socket, RESPONSE, flags, session_id, frame_id, capture_ns, payload)

    worker = threading.Thread(target=run_server)
    worker.start()
    result = client.infer(bytes(WARPED_BYTES), bytes(POLICY_INPUTS.size), frame_id=0,
                          capture_ns=time.monotonic_ns(), reset=True)
    worker.join(1)
    self.assertEqual(struct.unpack('<ff', result.output), (1.5, -2.0))
    self.assertEqual(client.fallback.state, AcceleratorState.ACTIVE)

  def test_timeout_latches_failure(self) -> None:
    client_socket, server_socket = socket.socketpair()
    self.addCleanup(client_socket.close)
    self.addCleanup(server_socket.close)
    client = AcceleratorClient('::1', deadline_ms=1)
    client.socket = client_socket
    client_socket.settimeout(.001)
    client.identity = AcceleratorIdentity(DEVICE_TYPE, 'METAL', 'checkpoint', 'a' * 64,
                                          {}, {'outputs': [1, 2]}, 2,
                                          WARPED_BYTES + POLICY_INPUTS.size)
    with self.assertRaisesRegex(RuntimeError, 'timed out'):
      client.infer(bytes(WARPED_BYTES), bytes(POLICY_INPUTS.size), frame_id=0,
                   capture_ns=time.monotonic_ns(), reset=True)
    self.assertEqual(client.fallback.state, AcceleratorState.FAILED)

  def test_slow_warmup_requires_consecutive_qualified_frames(self) -> None:
    client_socket, server_socket = socket.socketpair()
    self.addCleanup(client_socket.close)
    self.addCleanup(server_socket.close)
    # Keep a wide scheduling margin: this test may run beside Core ML imports on
    # a busy development Mac, while still proving that loading and active
    # deadlines are distinct.
    client = AcceleratorClient('::1', deadline_ms=50, qualification_frames=2,
                               qualification_timeout_ms=500)
    client.socket = client_socket
    client.identity = AcceleratorIdentity(DEVICE_TYPE, 'METAL', 'checkpoint', 'a' * 64,
                                          {}, {'outputs': [1, 2]}, 2,
                                          WARPED_BYTES + POLICY_INPUTS.size)

    def run_server():
      for delay in (0.1, 0.0, 0.0):
        flags, session_id, frame_id, capture_ns, _ = recv_message(server_socket, REQUEST)
        time.sleep(delay)
        now = time.monotonic_ns()
        payload = TIMINGS.pack(now, now, now) + b'\0' * 8
        send_message(server_socket, RESPONSE, flags, session_id, frame_id, capture_ns, payload)

    worker = threading.Thread(target=run_server)
    worker.start()
    for frame_id in range(3):
      client.infer(bytes(WARPED_BYTES), bytes(POLICY_INPUTS.size), frame_id=frame_id,
                   capture_ns=time.monotonic_ns(), reset=frame_id == 0)
    worker.join(1)
    self.assertEqual(client.fallback.state, AcceleratorState.ACTIVE)
    self.assertEqual(client.qualification_count, 2)


if __name__ == '__main__':
  unittest.main()
