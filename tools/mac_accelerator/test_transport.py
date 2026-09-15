#!/usr/bin/env python3

import socket
import threading
import time
import unittest
import zlib

from transport import HEADER, MAGIC, REQUEST, VERSION, ProtocolError, recv_message, send_message, send_message_parts


class TransportTest(unittest.TestCase):
  def test_slow_trickle_cannot_extend_absolute_deadline(self):
    sender, receiver = socket.socketpair()
    receiver.settimeout(.2)
    def trickle():
      try:
        for _ in range(20):
          sender.sendall(b'x')
          time.sleep(.03)
      except OSError:
        pass
    worker = threading.Thread(target=trickle)
    worker.start()
    started = time.monotonic()
    try:
      with self.assertRaises(TimeoutError):
        recv_message(receiver, REQUEST, deadline_ns=time.monotonic_ns() + 120_000_000)
      self.assertLess(time.monotonic() - started, .3)
    finally:
      receiver.close()
      sender.close()
      worker.join(1)

  def test_round_trip(self) -> None:
    sender, receiver = socket.socketpair()
    self.addCleanup(sender.close)
    self.addCleanup(receiver.close)
    payload = b'warped-input'
    send_message(sender, REQUEST, 1, 2, 3, 4, payload)
    self.assertEqual(recv_message(receiver, REQUEST), (1, 2, 3, 4, payload))

  def test_vectored_round_trip(self) -> None:
    sender, receiver = socket.socketpair()
    self.addCleanup(sender.close)
    self.addCleanup(receiver.close)
    send_message_parts(sender, REQUEST, 1, 2, 3, 4, (b'warped-', memoryview(b'input')))
    self.assertEqual(recv_message(receiver, REQUEST), (1, 2, 3, 4, b'warped-input'))

  def test_checksum_rejected(self) -> None:
    sender, receiver = socket.socketpair()
    self.addCleanup(sender.close)
    self.addCleanup(receiver.close)
    payload = b'bad-data'
    bad_checksum = (zlib.crc32(payload) + 1) & 0xFFFFFFFF
    sender.sendall(HEADER.pack(MAGIC, VERSION, REQUEST, 0, 1, 1, 1, len(payload), bad_checksum) + payload)
    with self.assertRaisesRegex(ProtocolError, 'checksum mismatch'):
      recv_message(receiver, REQUEST)

  def test_wrong_message_type_rejected(self) -> None:
    sender, receiver = socket.socketpair()
    self.addCleanup(sender.close)
    self.addCleanup(receiver.close)
    send_message(sender, REQUEST, 0, 1, 1, 1, b'')
    with self.assertRaisesRegex(ProtocolError, 'unexpected message type'):
      recv_message(receiver, REQUEST + 1)


if __name__ == '__main__':
  unittest.main()
