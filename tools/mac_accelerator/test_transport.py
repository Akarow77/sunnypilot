#!/usr/bin/env python3

import socket
import unittest
import zlib

from transport import HEADER, MAGIC, REQUEST, VERSION, ProtocolError, recv_message, send_message


class TransportTest(unittest.TestCase):
  def test_round_trip(self) -> None:
    sender, receiver = socket.socketpair()
    self.addCleanup(sender.close)
    self.addCleanup(receiver.close)
    payload = b'warped-input'
    send_message(sender, REQUEST, 1, 2, 3, 4, payload)
    self.assertEqual(recv_message(receiver, REQUEST), (1, 2, 3, 4, payload))

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
