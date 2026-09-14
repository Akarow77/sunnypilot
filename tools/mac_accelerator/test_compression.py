#!/usr/bin/env python3

import unittest

from compression import ZstdCodec, ZstdError


class CompressionTest(unittest.TestCase):
  def test_zstd_round_trip(self) -> None:
    codec = ZstdCodec()
    payload = bytes(range(256)) * 1536
    compressed = codec.compress(payload)
    self.assertLess(len(compressed), len(payload))
    self.assertEqual(codec.decompress(compressed, len(payload)), payload)

  def test_zstd_rejects_wrong_size(self) -> None:
    codec = ZstdCodec()
    compressed = codec.compress(b'payload')
    with self.assertRaises(ZstdError):
      codec.decompress(compressed, 8)


if __name__ == '__main__':
  unittest.main()
