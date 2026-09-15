import struct
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from macos_app.build_icns import CHUNKS, build_icns


class IcnsBuilderTest(unittest.TestCase):
  def test_builds_well_formed_png_chunk_container(self):
    with TemporaryDirectory() as directory:
      root = Path(directory)
      iconset = root / 'AppIcon.iconset'
      iconset.mkdir()
      png = b'\x89PNG\r\n\x1a\nexample'
      for _, filename in CHUNKS:
        (iconset / filename).write_bytes(png)
      output = root / 'AppIcon.icns'
      build_icns(iconset, output)
      data = output.read_bytes()
      self.assertEqual(data[:4], b'icns')
      self.assertEqual(struct.unpack('>I', data[4:8])[0], len(data))
      self.assertEqual(data.count(png), len(CHUNKS))


if __name__ == '__main__':
  unittest.main()
