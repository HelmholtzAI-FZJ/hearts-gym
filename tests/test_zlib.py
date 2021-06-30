import unittest
import zlib


class TestZlib(unittest.TestCase):
    def test_decompression(self):
        target = b'this is a test string'
        # compressed = zlib.compress(target)
        compressed = (
            b'x\x9c+\xc9\xc8,V\x00\xa2D\x85\x92\xd4\xe2\x12'
            b'\x85\xe2\x92\xa2\xcc\xbct\x00S\xe9\x07\xcd'
        )
        decompressed = zlib.decompress(compressed)
        self.assertEqual(decompressed, target)


if __name__ == '__main__':
    unittest.main()
