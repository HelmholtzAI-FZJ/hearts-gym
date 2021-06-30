"""
`test_zlib.py` without `unittest` dependence.
"""

import zlib


def test_decompression():
    target = b'this is a test string'
    # compressed = zlib.compress(target)
    compressed = (
        b'x\x9c+\xc9\xc8,V\x00\xa2D\x85\x92\xd4\xe2\x12'
        b'\x85\xe2\x92\xa2\xcc\xbct\x00S\xe9\x07\xcd'
    )
    decompressed = zlib.decompress(compressed)
    assert decompressed == target, 'decompressed result does not match target'


def main():
    test_decompression()
    print('OK')


if __name__ == '__main__':
    main()
