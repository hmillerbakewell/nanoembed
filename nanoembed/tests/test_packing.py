"""Tests for bit packing/unpacking at various bit widths."""

import numpy as np
import pytest

from nanoembed.packing import pack, unpack


@pytest.fixture(params=[1, 2, 4, 8])
def bits(request):
    return request.param


def _random_data(N: int, D: int, bits: int) -> np.ndarray:
    """Random uint8 array with values in [0, 2^bits)."""
    return np.random.randint(0, 1 << bits, size=(N, D), dtype=np.uint8)


class TestPackUnpackRoundtrip:
    """pack → unpack should recover the original data exactly."""

    def test_small(self, bits):
        data = _random_data(4, 16, bits)
        packed = pack(data, bits)
        recovered = unpack(packed, bits, D=16)
        np.testing.assert_array_equal(recovered, data)

    def test_large(self, bits):
        data = _random_data(100, 256, bits)
        packed = pack(data, bits)
        recovered = unpack(packed, bits, D=256)
        np.testing.assert_array_equal(recovered, data)

    def test_single_row(self, bits):
        data = _random_data(1, 64, bits)
        packed = pack(data, bits)
        recovered = unpack(packed, bits, D=64)
        np.testing.assert_array_equal(recovered, data)


class TestPackedSize:
    """Packed output should have the expected number of bytes."""

    def test_1bit_size(self):
        data = _random_data(10, 32, bits=1)
        packed = pack(data, bits=1)
        assert packed.shape == (10, 4)  # 32 bits / 8 = 4 bytes

    def test_2bit_size(self):
        data = _random_data(10, 32, bits=2)
        packed = pack(data, bits=2)
        assert packed.shape == (10, 8)  # 32 * 2 / 8 = 8 bytes

    def test_4bit_size(self):
        data = _random_data(10, 32, bits=4)
        packed = pack(data, bits=4)
        assert packed.shape == (10, 16)  # 32 * 4 / 8 = 16 bytes

    def test_8bit_noop(self):
        data = _random_data(10, 32, bits=8)
        packed = pack(data, bits=8)
        assert packed.shape == (10, 32)  # no packing


class TestEdgeCases:
    def test_all_zeros(self, bits):
        data = np.zeros((5, 16), dtype=np.uint8)
        packed = pack(data, bits)
        recovered = unpack(packed, bits, D=16)
        np.testing.assert_array_equal(recovered, data)

    def test_all_max(self, bits):
        max_val = (1 << bits) - 1
        data = np.full((5, 16), max_val, dtype=np.uint8)
        packed = pack(data, bits)
        recovered = unpack(packed, bits, D=16)
        np.testing.assert_array_equal(recovered, data)
