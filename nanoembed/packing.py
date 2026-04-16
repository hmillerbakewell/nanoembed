"""Bit packing utilities for quantized weight storage.

Packs arrays of small integers (1-8 bits per element) into dense uint8 arrays.
"""


import numpy as np


def pack(data: np.ndarray, bits: int) -> np.ndarray:
    """Pack a (N, D) uint8 array of values in [0, 2^bits) to a dense byte array.

    For bits=1, uses numpy.packbits (8 elements per byte, MSB first).
    For bits=2/4/8, packs multiple elements into each byte.
    For bits=3/5/6/7, falls back to storing one element per byte (no packing).
    """
    if bits == 1:
        return np.packbits(data, axis=1)
    elif bits == 2:
        N, D = data.shape
        assert D % 4 == 0, f"D={D} must be divisible by 4 for 2-bit packing"
        reshaped = data.reshape(N, D // 4, 4)
        packed = (reshaped[:, :, 0] << 6) | (reshaped[:, :, 1] << 4) | \
                 (reshaped[:, :, 2] << 2) | reshaped[:, :, 3]
        return packed.astype(np.uint8)
    elif bits == 4:
        N, D = data.shape
        assert D % 2 == 0, f"D={D} must be divisible by 2 for 4-bit packing"
        reshaped = data.reshape(N, D // 2, 2)
        packed = (reshaped[:, :, 0] << 4) | reshaped[:, :, 1]
        return packed.astype(np.uint8)
    elif bits == 8:
        return data.astype(np.uint8)
    else:
        return data.astype(np.uint8)


def unpack(packed: np.ndarray, bits: int, D: int) -> np.ndarray:
    """Unpack a dense byte array back to (N, D) uint8 values in [0, 2^bits).

    Args:
        packed: the packed array from pack().
        bits: the bit width per element.
        D: the original number of columns.
    """
    if bits == 1:
        return np.unpackbits(packed, axis=1)[:, :D]
    elif bits == 2:
        N = packed.shape[0]
        unpacked = np.empty((N, D), dtype=np.uint8)
        for i in range(4):
            shift = 6 - 2 * i
            unpacked[:, i::4] = (packed >> shift) & 0x03
        return unpacked
    elif bits == 4:
        N = packed.shape[0]
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        return np.stack([high, low], axis=-1).reshape(N, D)
    else:
        return packed[:, :D]
