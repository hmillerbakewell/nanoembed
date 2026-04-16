"""Tests for checkpoint save/load round-trip."""

import numpy as np
import pytest

from nanoembed.checkpoint import save_checkpoint, load_checkpoint
from nanoembed.packing import pack


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("ckpt")


def _make_checkpoint_data(N=100, D=32, bits=1):
    """Create valid checkpoint arrays for testing."""
    max_val = (1 << bits) - 1
    raw_codes = np.random.randint(0, max_val + 1, size=(N, D), dtype=np.uint8)
    packed_codes = pack(raw_codes, bits)
    scales = np.random.rand(N).astype(np.float32) + 0.1
    codebook = np.linspace(-0.5, 0.5, 1 << bits).astype(np.float32)
    return packed_codes, scales, codebook


class TestSaveLoadRoundtrip:
    @pytest.mark.parametrize("bits", [1, 2, 4])
    def test_roundtrip(self, tmp_path, bits):
        packed_codes, scales, codebook = _make_checkpoint_data(bits=bits)
        path = save_checkpoint(
            path=tmp_path / "model",
            packed_codes=packed_codes,
            scales=scales,
            codebook=codebook,
            embed_dim=32,
            vocab_size=100,
            bits=bits,
            tokenizer_name="test/tokenizer",
            method="turbo-lloyd",
            source_model="test/source",
        )

        loaded = load_checkpoint(path)

        np.testing.assert_array_equal(loaded["packed_codes"], packed_codes)
        np.testing.assert_array_almost_equal(loaded["scales"], scales)
        np.testing.assert_array_almost_equal(loaded["codebook"], codebook)
        assert loaded["embed_dim"] == 32
        assert loaded["vocab_size"] == 100
        assert loaded["bits"] == bits
        assert loaded["tokenizer_name"] == "test/tokenizer"
        assert loaded["method"] == "turbo-lloyd"
        assert loaded["source_model"] == "test/source"

    def test_npz_extension_added(self, tmp_path):
        packed_codes, scales, codebook = _make_checkpoint_data()
        path = save_checkpoint(
            path=tmp_path / "model",
            packed_codes=packed_codes,
            scales=scales,
            codebook=codebook,
            embed_dim=32,
            vocab_size=100,
            bits=1,
            tokenizer_name="test/tok",
        )
        assert path.suffix == ".npz"
        assert path.exists()

    def test_back_compat_codebook_val(self, tmp_path):
        """Old checkpoints stored codebook_val (scalar) instead of codebook (array)."""
        # Simulate old format
        path = tmp_path / "old_model.npz"
        np.savez(
            str(path),
            packed_codes=np.zeros((10, 4), dtype=np.uint8),
            scales=np.ones(10, dtype=np.float32),
            codebook_val=np.float32(0.0499),
            embed_dim=np.int64(32),
            vocab_size=np.int64(10),
            tokenizer_name=np.array("test/tok"),
        )

        loaded = load_checkpoint(path)
        assert loaded["codebook"].shape == (2,)
        np.testing.assert_almost_equal(loaded["codebook"][0], -0.0499, decimal=4)
        np.testing.assert_almost_equal(loaded["codebook"][1], 0.0499, decimal=4)
        assert loaded["bits"] == 1  # default
