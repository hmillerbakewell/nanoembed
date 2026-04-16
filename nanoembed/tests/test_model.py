"""Tests for Model encode and similarity."""

import numpy as np
import pytest

from nanoembed.checkpoint import save_checkpoint
from nanoembed.model import Model
from nanoembed.packing import pack


@pytest.fixture
def model_path(tmp_path_factory):
    """Create a tiny fake model checkpoint for testing.

    Uses a real tokenizer (bert-base-uncased, small vocab) so encode() works.
    The weights are random — we're testing the pipeline, not quality.
    """
    tmp = tmp_path_factory.mktemp("model")

    N = 30522  # bert-base-uncased vocab size
    D = 16
    bits = 1
    codebook = np.array([-0.05, 0.05], dtype=np.float32)

    raw_codes = np.random.randint(0, 2, size=(N, D), dtype=np.uint8)
    packed_codes = pack(raw_codes, bits)
    scales = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

    path = save_checkpoint(
        path=tmp / "test_model",
        packed_codes=packed_codes,
        scales=scales,
        codebook=codebook,
        embed_dim=D,
        vocab_size=N,
        bits=bits,
        tokenizer_name="google-bert/bert-base-uncased",
        method="turbo-lloyd",
        source_model="test",
    )
    return path


class TestModelLoad:
    def test_load(self, model_path):
        model = Model.load(model_path)
        assert model.embed_dim == 16
        assert model.vocab_size == 30522
        assert model.bits == 1

    def test_info(self, model_path):
        model = Model.load(model_path)
        info = model.info
        assert info.embed_dim == 16
        assert info.bits == 1
        assert info.source_model == "test"
        assert info.logical_size_mb > 0


class TestEncode:
    def test_single_sentence(self, model_path):
        model = Model.load(model_path)
        embs = model.encode(["hello world"])
        assert embs.shape == (1, 16)

    def test_batch(self, model_path):
        model = Model.load(model_path)
        embs = model.encode(["hello", "world", "foo bar"])
        assert embs.shape == (3, 16)

    def test_l2_normalised(self, model_path):
        model = Model.load(model_path)
        embs = model.encode(["hello world", "testing"])
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_deterministic(self, model_path):
        model = Model.load(model_path)
        embs1 = model.encode(["hello world"])
        embs2 = model.encode(["hello world"])
        np.testing.assert_array_equal(embs1, embs2)

    def test_different_sentences_differ(self, model_path):
        model = Model.load(model_path)
        embs = model.encode(["hello world", "completely unrelated text about quantum physics"])
        assert not np.allclose(embs[0], embs[1])

    def test_empty_string(self, model_path):
        model = Model.load(model_path)
        embs = model.encode([""])
        assert embs.shape == (1, 16)


class TestSimilarity:
    def test_similarity_shape(self, model_path):
        model = Model.load(model_path)
        sim = model.similarity(["hello", "world"], ["foo", "bar", "baz"])
        assert sim.shape == (2, 3)

    def test_self_similarity_is_one(self, model_path):
        model = Model.load(model_path)
        sim = model.similarity(["hello world"], ["hello world"])
        np.testing.assert_allclose(sim[0, 0], 1.0, atol=1e-5)
