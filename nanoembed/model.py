"""Quantized static embedding model.

Loads a packed .npz checkpoint and encodes sentences to L2-normalised dense
embeddings using only numpy and the HuggingFace tokenizers library.

No torch dependency. No GPU required.
"""


from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .checkpoint import load_checkpoint
from .packing import unpack


@dataclass(frozen=True)
class ModelInfo:
    """Read-only metadata about a loaded model."""
    source_model: str
    tokenizer_name: str
    vocab_size: int
    embed_dim: int
    bits: int
    method: str
    logical_size_mb: float

    def __str__(self) -> str:
        return (
            f"{self.source_model or 'unknown'}\n"
            f"  {self.vocab_size:,} tokens × {self.embed_dim}d, "
            f"{self.bits}-bit {self.method}, "
            f"{self.logical_size_mb:.1f} MB logical"
        )


class Model:
    """Quantized static embedding model."""

    def __init__(
        self,
        packed_codes: np.ndarray,
        scales: np.ndarray,
        codebook: np.ndarray,
        tokenizer: Tokenizer,
        embed_dim: int,
        vocab_size: int,
        bits: int,
        method: str = "",
        source_model: str = "",
    ) -> None:
        self.packed_codes = packed_codes    # (N, packed_cols) uint8
        self.scales = scales                # (N,) float32
        self.codebook = codebook            # (2^bits,) float32
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.bits = bits
        self._method = method
        self._source_model = source_model

    @classmethod
    def load(cls, path: str | Path) -> Model:
        """Load from a packed .npz checkpoint.

        The tokenizer is fetched from HuggingFace by the name stored in the
        checkpoint. First load needs internet; subsequent loads use the cache.
        """
        ckpt = load_checkpoint(path)

        tokenizer = Tokenizer.from_pretrained(ckpt["tokenizer_name"])
        tokenizer.no_padding()
        tokenizer.no_truncation()

        return cls(
            packed_codes=ckpt["packed_codes"],
            scales=ckpt["scales"],
            codebook=ckpt["codebook"],
            tokenizer=tokenizer,
            embed_dim=ckpt["embed_dim"],
            vocab_size=ckpt["vocab_size"],
            bits=ckpt["bits"],
            method=ckpt["method"],
            source_model=ckpt["source_model"],
        )

    @property
    def info(self) -> ModelInfo:
        """Metadata about this model."""
        nbits = self.vocab_size * self.embed_dim * self.bits
        scale_bytes = self.vocab_size * 4
        logical_bytes = (nbits + 7) // 8 + scale_bytes
        return ModelInfo(
            source_model=self._source_model,
            tokenizer_name=self.tokenizer.to_str()[:50] + "...",
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            bits=self.bits,
            method=self._method,
            logical_size_mb=logical_bytes / 1e6,
        )

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, sentences: list[str], batch_size: int = 256) -> np.ndarray:
        """Encode sentences to L2-normalised embeddings.

        Returns:
            (len(sentences), embed_dim) float32 array.
        """
        all_embs: list[np.ndarray] = []
        _iter = range(0, len(sentences), batch_size)

        try:
            from tqdm.auto import tqdm
            if len(sentences) > batch_size:
                _iter = tqdm(_iter, desc="encoding", unit="batch")
        except ImportError:
            pass

        for i in _iter:
            batch = sentences[i : i + batch_size]
            encoded = self.tokenizer.encode_batch(batch, add_special_tokens=False)
            ids_list = [enc.ids for enc in encoded]

            max_len = max(len(ids) for ids in ids_list)
            if max_len == 0:
                all_embs.append(np.zeros((len(batch), self.embed_dim), dtype=np.float32))
                continue

            padded_ids = np.zeros((len(batch), max_len), dtype=np.int64)
            mask = np.zeros((len(batch), max_len), dtype=np.float32)
            for j, ids in enumerate(ids_list):
                padded_ids[j, : len(ids)] = ids
                mask[j, : len(ids)] = 1.0

            tok_embs = self._decode_rows(padded_ids)
            tok_embs = tok_embs * mask[:, :, np.newaxis]
            summed = tok_embs.sum(axis=1)
            lengths = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
            mean_pooled = summed / lengths

            norms = np.maximum(np.linalg.norm(mean_pooled, axis=1, keepdims=True), 1e-8)
            all_embs.append(mean_pooled / norms)

        return np.concatenate(all_embs, axis=0)

    def similarity(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        """Cosine similarity between every pair from texts_a and texts_b.

        Returns:
            (len(texts_a), len(texts_b)) float32 similarity matrix.
        """
        return self.encode(texts_a) @ self.encode(texts_b).T

    # ── Internal ──────────────────────────────────────────────────────────────

    def _decode_rows(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode packed rows for the given token IDs.

        Args:
            token_ids: (B, L) int64 array.

        Returns:
            (B, L, D) float32 decoded embeddings.
        """
        B, L = token_ids.shape
        flat_ids = token_ids.ravel()

        packed = self.packed_codes[flat_ids]
        indices = unpack(packed, self.bits, self.embed_dim)

        decoded = self.scales[flat_ids, np.newaxis] * self.codebook[indices]
        return decoded.reshape(B, L, self.embed_dim)
