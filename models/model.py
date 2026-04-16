"""Core data types for static embedding models.

This module defines only the *representation* of models and their weight
formats. It knows how to decode (index into weights) but not how to encode
(quantize, compress). Encoding lives in compress/.

Weight types:
  - DenseWeights: fp32 matrix
  - QuantizedWeights: per-row scalar-quantized (uniform or codebook)
  - PQWeights: product-quantized (per-sub-vector codebook indices)

All three implement the same interface: shape, size_bytes, __getitem__, to_float.
"""


from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


# ── Weight types ──────────────────────────────────────────────────────────────

@runtime_checkable
class Weights(Protocol):
    """Anything that can be indexed like a tensor and reports its shape and size."""

    @property
    def shape(self) -> torch.Size: ...

    @property
    def size_bytes(self) -> int: ...

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor: ...


class DenseWeights:
    """Standard fp32 weight matrix."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def size_bytes(self) -> int:
        return self.data.nelement() * self.data.element_size()

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.data[idx]

    def to_float(self) -> torch.Tensor:
        return self.data


class UniformQuantizedWeights:
    """Per-row symmetric scalar quantization with evenly-spaced levels.

    int_data holds signed values in [-q_max-1, q_max].
    Decoded as ``scales * int_data / q_max``.
    """

    def __init__(self, int_data: torch.Tensor, scales: torch.Tensor,
                 bits: int = 8) -> None:
        self.int_data = int_data
        self.scales = scales
        self.bits = bits
        self._q_max = (1 << (bits - 1)) - 1

    @property
    def shape(self) -> torch.Size:
        return self.int_data.shape

    @property
    def size_bytes(self) -> int:
        numel = self.int_data.nelement()
        return (numel * self.bits + 7) // 8 + self.scales.nelement() * 4

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.scales[idx] * (self.int_data[idx].float() / self._q_max)

    def to_float(self) -> torch.Tensor:
        return self.scales * (self.int_data.float() / self._q_max)


class CodebookQuantizedWeights:
    """Per-row quantization with a fixed codebook (e.g. Max-Lloyd levels).

    int_data holds unsigned indices in [0, 2^bits).
    Decoded as ``scales * codebook[int_data]``.
    """

    def __init__(self, int_data: torch.Tensor, scales: torch.Tensor,
                 bits: int, codebook: torch.Tensor) -> None:
        self.int_data = int_data
        self.scales = scales
        self.bits = bits
        self.codebook = codebook

    @property
    def shape(self) -> torch.Size:
        return self.int_data.shape

    @property
    def size_bytes(self) -> int:
        numel = self.int_data.nelement()
        return (numel * self.bits + 7) // 8 + self.scales.nelement() * 4

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.scales[idx] * self.codebook[self.int_data[idx].long()]

    def to_float(self) -> torch.Tensor:
        return self.scales * self.codebook[self.int_data.long()]


# Back-compat alias — old code references QuantizedWeights as a union
QuantizedWeights = UniformQuantizedWeights | CodebookQuantizedWeights


class PQWeights:
    """Product-quantized weights: M sub-vector codebook indices per row.

    Reconstruction: ``concat(codebooks[m, codes[i, m]] for m in range(M))``.
    Limited to K ≤ 256 (uint8 codes).
    """

    def __init__(self, codes: torch.Tensor, codebooks: torch.Tensor) -> None:
        assert codes.dtype == torch.uint8, "codes must be uint8"
        self.codes = codes
        self.codebooks = codebooks
        self._m = codes.shape[1]
        self._d_sub = codebooks.shape[2]

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.codes.shape[0], self._m * self._d_sub])

    @property
    def size_bytes(self) -> int:
        return self.codes.nelement() + self.codebooks.nelement() * 4

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        selected_codes = self.codes[idx].long()
        parts = [self.codebooks[m, selected_codes[..., m]] for m in range(self._m)]
        return torch.cat(parts, dim=-1)

    def to_float(self) -> torch.Tensor:
        parts = [self.codebooks[m, self.codes[:, m].long()] for m in range(self._m)]
        return torch.cat(parts, dim=-1)


# ── Core model ────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingModel:
    """A static embedding model at any compression stage.

    For flat models:  weights is (vocab_size, D), token_to_row is identity.
    For clustered:    weights is (K, D), token_to_row maps token → centroid.
    """

    weights: DenseWeights | QuantizedWeights | PQWeights
    token_to_row: torch.Tensor
    old_to_new: dict[int, int]
    tokenizer_name: str
    embed_dim: int
    provenance: list[str] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return self.weights.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.token_to_row.shape[0]

    @property
    def size_bytes(self) -> int:
        return self.weights.size_bytes

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6

    @property
    def name(self) -> str:
        return "_".join(self.provenance) if self.provenance else "unnamed"

    def embed_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings, resolving through token_to_row mapping."""
        row_ids = self.token_to_row[input_ids]
        tok_embs = self.weights[row_ids]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (tok_embs * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return torch.nn.functional.normalize(summed / lengths, dim=-1)
