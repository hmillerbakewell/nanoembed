"""Scalar quantization of embedding weight matrices.

Two methods:
  - **uniform**: symmetric per-row scalar quantization with evenly-spaced levels.
  - **turbo-lloyd**: random orthogonal rotation then Max-Lloyd optimal Gaussian
    codebook. The rotation pre-conditions coordinates to ~N(0,1) so the
    codebook levels (derived from the Lloyd-Max algorithm) are near-optimal.
    Orthogonal rotations preserve dot products, so no derotation is needed
    at inference time.
"""


import torch

from models.model import (
    CodebookQuantizedWeights,
    DenseWeights,
    EmbeddingModel,
    UniformQuantizedWeights,
)
from .lloyd_max import lloyd_codebook

_ROTATION_SEED = 42


def _random_orthogonal(dim: int, seed: int = _ROTATION_SEED) -> torch.Tensor:
    """Fixed-seed random orthogonal (D, D) matrix via QR of a Gaussian."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, dim, generator=gen)
    q, _ = torch.linalg.qr(a)
    return q


def _rotate(weights: DenseWeights) -> DenseWeights:
    return DenseWeights(weights.data @ _random_orthogonal(weights.data.shape[1]))


def _quantize_uniform(weights: DenseWeights, bits: int) -> UniformQuantizedWeights:
    """Per-row symmetric scalar quantization with uniform levels."""
    data = weights.data
    q_max = (1 << (bits - 1)) - 1
    scales = data.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    int_data = (data / scales * q_max).round().clamp(-q_max - 1, q_max).to(torch.int8)
    return UniformQuantizedWeights(int_data=int_data, scales=scales, bits=bits)


def _quantize_with_codebook(weights: DenseWeights, bits: int,
                            codebook: torch.Tensor) -> CodebookQuantizedWeights:
    """Quantize by snapping each per-row-normalised value to the nearest codebook entry.

    The row is L2-normalised and scaled to unit variance so coordinates match
    the codebook's target distribution (N(0,1) for Max-Lloyd Gaussian).
    """
    data = weights.data
    N, D = data.shape

    norms = torch.linalg.vector_norm(data, dim=1, keepdim=True).clamp(min=1e-8)

    K = codebook.shape[0]
    chunk_size = max(1, 256 * 1024 * 1024 // (D * K * 4))

    codebook_view = codebook.view(1, 1, -1)
    indices = torch.empty(N, D, dtype=torch.int8)
    sqrt_d = D ** 0.5
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        unit = data[start:end] / norms[start:end]
        scaled = unit * sqrt_d
        dists = (scaled.unsqueeze(-1) - codebook_view).abs()
        indices[start:end] = dists.argmin(dim=-1).to(torch.int8)

    scaled_codebook = codebook / sqrt_d

    return CodebookQuantizedWeights(
        int_data=indices, scales=norms, bits=bits, codebook=scaled_codebook,
    )


def quantize(
    model: EmbeddingModel,
    bits: int = 4,
    method: str = "turbo-lloyd",
) -> EmbeddingModel:
    """Quantize an embedding model's weights.

    Args:
        model: The model to quantize.
        bits: Bit width per element (1-8).
        method: "uniform" for symmetric scalar quantization, or
                "turbo-lloyd" for rotation + Max-Lloyd optimal codebook.

    Returns:
        A new EmbeddingModel with QuantizedWeights.
    """
    if isinstance(model.weights, (UniformQuantizedWeights, CodebookQuantizedWeights)):
        return model

    if method == "uniform":
        tag = f"int{bits}"
        q = _quantize_uniform(model.weights, bits=bits)

    elif method == "turbo-lloyd":
        tag = f"turbo-lloyd-{bits}"
        q = _quantize_with_codebook(_rotate(model.weights), bits, lloyd_codebook(bits))

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'uniform' or 'turbo-lloyd'.")

    return EmbeddingModel(
        weights=q,
        token_to_row=model.token_to_row.clone(),
        old_to_new=model.old_to_new,
        tokenizer_name=model.tokenizer_name,
        embed_dim=model.embed_dim,
        provenance=model.provenance + [tag],
    )
