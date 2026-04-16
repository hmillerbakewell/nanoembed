"""Packed checkpoint format for quantized static embedding models.

Stored as a standard numpy .npz archive containing:

    packed_codes    (N, packed_cols)  uint8       bit-packed quantization codes
    scales          (N,)             float32     per-row decode scales
    codebook        (2^bits,)        float32     reconstruction levels
    embed_dim       scalar                       embedding dimension D
    vocab_size      scalar                       number of token rows N
    bits            scalar                       bit width per element (1-8)
    tokenizer_name  string                       HuggingFace tokenizer identifier
    method          string                       quantization method
    source_model    string                       original model id (provenance)
"""


from pathlib import Path

import numpy as np


def save_checkpoint(
    path: str | Path,
    packed_codes: np.ndarray,
    scales: np.ndarray,
    codebook: np.ndarray,
    embed_dim: int,
    vocab_size: int,
    bits: int,
    tokenizer_name: str,
    method: str = "turbo-lloyd",
    source_model: str = "",
) -> Path:
    """Write a packed .npz checkpoint."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    np.savez(
        str(path),
        packed_codes=packed_codes,
        scales=scales,
        codebook=codebook.astype(np.float32),
        embed_dim=np.int64(embed_dim),
        vocab_size=np.int64(vocab_size),
        bits=np.int64(bits),
        tokenizer_name=np.array(tokenizer_name),
        method=np.array(method),
        source_model=np.array(source_model),
    )
    return path


def load_checkpoint(path: str | Path) -> dict:
    """Load a packed .npz checkpoint and return its contents as a dict."""
    data = np.load(str(path), allow_pickle=True)

    # Back-compat: old format stored codebook_val (scalar) instead of codebook (array)
    if "codebook" in data:
        codebook = data["codebook"]
    else:
        codebook = np.array([-float(data["codebook_val"]), float(data["codebook_val"])],
                            dtype=np.float32)

    return {
        "packed_codes": data["packed_codes"],
        "scales": data["scales"],
        "codebook": codebook,
        "embed_dim": int(data["embed_dim"]),
        "vocab_size": int(data["vocab_size"]),
        "bits": int(data.get("bits", 1)),
        "tokenizer_name": str(data["tokenizer_name"]),
        "method": str(data.get("method", "turbo-lloyd")),
        "source_model": str(data.get("source_model", "")),
    }
