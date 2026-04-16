"""Export a compressed EmbeddingModel checkpoint to nanoembed's packed .npz format."""


from pathlib import Path

import numpy as np

from models.io import load_model
from models.model import CodebookQuantizedWeights, EmbeddingModel, UniformQuantizedWeights
from nanoembed.packing import pack


def export_to_nanoembed(
    model: EmbeddingModel,
    output_path: str | Path,
) -> Path:
    """Convert a quantized EmbeddingModel to a packed nanoembed .npz checkpoint.

    Supports both UniformQuantizedWeights and CodebookQuantizedWeights at any
    bit width. The codebook is stored in the checkpoint so nanoembed can decode
    without knowing the quantization method.
    """
    from nanoembed.checkpoint import save_checkpoint

    w = model.weights
    if not isinstance(w, (UniformQuantizedWeights, CodebookQuantizedWeights)):
        raise TypeError(
            f"Expected quantized weights, got {type(w).__name__}. "
            "Run a quantization step first."
        )

    int_data = w.int_data.numpy().astype(np.uint8)
    packed_codes = pack(int_data, w.bits)
    scales = w.scales.squeeze(-1).numpy() if w.scales.ndim > 1 else w.scales.numpy()

    # Build the codebook array for nanoembed
    if isinstance(w, CodebookQuantizedWeights):
        codebook = w.codebook.numpy()
    else:
        # Uniform: reconstruct the implicit codebook from q_max
        q_max = (1 << (w.bits - 1)) - 1
        levels = np.arange(-q_max - 1, q_max + 1, dtype=np.float32)
        codebook = levels / q_max

    source_model = model.provenance[0] if model.provenance else ""
    method = "turbo-lloyd" if isinstance(w, CodebookQuantizedWeights) else "uniform"

    path = save_checkpoint(
        path=output_path,
        packed_codes=packed_codes,
        scales=scales,
        codebook=codebook,
        embed_dim=model.embed_dim,
        vocab_size=model.vocab_size,
        bits=w.bits,
        tokenizer_name=model.tokenizer_name,
        method=method,
        source_model=source_model,
    )

    size_mb = path.stat().st_size / 1e6
    print(f"  exported: {path} ({size_mb:.1f} MB)")
    return path
