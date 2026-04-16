"""Save/load EmbeddingModel checkpoints."""


from pathlib import Path

import torch

from .model import (
    CodebookQuantizedWeights,
    DenseWeights,
    EmbeddingModel,
    PQWeights,
    UniformQuantizedWeights,
)


def save_model(model: EmbeddingModel, directory: str | Path = "checkpoints") -> Path:
    """Save an EmbeddingModel to disk, named by its provenance."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.name}.pt"

    w = model.weights
    if isinstance(w, CodebookQuantizedWeights):
        weight_data = {
            "type": "codebook_quantized",
            "int_data": w.int_data,
            "scales": w.scales,
            "bits": w.bits,
            "codebook": w.codebook,
        }
    elif isinstance(w, UniformQuantizedWeights):
        weight_data = {
            "type": "uniform_quantized",
            "int_data": w.int_data,
            "scales": w.scales,
            "bits": w.bits,
        }
    elif isinstance(w, PQWeights):
        weight_data = {
            "type": "pq",
            "codes": w.codes,
            "codebooks": w.codebooks,
        }
    else:
        weight_data = {"type": "dense", "data": w.data}

    d = {
        "weight_data": weight_data,
        "token_to_row": model.token_to_row,
        "old_to_new": model.old_to_new,
        "tokenizer_name": model.tokenizer_name,
        "embed_dim": model.embed_dim,
        "provenance": model.provenance,
    }
    torch.save(d, path)
    return path


def load_model(path: str | Path) -> EmbeddingModel:
    """Load an EmbeddingModel from a checkpoint."""
    d = torch.load(path, map_location="cpu", weights_only=False)

    wd = d["weight_data"]
    wtype = wd["type"]

    if wtype == "codebook_quantized":
        weights = CodebookQuantizedWeights(
            int_data=wd["int_data"], scales=wd["scales"],
            bits=wd["bits"], codebook=wd["codebook"],
        )
    elif wtype == "uniform_quantized":
        weights = UniformQuantizedWeights(
            int_data=wd["int_data"], scales=wd["scales"], bits=wd["bits"],
        )
    elif wtype == "quantized":
        # Back-compat: old checkpoints used a single "quantized" type
        int_data = wd.get("int_data", wd.get("int8_data"))
        bits = wd.get("bits", 8)
        if wd.get("has_codebook", False) or wd.get("codebook") is not None:
            codebook = wd["codebook"]
            if codebook is None:
                from compress.lloyd_max import lloyd_codebook
                codebook = lloyd_codebook(bits) / (int_data.shape[1] ** 0.5)
            weights = CodebookQuantizedWeights(
                int_data=int_data, scales=wd["scales"], bits=bits, codebook=codebook,
            )
        else:
            weights = UniformQuantizedWeights(
                int_data=int_data, scales=wd["scales"], bits=bits,
            )
    elif wtype == "pq":
        weights = PQWeights(codes=wd["codes"], codebooks=wd["codebooks"])
    else:
        weights = DenseWeights(data=wd["data"])

    return EmbeddingModel(
        weights=weights,
        token_to_row=d["token_to_row"],
        old_to_new=d["old_to_new"],
        tokenizer_name=d["tokenizer_name"],
        embed_dim=d["embed_dim"],
        provenance=d["provenance"],
    )
