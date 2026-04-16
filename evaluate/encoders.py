"""Encoder factories for evaluating external models.

Wrap a model's native encode method as an Encoder callable for use with
evaluate.metrics functions. Unlike compress.importer, these don't convert
the model to our format — they just wrap the native encode path.
"""


import numpy as np

from evaluate.metrics import Encoder


def make_m2v_native_encoder(model_id: str) -> tuple[Encoder, float]:
    """Wrap a model2vec StaticModel's native encode as an Encoder.

    Returns (encoder, size_mb) where size_mb is the embedding table size.
    """
    from model2vec import StaticModel

    m = StaticModel.from_pretrained(model_id)

    raw = m.embedding
    size_bytes = (
        raw.nbytes if hasattr(raw, "nbytes") else np.asarray(raw).nbytes
    )
    size_mb = size_bytes / 1e6

    def encode(sentences: list[str]) -> np.ndarray:
        return np.asarray(m.encode(sentences))

    return encode, size_mb


def make_transformer_encoder(model_id: str, cache_dir: str | None = None) -> tuple[Encoder, float]:
    """Wrap a SentenceTransformer as an Encoder.

    Used for transformer baselines (bge-m3, LaBSE, multilingual-e5, etc.).
    Size is reported as the total parameter footprint in fp32 bytes.
    """
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_id, cache_folder=cache_dir)

    # Total parameter size in fp32 bytes
    param_bytes = sum(
        p.nelement() * p.element_size() for p in m.parameters()
    )
    size_mb = param_bytes / 1e6

    def encode(sentences: list[str]) -> np.ndarray:
        return np.asarray(
            m.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        )

    return encode, size_mb
