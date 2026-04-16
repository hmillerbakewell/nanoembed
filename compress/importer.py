"""Import pre-trained static embedding models into our EmbeddingModel format.

Supports:
  - model2vec StaticModels (minishlab/potion-*, blobbybob/*)
  - SentenceTransformer StaticEmbedding models (stephantulkens/NIFE-*)
"""


from pathlib import Path

import numpy as np
import torch

from models.io import save_model
from models.model import DenseWeights, EmbeddingModel


def import_model2vec(model_id: str, cache_dir: str | None = None) -> EmbeddingModel:
    """Load a model2vec StaticModel and wrap its embedding table as an EmbeddingModel.

    The tokenizer is loaded via HuggingFace AutoTokenizer from the same model repo,
    so our pipeline's existing `make_local_encoder` can re-instantiate it consistently.

    Args:
        model_id: HuggingFace id, e.g. "minishlab/potion-multilingual-128M".
        cache_dir: HF cache directory.

    Returns:
        An EmbeddingModel with DenseWeights from the static model's embedding matrix.
    """

    from model2vec import StaticModel

    print(f"Loading {model_id}...")
    m = StaticModel.from_pretrained(model_id)

    # Extract embedding matrix — handle both numpy and torch storage
    raw = m.embedding
    if isinstance(raw, torch.Tensor):
        weights_np = raw.detach().cpu().numpy()
    else:
        weights_np = np.asarray(raw)

    weights = torch.from_numpy(weights_np).float()
    vocab_size, embed_dim = weights.shape
    print(f"  vocab_size={vocab_size}, embed_dim={embed_dim}, "
          f"size={weights.nelement() * 4 / 1e6:.1f} MB")

    # Fold per-token pooling weights (model2vec importance scores) into the
    # embedding rows. This bakes the weighting into the row norms so downstream
    # quantization scales capture both magnitude and importance in one number.
    if m.weights is not None:
        tw = torch.from_numpy(np.asarray(m.weights)).float()
        weights = weights * tw.unsqueeze(-1)
        print(f"  folded {tw.shape[0]} token_weights into embeddings")

    return EmbeddingModel(
        weights=DenseWeights(weights),
        token_to_row=torch.arange(vocab_size),
        old_to_new={i: i for i in range(vocab_size)},
        tokenizer_name=model_id,
        embed_dim=embed_dim,
        provenance=[model_id.replace("/", "_")],
    )


def import_sentence_transformer(model_id: str) -> EmbeddingModel:
    """Load a SentenceTransformer static embedding model (e.g. NIFE).

    These use StaticEmbedding with an EmbeddingBag, not model2vec's format.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading {model_id}...")
    st = SentenceTransformer(model_id, device="cpu")

    static_module = st[0]
    weight = static_module.embedding.weight.detach().float()
    vocab_size, embed_dim = weight.shape
    print(f"  vocab_size={vocab_size}, embed_dim={embed_dim}, "
          f"size={weight.nelement() * 4 / 1e6:.1f} MB")

    # Fold per-token importance weights into embedding rows (if present)
    if hasattr(static_module, 'token_weights') and static_module.token_weights is not None:
        tw = static_module.token_weights.detach().float()
        weight = weight * tw.unsqueeze(-1)
        print(f"  folded {tw.shape[0]} token_weights into embeddings")

    return EmbeddingModel(
        weights=DenseWeights(weight),
        token_to_row=torch.arange(vocab_size),
        old_to_new={i: i for i in range(vocab_size)},
        tokenizer_name=model_id,
        embed_dim=embed_dim,
        provenance=[model_id.replace("/", "_")],
    )


def import_model(model_id: str, cache_dir: str | None = None) -> EmbeddingModel:
    """Import any supported static model. Auto-detects model2vec vs SentenceTransformer."""
    try:
        return import_model2vec(model_id, cache_dir=cache_dir)
    except Exception:
        return import_sentence_transformer(model_id)


def import_and_save(model_id: str, checkpoint_dir: Path, cache_dir: str | None = None) -> Path:
    """Import a model and save it to disk as the base checkpoint."""
    model = import_model(model_id, cache_dir=cache_dir)
    return save_model(model, checkpoint_dir)
