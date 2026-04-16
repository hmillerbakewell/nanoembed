"""PCA dimension reduction on the weight matrix."""


import torch
import torch.nn.functional as F

from models.model import DenseWeights, EmbeddingModel


def pca(model: EmbeddingModel, dim: int) -> EmbeddingModel:
    """Project weight matrix to top-`dim` principal components."""
    weights = model.weights.to_float()
    mean = weights.mean(dim=0)
    centered = weights - mean
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:dim].T
    projected = F.normalize(projected, dim=-1)

    print(f"  pca: {model.embed_dim}d → {dim}d "
          f"({model.size_mb:.1f} MB → {projected.nelement() * 4 / 1e6:.1f} MB)")

    return EmbeddingModel(
        weights=DenseWeights(projected),
        token_to_row=model.token_to_row.clone(),
        old_to_new=model.old_to_new,
        tokenizer_name=model.tokenizer_name,
        embed_dim=dim,
        provenance=model.provenance + [f"pca-{dim}d"],
    )
