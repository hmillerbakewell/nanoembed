"""Vocabulary clustering: collapse tokens to K centroids via k-means."""

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from models.model import DenseWeights, EmbeddingModel


def cluster_global(model: EmbeddingModel, k: int) -> EmbeddingModel:
    """Cluster all token embeddings to K centroids via global k-means."""
    # Resolve token embeddings through token_to_row (dequantizes if needed)
    embeddings = model.weights[model.token_to_row].numpy()

    if k >= len(embeddings):
        print(f"  cluster: k={k} >= vocab_size={len(embeddings)}, skipping")
        return EmbeddingModel(
            weights=DenseWeights(torch.from_numpy(embeddings)),
            token_to_row=model.token_to_row.clone(),
            old_to_new=model.old_to_new,
            tokenizer_name=model.tokenizer_name,
            embed_dim=model.embed_dim,
            provenance=model.provenance + [f"cluster-k{k}"],
        )

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
    assignments = kmeans.fit_predict(embeddings)
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    token_to_row = torch.from_numpy(assignments).long()

    print(f"  cluster: {model.num_rows} rows → {k} centroids "
          f"({model.size_mb:.1f} MB → {centroids.nelement() * 4 / 1e6:.1f} MB)")

    return EmbeddingModel(
        weights=DenseWeights(centroids),
        token_to_row=token_to_row,
        old_to_new=model.old_to_new,
        tokenizer_name=model.tokenizer_name,
        embed_dim=model.embed_dim,
        provenance=model.provenance + [f"cluster-k{k}"],
    )
