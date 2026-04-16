"""Product quantization step.

Splits each D-dim weight vector into M sub-vectors of D/M dims, fits an
independent k-means codebook per sub-vector, and replaces the row with M
codebook indices. Unlike global clustering (which collapses a whole row to
one centroid index), PQ keeps per-sub-vector resolution so it can represent
K^M distinct encodings with only M bytes per row.
"""


import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from models.model import DenseWeights, EmbeddingModel, PQWeights


def _encode_pq(weights: DenseWeights, num_subvectors: int,
               centroids_per_sub: int = 256, seed: int = 42) -> PQWeights:
    """Fit M k-means codebooks on sub-vectors and encode rows as indices."""
    data = weights.data.numpy()
    n, d = data.shape
    if d % num_subvectors != 0:
        raise ValueError(f"embed_dim={d} not divisible by num_subvectors={num_subvectors}")
    if centroids_per_sub > 256:
        raise ValueError("centroids_per_sub must be ≤ 256 (uint8 codes)")

    d_sub = d // num_subvectors
    codes = np.zeros((n, num_subvectors), dtype=np.uint8)
    codebooks = np.zeros((num_subvectors, centroids_per_sub, d_sub), dtype=np.float32)

    for m in range(num_subvectors):
        sub = data[:, m * d_sub:(m + 1) * d_sub]
        km = MiniBatchKMeans(n_clusters=centroids_per_sub, random_state=seed + m, batch_size=1024)
        codes[:, m] = km.fit_predict(sub).astype(np.uint8)
        codebooks[m] = km.cluster_centers_

    return PQWeights(codes=torch.from_numpy(codes), codebooks=torch.from_numpy(codebooks))


def pq(
    model: EmbeddingModel,
    num_subvectors: int,
    centroids_per_sub: int = 256,
) -> EmbeddingModel:
    """Apply product quantization to the weight matrix."""
    dense = DenseWeights(model.weights.to_float())
    pq_weights = _encode_pq(dense, num_subvectors=num_subvectors, centroids_per_sub=centroids_per_sub)

    print(f"  pq: m={num_subvectors} k={centroids_per_sub}  "
          f"{model.size_mb:.2f} MB → {pq_weights.size_bytes / 1e6:.3f} MB")

    return EmbeddingModel(
        weights=pq_weights,
        token_to_row=model.token_to_row.clone(),
        old_to_new=model.old_to_new,
        tokenizer_name=model.tokenizer_name,
        embed_dim=model.embed_dim,
        provenance=model.provenance + [f"pq-m{num_subvectors}-k{centroids_per_sub}"],
    )
