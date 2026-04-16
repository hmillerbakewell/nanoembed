"""Named compression recipes.

Each recipe is a list of (step_name, kwargs) tuples passed to
compress.runner.run_pipeline.
"""


from typing import Any


Recipe = list[tuple[str, dict[str, Any]]]


RECIPES: dict[str, Recipe] = {
    # ── Identity ──────────────────────────────────────────────────────────
    "baseline": [],

    # ── Uniform scalar quantization ───────────────────────────────────────
    "int8": [("quantize", {"bits": 8, "method": "uniform"})],
    "int4": [("quantize", {"bits": 4, "method": "uniform"})],
    "int2": [("quantize", {"bits": 2, "method": "uniform"})],

    # ── TurboQuant (rotation + Max-Lloyd optimal codebook) ────────────────
    "turbo-lloyd-1": [("quantize", {"bits": 1, "method": "turbo-lloyd"})],
    "turbo-lloyd-2": [("quantize", {"bits": 2, "method": "turbo-lloyd"})],
    "turbo-lloyd-3": [("quantize", {"bits": 3, "method": "turbo-lloyd"})],
    "turbo-lloyd-4": [("quantize", {"bits": 4, "method": "turbo-lloyd"})],

    # ── PCA ───────────────────────────────────────────────────────────────
    "pca-128d": [("pca", {"dim": 128})],
    "pca-64d":  [("pca", {"dim": 64})],

    # ── PCA + quantization ────────────────────────────────────────────────
    "pca-128d_int4": [
        ("pca", {"dim": 128}),
        ("quantize", {"bits": 4, "method": "uniform"}),
    ],
    "pca-64d_int4": [
        ("pca", {"dim": 64}),
        ("quantize", {"bits": 4, "method": "uniform"}),
    ],

    # ── Clustering (blobbybob reproduction) ───────────────────────────────
    "blobbybob-style": [
        ("cluster_global", {"k": 2000}),
        ("quantize", {"bits": 8, "method": "uniform"}),
    ],

    # ── Product quantization ──────────────────────────────────────────────
    "pq-m32-k256": [("pq", {"num_subvectors": 32, "centroids_per_sub": 256})],
    "pq-m16-k256": [("pq", {"num_subvectors": 16, "centroids_per_sub": 256})],
    "pq-m8-k256":  [("pq", {"num_subvectors": 8,  "centroids_per_sub": 256})],
    "pq-m4-k256":  [("pq", {"num_subvectors": 4,  "centroids_per_sub": 256})],
    "pq-m32-k64":  [("pq", {"num_subvectors": 32, "centroids_per_sub": 64})],
    "pq-m16-k64":  [("pq", {"num_subvectors": 16, "centroids_per_sub": 64})],

    # ── PCA + product quantization ────────────────────────────────────────
    "pca-128d_pq-m8-k256":  [("pca", {"dim": 128}), ("pq", {"num_subvectors": 8,  "centroids_per_sub": 256})],
    "pca-128d_pq-m16-k256": [("pca", {"dim": 128}), ("pq", {"num_subvectors": 16, "centroids_per_sub": 256})],
    "pca-128d_pq-m32-k256": [("pca", {"dim": 128}), ("pq", {"num_subvectors": 32, "centroids_per_sub": 256})],
    "pca-64d_pq-m8-k256":  [("pca", {"dim": 64}),  ("pq", {"num_subvectors": 8,  "centroids_per_sub": 256})],
    "pca-64d_pq-m16-k256": [("pca", {"dim": 64}),  ("pq", {"num_subvectors": 16, "centroids_per_sub": 256})],
}
