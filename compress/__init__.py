"""Compress static embedding models.

Transforms: PCA, product quantization, scalar quantization (uniform and TurboQuant),
k-means clustering. Plus model importers and named recipes.

Entry points:
  - compress.runner.compress_external_model(cfg)
  - compress.runner.run_pipeline(start, steps)
  - compress.importer.import_model(model_id)
"""

from .config import CompressorConfig
from .runner import compress_external_model, run_pipeline

__all__ = ["CompressorConfig", "compress_external_model", "run_pipeline"]
