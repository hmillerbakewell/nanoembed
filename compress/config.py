"""Configuration for compression-only runs on external pretrained static models."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CompressorConfig:
    """Parameters for compress_external_model().

    Attributes:
        model_id: HuggingFace model id to import, e.g. 'minishlab/potion-multilingual-128M'.
        recipes: List of recipe names to apply (keys in compressor.recipes.RECIPES).
        checkpoint_dir: Where to save the imported base and each compressed variant.
        cache_dir: HuggingFace cache directory for downloads (None = HF default).
    """

    model_id: str = "minishlab/potion-multilingual-128M"
    recipes: tuple[str, ...] = field(default_factory=tuple)
    checkpoint_dir: Path = Path("checkpoints/compressed")
    cache_dir: str | None = None

    @property
    def model_slug(self) -> str:
        """Filename-friendly version of the model id."""
        return self.model_id.replace("/", "_")
