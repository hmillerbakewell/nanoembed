"""Compression runner: apply named transforms to an embedding model.

Every step is a pure function: (EmbeddingModel, **kwargs) -> EmbeddingModel.
No teacher data, no evaluation — just compression.
"""


import time
from pathlib import Path
from typing import Any, Callable

from models.io import load_model, save_model
from models.model import EmbeddingModel

from .config import CompressorConfig
from .recipes import RECIPES


# ── Step registry ─────────────────────────────────────────────────────────────

StepSpec = tuple[str, dict[str, Any]]
StepFn = Callable[..., EmbeddingModel]

_STEPS: dict[str, StepFn] = {}


def register_step(name: str, fn: StepFn) -> None:
    _STEPS[name] = fn


def _register_defaults() -> None:
    from .cluster import cluster_global
    from .pca import pca
    from .pq import pq
    from .quantize import quantize

    for name, fn in [
        ("cluster_global", cluster_global),
        ("pca", pca),
        ("pq", pq),
        ("quantize", quantize),
    ]:
        register_step(name, fn)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(
    start: EmbeddingModel | str | Path,
    steps: list[StepSpec],
    checkpoint_dir: str | Path = "checkpoints",
) -> EmbeddingModel:
    """Apply a sequence of named compression steps to a model.

    Each step transforms the model and saves a checkpoint. Returns the final model.
    """
    if not _STEPS:
        _register_defaults()

    if isinstance(start, (str, Path)):
        print(f"Loading starting model from {start}")
        model = load_model(start)
    else:
        model = EmbeddingModel(
            weights=start.weights,
            token_to_row=start.token_to_row,
            old_to_new=start.old_to_new,
            tokenizer_name=start.tokenizer_name,
            embed_dim=start.embed_dim,
            provenance=list(start.provenance),
        )

    print(f"\nStart: {model.name} ({model.size_mb:.2f} MB)")

    for step_name, kwargs in steps:
        if step_name not in _STEPS:
            raise ValueError(
                f"Unknown step: {step_name}. Available: {sorted(_STEPS.keys())}"
            )

        print(f"\n  {step_name} {kwargs if kwargs else ''}...", end="", flush=True)
        t0 = time.perf_counter()
        model = _STEPS[step_name](model=model, **kwargs)
        dt = time.perf_counter() - t0
        print(f" {model.size_mb:.2f} MB ({dt:.1f}s)")

        save_model(model, checkpoint_dir)

    # Summary
    print(f"\nDone: {model.name} ({model.size_mb:.2f} MB)")
    return model


# ── External model compression ────────────────────────────────────────────────

def compress_external_model(cfg: CompressorConfig) -> list[Path]:
    """Import an external model and apply each configured compression recipe."""
    from .importer import import_and_save

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Importing {cfg.model_id}")
    base_path = import_and_save(
        cfg.model_id,
        checkpoint_dir=cfg.checkpoint_dir,
        cache_dir=cfg.cache_dir,
    )
    print(f"Base: {base_path}")
    saved: list[Path] = [base_path]

    for recipe_name in cfg.recipes:
        if recipe_name not in RECIPES:
            raise ValueError(
                f"Unknown recipe '{recipe_name}'. Available: {sorted(RECIPES.keys())}"
            )
        steps = RECIPES[recipe_name]
        if not steps:
            continue

        print(f"\n--- {recipe_name} ---")
        result = run_pipeline(
            start=base_path,
            steps=steps,
            checkpoint_dir=cfg.checkpoint_dir,
        )
        saved.append(cfg.checkpoint_dir / f"{result.name}.pt")

    return saved
