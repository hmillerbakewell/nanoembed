"""MTEB adapter: wrap our Encoder callable for use with mteb >= 2.12.

MTEB 2.12+ uses a runtime-checkable EncoderProtocol that requires:
  - encode(inputs: DataLoader[BatchedInput], *, task_metadata, hf_split, ...)
  - similarity(embeddings1, embeddings2) -> matrix
  - similarity_pairwise(embeddings1, embeddings2) -> vector

We implement these by wrapping a simple ``list[str] -> np.ndarray`` callable.

Usage:
    from evaluate.mteb_adapter import MTEBModelWrapper
    from evaluate.metrics import make_local_encoder
    from models.io import load_model

    model = load_model("checkpoints/some_model.pt")
    encoder = make_local_encoder(model, max_length=256)
    wrapper = MTEBModelWrapper(encoder)

    import mteb
    tasks = mteb.get_tasks(tasks=["STS12", ...])
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(wrapper, output_folder="results/")
"""

from typing import Any, Callable
from mteb.models.model_meta import ModelMeta
from mteb.models.model_meta import ScoringFunction

import numpy as np
import torch
from torch.utils.data import DataLoader


Encoder = Callable[[list[str]], np.ndarray]


class MTEBModelWrapper:
    """Adapts a simple Encoder callable to the MTEB 2.12+ EncoderProtocol."""

    def __init__(
        self,
        encoder: Encoder,
        model_name: str = "static_model",
        revision: str | None = None,
        *,
        device: str | None = None,
        batch_size: int = 512,
        **kwargs: Any,
    ) -> None:
        self.encoder = encoder
        self.model_name = model_name
        self.revision = revision
        self._batch_size = batch_size

    def encode(
        self,
        inputs: DataLoader,
        *,
        task_metadata: Any = None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode batched inputs from a DataLoader."""
        all_embs: list[np.ndarray] = []
        for batch in inputs:
            # BatchedInput is a dict with "text" key (list of strings)
            if isinstance(batch, dict):
                sentences = batch.get("text", batch.get("sentences", []))
            elif isinstance(batch, (list, tuple)):
                sentences = list(batch)
            else:
                sentences = [str(batch)]

            if sentences:
                all_embs.append(self.encoder(sentences))

        if not all_embs:
            return np.array([])
        return np.concatenate(all_embs, axis=0)

    def similarity(
        self,
        embeddings1: np.ndarray | torch.Tensor,
        embeddings2: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity matrix between two sets of embeddings."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        e1 = torch.nn.functional.normalize(embeddings1.float(), dim=-1)
        e2 = torch.nn.functional.normalize(embeddings2.float(), dim=-1)
        return e1 @ e2.T

    def similarity_pairwise(
        self,
        embeddings1: np.ndarray | torch.Tensor,
        embeddings2: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity between corresponding pairs."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        e1 = torch.nn.functional.normalize(embeddings1.float(), dim=-1)
        e2 = torch.nn.functional.normalize(embeddings2.float(), dim=-1)
        return (e1 * e2).sum(dim=-1)

    @property
    def mteb_model_meta(self):
        return ModelMeta(
            loader=None,
            name=self.model_name,
            revision=self.revision,
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False,
            training_datasets=None,
        )