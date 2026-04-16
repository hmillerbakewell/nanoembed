"""nanoembed — tiny quantized static embeddings.

Load a model, encode text, get vectors. No torch, no GPU.

Usage:
    import nanoembed

    model = nanoembed.load("path/to/model.npz")  # local file
    model = nanoembed.load("org/model-name")      # from HuggingFace (pip install nanoembed[hf])

    embeddings = model.encode(["hello world", "مرحبا بالعالم"])
    print(model.info)
"""


from pathlib import Path

from .model import Model, ModelInfo


def load(path_or_id: str) -> Model:
    """Load a nanoembed model from a local file or HuggingFace model ID.

    Args:
        path_or_id: Either a local .npz file path, or a HuggingFace model ID
            (e.g. "org/model-name"). HuggingFace downloads require
            ``pip install nanoembed[hf]``.

    Returns:
        A ready-to-use Model.
    """
    path = Path(path_or_id)
    if path.exists():
        return Model.load(path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"'{path_or_id}' is not a local file. To download from HuggingFace, "
            "install huggingface_hub: pip install nanoembed[hf]"
        ) from None

    local = hf_hub_download(path_or_id, filename="model.npz")
    return Model.load(local)


__all__ = ["Model", "ModelInfo", "load"]
