# nanoembed

Tiny quantized static embeddings. Load a 1-bit model (~18 MB), encode text, get vectors. No torch, no GPU.

## Install

```bash
pip install nanoembed
```

## Usage

```python
from nanoembed import TurboModel

model = TurboModel.load("model.npz")
embeddings = model.encode(["hello world", "مرحبا بالعالم"])
similarity = float(embeddings[0] @ embeddings[1])
```

## What it does

1. Tokenizes text (HuggingFace `tokenizers` library)
2. Looks up each token's 1-bit packed embedding row
3. Decodes: `scale * codebook_val * (2 * bit - 1)`
4. Mean-pools across tokens
5. L2-normalizes

The result is a dense float32 embedding vector suitable for cosine similarity search.

## Dependencies

- `numpy` — decoding and pooling
- `tokenizers` — text tokenization (the Rust-backed HuggingFace library, not the full `transformers` package)

## Model format

Models are standard `.npz` files containing:

| Field | Shape | Description |
|---|---|---|
| `packed_codes` | `(N, D/8)` | Bit-packed quantization codes |
| `scales` | `(N,)` | Per-row decode scales |
| `codebook_val` | scalar | Signed reconstruction level |
| `embed_dim` | scalar | Embedding dimension D |
| `vocab_size` | scalar | Number of tokens N |
| `tokenizer_name` | string | HuggingFace tokenizer to load |
| `bits` | scalar | Bit width (1, 2, 3, or 4) |
| `method` | string | Quantization method used |
| `source_model` | string | Original model this was compressed from |
