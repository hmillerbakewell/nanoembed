# nanoembed

Compress static embedding models to 1-bit and run them with numpy, tiny footprint, runs in CPU.

Takes model2vec / NIFE / SentenceTransformer models, applies TurboQuant (rotation + Max-Lloyd optimal codebook), and produces packed `.npz` checkpoints that are ~28x smaller with zero quality loss on MIRACL multilingual retrieval.

This research was funded by [Health Data Avatar](https://healthdataavatar.com).

## Quick start (running the model)

```bash
pip install nanoembed
```

```python
import nanoembed

model = nanoembed.load("model.npz")          # local file
model = nanoembed.load("org/model-name")      # from HuggingFace (pip install nanoembed[hf])

embeddings = model.encode(["hello world", "مرحبا بالعالم"])
sim = model.similarity(["query"], ["doc a", "doc b"])
```

## Compress and evaluate models

Compress potion-multilingual-128M to 1-bit and evaluate on MIRACL Arabic:

```bash
# Install deps
pip install torch model2vec sentence-transformers scikit-learn transformers

# Compress + export
python run_compress.py --model minishlab/potion-multilingual-128M --recipes turbo-lloyd-1 --export

# Evaluate (.pt and .npz side by side)
python run_eval_miracl.py \
  --checkpoint "checkpoints/minishlab_potion-multilingual-128M/*turbo-lloyd-1.pt" \
  --nanoembed "artifacts/*turbo-lloyd-1.npz" \
  --language ar
```

Run all 15 MIRACL languages with `--all-languages` instead of `--language ar`.

## Project layout

```
nanoembed/     Standalone inference package (numpy + tokenizers only)
compress/      Compression pipeline — PCA, PQ, TurboQuant, clustering, export
evaluate/      MIRACL and MTEB evaluation harness
models/        Model types, save/load for .pt checkpoints
artifacts/     Where the compressed models are stored
```

The following folders are for cached / local data

```
checkpoints/   Intermediate steps in the compression process are stored here
data/          Data used in the MIRACL evaluation
```

Scripts:
- `run_compress.py` — import a public model, compress it, optionally export to `.npz`
- `run_eval_miracl.py` — score models on MIRACL multilingual retrieval
- `run_eval_mteb.py` — score models on MTEB English benchmarks

## Compression recipes

| Recipe | What it does |
|---|---|
| `turbo-lloyd-1` | 1-bit TurboQuant with Max-Lloyd codebook (28x compression) |
| `turbo-lloyd-{2,3,4}` | Higher bit-width TurboQuant |
| `int{2,4,8}` | Uniform scalar quantization |
| `pca-{64,128}d` | PCA dimensionality reduction |
| `pq-m{N}-k{K}` | Product quantization |

Recipes compose: `pca-128d_int4` applies PCA then quantizes. See `compress/recipes.py` for the full list.

## Checkpoint format

Exported models are standard numpy `.npz` files containing:
- `packed_codes` — bit-packed quantization indices (uint8)
- `scales` — per-row decode scales (float32)
- `codebook` — reconstruction levels (float32)
- Metadata: `embed_dim`, `vocab_size`, `bits`, `tokenizer_name`, `method`, `source_model`

## License

MIT
