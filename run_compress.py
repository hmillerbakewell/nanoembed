"""Compress a static embedding model.

Import a public model (model2vec or SentenceTransformer), apply compression
recipes, and optionally export to nanoembed's packed .npz format.

Usage:
    python compress.py --model minishlab/potion-multilingual-128M --recipes turbo-lloyd-1
    python compress.py --model minishlab/potion-multilingual-128M --recipes turbo-lloyd-1 --export
    python compress.py --list-recipes
"""


import argparse
from pathlib import Path

from compress import CompressorConfig, compress_external_model
from compress.recipes import RECIPES


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a static embedding model")
    parser.add_argument("--model", type=str, help="HuggingFace model ID")
    parser.add_argument(
        "--recipes", nargs="+", default=["baseline", "turbo-lloyd-1"],
        help="Compression recipes to apply (default: baseline turbo-lloyd-1)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Output directory (default: checkpoints/<model-slug>)",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Also export quantized checkpoints to nanoembed .npz format",
    )
    parser.add_argument(
        "--export-dir", type=str, default="artifacts",
        help="Directory for exported .npz files (default: artifacts/)",
    )
    parser.add_argument(
        "--list-recipes", action="store_true",
        help="List all available recipes and exit",
    )
    args = parser.parse_args()

    if args.list_recipes:
        print("Available recipes:")
        for name, steps in sorted(RECIPES.items()):
            desc = " → ".join(s[0] for s in steps) if steps else "(identity)"
            print(f"  {name:<30s}  {desc}")
        return

    if not args.model:
        parser.error("--model is required (or use --list-recipes)")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        Path("checkpoints") / args.model.replace("/", "_")
    )

    cfg = CompressorConfig(
        model_id=args.model,
        recipes=tuple(args.recipes),
        checkpoint_dir=checkpoint_dir,
    )
    saved = compress_external_model(cfg)

    print(f"\nDone. {len(saved)} checkpoints in {checkpoint_dir}/")
    for p in saved:
        print(f"  {p.name}")

    if args.export:
        from compress.export import export_to_nanoembed
        from models.io import load_model
        from models.model import CodebookQuantizedWeights, UniformQuantizedWeights

        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting to nanoembed format in {export_dir}/")
        for pt_path in saved:
            model = load_model(pt_path)
            if isinstance(model.weights, (CodebookQuantizedWeights, UniformQuantizedWeights)):
                export_to_nanoembed(model, export_dir / f"{model.name}.npz")
            else:
                print(f"  skipping {model.name} (not quantized)")


if __name__ == "__main__":
    main()
