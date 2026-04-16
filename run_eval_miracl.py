"""Score static models on MIRACL multilingual retrieval.

Two modes:
  - **rerank** (default): rank ~30K pre-annotated passages per language. Fast.
  - **full** (--full): stream entire corpus, running top-K. Actual MIRACL protocol.

Usage:
    python run_eval_miracl.py --checkpoint "checkpoints/*.pt" --language ar
    python run_eval_miracl.py --checkpoint "checkpoints/*.pt" --all-languages
    python run_eval_miracl.py --checkpoint "checkpoints/*.pt" --language ar --full
    python run_eval_miracl.py --checkpoint "checkpoints/*.pt" --include-m2v minishlab/potion-multilingual-128M
"""


import argparse
import time
from glob import glob
from pathlib import Path

import numpy as np

from evaluate.metrics import (
    evaluate_miracl,
    evaluate_miracl_full,
    make_local_encoder,
)
from models.io import load_model
from evaluate.miracl import MiraclEvalSet, load_miracl
from evaluate.results import Run, RunResult


# 15 MIRACL languages excluding English. English corpus (~30 GB) is deferred to Colab.
ALL_MIRACL_NON_EN: tuple[str, ...] = (
    "ar", "bn", "es", "fa", "fi", "fr", "hi", "id",
    "ja", "ko", "ru", "sw", "te", "th", "zh",
)


def _score_rerank_encoder(
    name: str,
    encoder,
    size_mb: float,
    languages: list[str],
    eval_sets: dict[str, MiraclEvalSet],
    run: Run,
    provenance: list[str] | None = None,
) -> None:
    """Rerank-style eval for any encoder callable."""
    run.log(f"\n--- {name} ---")
    t0 = time.perf_counter()

    per_lang: dict[str, dict[str, float]] = {}
    per_lang_times: dict[str, float] = {}
    for lang in languages:
        t_lang = time.perf_counter()
        per_lang[lang] = evaluate_miracl(encoder, eval_sets[lang])
        per_lang_times[lang] = time.perf_counter() - t_lang

    eval_s = time.perf_counter() - t0

    macro_ndcg = float(np.mean([per_lang[l]["ndcg@10"] for l in languages]))
    macro_r10 = float(np.mean([per_lang[l]["recall@10"] for l in languages]))
    macro_r3 = float(np.mean([per_lang[l]["recall@3"] for l in languages]))
    total_queries = int(sum(per_lang[l]["num_queries"] for l in languages))

    run.log(f"  nDCG@10={macro_ndcg:.4f}  R@10={macro_r10:.4f}  ({eval_s:.1f}s)")

    flat: dict[str, float] = {
        "ndcg@10": macro_ndcg,
        "recall@10": macro_r10,
        "recall@3": macro_r3,
        "eval_s": eval_s,
    }
    for lang in languages:
        flat[f"ndcg@10_{lang}"] = per_lang[lang]["ndcg@10"]

    run.add_result(RunResult(
        name=name, size_mb=size_mb, metrics=flat,
        provenance=provenance or [name],
    ))


def _score_rerank(path, languages, eval_sets, run):
    """Load a checkpoint and score it via rerank."""
    model = load_model(path)
    encoder = make_local_encoder(model, max_length=256)
    _score_rerank_encoder(
        name=path.stem, encoder=encoder, size_mb=model.size_mb,
        languages=languages, eval_sets=eval_sets, run=run,
        provenance=model.provenance,
    )


def _score_full(
    path: Path,
    languages: list[str],
    run: Run,
    data_dir: str,
    batch_size: int,
    top_k: int,
) -> None:
    """Full-corpus eval: stream the corpus per language, running top-K per query."""
    run.log(f"\n--- {path.name} ---")
    t_load = time.perf_counter()
    model = load_model(path)
    encoder = make_local_encoder(model, max_length=256)
    t_eval_start = time.perf_counter()
    load_s = t_eval_start - t_load

    per_lang: dict[str, dict[str, float]] = {}
    per_lang_times: dict[str, float] = {}
    for lang in languages:
        run.log(f"  [{lang}] streaming corpus...")
        t_lang = time.perf_counter()
        try:
            per_lang[lang] = evaluate_miracl_full(
                encoder=encoder,
                language=lang,
                data_dir=data_dir,
                batch_size=batch_size,
                top_k=top_k,
            )
        except FileNotFoundError as e:
            run.log(f"  [{lang}] skip: {e}")
            continue
        per_lang_times[lang] = time.perf_counter() - t_lang
        scores = per_lang[lang]
        run.log(
            f"  [{lang}] nDCG@10={scores['ndcg@10']:.4f} "
            f"R@100={scores['recall@100']:.4f} "
            f"({int(scores['total_passages']):,} passages, "
            f"{per_lang_times[lang]:.1f}s)"
        )

    eval_s = time.perf_counter() - t_eval_start

    if not per_lang:
        run.log("  no languages scored")
        return

    scored_langs = list(per_lang.keys())
    macro_ndcg = float(np.mean([per_lang[l]["ndcg@10"] for l in scored_langs]))
    macro_r100 = float(np.mean([per_lang[l]["recall@100"] for l in scored_langs]))
    total_queries = int(sum(per_lang[l]["num_queries"] for l in scored_langs))
    total_passages = int(sum(per_lang[l]["total_passages"] for l in scored_langs))

    run.log(f"  ---")
    run.log(f"  macro nDCG@10:    {macro_ndcg:.4f}  (full-corpus, across {len(scored_langs)} languages)")
    run.log(f"  macro recall@100: {macro_r100:.4f}")
    run.log(f"  total queries:    {total_queries}")
    run.log(f"  total passages:   {total_passages:,}")
    run.log(f"  total eval:       {eval_s:.1f}s")

    flat: dict[str, float] = {
        "ndcg@10": macro_ndcg,
        "recall@100": macro_r100,
        "num_languages": float(len(scored_langs)),
        "total_queries": float(total_queries),
        "total_passages": float(total_passages),
        "load_s": load_s,
        "eval_s": eval_s,
        "mode": 1.0,
    }
    for lang in scored_langs:
        flat[f"ndcg@10_{lang}"] = per_lang[lang]["ndcg@10"]
        flat[f"recall@100_{lang}"] = per_lang[lang]["recall@100"]
        flat[f"passages_{lang}"] = per_lang[lang]["total_passages"]
        flat[f"eval_s_{lang}"] = per_lang_times[lang]

    run.add_result(RunResult(
        name=path.stem,
        size_mb=model.size_mb,
        metrics=flat,
        provenance=model.provenance,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", default=[],
                        help="Pipeline .pt checkpoint path(s) or glob(s)")
    parser.add_argument("--language", nargs="+", default=["ar"],
                        help="MIRACL language code(s). Default: ar")
    parser.add_argument("--all-languages", action="store_true",
                        help=f"Shortcut for 15 non-English MIRACL languages: "
                             f"{' '.join(ALL_MIRACL_NON_EN)}")
    parser.add_argument("--full", action="store_true",
                        help="Full-corpus retrieval (streaming). Default: rerank-style.")
    parser.add_argument("--data-dir", default="data",
                        help="Root of the data directory")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="Full-corpus mode: passages encoded per batch (default 10000)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Full-corpus mode: running top-K per query (default 100)")
    parser.add_argument("--nanoembed", nargs="*", default=[],
                        help="Packed .npz model path(s) or glob(s) to evaluate")
    parser.add_argument("--include-m2v", nargs="*", default=[],
                        help="model2vec model IDs to include as baselines")
    parser.add_argument("--tag", default=None,
                        help="Filter to checkpoints whose filename starts with this tag")
    args = parser.parse_args()

    # Resolve language list
    languages: list[str]
    if args.all_languages:
        languages = list(ALL_MIRACL_NON_EN)
    else:
        languages = list(args.language)

    # Expand globs for .pt checkpoints
    paths: list[Path] = []
    for pattern in args.checkpoint:
        matches = sorted(Path(p) for p in glob(pattern))
        if not matches:
            print(f"Warning: no matches for {pattern}")
        paths.extend(matches)

    if args.tag:
        prefix = f"{args.tag}_"
        paths = [p for p in paths if p.stem.startswith(prefix)]
        print(f"Filtered to {len(paths)} checkpoints with tag '{args.tag}'")

    # Expand globs for .npz nanoembed models
    nano_paths: list[Path] = []
    for pattern in args.nanoembed:
        matches = sorted(Path(p) for p in glob(pattern))
        if not matches:
            print(f"Warning: no matches for {pattern}")
        nano_paths.extend(matches)

    if not paths and not nano_paths and not args.include_m2v:
        raise SystemExit("No models to evaluate")

    mode_label = "full" if args.full else "rerank"
    if len(languages) == 1:
        lang_label = languages[0]
    elif args.all_languages:
        lang_label = "all"
    else:
        lang_label = "+".join(languages)
    label = f"miracl_{mode_label}_{lang_label}"
    if args.tag:
        label += f"_{args.tag}"

    run = Run(
        label=label,
        metadata={
            "kind": "miracl_eval",
            "mode": mode_label,
            "languages": languages,
            "tag": args.tag,
            "num_checkpoints": len(paths),
            "checkpoints": [str(p) for p in paths],
        },
    )
    run.log(f"Mode:          {mode_label}")
    run.log(f"Languages:     {' '.join(languages)}")
    run.log(f"Checkpoints:   {len(paths)}")
    run.log(f"Run directory: {run.dir}")

    # ── Rerank mode: pre-load all eval sets ──────────────────────────────────
    if not args.full:
        run.log(f"\nLoading {len(languages)} rerank eval set(s)...")
        t0 = time.perf_counter()
        eval_sets: dict[str, MiraclEvalSet] = {}
        for lang in languages:
            try:
                eval_sets[lang] = load_miracl(lang, data_dir=args.data_dir)
            except FileNotFoundError as e:
                run.log(f"  [skip] {lang}: {e}")
        load_all_s = time.perf_counter() - t0

        if not eval_sets:
            raise SystemExit("No eval sets could be loaded")

        total_queries = sum(es.num_queries for es in eval_sets.values())
        total_passages = sum(es.num_passages for es in eval_sets.values())
        run.log(f"\nLoaded {len(eval_sets)} languages: "
                f"{total_queries} queries, {total_passages} passages "
                f"({load_all_s:.2f}s)")

        languages = list(eval_sets.keys())  # drop ones that failed
        for path in paths:
            _score_rerank(path, languages, eval_sets, run)

        # nanoembed models (rerank)
        for np_path in nano_paths:
            import nanoembed
            nm = nanoembed.load(str(np_path))
            _score_rerank_encoder(
                name=np_path.stem, encoder=nm.encode, size_mb=nm.info.logical_size_mb,
                languages=languages, eval_sets=eval_sets, run=run,
            )

        # m2v baselines (rerank)
        if args.include_m2v:
            from evaluate.encoders import make_m2v_native_encoder
            for m2v_id in args.include_m2v:
                try:
                    encoder, size_mb = make_m2v_native_encoder(m2v_id)
                except Exception as e:
                    run.log(f"\nFailed to load {m2v_id}: {e}")
                    continue
                _score_rerank_encoder(
                    name=m2v_id, encoder=encoder, size_mb=size_mb,
                    languages=languages, eval_sets=eval_sets, run=run,
                )

    # ── Full-corpus mode: stream per checkpoint, per language ────────────────
    else:
        run.log(f"\nFull-corpus mode: streaming each language's corpus per checkpoint.")
        run.log(f"Batch size: {args.batch_size}, top-K: {args.top_k}")
        for path in paths:
            _score_full(
                path=path,
                languages=languages,
                run=run,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                top_k=args.top_k,
            )

    # ── Summary table ────────────────────────────────────────────────────────
    run.log(f"\n{'='*120}")
    run.log(f"MIRACL summary — {mode_label}, {len(languages)} language(s), macro-averaged")
    run.log(f"{'='*120}\n")

    if args.full:
        # Full-corpus table: macro nDCG@10, recall@100, per-lang nDCG@10
        header_macro = (f"{'Checkpoint':<55} {'MB':>8} {'macro':>7} "
                        f"{'R@100':>8} {'eval_s':>8} ")
        header_langs = "  ".join(f"{l:>5}" for l in languages)
        run.log(header_macro + header_langs)
        run.log("-" * (len(header_macro) + len(header_langs)))
        for r in run.results:
            macro_block = (f"{r.name:<55} {r.size_mb:>8.2f} "
                           f"{r.metrics['ndcg@10']:>7.4f} "
                           f"{r.metrics.get('recall@100', 0.0):>8.4f} "
                           f"{r.metrics.get('eval_s', 0.0):>8.1f} ")
            per_lang_block = "  ".join(
                f"{r.metrics.get(f'ndcg@10_{l}', 0.0):>5.3f}" for l in languages
            )
            run.log(macro_block + per_lang_block)
    elif len(languages) == 1:
        run.log(f"{'Checkpoint':<55} {'MB':>8} {'nDCG@10':>9} {'R@10':>8} {'R@3':>8} {'eval_s':>8}")
        run.log("-" * 110)
        for r in run.results:
            run.log(f"{r.name:<55} {r.size_mb:>8.2f} {r.metrics['ndcg@10']:>9.4f} "
                    f"{r.metrics['recall@10']:>8.4f} {r.metrics['recall@3']:>8.4f} "
                    f"{r.metrics.get('eval_s', 0.0):>8.2f}")
    else:
        header_macro = f"{'Checkpoint':<55} {'MB':>8} {'macro':>7} {'R@10':>8} {'R@3':>8} {'eval_s':>8} "
        header_langs = "  ".join(f"{l:>5}" for l in languages)
        run.log(header_macro + header_langs)
        run.log("-" * (len(header_macro) + len(header_langs)))
        for r in run.results:
            macro_block = (f"{r.name:<55} {r.size_mb:>8.2f} "
                           f"{r.metrics['ndcg@10']:>7.4f} "
                           f"{r.metrics['recall@10']:>8.4f} "
                           f"{r.metrics['recall@3']:>8.4f} "
                           f"{r.metrics.get('eval_s', 0.0):>8.2f} ")
            per_lang_block = "  ".join(
                f"{r.metrics.get(f'ndcg@10_{l}', 0.0):>5.3f}" for l in languages
            )
            run.log(macro_block + per_lang_block)

    run.finalise(primary_metric="ndcg@10", color_by=None)


if __name__ == "__main__":
    main()
