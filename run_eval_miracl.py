"""Score static models on MIRACL multilingual retrieval.

Two modes:
  - **rerank** (default): rank ~30K pre-annotated passages per language. Fast.
  - **full**: stream entire corpus, running top-K. Actual MIRACL protocol.

Usage:
    python run_eval_miracl.py --model checkpoint.pt --languages ar
    python run_eval_miracl.py --model model.npz org/model-name --languages all
    python run_eval_miracl.py --model model.npz --languages ar sw fi --mode full
"""


import argparse
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path

from tqdm import tqdm

import numpy as np

from evaluate.metrics import (
    Encoder,
    evaluate_miracl,
    evaluate_miracl_full,
    make_local_encoder,
)
from evaluate.miracl import MiraclEvalSet, load_miracl
from evaluate.results import Run, RunResult


# 15 MIRACL languages excluding English. English corpus (~30 GB) is deferred to Colab.
ALL_MIRACL_NON_EN: tuple[str, ...] = (
    "ar", "bn", "es", "fa", "fi", "fr", "hi", "id",
    "ja", "ko", "ru", "sw", "te", "th", "zh",
)


@dataclass
class LoadedModel:
    """A model ready for evaluation."""
    name: str
    encoder: Encoder
    size_mb: float
    provenance: list[str]


def load_any_model(path_or_id: str) -> LoadedModel:
    """Load a model from any supported source.

    - .pt file  → pytorch checkpoint (via models.io)
    - .npz file → nanoembed packed format
    - otherwise → model2vec from HuggingFace
    """
    p = Path(path_or_id)

    if p.suffix == ".pt" and p.exists():
        from models.io import load_model
        model = load_model(p)
        encoder = make_local_encoder(model, max_length=256)
        return LoadedModel(
            name=p.stem, encoder=encoder,
            size_mb=model.size_mb, provenance=model.provenance,
        )

    if p.suffix == ".npz" and p.exists():
        import nanoembed
        nm = nanoembed.load(str(p))
        return LoadedModel(
            name=p.stem, encoder=nm.encode,
            size_mb=nm.info.logical_size_mb, provenance=[p.stem],
        )

    # HuggingFace model2vec
    from evaluate.encoders import make_m2v_native_encoder
    encoder, size_mb = make_m2v_native_encoder(path_or_id)
    return LoadedModel(
        name=path_or_id, encoder=encoder,
        size_mb=size_mb, provenance=[path_or_id],
    )


def _score_rerank(
    model: LoadedModel,
    languages: list[str],
    eval_sets: dict[str, MiraclEvalSet],
    run: Run,
) -> None:
    """Rerank-style eval for a loaded model."""
    run.log(f"\n--- {model.name} ---")
    t0 = time.perf_counter()

    per_lang: dict[str, dict[str, float]] = {}
    for lang in tqdm(languages, desc=f"  {model.name}", unit="lang"):
        per_lang[lang] = evaluate_miracl(model.encoder, eval_sets[lang])

    eval_s = time.perf_counter() - t0

    macro_ndcg = float(np.mean([per_lang[l]["ndcg@10"] for l in languages]))
    macro_r10 = float(np.mean([per_lang[l]["recall@10"] for l in languages]))
    macro_r3 = float(np.mean([per_lang[l]["recall@3"] for l in languages]))

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
        name=model.name, size_mb=model.size_mb, metrics=flat,
        provenance=model.provenance,
    ))


def _score_full(
    model: LoadedModel,
    languages: list[str],
    run: Run,
    data_dir: str,
    batch_size: int,
    top_k: int,
) -> None:
    """Full-corpus eval: stream the corpus per language, running top-K per query.

    Results are checkpointed per-language to {data_dir}/miracl_full_progress/{model.name}.json.
    Re-running the same command automatically resumes where it left off.
    """
    import json

    run.log(f"\n--- {model.name} ---")
    t0 = time.perf_counter()

    # Load checkpoint if resuming
    progress_dir = Path(data_dir) / "miracl_full_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"{model.name}.json"
    per_lang: dict[str, dict[str, float]] = {}
    per_lang_times: dict[str, float] = {}
    if progress_path.exists():
        saved = json.loads(progress_path.read_text())
        per_lang = saved.get("per_lang", {})
        per_lang_times = saved.get("per_lang_times", {})
        run.log(f"  resuming: {len(per_lang)} languages already done")

    remaining = [l for l in languages if l not in per_lang]
    lang_iter = tqdm(remaining, desc=f"  {model.name}", unit="lang",
                     initial=len(per_lang), total=len(languages))
    for lang in lang_iter:
        lang_iter.set_postfix(lang=lang)
        t_lang = time.perf_counter()
        try:
            per_lang[lang] = evaluate_miracl_full(
                encoder=model.encoder,
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

        # Checkpoint after each language
        progress_path.write_text(json.dumps({
            "per_lang": per_lang,
            "per_lang_times": per_lang_times,
        }, indent=2))

    eval_s = time.perf_counter() - t0

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
        "eval_s": eval_s,
        "mode": 1.0,
    }
    for lang in scored_langs:
        flat[f"ndcg@10_{lang}"] = per_lang[lang]["ndcg@10"]
        flat[f"recall@100_{lang}"] = per_lang[lang]["recall@100"]
        flat[f"passages_{lang}"] = per_lang[lang]["total_passages"]
        flat[f"eval_s_{lang}"] = per_lang_times[lang]

    run.add_result(RunResult(
        name=model.name, size_mb=model.size_mb, metrics=flat,
        provenance=model.provenance,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score static models on MIRACL multilingual retrieval",
    )
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="Model(s) to evaluate: .pt checkpoint, .npz nanoembed file, "
             "or HuggingFace model ID. Globs are expanded.",
    )
    parser.add_argument(
        "--languages", nargs="+", default=["ar"],
        help="MIRACL language code(s), or 'all' for 15 non-English languages. "
             "Default: ar",
    )
    parser.add_argument(
        "--mode", choices=["rerank", "full"], default="rerank",
        help="rerank (fast, ~30K passages) or full (stream entire corpus). "
             "Default: rerank",
    )
    parser.add_argument("--data-dir", default="data",
                        help="Root of the data directory")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="Full mode: passages encoded per batch (default 10000)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Full mode: running top-K per query (default 100)")
    args = parser.parse_args()

    # Resolve language list
    if args.languages == ["all"]:
        languages = list(ALL_MIRACL_NON_EN)
    else:
        languages = list(args.languages)

    # Expand globs and resolve models
    model_specs: list[str] = []
    for pattern in args.model:
        matches = sorted(glob(pattern))
        if matches and matches != [pattern]:
            model_specs.extend(matches)
        else:
            model_specs.append(pattern)

    if not model_specs:
        raise SystemExit("No models to evaluate")

    mode_label = args.mode
    if len(languages) == 1:
        lang_label = languages[0]
    elif len(languages) == len(ALL_MIRACL_NON_EN):
        lang_label = "all"
    else:
        lang_label = "+".join(languages)
    label = f"miracl_{mode_label}_{lang_label}"

    run = Run(
        label=label,
        metadata={
            "kind": "miracl_eval",
            "mode": mode_label,
            "languages": languages,
            "models": model_specs,
        },
    )
    run.log(f"Mode:       {mode_label}")
    run.log(f"Languages:  {' '.join(languages)}")
    run.log(f"Models:     {len(model_specs)}")
    run.log(f"Run dir:    {run.dir}")

    # Load all models upfront
    models: list[LoadedModel] = []
    for spec in model_specs:
        try:
            models.append(load_any_model(spec))
            run.log(f"  loaded {models[-1].name} ({models[-1].size_mb:.1f} MB)")
        except Exception as e:
            import traceback
            run.log(f"  failed to load {spec}: {e}")
            traceback.print_exc()

    if not models:
        raise SystemExit("No models could be loaded")

    # ── Rerank mode ───────────────────────────────────────────────────────────
    if args.mode == "rerank":
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
        for m in models:
            _score_rerank(m, languages, eval_sets, run)

    # ── Full-corpus mode ──────────────────────────────────────────────────────
    else:
        run.log(f"\nFull-corpus mode: streaming each language's corpus per model.")
        run.log(f"Batch size: {args.batch_size}, top-K: {args.top_k}")
        for m in models:
            _score_full(
                model=m,
                languages=languages,
                run=run,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                top_k=args.top_k,
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    run.log(f"\n{'='*120}")
    run.log(f"MIRACL summary — {mode_label}, {len(languages)} language(s), macro-averaged")
    run.log(f"{'='*120}\n")

    if args.mode == "full":
        header_macro = (f"{'Model':<55} {'MB':>8} {'macro':>7} "
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
        run.log(f"{'Model':<55} {'MB':>8} {'nDCG@10':>9} {'R@10':>8} {'R@3':>8} {'eval_s':>8}")
        run.log("-" * 110)
        for r in run.results:
            run.log(f"{r.name:<55} {r.size_mb:>8.2f} {r.metrics['ndcg@10']:>9.4f} "
                    f"{r.metrics['recall@10']:>8.4f} {r.metrics['recall@3']:>8.4f} "
                    f"{r.metrics.get('eval_s', 0.0):>8.2f}")
    else:
        header_macro = f"{'Model':<55} {'MB':>8} {'macro':>7} {'R@10':>8} {'R@3':>8} {'eval_s':>8} "
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
