"""Evaluate compressed checkpoints on MTEB English tasks.

Matches blobbybob's evaluation: STS + Classification + PairClassification.
Produces per-task JSON results + a summary table.

Usage:
    # Score all mxbai-compressed checkpoints on STS only (fast, ~5 min/model)
    python run_mteb_eval.py --checkpoint "checkpoints/mxbai-compressed/*.pt" --tasks sts

    # Score on the full blobbybob-matched task set (~30 min/model)
    python run_mteb_eval.py --checkpoint "checkpoints/mxbai-compressed/*.pt" --tasks blobbybob

    # Also include the original blobbybob-micro for head-to-head
    python run_mteb_eval.py --checkpoint "checkpoints/mxbai-compressed/*.pt" \\
        --include-m2v blobbybob/potion-mxbai-micro \\
        --tasks blobbybob
"""


import argparse
import time
from glob import glob
from pathlib import Path

import numpy as np

from evaluate.metrics import make_local_encoder
from models.io import load_model
from evaluate.mteb_adapter import MTEBModelWrapper
from evaluate.results import Run, RunResult

import mteb


# ── Task sets ────────────────────────────────────────────────────────────────

TASK_SETS: dict[str, list[str]] = {
    # Quick sanity check
    "sts": [
        "STS12", "STS13", "STS14", "STS15", "STS16",
        "STSBenchmark", "SICK-R",
    ],
    # NanoBEIR retrieval tasks (13) — matches NIFE's published benchmark
    "nanobeir": [
        "NanoArguAnaRetrieval", "NanoClimateFeverRetrieval", "NanoDBPediaRetrieval",
        "NanoFEVERRetrieval", "NanoFiQA2018Retrieval", "NanoHotpotQARetrieval",
        "NanoMSMARCORetrieval", "NanoNFCorpusRetrieval", "NanoNQRetrieval",
        "NanoQuoraRetrieval", "NanoSCIDOCSRetrieval", "NanoSciFactRetrieval",
        "NanoTouche2020Retrieval",
    ],
    # Fast subset: STS + PairClassification + small classification (~5 min/model)
    "fast": [
        # STS (7)
        "STS12", "STS13", "STS14", "STS15", "STS16",
        "STSBenchmark", "SICK-R",
        # PairClassification (3)
        "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
        # Small classification (6) — these fit in memory and train quickly
        "Banking77Classification", "EmotionClassification",
        "MTOPDomainClassification", "MTOPIntentClassification",
        "ToxicConversationsClassification", "TweetSentimentExtractionClassification",
    ],
    # Matches blobbybob's reported 25 tasks (10 STS + 12 Classification + 3 PairClassification)
    "blobbybob": [
        # STS (10)
        "STS12", "STS13", "STS14", "STS15", "STS16",
        "STSBenchmark", "SICK-R",
        "STS17", "STS22",
        "STSBenchmarkMultilingualSTS",
        # Classification (12)
        "AmazonCounterfactualClassification", "AmazonPolarityClassification",
        "AmazonReviewsClassification", "Banking77Classification",
        "EmotionClassification", "ImdbClassification",
        "MassiveIntentClassification", "MassiveScenarioClassification",
        "MTOPDomainClassification", "MTOPIntentClassification",
        "ToxicConversationsClassification", "TweetSentimentExtractionClassification",
        # PairClassification (3)
        "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
    ],
}


def _score_with_mteb(
    wrapper: MTEBModelWrapper,
    task_names: list[str],
    output_dir: Path,
) -> dict[str, float]:
    """Run MTEB tasks and return {task_name: main_score} dict."""

    tasks = mteb.get_tasks(tasks=task_names)
    result = mteb.evaluate(
        wrapper,
        tasks=tasks,
        prediction_folder=str(output_dir),
        overwrite_strategy="always",
    )

    scores: dict[str, float] = {}
    for task_result in results:
        task_name = task_result.task_name
        # MTEB stores the main score in the test split's main_score
        for split_name, split_scores in task_result.scores.items():
            if split_name == "test" and split_scores:
                scores[task_name] = split_scores[0].get("main_score", 0.0)
                break
        if task_name not in scores:
            # Fallback: try any split
            for split_name, split_scores in task_result.scores.items():
                if split_scores:
                    scores[task_name] = split_scores[0].get("main_score", 0.0)
                    break

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", default=[],
                        help="Pipeline .pt checkpoint path(s) or glob(s)")
    parser.add_argument("--nanoembed", nargs="*", default=[],
                        help="Packed .npz model path(s) or glob(s)")
    parser.add_argument("--tasks", default="sts", choices=list(TASK_SETS.keys()),
                        help="Task set to evaluate (default: sts)")
    parser.add_argument("--include-m2v", nargs="*", default=[],
                        help="model2vec model IDs to include as baselines")
    parser.add_argument("--tag", default=None,
                        help="Filter checkpoints by filename prefix")
    args = parser.parse_args()

    task_names = TASK_SETS[args.tasks]

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

    # Expand globs for .npz nanoembed models
    nano_paths: list[Path] = []
    for pattern in (args.nanoembed or []):
        matches = sorted(Path(p) for p in glob(pattern))
        if not matches:
            print(f"Warning: no matches for {pattern}")
        nano_paths.extend(matches)

    if not paths and not nano_paths and not args.include_m2v:
        raise SystemExit("No models to evaluate")

    run = Run(
        label=f"mteb_{args.tasks}",
        metadata={
            "kind": "mteb_eval",
            "task_set": args.tasks,
            "task_names": task_names,
            "num_checkpoints": len(paths),
            "m2v_baselines": args.include_m2v,
        },
    )
    run.log(f"Task set:      {args.tasks} ({len(task_names)} tasks)")
    run.log(f"Checkpoints:   {len(paths)}")
    run.log(f"M2V baselines: {args.include_m2v or 'none'}")
    run.log(f"Run directory:  {run.dir}")

    results_dir = run.dir / "mteb_results"

    # ── Score local checkpoints ──────────────────────────────────────────────
    for path in paths:
        run.log(f"\n--- {path.stem} ---")
        t0 = time.perf_counter()
        model = load_model(path)
        encoder = make_local_encoder(model, max_length=256)
        wrapper = MTEBModelWrapper(encoder, model_name="healthdataavatar/"+path.stem, size_mb=model.size_mb)

        scores = _score_with_mteb(wrapper, task_names, results_dir / path.stem)
        elapsed = time.perf_counter() - t0

        avg = float(np.mean(list(scores.values()))) if scores else 0.0
        run.log(f"  avg: {avg:.4f}  ({elapsed:.1f}s)")
        for task, score in sorted(scores.items()):
            run.log(f"    {task}: {score:.4f}")

        flat = {"mteb_avg": avg, "eval_s": elapsed, **scores}
        run.add_result(RunResult(
            name=path.stem,
            size_mb=model.size_mb,
            metrics=flat,
            provenance=model.provenance,
        ))

    # ── Score nanoembed models ────────────────────────────────────────────────
    for np_path in nano_paths:
        run.log(f"\n--- {np_path.stem} (nanoembed) ---")
        t0 = time.perf_counter()

        import nanoembed
        nm = nanoembed.load(str(np_path))
        wrapper = MTEBModelWrapper(
            nm.encode, model_name=f"nanoembed/{np_path.stem}",
            size_mb=nm.info.logical_size_mb,
        )
        scores = _score_with_mteb(wrapper, task_names, results_dir / np_path.stem)
        elapsed = time.perf_counter() - t0

        avg = float(np.mean(list(scores.values()))) if scores else 0.0
        run.log(f"  avg: {avg:.4f}  ({elapsed:.1f}s)")
        for task, score in sorted(scores.items()):
            run.log(f"    {task}: {score:.4f}")

        flat = {"mteb_avg": avg, "eval_s": elapsed, **scores}
        run.add_result(RunResult(
            name=np_path.stem, size_mb=nm.info.logical_size_mb,
            metrics=flat, provenance=[np_path.stem],
        ))

    # ── Score model2vec baselines ────────────────────────────────────────────
    for m2v_id in args.include_m2v:
        run.log(f"\n--- {m2v_id} (native m2v) ---")
        t0 = time.perf_counter()

        from evaluate.encoders import make_m2v_native_encoder
        encoder, size_mb = make_m2v_native_encoder(m2v_id)

        wrapper = MTEBModelWrapper(encoder, model_name=m2v_id, size_mb=size_mb)
        scores = _score_with_mteb(wrapper, task_names, results_dir / m2v_id.replace("/", "_"))
        elapsed = time.perf_counter() - t0

        avg = float(np.mean(list(scores.values()))) if scores else 0.0
        run.log(f"  avg: {avg:.4f}  ({elapsed:.1f}s)")
        for task, score in sorted(scores.items()):
            run.log(f"    {task}: {score:.4f}")

        flat = {"mteb_avg": avg, "eval_s": elapsed, **scores}
        run.add_result(RunResult(
            name=m2v_id, size_mb=size_mb, metrics=flat, provenance=[m2v_id],
        ))

    # ── Summary table ────────────────────────────────────────────────────────
    run.log(f"\n{'='*80}")
    run.log(f"MTEB {args.tasks} summary")
    run.log(f"{'='*80}\n")
    run.log(f"{'Model':<55} {'MB':>8} {'avg':>8} {'eval_s':>8}")
    run.log("-" * 82)
    for r in sorted(run.results, key=lambda r: -r.metrics.get("mteb_avg", 0)):
        run.log(f"{r.name:<55} {r.size_mb:>8.2f} "
                f"{r.metrics.get('mteb_avg', 0):>8.4f} "
                f"{r.metrics.get('eval_s', 0):>8.1f}")

    run.finalise(primary_metric="mteb_avg", color_by=None)


if __name__ == "__main__":
    main()
