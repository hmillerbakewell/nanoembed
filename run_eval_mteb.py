"""Evaluate static embedding models on MTEB tasks.

Usage:
    python run_eval_mteb.py --model model.npz --tasks sts
    python run_eval_mteb.py --model model.npz checkpoint.pt org/model-name --tasks blobbybob
    python run_eval_mteb.py --model model.npz --tasks eng-v2
    python run_eval_mteb.py --model model.npz --tasks multilingual
"""


import argparse
import time
from glob import glob
from pathlib import Path

import numpy as np

from evaluate.metrics import Encoder
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

# MMTEB benchmarks — loaded via mteb.get_benchmark() instead of task name lists.
BENCHMARKS: dict[str, str] = {
    "eng-v2": "MTEB(eng, v2)",
    "multilingual": "MTEB(Multilingual, v1)",
}

ALL_TASK_CHOICES = list(TASK_SETS.keys()) + list(BENCHMARKS.keys())


def _resolve_tasks(task_set: str):
    """Resolve a task set name to mteb task objects."""
    if task_set in BENCHMARKS:
        benchmark = mteb.get_benchmark(BENCHMARKS[task_set])
        return benchmark.tasks
    return mteb.get_tasks(tasks=TASK_SETS[task_set])


def load_any_model(path_or_id: str) -> tuple[str, Encoder, float]:
    """Load a model from any supported source.

    Returns (name, encoder_callable, size_mb).

    - .pt file  → pytorch checkpoint
    - .npz file → nanoembed packed format
    - otherwise → model2vec from HuggingFace
    """
    p = Path(path_or_id)

    if p.suffix == ".pt" and p.exists():
        from evaluate.metrics import make_local_encoder
        from models.io import load_model
        model = load_model(p)
        encoder = make_local_encoder(model, max_length=256)
        return p.stem, encoder, model.size_mb

    if p.suffix == ".npz" and p.exists():
        import nanoembed
        nm = nanoembed.load(str(p))
        return p.stem, nm.encode, nm.info.logical_size_mb

    # HuggingFace model2vec
    from evaluate.encoders import make_m2v_native_encoder
    encoder, size_mb = make_m2v_native_encoder(path_or_id)
    return path_or_id, encoder, size_mb


def _score_with_mteb(
    wrapper: MTEBModelWrapper,
    tasks,
    output_dir: Path,
) -> dict[str, float]:
    """Run MTEB tasks and return {task_name: main_score} dict."""
    result = mteb.evaluate(
        wrapper,
        tasks,
        prediction_folder=str(output_dir),
        overwrite_strategy="always",
    )

    scores: dict[str, float] = {}
    for task_result in result.task_results:
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
    parser = argparse.ArgumentParser(
        description="Evaluate static embedding models on MTEB tasks",
    )
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="Model(s) to evaluate: .pt checkpoint, .npz nanoembed file, "
             "or HuggingFace model ID. Globs are expanded.",
    )
    parser.add_argument(
        "--tasks", default="sts", choices=ALL_TASK_CHOICES,
        help="Task set to evaluate (default: sts). "
             "eng-v2 and multilingual use MMTEB benchmarks.",
    )
    args = parser.parse_args()

    tasks = _resolve_tasks(args.tasks)
    task_names = [t.metadata.name for t in tasks]

    # Expand globs and resolve model specs
    model_specs: list[str] = []
    for pattern in args.model:
        matches = sorted(glob(pattern))
        if matches and matches != [pattern]:
            model_specs.extend(matches)
        else:
            model_specs.append(pattern)

    if not model_specs:
        raise SystemExit("No models to evaluate")

    run = Run(
        label=f"mteb_{args.tasks}",
        metadata={
            "kind": "mteb_eval",
            "task_set": args.tasks,
            "task_names": task_names,
            "models": model_specs,
        },
    )
    run.log(f"Task set:   {args.tasks} ({len(task_names)} tasks)")
    run.log(f"Models:     {len(model_specs)}")
    run.log(f"Run dir:    {run.dir}")

    results_dir = run.dir / "mteb_results"

    for spec in model_specs:
        try:
            name, encoder, size_mb = load_any_model(spec)
        except Exception as e:
            import traceback
            run.log(f"\n--- {spec} ---")
            run.log(f"  failed to load: {e}")
            traceback.print_exc()
            continue

        run.log(f"\n--- {name} ({size_mb:.1f} MB) ---")
        t0 = time.perf_counter()

        wrapper = MTEBModelWrapper(encoder, model_name=name, size_mb=size_mb)
        scores = _score_with_mteb(wrapper, tasks, results_dir / name.replace("/", "_"))
        elapsed = time.perf_counter() - t0

        avg = float(np.mean(list(scores.values()))) if scores else 0.0
        run.log(f"  avg: {avg:.4f}  ({elapsed:.1f}s)")
        for task, score in sorted(scores.items()):
            run.log(f"    {task}: {score:.4f}")

        flat = {"mteb_avg": avg, "eval_s": elapsed, **scores}
        run.add_result(RunResult(
            name=name, size_mb=size_mb, metrics=flat, provenance=[name],
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
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
