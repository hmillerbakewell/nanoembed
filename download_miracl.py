"""Download MIRACL topics, qrels, and corpus from HuggingFace.

Usage:
    # Download everything for all 15 non-English languages
    python download_miracl.py

    # Download specific languages
    python download_miracl.py --languages ar sw fi

    # Topics + qrels only (small, skip the large corpus)
    python download_miracl.py --skip-corpus

    # Download to a custom directory
    python download_miracl.py --data-dir /path/to/data
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


DEFAULT_LANGUAGES = (
    "ar", "bn", "es", "fa", "fi", "fr", "hi", "id",
    "ja", "ko", "ru", "sw", "te", "th", "zh",
)


def download_topics_and_qrels(language: str, data_dir: Path) -> None:
    """Download topics and qrels TSVs for one language."""
    repo_id = "miracl/miracl"
    prefix = f"miracl-v1.0-{language}"

    for subdir, pattern in [("topics", "topics"), ("qrels", "qrels")]:
        files = [
            f for f in list_repo_files(repo_id, repo_type="dataset")
            if f.startswith(f"{prefix}/{subdir}/{pattern}") and f.endswith(".tsv")
        ]
        for fname in files:
            out_path = data_dir / "miracl" / fname
            if out_path.exists():
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id, fname, repo_type="dataset",
            )
            # Copy from HF cache to our data dir
            import shutil
            shutil.copy2(downloaded, out_path)
            print(f"  {out_path}")


def download_corpus(language: str, data_dir: Path) -> None:
    """Download corpus jsonl.gz shards for one language."""
    repo_id = "miracl/miracl-corpus"
    prefix = f"miracl-corpus-v1.0-{language}"

    files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".jsonl.gz")
    ]
    for fname in files:
        out_path = data_dir / "miracl-corpus" / fname
        if out_path.exists():
            print(f"  {out_path} (exists)")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id, fname, repo_type="dataset",
        )
        import shutil
        shutil.copy2(downloaded, out_path)
        print(f"  {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MIRACL data from HuggingFace")
    parser.add_argument(
        "--languages", nargs="+", default=list(DEFAULT_LANGUAGES),
        help=f"Languages to download. Default: all 15",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--skip-corpus", action="store_true",
        help="Only download topics + qrels (small), skip the large corpus",
    )
    args = parser.parse_args()

    for lang in args.languages:
        print(f"\n[{lang}] topics + qrels")
        download_topics_and_qrels(lang, args.data_dir)

        if not args.skip_corpus:
            print(f"[{lang}] corpus")
            download_corpus(lang, args.data_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
