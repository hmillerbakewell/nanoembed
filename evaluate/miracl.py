"""MIRACL data loading and retrieval metrics.

Supports the rerank-style evaluation: for each query, we only rank over the
passages that were pre-annotated in the dev qrels (mix of relevant/non-relevant).
This is the same approximation MTEB's MIRACLReranking subset uses. It's not
full-corpus retrieval but it's a legitimate signal that runs on a laptop.
"""


import gzip
import json
import math
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class MiraclQuery:
    query_id: str
    text: str
    qrels: dict[str, int]   # docid → relevance (0 or 1)


@dataclass
class MiraclEvalSet:
    language: str
    queries: list[MiraclQuery]
    passages: dict[str, str]   # docid → "title. text"

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_passages(self) -> int:
        return len(self.passages)


# ── Loader ────────────────────────────────────────────────────────────────────

def _parse_topics(path: Path) -> dict[str, str]:
    """Parse topics TSV: `query_id \\t query_text`."""
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        qid, text = line.split("\t", 1)
        result[qid] = text
    return result


def _parse_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse TREC qrels TSV: `query_id \\t Q0 \\t docid \\t relevance`.

    Returns query_id → (docid → relevance).
    """
    result: dict[str, dict[str, int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        qid, _, docid, rel = parts
        result.setdefault(qid, {})[docid] = int(rel)
    return result


def _stream_corpus_passages(
    corpus_dir: Path,
    needed_docids: set[str],
) -> dict[str, str]:
    """Scan all docs-*.jsonl.gz files, keep only passages in needed_docids.

    Returns docid → "title. text".
    """
    result: dict[str, str] = {}
    shards = sorted(corpus_dir.glob("docs-*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"No docs-*.jsonl.gz shards found in {corpus_dir}")

    for shard in shards:
        print(f"  scanning {shard.name}...")
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docid = obj["docid"]
                if docid in needed_docids:
                    title = obj.get("title", "") or ""
                    text = obj.get("text", "") or ""
                    result[docid] = f"{title}. {text}" if title else text
                    if len(result) == len(needed_docids):
                        return result
    return result


def iter_corpus_batches(
    corpus_dir: Path,
    batch_size: int = 10_000,
) -> Iterator[tuple[list[str], list[str]]]:
    """Stream all passages from a MIRACL corpus directory in batches.

    Yields (docids, texts) tuples, each of length up to batch_size. Texts are
    formatted as "title. text" (matching rerank-style eval).
    """
    shards = sorted(corpus_dir.glob("docs-*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"No docs-*.jsonl.gz shards found in {corpus_dir}")

    batch_docids: list[str] = []
    batch_texts: list[str] = []
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docid = obj["docid"]
                title = obj.get("title", "") or ""
                text = obj.get("text", "") or ""
                passage = f"{title}. {text}" if title else text
                batch_docids.append(docid)
                batch_texts.append(passage)
                if len(batch_docids) >= batch_size:
                    yield batch_docids, batch_texts
                    batch_docids = []
                    batch_texts = []
    if batch_docids:
        yield batch_docids, batch_texts


def corpus_dir_for(language: str, data_dir: str | Path = "data") -> Path:
    """Path to the MIRACL corpus directory for a given language."""
    return Path(data_dir) / "miracl-corpus" / f"miracl-corpus-v1.0-{language}"


def load_dev_topics_and_qrels(
    language: str,
    data_dir: str | Path = "data",
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Load dev queries + qrels for a language. Used by both rerank and full eval."""
    data_dir = Path(data_dir)
    topics_path = (
        data_dir / "miracl" / f"miracl-v1.0-{language}"
        / "topics" / f"topics.miracl-v1.0-{language}-dev.tsv"
    )
    qrels_path = (
        data_dir / "miracl" / f"miracl-v1.0-{language}"
        / "qrels" / f"qrels.miracl-v1.0-{language}-dev.tsv"
    )
    if not topics_path.exists():
        raise FileNotFoundError(f"Missing MIRACL topics: {topics_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing MIRACL qrels: {qrels_path}")
    return _parse_topics(topics_path), _parse_qrels(qrels_path)


def load_miracl(
    language: str,
    data_dir: str | Path = "data",
    cache_dir: str | Path = "data/miracl_cache",
) -> MiraclEvalSet:
    """Load MIRACL dev evaluation set for a language.

    Uses a pickle cache to skip the corpus scan on subsequent loads.
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"miracl_{language}_dev.pkl"

    if cache_path.exists():
        print(f"Loading cached {language} eval set from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    topics_path = data_dir / "miracl" / f"miracl-v1.0-{language}" / "topics" / f"topics.miracl-v1.0-{language}-dev.tsv"
    qrels_path = data_dir / "miracl" / f"miracl-v1.0-{language}" / "qrels" / f"qrels.miracl-v1.0-{language}-dev.tsv"
    corpus_dir = data_dir / "miracl-corpus" / f"miracl-corpus-v1.0-{language}"

    for p in [topics_path, qrels_path, corpus_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing MIRACL file: {p}")

    print(f"Loading {language} topics from {topics_path.name}")
    topics = _parse_topics(topics_path)
    print(f"  {len(topics)} queries")

    print(f"Loading {language} qrels from {qrels_path.name}")
    qrels = _parse_qrels(qrels_path)

    needed_docids: set[str] = set()
    for q_qrels in qrels.values():
        needed_docids.update(q_qrels.keys())
    print(f"  {len(needed_docids)} unique passages needed")

    print(f"Scanning corpus for needed passages...")
    passages = _stream_corpus_passages(corpus_dir, needed_docids)
    print(f"  found {len(passages)} passages ({len(needed_docids) - len(passages)} missing)")

    # Build queries list, keeping only those that have qrels
    queries: list[MiraclQuery] = []
    for qid, text in topics.items():
        if qid in qrels:
            queries.append(MiraclQuery(query_id=qid, text=text, qrels=qrels[qid]))

    eval_set = MiraclEvalSet(language=language, queries=queries, passages=passages)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(eval_set, f)
    print(f"Cached to {cache_path}")

    return eval_set


# ── Metrics ───────────────────────────────────────────────────────────────────

def dcg(rels: list[int]) -> float:
    """Discounted cumulative gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(ranked_rels: list[int], k: int) -> float:
    """Normalised DCG at rank k.

    Args:
        ranked_rels: relevance labels in model-ranked order
        k: truncation rank

    Returns:
        nDCG@k in [0, 1]
    """
    if not ranked_rels:
        return 0.0
    dcg_at_k = dcg(ranked_rels[:k])
    ideal = sorted(ranked_rels, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg_at_k / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_rels: list[int], num_relevant: int, k: int) -> float:
    """Fraction of relevant items appearing in the top k.

    Args:
        ranked_rels: relevance labels in model-ranked order
        num_relevant: total number of relevant items for this query
        k: truncation rank
    """
    if num_relevant == 0:
        return 0.0
    found = sum(1 for rel in ranked_rels[:k] if rel > 0)
    return found / num_relevant
