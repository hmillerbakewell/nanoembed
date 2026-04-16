"""Evaluation metrics for static embedding models.

An Encoder is any ``list[str] -> np.ndarray`` callable returning L2-normalised
sentence embeddings. All metric functions take an Encoder, decoupling evaluation
from how embeddings are produced.
"""


from pathlib import Path
from typing import Callable

import numpy as np
import torch
from transformers import AutoTokenizer

from .miracl import (
    MiraclEvalSet,
    corpus_dir_for,
    iter_corpus_batches,
    load_dev_topics_and_qrels,
    ndcg_at_k,
    recall_at_k,
)
from models.model import EmbeddingModel

Encoder = Callable[[list[str]], np.ndarray]


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def make_local_encoder(
    model: EmbeddingModel,
    max_length: int = 128,
    cache_dir: str | None = None,
) -> Encoder:
    """Wrap a local EmbeddingModel as an Encoder callable.

    Uses the tokenizers library directly (no padding, no special tokens),
    matching the SentenceTransformer StaticEmbedding encode path exactly.
    Variable-length sequences are padded to the batch max and masked.
    """
    from tokenizers import Tokenizer as HFTokenizer

    # Try loading as a tokenizers.Tokenizer first (works for both model2vec
    # and NIFE models). Fall back to transformers AutoTokenizer if needed.
    try:
        tokenizer = HFTokenizer.from_pretrained(model.tokenizer_name)
        tokenizer.no_padding()
        tokenizer.no_truncation()
        use_hf_tokenizers = True
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_name, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        use_hf_tokenizers = False

    def encode(sentences: list[str]) -> np.ndarray:
        # Tokenize — no special tokens, no padding
        if use_hf_tokenizers:
            encoded = tokenizer.encode_batch(sentences, add_special_tokens=False)
            ids_list = [enc.ids for enc in encoded]
        else:
            ids_list = [
                tokenizer.encode(s, add_special_tokens=False)
                for s in sentences
            ]

        # Pad to max length in batch and build mask
        max_len = max(len(ids) for ids in ids_list) if ids_list else 0
        if max_len == 0:
            return np.zeros((len(sentences), model.embed_dim), dtype=np.float32)

        padded_ids = torch.zeros(len(sentences), max_len, dtype=torch.long)
        mask = torch.zeros(len(sentences), max_len)
        for j, ids in enumerate(ids_list):
            length = len(ids)
            padded_ids[j, :length] = torch.tensor(ids, dtype=torch.long)
            mask[j, :length] = 1.0

        # Remap token IDs if needed (for models with pruned vocab)
        if model.old_to_new:
            padded_ids.apply_(lambda x: model.old_to_new.get(x, 0))

        with torch.no_grad():
            embs = model.embed_ids(padded_ids, mask)
        return embs.numpy()

    return encode


# ── MIRACL rerank-style retrieval ─────────────────────────────────────────────

def _batched_encode(encoder: Encoder, texts: list[str], batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunks.append(encoder(texts[i:i + batch_size]))
    return np.concatenate(chunks, axis=0)


def evaluate_miracl(
    encoder: Encoder,
    eval_set: MiraclEvalSet,
    batch_size: int = 512,
) -> dict[str, float]:
    """Pooled-qrels MIRACL eval: rank each query against ALL pooled passages.

    For each query, every passage in the pool is a candidate. Passages not in
    the query's qrels are treated as implicit non-relevant. Other queries'
    annotated passages become hard distractors.

    Returns nDCG@10, recall@10, recall@3, num_queries. Macro-averaged.
    """
    print(f"  embedding {eval_set.num_passages} passages...")
    docids = list(eval_set.passages.keys())
    passage_texts = [eval_set.passages[d] for d in docids]
    passage_embs = _l2_normalise(_batched_encode(encoder, passage_texts, batch_size))

    print(f"  embedding {eval_set.num_queries} queries...")
    query_texts = [q.text for q in eval_set.queries]
    query_embs = _l2_normalise(_batched_encode(encoder, query_texts, batch_size))

    print(f"  ranking {eval_set.num_queries} queries against {eval_set.num_passages} passages...")
    sim_matrix = query_embs @ passage_embs.T   # both L2-normalised

    top_k = 10
    top_partition = np.argpartition(-sim_matrix, top_k, axis=1)[:, :top_k]
    top_sims = np.take_along_axis(sim_matrix, top_partition, axis=1)
    sorted_within = np.argsort(-top_sims, axis=1)
    top_indices = np.take_along_axis(top_partition, sorted_within, axis=1)

    ndcgs: list[float] = []
    recall_10: list[float] = []
    recall_3: list[float] = []
    skipped = 0

    for i, query in enumerate(eval_set.queries):
        num_relevant = sum(1 for r in query.qrels.values() if r > 0)
        if num_relevant == 0:
            skipped += 1
            continue

        ranked_rels: list[int] = []
        for row_idx in top_indices[i]:
            doc = docids[row_idx]
            ranked_rels.append(query.qrels.get(doc, 0))

        ndcgs.append(ndcg_at_k(ranked_rels, 10))
        recall_10.append(recall_at_k(ranked_rels, num_relevant, 10))
        recall_3.append(recall_at_k(ranked_rels, num_relevant, 3))

    return {
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall@10": float(np.mean(recall_10)) if recall_10 else 0.0,
        "recall@3": float(np.mean(recall_3)) if recall_3 else 0.0,
        "num_queries": float(len(ndcgs)),
        "skipped": float(skipped),
    }


# ── Full-corpus MIRACL retrieval (streaming) ──────────────────────────────────

def evaluate_miracl_full(
    encoder: Encoder,
    language: str,
    data_dir: str | Path = "data",
    batch_size: int = 10_000,
    top_k: int = 100,
    progress_every: int = 10,
) -> dict[str, float]:
    """Full-corpus MIRACL retrieval for one language, streaming the corpus.

    Unlike `evaluate_miracl` (rerank-style, ranks only ~30K pre-annotated
    passages), this streams the entire corpus through the encoder in batches,
    maintains a running top-K per query, and computes nDCG@10 + recall@100
    over the true ranking.

    Memory footprint:
      - query embs: Q × D × 4 bytes
      - running top-K: Q × top_k × 8 bytes
      - batch workspace: batch_size × D × 4 bytes + small overhead

    For Q=5000, D=256, batch_size=10000, top_k=100 → ~45 MB peak.

    Returns: nDCG@10, recall@100, num_queries, total_passages, total_s.
    """
    import time

    # ── 1. Load queries + qrels ──────────────────────────────────────────────
    topics, qrels = load_dev_topics_and_qrels(language, data_dir)
    corpus_dir = corpus_dir_for(language, data_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Missing MIRACL corpus dir: {corpus_dir}")

    # Keep only queries that have qrels
    q_items = [(qid, topics[qid]) for qid in topics if qid in qrels]
    if not q_items:
        return {
            "ndcg@10": 0.0, "recall@100": 0.0,
            "num_queries": 0.0, "total_passages": 0.0, "total_s": 0.0,
            "skipped": 0.0,
        }

    qids = [q[0] for q in q_items]
    query_texts = [q[1] for q in q_items]
    Q = len(qids)

    # ── 2. Embed all queries once ────────────────────────────────────────────
    t0 = time.perf_counter()
    query_embs = encoder(query_texts)
    query_embs = _l2_normalise(query_embs)
    query_t = torch.from_numpy(query_embs)                   # (Q, D)

    # ── 3. Stream corpus, maintain running top-K ─────────────────────────────
    top_k_scores = torch.full((Q, top_k), float("-inf"), dtype=torch.float32)
    top_k_gids = torch.full((Q, top_k), -1, dtype=torch.int64)
    all_docids: list[str] = []

    total_passages = 0
    batches_done = 0
    for batch_docids, batch_texts in iter_corpus_batches(corpus_dir, batch_size):
        B = len(batch_docids)
        batch_embs = _l2_normalise(encoder(batch_texts))
        batch_t = torch.from_numpy(batch_embs)               # (B, D)

        sims = query_t @ batch_t.T                           # (Q, B)

        gid_start = len(all_docids)
        batch_gids = torch.arange(gid_start, gid_start + B, dtype=torch.int64)
        all_docids.extend(batch_docids)

        # Combine running top-K with the new batch, keep top-K overall
        combined_scores = torch.cat([top_k_scores, sims], dim=1)            # (Q, K+B)
        combined_gids = torch.cat([
            top_k_gids,
            batch_gids.unsqueeze(0).expand(Q, -1),
        ], dim=1)
        new_scores, new_idx = torch.topk(combined_scores, top_k, dim=1)
        top_k_scores = new_scores
        top_k_gids = torch.gather(combined_gids, 1, new_idx)

        total_passages += B
        batches_done += 1
        if batches_done % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = total_passages / elapsed
            print(f"    {language}: {total_passages:>10,} passages processed "
                  f"({rate:>7.0f}/s)")

    # ── 4. Compute metrics per query ─────────────────────────────────────────
    ndcgs: list[float] = []
    recalls_100: list[float] = []
    skipped = 0
    for i, qid in enumerate(qids):
        q_qrels = qrels[qid]
        num_relevant = sum(1 for r in q_qrels.values() if r > 0)
        if num_relevant == 0:
            skipped += 1
            continue
        ranked_rels = [
            q_qrels.get(all_docids[gid], 0) for gid in top_k_gids[i].tolist()
        ]
        ndcgs.append(ndcg_at_k(ranked_rels, 10))
        recalls_100.append(recall_at_k(ranked_rels, num_relevant, 100))

    total_s = time.perf_counter() - t0

    return {
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall@100": float(np.mean(recalls_100)) if recalls_100 else 0.0,
        "num_queries": float(len(ndcgs)),
        "total_passages": float(total_passages),
        "total_s": float(total_s),
        "skipped": float(skipped),
    }
