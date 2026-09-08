"""Hybrid review retrieval with identity based on review_id AND chunk_index."""
import re
import threading
import time
from src.agent.schemas import ReviewQuery
from src.core import config
from src.core.logging import get_logger
from src.ml import reranker
from src.services.review_store import get_store

log = get_logger(__name__)
_cache = None
_lock = threading.Lock()


def where_filter(q: ReviewQuery):
    clauses = []
    if q.product_ids: clauses.append({"parent_asin": {"$in": q.product_ids}})
    if q.min_rating is not None: clauses.append({"rating": {"$gte": q.min_rating}})
    if q.max_rating is not None: clauses.append({"rating": {"$lte": q.max_rating}})
    return {"$and": clauses} if len(clauses) > 1 else clauses[0] if clauses else None


def row(text, meta, identity):
    return {"id": identity, "text": text, **meta}


def tokens(text):
    return re.findall(r"\w+", text.lower())


def keyword_rows(store):
    global _cache
    count = store._collection.count()
    with _lock:
        if _cache is None or _cache[0] != count or time.monotonic() - _cache[3] > 300:
            from rank_bm25 import BM25Okapi
            rows = []
            for offset in range(0, count, 1000):
                got = store.get(limit=1000, offset=offset, include=["documents", "metadatas"])
                rows.extend(row(t, m, i) for t, m, i in zip(got["documents"], got["metadatas"], got["ids"]))
            _cache = (count, rows, BM25Okapi([tokens(r["text"]) for r in rows]) if rows else None, time.monotonic())
        return _cache[1:3]


def matches(r, q):
    rating = r.get("rating")
    return ((not q.product_ids or r.get("parent_asin") in q.product_ids)
            and (q.min_rating is None or rating is not None and rating >= q.min_rating)
            and (q.max_rating is None or rating is not None and rating <= q.max_rating))


def search(q: ReviewQuery) -> dict:
    store = get_store()
    if not store._collection.count():
        return {"count": 0, "results": []}
    pairs = store.similarity_search_with_score(q.question, k=config.RETRIEVAL_CANDIDATES, filter=where_filter(q))
    dense = []
    for doc, distance in pairs:
        m = doc.metadata
        identity = f"{m['review_id']}::c{m['chunk_index']}"
        dense.append({**row(doc.page_content, m, identity), "similarity": max(0, 1-distance)})
    lists = [dense]
    if config.HYBRID_ENABLED and config.KEYWORD_SEARCH == "on":
        try:
            rows, index = keyword_rows(store)
            if index:
                ranked = sorted(zip(rows, index.get_scores(tokens(q.question))), key=lambda p: p[1], reverse=True)
                lists.append([r for r, score in ranked if score > 0 and matches(r, q)][:config.RETRIEVAL_CANDIDATES])
        except Exception:
            log.warning("Review keyword search unavailable; using dense search", exc_info=True)
    fused, scores = {}, {}
    for ranked in lists:
        for rank, r in enumerate(ranked, 1):
            fused.setdefault(r["id"], r)
            scores[r["id"]] = scores.get(r["id"], 0) + 1 / (config.RRF_K + rank)
    candidates = [fused[k] for k in sorted(scores, key=scores.get, reverse=True)][:config.RETRIEVAL_CANDIDATES]
    rescored = None
    if config.RERANK_ENABLED:
        try:
            rescored = reranker.rerank(q.question, candidates)
        except Exception:
            log.warning("Review reranking failed; using dense relevance floor", exc_info=True)
    selected = ([r for r, score in rescored if score >= config.MIN_RERANK_SCORE] if rescored is not None
                else [r for r in candidates if r.get("similarity", 0) >= config.MIN_SIMILARITY])[:q.limit]
    # Adjacent chunks belong to one review, not merely one product. Otherwise one
    # customer's statement would be concatenated with an unrelated customer's review.
    expanded = {r["id"]: r for r in selected}
    for r in selected:
        try:
            if config.NEIGHBOR_EXPANSION > 0:
                indices = list(range(max(0, r["chunk_index"]-config.NEIGHBOR_EXPANSION), r["chunk_index"]+config.NEIGHBOR_EXPANSION+1))
                got = store.get(where={"$and": [{"review_id": r["review_id"]}, {"parent_asin": r["parent_asin"]}, {"chunk_index": {"$in": indices}}]}, include=["documents", "metadatas"])
                for text, meta, identity in zip(got["documents"], got["metadatas"], got["ids"]):
                    expanded.setdefault(identity, row(text, meta, identity))
        except Exception:
            log.warning("Review neighbor expansion failed", exc_info=True)
    budget = config.MAX_CONTEXT_CHARS
    results = []
    for r in expanded.values():
        excerpt = r["text"][:min(budget, config.REVIEW_EXCERPT_CHARS)]
        if not excerpt or len(results) >= config.REVIEW_MAX_RESULTS_TO_MODEL: break
        results.append({"id": r["id"], "parent_asin": r["parent_asin"], "review_id": r["review_id"], "excerpt": excerpt, "rating": r.get("rating")})
        budget -= len(excerpt)
    return {"count": len(results), "results": results}


def warm_up() -> None:
    """Build the local review search dependencies off the request path."""
    store = get_store()
    if config.HYBRID_ENABLED and config.KEYWORD_SEARCH == "on":
        keyword_rows(store)
