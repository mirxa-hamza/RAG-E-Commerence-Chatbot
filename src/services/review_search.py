"""Hybrid review retrieval with identity based on review_id AND chunk_index."""
import re
from collections import Counter
from src.agent.schemas import ReviewQuery
from src.core import config
from src.core.logging import get_logger
from src.ml import reranker
from src.services.review_store import get_store

log = get_logger(__name__)

# Words that carry no lexical signal worth a literal document lookup. Kept deliberately
# small: this only decides which terms are worth asking the store about, and a term that
# slips through costs one extra substring filter, not a wrong answer.
_STOPWORDS = frozenset("""
a an and are as at be but by do does for from has have how i in is it its me my of on
or should that the their them there these they this to was were what when where which
who why will with you your about any can could would""".split())


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


def keyword_terms(question: str, limit: int = 6) -> list:
    """The content words from a question that are worth a literal document lookup."""
    picked = []
    for term in tokens(question):
        if len(term) < 3 or term in _STOPWORDS or term in picked:
            continue
        picked.append(term)
        if len(picked) >= limit:
            break
    return picked


def keyword_candidates(store, q: ReviewQuery) -> list:
    """The lexical half of hybrid retrieval, over a BOUNDED candidate set.

    This used to hold a BM25 index over the ENTIRE collection in process memory. At
    308k review chunks that meant gigabytes of resident Python tokens, a rebuild
    measured in minutes that every concurrent review query queued behind one lock, and
    an O(n log n) sort of the whole corpus to pick ten rows. Chroma already stores the
    documents on disk and can filter them by substring, so ask IT for the chunks
    containing the question's content words, and rank only those.

    Ranking is term COVERAGE (how many of the question's distinct content words a chunk
    contains, then how often), not BM25. BM25 would be actively wrong here: its IDF is a
    corpus statistic, and every candidate in this set already contains a query term by
    construction, so those terms look maximally common and their IDF collapses toward
    zero - on a small candidate set BM25 literally scores everything 0. Coverage needs no
    corpus statistics, which is the whole point of not holding the corpus in memory.
    Fusion only consumes rank order, and the cross-encoder downstream does the precision
    work regardless.
    """
    terms = keyword_terms(q.question)
    if not terms:
        return []
    contains = [{"$contains": term} for term in terms]
    got = store.get(where=where_filter(q),
                    where_document=contains[0] if len(contains) == 1 else {"$or": contains},
                    limit=config.KEYWORD_CANDIDATE_LIMIT,
                    include=["documents", "metadatas"])
    scored = []
    for text, meta, identity in zip(got.get("documents") or [], got.get("metadatas") or [],
                                    got.get("ids") or []):
        counts = Counter(tokens(text))
        present = [counts[term] for term in terms if counts[term]]
        if not present:
            continue
        scored.append(((len(present), sum(present)), row(text, meta, identity)))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [candidate for _, candidate in scored][:config.RETRIEVAL_CANDIDATES]


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
            keyword = keyword_candidates(store, q)
            if keyword:
                lists.append(keyword)
        except Exception:
            # Fails open, as before: a store that cannot do document filtering costs
            # lexical recall, not the answer.
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
    """Load the embedding model off the request path.

    Keyword search no longer builds anything shared, so there is nothing else to warm
    here - and nothing for a live request to queue behind while this runs.
    """
    get_store()
