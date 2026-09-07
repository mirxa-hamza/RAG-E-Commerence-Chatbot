"""
Re-ranking - the precision stage of retrieval. Runs on this machine.

A bi-encoder (the embedding model) compresses a chunk into one vector *before* it has ever
seen the question, so ranking by vector distance is inherently coarse. A cross-encoder
reads the question and the chunk together and scores that specific pair, which is far more
accurate and far too slow to run over a whole corpus. Hence the standard shape: retrieve a
wide candidate set cheaply, then re-rank it properly and keep the best few.

The model is loaded lazily on first use (~80MB) so that startup, ingestion and the test
suite never pay for it when no question has been asked, and warmed on a background thread
from src/main.py so the first real question does not either.

It fails OPEN: if the model cannot load, rerank() returns None and the caller keeps its
fused order. Worse ranking beats a failed question - but it is invisible, which is why
is_available() is reported on /info.

Scores are unbounded logits: ms-marco models put clearly-irrelevant pairs well below zero.
Callers compare against score_floor() rather than a literal.
"""
import threading
import time
from typing import Dict, List, Optional, Tuple

from src.core.config import MIN_RERANK_SCORE, RERANK_MODEL
from src.core.logging import get_logger, timed

log = get_logger(__name__)

# How long to wait before retrying a failed model load. A transient failure (no network on
# the very first question, a half-written cache) used to disable re-ranking - the single
# biggest quality stage - for the entire life of the process, silently.
_RETRY_AFTER_SECONDS = 300

_lock = threading.Lock()
_model = None
_next_retry_at = 0.0


def _get_model():
    global _model, _next_retry_at
    if _model is not None:
        return _model
    if time.monotonic() < _next_retry_at:
        return None

    with _lock:
        if _model is None and time.monotonic() >= _next_retry_at:
            try:
                from sentence_transformers import CrossEncoder
                log.info("Loading re-ranker '%s' (first use downloads it)...", RERANK_MODEL)
                _model = CrossEncoder(RERANK_MODEL)
                log.info("Re-ranker loaded.")
            except Exception as exc:
                # No model, no network, or a stubbed sentence_transformers in tests:
                # degrade to fusion-only ranking rather than failing the request, and try
                # again later instead of giving up permanently.
                _next_retry_at = time.monotonic() + _RETRY_AFTER_SECONDS
                log.warning(
                    "Re-ranker unavailable (%s) - using fused ranking; retrying in %ds.",
                    exc, _RETRY_AFTER_SECONDS,
                )
    return _model


def warm_up() -> None:
    """
    Load the cross-encoder ahead of the first question.

    Measured cost of NOT doing this: ~15 seconds, paid by whoever asks the first question of
    a fresh process, on top of everything else retrieval does. The model is a lazy singleton
    for a good reason - loading it at import kept the port closed and gave the browser
    ERR_CONNECTION_REFUSED - but "lazy" only has to mean "not at import", not "on the
    critical path of a user request".
    """
    with timed(log, "re-ranker warm-up"):
        _get_model()


def provider_name() -> str:
    return "local"


def score_floor() -> float:
    """The minimum score a chunk must reach to be kept."""
    return MIN_RERANK_SCORE


def available() -> bool:
    """Whether the model can be loaded - loads it if it has not been tried yet."""
    return _get_model() is not None


def is_available() -> bool:
    """
    Whether re-ranking is working right now, WITHOUT triggering a load.

    Exposed on /info because this stage fails OPEN: if it cannot run, retrieval quietly
    falls back to a similarity floor and answers get worse with no error anywhere. That is
    the right runtime behaviour and the wrong thing to leave invisible.
    """
    return _model is not None


def reset_model() -> None:
    """Drop the cached model. Only for tests."""
    global _model, _next_retry_at
    with _lock:
        _model = None
        _next_retry_at = 0.0


def rerank(question: str, chunks: List[Dict]) -> Optional[List[Tuple[Dict, float]]]:
    """
    Returns [(chunk, score), ...] sorted best first, or None if re-ranking isn't available
    (caller then keeps its existing order).

    Compare the scores against score_floor(), not against a literal.
    """
    if not chunks:
        return None

    model = _get_model()
    if model is None:
        return None

    pairs = [(question, c["text"]) for c in chunks]
    with timed(log, f"re-rank {len(pairs)} candidates"):
        scores = model.predict(pairs)

    return sorted(zip(chunks, (float(s) for s in scores)), key=lambda p: p[1], reverse=True)
