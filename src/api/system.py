"""Liveness and diagnostics."""
import uuid

from fastapi import APIRouter, Response

from src.core.config import (
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    GROQ_MODEL,
    LLM_PROVIDER,
    RERANK_MODEL,
)
from src.services import review_store
from src.services import answer_cache

router = APIRouter(tags=["system"])

# A new identity for every server start. The browser stores the one it signed in under and
# compares it on load: same id means "this is the same running server, stay signed in";
# a different id means the project was restarted, which is when a fresh sign-in is wanted.
# It is not a secret - it is a random label with nothing derived from it.
BOOT_ID = uuid.uuid4().hex


@router.get("/health")
def health():
    """Process liveness and diagnostic status of the lazily loaded embedding model."""
    return {"status": "ok", "embedding_model_ready": review_store.is_ready(),
            "boot_id": BOOT_ID}


@router.get("/ready")
async def ready(response: Response):
    """Check stores and configured credentials; models load lazily on the first request.

    This does not call the provider or prove model downloads will succeed. Use the explicit
    warm-up/provider checks before deployment; empty/unavailable stores return 503.
    """
    from src.services import database

    model_ready = review_store.is_ready()
    try:
        await database.ping()
        db_ready = True
    except Exception:
        db_ready = False

    from src.services import catalog
    from src.core import config
    import asyncio
    try:
        review_count = await asyncio.to_thread(review_store.count)
        product_count = await catalog.collection().count_documents({}) if db_ready else 0
    except Exception:
        review_count = product_count = 0
    credentials_ready = bool(config.GROQ_API_KEY if config.LLM_PROVIDER == "groq" else config.GEMINI_API_KEY)
    ok = db_ready and review_count > 0 and product_count > 0 and credentials_ready
    if not ok:
        response.status_code = 503
    return {
        "ready": ok,
        "embedding_model_ready": model_ready,
        "database_ready": db_ready,
        "catalog_ready": product_count > 0,
        "reviews_ready": review_count > 0,
        "provider_configured": credentials_ready,
    }


@router.get("/api/health/auth")
async def auth_health():
    """
    Whether MongoDB is reachable. The login screen calls this so a failure says "the
    database is down" rather than "incorrect username or password".
    """
    from src.services import database

    try:
        await database.ping()
        return {"database": "ok"}
    except database.DatabaseUnavailable as exc:
        return {"database": "unavailable", "detail": str(exc)}


@router.get("/info")
async def info():
    """Which models this instance is actually running - the first thing to check when
    answers look different from what you expected."""
    from src.ml import reranker

    from src.services import catalog, review_store
    from src.core import config
    import asyncio
    try:
        products = await catalog.collection().count_documents({})
    except Exception:
        products = None
    try:
        reviews = await asyncio.to_thread(review_store.count)
    except Exception:
        reviews = None
    return {
        "agent_framework": "langchain",
        "mongo_products_count": products,
        "chroma_reviews_count": reviews,
        "subset_size": config.PRODUCT_SUBSET_SIZE,
        "chroma_mode": config.CHROMA_MODE,
        "streaming_mode": "provisional_text_then_validated_result",
        "chunking": "semantic",
        "embeddings_provider": "langchain-huggingface (local)",
        "reranker_provider": reranker.provider_name(),
        "vector_store": "langchain-chroma",
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "llm_provider": LLM_PROVIDER,
        "llm_model": GEMINI_MODEL if LLM_PROVIDER == "gemini" else GROQ_MODEL,
        # Re-ranking is the largest single quality stage, and it degrades SILENTLY to a
        # cosine floor if the model cannot load. Without this, the only evidence is one log
        # line at startup and noticeably worse answers weeks later.
        "reranker_available": reranker.is_available(),
        "answer_cache": {"enabled": config.ANSWER_CACHE_ENABLED, **answer_cache.stats()},
    }
