"""API-only LangChain shopping backend. Run one worker; see README.md."""
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from src.api import api_router
from src.core import config
from src.core.logging import get_logger, new_request_id, quiet_access_log, request_id
from src.services import catalog, database, sessions

log = get_logger(__name__)


def warm_up():
    from src.services import review_search
    from src.ml import reranker
    # Loading the embedding/reranker models is CPU-heavy. Give authentication and
    # the first page render a head start instead of competing with them immediately.
    time.sleep(config.APP_WARMUP_DELAY_SECONDS)
    for label, callback in (("review embeddings and keyword index", review_search.warm_up),
                            ("reranker", reranker.warm_up)):
        try:
            callback()
        except Exception:
            log.exception("%s warm-up failed", label)


@asynccontextmanager
async def lifespan(app: FastAPI):
    quiet_access_log()
    log.info("LangChain shopping API: llm=%s, chroma=%s, answers=%s, warmup=%s",
             config.LLM_PROVIDER, config.CHROMA_MODE, config.ANSWER_STYLE, config.APP_WARMUP_ENABLED)
    for warning in config.CONFIG_WARNINGS:
        log.warning("%s", warning)
    try:
        await database.ping()
        await database.ensure_indexes()
        await sessions.ensure_indexes()
        await catalog.ensure_indexes()
    except Exception:
        log.exception("MongoDB setup failed; authentication and catalog queries need MongoDB")
    if config.APP_WARMUP_ENABLED:
        threading.Thread(target=warm_up, name="model-warm-up", daemon=True).start()
    yield
    database.close()


app = FastAPI(title="FitFinder AI — Fashion Shopping Assistant", version="6.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    token = request_id.set(new_request_id())
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id.get("")
        return response
    finally:
        request_id.reset(token)


@app.exception_handler(database.DatabaseUnavailable)
async def database_unavailable(request: Request, exc: database.DatabaseUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


if config.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                       allow_methods=["GET", "POST", "PATCH", "DELETE"],
                       allow_headers=["Authorization", "Content-Type"])
app.include_router(api_router)
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
