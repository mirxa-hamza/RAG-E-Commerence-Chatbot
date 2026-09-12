# CLAUDE.md — FitFinder AI engineering guide

Updated 2026-09-10. This document describes the active ecommerce shopping assistant. Do not
apply the older PDF-RAG/Next.js notes that used to live here.

## Product and architecture

FitFinder AI is a FastAPI application with a static browser UI. The backend exposes the chat,
auth, session, preference, system, and ingestion APIs; the UI is served from `src/static/`.
There is no active `frontend/` or Next.js application.

The main request path is:

```text
src/static/app.js
  -> POST /chat/stream
  -> src/api/shopping.py
  -> direct_answers.py for deterministic shortcuts, otherwise src/agent/shopper.py
  -> LangChain tools (Mongo catalog, Chroma reviews, Mongo memory)
  -> grounded AnswerDraft
  -> hydrate/validate product cards and evidence
  -> SSE response and persistence in the session
```

Keep API modules thin. Business logic belongs in `src/services/`; agent behavior belongs in
`src/agent/`; configuration belongs in `src/core/config.py`.

## Data stores and identifiers

- MongoDB (`ecommerce_agent` by default) stores users, sessions/messages, durable shopping
  preferences, audit records, and the canonical product catalog (`products`). Product cards
  must be hydrated from Mongo by `parent_asin`; never trust a model-generated title or price.
- ChromaDB (`storage/chroma_db`, collection `amazon_fashion_reviews_384`) stores embedded review
  chunks for semantic retrieval. It is evidence, not the source of product truth.
- Raw ingestion inputs default to `data/meta_Amazon_Fashion.jsonl` and
  `data/Amazon_Fashion.jsonl`. Review IDs are generated as `<parent_asin>::r<index>` and chunk
  IDs as `<review_id>::c<chunk_index>`. Metadata includes `parent_asin`, `review_id`, rating,
  title, timestamp, and chunk index.
- Use `scripts/ingest_ecommerce.py` to build/update the stores and
  `scripts/verify_ecommerce.py` to verify counts and identifiers. Never edit Chroma files by
  hand.

## Agentic RAG and privacy rules

`src/agent/shopper.py` creates the LangChain agent, collects evidence, and produces the
structured `AnswerDraft` used by the UI. The active tools are:

- `catalog_search`: exact/filtered product retrieval from MongoDB.
- `semantic_review_search`: hybrid Chroma review retrieval (dense + keyword/RRF, with optional
  cross-encoder reranking).
- `memory_lookup` and `remember_preference`: durable preference reads/writes in MongoDB.

Tool calls must use the authenticated user and session IDs closed over by the server. Never let
the model select or override those IDs. `hydrate()` must reject unknown product IDs, discard
stale evidence, and ensure every displayed price/title comes from the current tool results.
Answers must be natural prose (no Python list repr such as `['black']`, no raw internal fields,
and no invented products). If exact filters return no products, perform a clearly labelled,
relaxed alternative search when the user asks for alternatives; otherwise explain which
constraint failed.

Memory is user-scoped and durable in MongoDB. LangGraph checkpointing/store settings are useful
for the running process, but must not be treated as a replacement for Mongo persistence.

## Providers and configuration

The provider is selected in `.env` through the project settings (Groq or Google Gemini). Model,
timeouts, retry limits, retrieval counts, reranking, and warmup are all configurable in
`src/core/config.py`. Restart Uvicorn after changing `.env`. Never commit, print, or paste API
keys; rotate any credential that has appeared in logs or chat history.

Hosted models use a tool-enabled phase followed by schema-only finalization where required. Do
not combine JSON mode/`response_format` with tool calling on providers that reject that request.
Keep retries bounded and preserve the fallback/direct-answer path for simple local queries.

## Performance and operations

The BGE embedding model and optional cross-encoder are CPU-heavy and load lazily. Startup warmup
runs in a background thread (`APP_WARMUP_ENABLED`, with a default delay of 20 seconds) so login
is not blocked; the first retrieval after a cold start can still be slower. Run one Uvicorn
worker when using embedded Chroma. Keep `RETRIEVAL_CANDIDATES`, `REVIEW_MAX_RESULTS_TO_MODEL`,
`AGENT_MAX_TOOL_ROUNDTRIPS`, and `AGENT_MODEL_RETRIES` small enough for the configured provider.
Use `/info` to inspect model, store, count, reranker, and cache status.

When changing embedding models, dimensions, chunking, or review metadata, ingest into a new
collection (or explicitly force a rebuild) and verify it before switching production settings.
Do not open the same embedded Chroma directory for concurrent write/indexing processes.

## Frontend conventions

The browser UI lives in `src/static/` (`index.html`, `app.js`, and CSS files). Theme styles must
define both light and dark values for backgrounds, text, buttons, product/image panels, review
evidence, hover/focus states, and the composer. Keep the composer height and positioning
independent of theme. After static changes, update the query-string cache version in
`src/static/index.html` when browser caching would otherwise hide the fix.

## Testing and useful commands

Use the application virtual environment:

```powershell
.\.venv-app\Scripts\python.exe -m pytest tests -q
python -m uvicorn src.main:app --port 8000 --workers 1
python scripts/verify_ecommerce.py
python evaluation/run.py
```

Tests should use isolated fixtures/fake providers and must not require live Groq/Gemini,
MongoDB, or a pre-existing Chroma index unless explicitly marked as an integration test. Add
regression coverage for provider failures, malformed tool arguments, no-match searches, memory
privacy, product price/title grounding, and light/dark rendering behavior.

## Change checklist

Before handing off a change:

1. Run `git diff --check` and the focused tests, then the full test suite when practical.
2. Confirm new product claims are backed by Mongo results and review claims by Chroma evidence.
3. Check that loading, timeout, and provider errors remain user-readable and do not leak keys or
   internal IDs.
4. Keep generated indexes, local databases, `.env`, and downloaded model caches out of commits.
