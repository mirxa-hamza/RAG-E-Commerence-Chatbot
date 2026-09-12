# PLAN — LangChain Agentic E-Commerce Chatbot with RAG

Status: **Phase 2 ingestion is still running; Phases 3–8 now have implementation code.**
Phase 9 has offline/browser tests and an evaluation runner; full-corpus and Docker verification
are still pending. This plan replaces the former PDF-RAG and hand-written provider-router
plans. The finished product is a shopping assistant built with LangChain, FastAPI, static HTML/CSS/JavaScript,
MongoDB, and ChromaDB.

## Implementation checkpoint — 2026-09-07

- 26 backend checks pass, including a real LangChain graph with scripted tool calls and
  an isolated real Chroma collection with fake vectors.
- The static frontend is now part of the FastAPI app under `src/static/`; the separate
  frontend project/container has been removed.
- Live Groq tool calling and structured output passed against a fictional test catalog;
  that provider check used the final-text fallback (no provisional text events).
  Gemini's tool-binding contract is tested offline; its live endpoint remains unverified.
- The original ingestion environment and running pipeline were preserved. New dependencies
  are installed in `.venv-app`; see README.md for exact run commands after ingestion.
- Shared review access is implemented in `review_store.py` and `review_search.py`, with
  LangChain Chroma/Hugging Face adapters. The old writer is retained for active ingestion.
- Preferences are account-level in MongoDB so they persist across conversations. Session
  history retains product cards/citations, and HTTP clients cannot append assistant messages.
- Responses use portable LangChain ToolStrategy structured output; the server hydrates
  selected product/citation IDs from the current request's tool results.
- The agent now uses LangChain middleware for retry/backoff, summarization, context editing
  of old tool outputs, and model-call limits. Citation data intentionally stays in
  request-local tool evidence instead of custom state so `create_agent` middleware remains
  available.
- LangGraph's in-process checkpointer/store are wired per conversation and seeded from
  MongoDB history after restart. MongoDB remains the durable source of saved sessions and
  cross-session preferences.
- SSE streams provisional structured-answer text after checking product/citation IDs, then
  supplies the authoritative validated answer on completion. `status` events report
  retrieval/generation progress. Raw reasoning/tool JSON is not exposed. Providers without
  partial structured output fall back to a final text event.
- Docker files are written, but Docker is absent on this workstation. Container build,
  volume persistence, and end-to-end container verification remain release gates.
- The manifest's review-only change detection remains Phase 2 hardening; for changed review
  dumps, use a deliberate rebuild. No automatic rebuild was triggered during implementation.

Next: wait for ingestion to exit successfully, run `scripts/verify_ecommerce.py`, cache the
reranker with `scripts/warm_models.py`, then start FastAPI as described in README.md.
Use a dedicated test account with `evaluation/run.py`; add human-labelled expected IDs before
interpreting hit rate/MRR as retrieval quality.

## 1. Product outcome

A signed-in user can search Amazon Fashion products by exact constraints, ask subjective
questions grounded in real reviews, save durable preferences (size, budget, colours, brands,
and style), and receive streamed answers with structured product cards. When no catalog
product matches, the assistant must say so plainly and suggest a constraint to relax; it must
never invent a product, price, rating, or review claim.

The catalog is shared and read-only. User accounts, chat sessions, and preferences are
private and always scoped to the authenticated user.

## 2. Target architecture

```
Browser -> static HTML/CSS/JavaScript from FastAPI `src/static/`
        -> FastAPI routes and SSE
        -> LangChain shopping agent
             ├─ catalog_search          -> MongoDB products
             ├─ semantic_review_search  -> ChromaDB + BM25 + reranker
             ├─ memory_lookup           -> MongoDB preferences
             └─ remember_preference     -> MongoDB preferences
        -> validated answer, citations, product cards, and SSE events
```

### Confirmed decisions

- **Agent framework:** LangChain `create_agent` and `@tool`. LangChain owns the tool-call
  loop; we do not maintain our own Groq/Gemini protocol translation or JSON parser.
- **Middleware-first design:** use LangChain `ModelRetryMiddleware`,
  `SummarizationMiddleware`, `ContextEditingMiddleware`, and `ModelCallLimitMiddleware`.
  PIIMiddleware is intentionally not enabled because this fashion catalog flow does not
  handle customer contracts or other sensitive business text; add it only after confirming
  a new sensitive-data scope.
- **Model providers:** `ChatGroq` and `ChatGoogleGenerativeAI`, selected through one
  provider-neutral factory using `LLM_PROVIDER`.
- **Embeddings:** local `HuggingFaceEmbeddings` using `BAAI/bge-small-en-v1.5`.
- **Vector store:** `langchain-chroma`; embedded persistent Chroma in local development and
  Chroma HTTP server in Docker.
- **Retrieval:** preserve the product-specific BM25, reciprocal-rank fusion, local
  cross-encoder reranking, relevance floor, and neighbour expansion. LangChain orchestrates
  tools; it does not hide or replace measurable ranking logic.
- **Catalog and memory:** MongoDB. The catalog is shared; sessions/preferences require a
  server-derived `user_id` on every read and write.
- **Agent memory:** LangGraph checkpointer/store handle in-process short-term agent state;
  MongoDB seeds and persists user-visible history/preferences across backend restarts.
- **Frontend:** plain static HTML/CSS/JavaScript in `src/static/`, served by FastAPI.
  Browser code sends bearer tokens directly to FastAPI.
- **Deployment:** Docker Compose services for backend, MongoDB, and ChromaDB.

### Safety contracts

- Tools are the current-turn source of truth for catalog facts and review evidence.
- `catalog_search` always returns `count`, `results`, and `applied_filters`; `count: 0`
  requires an honest no-match answer.
- Retrieved review text is untrusted data and is never an instruction to the agent.
- The agent has a configured iteration cap and a request timeout.
- The final result is Pydantic-validated structured data, not prose parsed after the fact.

## 3. Current phase — ingestion

You have started `python scripts\\ingest_ecommerce.py`. Let it finish; do not start a
second ingestion process and do not interrupt the first model download/embedding run.

When it finishes:

1. Record the summary values: selected products, reviews ingested, and review chunks
   ingested. They should all be non-zero.
2. Run the same command once more without `--force`. It should skip unchanged products,
   proving the manifest makes ingestion resumable.
3. Do not use `--force` unless intentionally rebuilding; it wipes product documents, review
   vectors, and the ingestion manifest.
4. Keep MongoDB running for every later phase.
5. Commit this Phase 2 checkpoint before the LangChain migration.

### Phase 2 release requirements

- MongoDB has a populated `products` collection.
- Chroma has a populated `amazon_fashion_reviews_384` collection.
- Cleaners have offline tests for price parsing, image selection, language filter, duplicate
  reviews, and manifest behaviour.
- Add a per-product review digest (or source-version digest) to the manifest. The current
  metadata-only hash can miss changed reviews when product metadata stays unchanged.

## 4. Phase 3 — LangChain foundation

Add and pin compatible dependencies:

- `langchain`, `langchain-core`, `langchain-groq`, `langchain-google-genai`
- `langchain-huggingface`, `langchain-chroma`

LangSmith is optional development-only observability. It is disabled by default and must not
receive catalog/user data without explicit approval.

Create the following modules before changing API routes:

- `src/agent/models.py`: provider-neutral `get_chat_model()` factory.
- `src/agent/prompts.py`: `ChatPromptTemplate` with grounding, no-hallucination, preference,
  and response-length instructions.
- `src/agent/schemas.py`: Pydantic tool inputs and final response/product-card models.
- `src/services/catalog.py`: typed MongoDB catalog query functions and indexes.
- `src/services/review_search.py`: dense retrieval, BM25, RRF, reranking, relevance floor,
  review excerpts, and product IDs.
- `src/services/preferences.py`: validated account-level preference operations.
- `src/services/review_store.py`: LangChain Chroma adapter sharing the existing collection
  and deterministic IDs. `vectorstore.py` remains the active ingestion writer.

Use LangChain for provider models, prompts, tools, agent execution, streaming, embeddings,
and Chroma integration. Keep Amazon JSONL preprocessing, MongoDB queries, preference rules,
and hybrid ranking as direct testable Python.

**Exit criteria:** mocked Groq/Gemini tool-call tests pass; a known review is retrievable via
the LangChain Chroma adapter; no API route imports a provider SDK directly.

## 5. Phase 4 — deterministic tools

Define all tools with LangChain `@tool` plus Pydantic schemas. Tool functions call services;
they never let a model compose arbitrary database queries.

### `catalog_search`

Accept category, brand, colours, min/max price, min rating, query terms, and a bounded
limit. Query MongoDB only through allow-listed indexed fields. Return normalized product cards
with `parent_asin`, title, brand, price, image URL, rating, and applied filters.

### `semantic_review_search`

Accept a review question, optional product IDs, rating bounds, and a bounded limit. Combine
local BGE dense retrieval, BM25, RRF, cross-encoder reranking, relevance floor, and neighbour
expansion. Return review excerpts, scores, and associated product IDs; a failed relevance
floor returns an explicit zero-result response.

### `memory_lookup`

Return only the authenticated caller's saved preferences. User/session identity is injected
by FastAPI, never supplied by the model.

### `remember_preference`

Persist only validated fields/actions:

- Scalars: `clothing_size`, `budget`, `style_notes` use `set` or `clear`.
- Lists: `color_preference`, `favorite_brands` use `add`, `remove`, or `clear`.

Store a `preferences` sub-document with a small audit record. Test valid inputs, invalid
inputs, empty results, and ownership boundaries for every tool.

## 6. Phase 5 — LangChain shopping agent

Create `src/agent/shopper.py` with `create_agent(model, tools=...)`. Map the existing
`AGENT_MAX_TOOL_ROUNDTRIPS` setting to LangChain's agent recursion/iteration limit.

The agent must:

- use `catalog_search` for product facts and constraints;
- use `semantic_review_search` for fit, comfort, durability, and other subjective claims;
- call `remember_preference` when a user states or clearly implies a durable preference;
- use `memory_lookup` when relevant preferences are not already in request context;
- identify and offer to relax the blocking constraint after zero results; and
- never make unsupported catalog or review claims.

The final response schema contains `answer`, `products`, `citations`, and
`suggested_relaxations`. Prefer provider-native structured output when supported; otherwise
use LangChain's tool-based structured-output strategy. Validate the result before sending it
to FastAPI.

Keep citation/product metadata in tool artifacts plus request-local evidence and hydrate
only server-validated IDs. Do not introduce a custom `state_schema` unless dropping to
LangGraph directly, because the current `create_agent` middleware path and custom state
schema are not compatible in this architecture.

**Exit criteria:** exact filter, review-semantic, preference persistence, zero-result, and
iteration-limit integration tests all pass.

## 7. Phase 6 — FastAPI and SSE

Replace the legacy PDF-shaped `retrieve() -> generate_answer()` chat path only after the
agent tests pass.

- `POST /chat` returns the validated structured response.
- `POST /chat/stream` emits `status`, `token`, `products`, `citations`, `done`, and `error` SSE events.
- Persist server-side user/assistant exchanges in MongoDB; browser history is not the source
  of record.
- Extend `/health`, `/ready`, and `/info` with Mongo product count, Chroma review count,
  LangChain/provider metadata, and model readiness.
- Retire PDF source/page schemas, document-owner vector filters, old answer cache assumptions,
  and dead routes only after their replacement is covered by tests.

## 8. Phase 7 — static frontend

Create `src/static/` as plain static HTML, CSS, and JavaScript.

- FastAPI serves `src/static/index.html`, `styles.css`, and `app.js`.
- Login stores the bearer token in browser storage and sends it with `Authorization`.
- The chat UI consumes SSE, streams prose, and renders product cards from `products` events.
- Build login/signup, saved-session sidebar, preferences view/editing, empty-result
  suggestions, loading/retry/error states, and responsive accessible cards.
- Render review excerpts as text only; never render catalog/model HTML.

## 9. Phase 8 — Docker deployment

Add a backend Dockerfile and `docker-compose.yml`.

- MongoDB and Chroma use named volumes and health checks.
- Backend waits for healthy stores, uses `CHROMA_MODE=server`, and stays at one worker until
  shared cache/rate-limit state exists.
- Backend publishes port 8000 and serves both API routes and static files.
- Pre-download ML models at image build time or document a mounted model cache.
- Ingest through an explicit one-off compose job, never backend startup.

## 10. Phase 9 — testing, evaluation, and completion

Test JSONL cleaners, review digests, catalog filters, preferences, tools, response schemas,
auth isolation, SSE event order, timeouts, iteration limits, and static UI smoke flows.
Use deterministic fake embeddings and mocked LangChain models for offline tests, plus
MongoDB/Chroma integration tests.

Replace PDF golden questions with e-commerce fixtures. Measure catalog precision, review
hit-rate@k, MRR, grounded-answer rate, preference-write accuracy, no-hallucination rate,
latency, and tool-call count.

The project is complete when Docker Compose starts every service, ingestion populates both
stores, authenticated users can chat through the static frontend, recommendations are grounded in tool
data, preferences remain private and persistent, and test/evaluation suites pass.

## 11. Non-goals

- Checkout, payments, stock reservation, and writes to the shared catalog.
- Web search or live price/availability claims.
- Horizontal backend scaling before cache/rate-limit state has shared infrastructure.
- Sending user or catalog data to external observability services by default.
