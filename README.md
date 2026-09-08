# FitFinder AI — LangChain fashion shopping assistant

A static HTML/CSS/JavaScript shopping interface backed by FastAPI and a LangChain agent. It searches a shared
Amazon Fashion catalog in MongoDB and uses locally embedded customer reviews in Chroma.
Preferences persist privately in the user's account across conversations.

## Current implementation

- LangChain agent with Groq and Gemini adapters, typed catalog/review/memory tools, an
  explicit model-call limit, timeouts, and validated answers.
- Local BGE embeddings compatible with the existing ingestion vectors, hybrid BM25/RRF
  retrieval, cross-encoder reranking, relevance floor, and same-review neighbour expansion.
- Account authentication, atomic saved exchanges, account-owned preferences, and SSE.
- Static frontend served by FastAPI, with bearer auth, session history, product cards and citations.
- Three-service Docker Compose setup, offline backend tests, and a live evaluation CLI.

The running Phase 2 ingestion must finish before opening its embedded Chroma index from the
API or verification script. Do not run a second ingestion process against the same store.
The original ingestion environment is `venv`; this development session installed the
application dependencies in `.venv-app`, which shares read-only access to the existing ML
packages and has its own LangChain packages.

## Run locally after ingestion finishes

Keep the MongoDB Windows service running. From the project root:

```powershell
.\.venv-app\Scripts\python.exe scripts\verify_ecommerce.py
.\.venv-app\Scripts\python.exe scripts\warm_models.py
.\.venv-app\Scripts\python.exe -m uvicorn src.main:app --port 8000 --workers 1
```

Open http://localhost:8000 and create an account. API documentation is at
http://localhost:8000/docs. Stop the API before any embedded-store re-ingestion.
The model warm-up command may download the reranker on first use.

For a fresh checkout, create a normal virtual environment and install
`requirements-agent.txt`; `.venv-app` is a local convenience and is not committed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-agent.txt
```

Copy `.env.example` to `.env` if starting fresh. Set `LLM_PROVIDER` and its API key,
`MONGO_URI`, and a persistent `JWT_SECRET`. Settings are read once per process.
The static frontend lives in `src/static/` and is served by FastAPI. There is no separate
frontend project or Node development server in the runtime path.

## API and data contracts

- `POST /api/signup`, `POST /api/login`, `GET /api/me`: accounts and bearer tokens.
- `GET/POST /api/sessions`, `GET/PATCH/DELETE /api/sessions/{id}`: private history.
- `GET/PATCH /api/preferences`: persistent account memory.
- `POST /chat`: `{"question":"Find dresses under $40","session_id":null}`.
- `POST /chat/stream`: same request; SSE events are `session`, `status`, `products`,
  `citations`, `token`, `done`, or `error`, with heartbeat comments during generation.
- `GET /health`, `/ready`, `/info`: process, store readiness and counts.

Chat responses contain `session_id`, `answer`, `products`, `citations`, and
`suggested_relaxations`. The backend accepts only the question/session ID; history is
read from MongoDB. Browser clients cannot append arbitrary assistant messages.

The static frontend stores the bearer token in `localStorage` and sends it in the
`Authorization` header. This is the normal tradeoff for a pure browser-only static client;
the previous httpOnly cookie protection depended on the removed Next.js server proxy.

`status` reports `retrieving` and `generating` phases so the UI can show progress before
validated text is ready. The SSE transport streams provisional answer text from the
structured response after its product/citation IDs can be checked. It never exposes
reasoning or other tool JSON. The `done` event supplies the authoritative, validated
answer; errors discard provisional text.
Providers without partial structured output deliver the answer in one final `token` event.
Cards are hydrated from returned tool data, never from model-supplied prices or image URLs.
All-empty catalog searches have a deterministic no-match answer.
Prose grounding still depends on model instruction; review the actual citations.

The agent uses LangChain 1.x `create_agent` with middleware for model-call limits, model
retry/backoff, conversation summarisation, and clearing old retrieved tool payloads from
context. Citation metadata stays in request-local tool evidence and is validated by the
server; this avoids the current `create_agent` limitation around combining custom state
schemas with middleware. LangGraph's in-process checkpointer/store are enabled for short
term agent memory, while MongoDB remains the durable source of sessions and preferences.
Prompt caching is not claimed for Groq; cost control comes from summarisation, trimming,
bounded tools, and short answer budgets.

Color matching uses literal text in product titles/features/descriptions because the current
ingestion does not retain a normalized color field. Clothing size is a preference, not a
verified inventory filter. Prices come from the historical dataset, not live Amazon stock.

## Ingestion

`scripts/ingest_ecommerce.py` is the existing three-pass preprocessing/chunking writer.
Do not change its vector dimensions, query prefix or normalization without a new index.
The app reads the same collection through `src/services/review_store.py`; ingestion keeps
its original writer until the running batch completes.

The current ingestion manifest detects product metadata changes, not review-only changes.
For an updated review dump, a deliberate full rebuild is required. No rebuild is performed
automatically. `--force` deletes generated catalog/review data; use it only deliberately.

## Tests and evaluation

```powershell
.\.venv-app\Scripts\python.exe -m pytest -q
.\.venv-app\Scripts\python.exe scripts\check_provider.py
```

A fresh test environment should install `requirements-test.txt`.
Backend tests use a real LangChain graph with a scripted model, isolated Mongo mocks, and
a real ephemeral Chroma collection with fake vectors. Browser tests were removed with the
separate frontend project; use FastAPI at `/` for manual UI smoke testing after ingestion.
`scripts/check_provider.py` makes one small live provider test using fictional data.

After ingestion, create a dedicated evaluation account and run:

```powershell
.\.venv-app\Scripts\python.exe evaluation\run.py --username evaluation-user
```

This spends LLM tokens and saves preferences/conversations to that account. Add
`expected_product_ids` and `expected_review_ids` to evaluation cases after manually
labelling your corpus to enable hit rate and MRR. Those metrics report null when unlabelled.
Passing mocked tests is not evidence of answer quality on the actual corpus.

## Docker

Docker Desktop with Linux containers is required. This workstation did not have Docker
available during implementation, so a container build/run still needs verification.

```powershell
docker compose up --build -d
docker compose run --rm backend python scripts/warm_models.py
docker compose run --rm backend python scripts/ingest_container.py
docker compose run --rm backend python scripts/verify_ecommerce.py
```

Open http://localhost:8000. Named volumes preserve MongoDB, Chroma, model cache, and the
manifest. Docker uses separate stores from your existing local ingestion, so the Docker
ingestion is a new batch; it does not reuse local MongoDB automatically.

Only the FastAPI port is published. FastAPI serves both the API and `src/static/`. TLS
termination and production secrets management remain infrastructure work.
LangSmith tracing is disabled unless `AGENT_TRACING_ENABLED=true`; its SDK is a transitive
LangChain dependency but no tracing subscription is required.

## Architecture references

Provider/agent integration follows the official
[LangChain agent documentation](https://docs.langchain.com/oss/python/langchain/agents),
[Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma),
and the standard FastAPI static-files pattern.
