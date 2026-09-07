# Marginalia — a from-scratch RAG system

*Answers from your own pages, with the citations to prove it.*

PDF in → **semantically chunked and embedded on your machine** → stored in a local ChromaDB
folder → retrieved at question time → answered by an LLM, grounded only in what's in the PDF.

No LangChain, no LlamaIndex — every step is plain Python so you can see exactly what happens
at each stage.

**Everything runs locally except the final answer.** Text extraction, chunking, embedding,
the vector store and re-ranking all happen in this process, on this CPU, with no API key and
no network. The one exception is answer generation: the passages that matched your question
(up to ~24,000 characters) are sent to Groq or the Gemini API to be written up. Nothing else
leaves the machine.

**It is multi-tenant.** Accounts live in a local MongoDB; documents belong to the account
that uploaded them, and no filter, search or answer ever crosses that line. Tokens are
revocable (`token_version`), requests are rate limited, and each account has a storage quota.
The API is plain HTTP, so put TLS in front of it before it leaves localhost.

## Stack

| Stage               | Tool                                                                  | Where |
|---------------------|-----------------------------------------------------------------------|-------|
| PDF text extraction | `PyMuPDF`                                                             | local |
| Chunking            | semantic (embedding-breakpoint) chunker, `src/services/chunking.py`    | local |
| Embeddings          | `sentence-transformers` — `BAAI/bge-small-en-v1.5`                     | local |
| Vector store        | ChromaDB, persistent folder, no server to run                          | local |
| Keyword search      | `rank-bm25`, fused with vector search via Reciprocal Rank Fusion       | local |
| Re-ranking          | `cross-encoder/ms-marco-MiniLM-L-6-v2`                                 | local |
| Answer generation   | Groq (default) or the Gemini API                                       | remote |
| Accounts / history  | MongoDB                                                                | local |
| Backend             | FastAPI + Uvicorn                                                      | local |
| Frontend            | Plain HTML/CSS/JS, no build step                                       | local |

> **License note:** `PyMuPDF` is AGPL-3.0 (unlike the permissively-licensed libraries above).
> Fine for personal/learning use; if you open-source or sell this, either comply with AGPL or
> buy Artifex's commercial PyMuPDF licence.

## Project layout

```
.
├── src/
│   ├── main.py                app assembly: router, CORS, lifespan, static mount
│   ├── api/                   HTTP layer — thin handlers
│   │   ├── auth.py            signup, login, password change, account deletion
│   │   ├── chat.py            /chat, /chat/stream (SSE)
│   │   ├── documents.py       /ingest, /stats, /upload, /reset, DELETE /documents
│   │   ├── sessions.py        saved conversations
│   │   └── system.py          /health, /ready, /info
│   ├── core/
│   │   ├── config.py          every setting, loaded once from .env
│   │   ├── logging.py         logging config + per-stage timing helper
│   │   └── ratelimit.py       in-memory sliding-window limiter
│   ├── models/schemas.py      pydantic request/response shapes
│   ├── ml/
│   │   ├── embeddings.py      sentence-transformers + truncation guard
│   │   ├── reranker.py        lazy cross-encoder singleton
│   │   ├── llm.py             grounded prompt, provider dispatch, streaming
│   │   └── gemini.py          the Gemini API backend
│   ├── services/
│   │   ├── ingestion.py       the ONLY path documents enter the system
│   │   ├── pdf.py             extraction + sentence splitting
│   │   ├── chunking.py        the semantic chunker
│   │   ├── vectorstore.py     ChromaDB add / query / neighbours / delete / reset
│   │   ├── bm25.py            per-user in-memory BM25 keyword index
│   │   ├── retrieval.py       fusion → re-rank → floor → neighbour expansion
│   │   ├── manifest.py        what's ingested, with content hashes
│   │   ├── answer_cache.py    per-user cache of finished answers
│   │   ├── uploads.py         the request-bytes → trusted-file boundary
│   │   ├── ownership.py       document ownership and account deletion
│   │   ├── database.py        MongoDB client, indexes, audit log
│   │   ├── security.py        JWT + Argon2id
│   │   └── sessions.py        chat history
│   └── static/                the web UI, served at "/" by FastAPI
├── scripts/
│   ├── run.py                 start the server and open the browser when it answers
│   ├── ingest.py              build the index without starting the API
│   ├── verify_index.py        sanity-check the store against the data folder
│   └── backup.py              archive data/, the index and the database
├── eval/                      golden questions + hit-rate/MRR harness
├── tests/                     offline checks, no model download and no network
├── data/                      your PDFs go here
├── storage/chroma_db/         generated index state (gitignored)
└── requirements.txt · .env.example
```

## How a question gets answered

**At ingestion time** (CLI, server startup, or `POST /ingest`):

1. The backend scans `data/` and fingerprints each PDF (SHA-256). New or changed files are
   ingested; unchanged ones are skipped.
2. `PyMuPDF` extracts text page by page, preserving paragraph breaks, and it is split into
   sentences that each remember their page.
3. **Semantic chunking.** Every sentence is embedded together with its neighbours; the
   cosine distance between consecutive windows is measured; a boundary is placed wherever
   that distance exceeds the 95th percentile of *this document's own* distances. Chunks are
   then split if over `SEMANTIC_MAX_CHUNK_WORDS` (at their weakest internal seam) and merged
   if under `SEMANTIC_MIN_CHUNK_WORDS` (into the more similar neighbour).
4. Anything still past the embedding model's token window is split rather than silently
   truncated.
5. Chunk text + vector + page range go into ChromaDB, in batches.

**At question time:**

1. With conversation history, the question is first rewritten into a standalone one — "what
   about the second one?" retrieves nothing useful as written.
2. Two searches run over the same corpus: **vector** (semantic, good at paraphrase) and
   **BM25** (lexical, good at exact terms like "A* search"). The ranked lists are merged with
   Reciprocal Rank Fusion.
3. A **cross-encoder re-ranks** the ~30 fused candidates by reading each (question, chunk)
   pair together; the best `top_k` survive.
4. A **relevance floor** applies: if nothing scores well enough, the app answers "not in
   these documents" *without calling the LLM at all*.
5. Each surviving chunk is returned with its **neighbouring chunks**. Semantic chunks do not
   overlap by design, so this is what stops the model reading one side of a boundary alone.
6. Those chunks go into the prompt as CONTEXT under a character budget, with a system prompt
   instructing the model to answer only from that context.
7. The LLM writes the answer — streamed token by token — with the documents and pages it drew
   from.

## Setup

Run everything from the **project root**.

```bash
python -m venv venv
venv\Scripts\activate           # Windows;  macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
copy .env.example .env          # Windows;  macOS/Linux: cp .env.example .env
```

Then open `.env` and set **one** LLM key:

- `GROQ_API_KEY` — free, no card: <https://console.groq.com/keys>
- or `LLM_PROVIDER=gemini` plus `GEMINI_API_KEY` — free, no card:
  <https://aistudio.google.com/apikey>

You also need **MongoDB running locally** for accounts and chat history (the default
`mongodb://localhost:27017`). Without it the app starts and serves pages, but nobody can
sign in.

The first ingestion downloads the embedding model (~130MB) and the first question downloads
the re-ranker (~80MB). Both are then cached.

## Adding documents

Put PDFs in `data/`, or upload them from the web UI (which writes them into
`data/users/<your-id>/`). Then:

```bash
python scripts/ingest.py           # ingest new / changed PDFs, prune deleted ones
python scripts/ingest.py --status  # show what's in the store
python scripts/ingest.py --force   # wipe and rebuild from scratch
```

Subfolders work — `data/textbooks/norvig.pdf` is ingested and identified by that relative
path.

Files are fingerprinted by content hash, so re-running is always safe: unchanged files are
skipped, and an edited PDF is re-ingested (its old chunks deleted first, not duplicated). The
same scan runs at startup, on `POST /ingest`, and behind the UI's "Refresh" button.

**Semantic chunking embeds every sentence**, which is roughly 15× the embedding calls of a
fixed-size packer. On a local model that is CPU time and no money, but a 900-page textbook is
genuinely several minutes. The progress bar in the UI is real.

## Running it

```bash
python scripts/run.py       # starts uvicorn and opens the browser once /health answers
```

or plainly:

```bash
uvicorn src.main:app --reload --port 8000
```

Then open **<http://localhost:8000>**. The web UI is served by the same FastAPI process from
`src/static/`, so there is no separate file to open and no CORS hop.
`http://localhost:8000/docs` gives the interactive API reference.

Run it with **one worker**. The BM25 cache, the answer cache, the rate limiter and the
ingestion job are all in-process, and Chroma's persistent client is single-process.

## API

| Method | Path                         | Purpose |
|--------|------------------------------|---------|
| GET    | `/`                          | the web UI |
| GET    | `/health`                    | liveness + whether the embedding model has loaded |
| GET    | `/ready`                     | readiness: 503 until the model is loaded and MongoDB answers |
| GET    | `/info`                      | which chunker, models and provider this instance is running |
| POST   | `/api/signup`                | create an account, returns a JWT (201) |
| POST   | `/api/login`                 | exchange credentials for a JWT |
| GET    | `/api/me`                    | the signed-in user |
| POST   | `/api/me/password`           | change password; invalidates every other session |
| POST   | `/api/me/signout-everywhere` | invalidate all tokens for this account |
| DELETE | `/api/me`                    | delete the account and everything it owns |
| POST   | `/chat`                      | `{"question": "...", "top_k": 4, "sources": [], "history": []}` → answer + sources |
| POST   | `/chat/stream`               | same body, streamed as SSE (`sources`, then `token`s, then `done`) |
| POST   | `/upload`                    | upload one or more PDFs (multipart `files`), then index them (202) |
| POST   | `/ingest`                    | start a background scan of `data/` (202) |
| GET    | `/ingest/status`             | progress of the current/last ingestion job |
| GET    | `/api/documents`             | the caller's documents |
| GET    | `/stats`                     | chunk count, per-document pages/chunks, quota, ingestion state |
| DELETE | `/documents/{name}`          | remove a document: its vectors, manifest entry, and the PDF |
| POST   | `/reset`                     | rebuild the caller's documents from their files on disk (202) |

## Testing

```bash
pip install -r requirements-dev.txt
python tests/test_chunking_offline.py
python tests/test_answer_length_offline.py
```

The chunking suite stubs the embedding model with a deterministic fake whose vectors encode a
known topic structure, so every "did it cut in the right place" assertion is checkable rather
than a vibe. No model download, no network, no API key.

## Measuring answer quality

The tests prove the plumbing works. The eval harness measures whether the answers are any
*good* — and, more usefully, whether a change made them better:

```bash
python eval/run_eval.py                 # hit-rate@k, MRR, refusal rate (no API key needed)
python eval/run_eval.py --judge         # + LLM-as-judge correctness/groundedness/relevance
python eval/run_eval.py --no-rerank     # A/B: what is the cross-encoder actually worth?
python eval/run_eval.py --no-hybrid     # A/B: what is BM25 worth?
```

Replace `eval/golden_questions.json` with 20–30 questions about *your* documents. That is
what turns tuning (`SEMANTIC_BREAKPOINT_PERCENTILE`, `TOP_K`, the relevance floor) from
guesswork into measurement.

## Tuning the chunker

`SEMANTIC_BREAKPOINT_PERCENTILE` is the main dial: lower means more, smaller chunks. The size
bounds are guard rails, not targets.

**Changing any chunking setting requires a full re-ingest** (`python scripts/ingest.py
--force`), ideally into a fresh `CHROMA_COLLECTION`. Different settings produce different
chunk text and therefore different vectors; mixing two generations in one store does not
error, it just quietly ruins every number you measure afterwards.

Be honest about what to expect. Published comparisons mostly find semantic chunking inside
the noise of a well-tuned fixed chunker on well-structured documents, and clearly ahead only
where formatting carries no signal — transcripts, chat logs, OCR with the paragraph breaks
gone. Textbooks with intact paragraphs are the hard case. Measure on your own corpus.

## Known limitations

- **OCR is off by default** — set `OCR_ENABLED=true` and install `pytesseract`, `pillow` and
  the Tesseract binary; otherwise scanned/image PDFs are reported as skipped.
- **Page ranges are chunk-level**, not sentence-level — good for "roughly where to look".
- **Single worker only.** See the note under "Running it".
- **No TLS.** The API is plain HTTP; put a reverse proxy in front before exposing it.
