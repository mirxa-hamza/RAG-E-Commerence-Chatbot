# STATUS — where this project is right now

> Superseded snapshot: this document below describes the old PDF cleanup. For the current
> LangChain shopping implementation, read README.md and PLAN.md's implementation checkpoint.
> As of 2026-09-07 the new backend, Next.js frontend and Docker configuration are written;
> tests pass, but current ingestion and full-corpus/container verification are still pending.

Written 2026-09-07, immediately after the local-only refactor. Read this first if you are
picking the project up cold (human or AI). `README.md` is how to run it, `CLAUDE.md` is the
rules for changing it, this file is **what state it is in today**.

---

## 0. TL;DR

A working, from-scratch RAG system over PDFs. Chunking, embedding, the vector DB and
re-ranking all run locally; only answer generation calls out.

**The code is complete and clean. It does not currently start.** One stale value in `.env`
(`LLM_PROVIDER=vertex`) is rejected by the new config. Fix that line and it runs.

**There are no documents and no index.** `data/` and `storage/` are both empty, so even
after it starts there is nothing to ask questions about until you add a PDF and ingest.

---

## 1. Blocking issue — fix before anything else

`src/core/config.py` validates `LLM_PROVIDER` at import and raises:

```
ValueError: LLM_PROVIDER must be 'groq' or 'gemini', got 'vertex'
```

`.env` still says `LLM_PROVIDER=vertex`, which was removed in the refactor. Edit `.env`:

```diff
- LLM_PROVIDER=vertex
+ LLM_PROVIDER=groq        # or gemini
```

Both keys are already present in `.env` (`GROQ_API_KEY` and `GEMINI_API_KEY` are set), so
either value works with no further changes.

### Also in `.env`, non-blocking but dead

37 settings remain that the new code never reads. They are inert — nothing errors, they are
simply ignored — but they are misleading to anyone reading the file, and a couple are
actively confusing:

| Setting | Why it is dead |
|---|---|
| `RAG_MODE=cloud`, `VECTOR_STORE`, `EMBEDDINGS_PROVIDER=gemini`, `RERANKER_PROVIDER` | the mode/provider switches were removed; everything is local now |
| `CHROMA_BACKEND=cloud`, `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` | Chroma Cloud removed; the store is always a local folder |
| `PINECONE_*` (8 keys) | Pinecone removed |
| `CLOUDINARY_*` (4 keys) | Cloudinary removed |
| `VERTEX_*` (3 keys), `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Vertex AI removed |
| `GEMINI_EMBED_MODEL`, `GEMINI_EMBED_DIM`, `GEMINI_EMBED_BATCH` | Gemini **embeddings** removed. `GEMINI_API_KEY` now means the **answering** model |
| `CHUNK_STRATEGY`, `CHUNK_SIZE_WORDS`, `CHUNK_OVERLAP_WORDS` | semantic is the only chunker; use `SEMANTIC_*` |
| `HIER_*` (3 keys), `PARENT_CONTEXT` | hierarchical chunking removed |
| `PROVIDER_TLS_MAX_VERSION` | belonged to the deleted HTTP provider layer |
| `GROQ_MAX_TOKENS`, `GROQ_TEMPERATURE` | **renamed** to `LLM_MAX_TOKENS` / `LLM_TEMPERATURE`. Your `GROQ_TEMPERATURE=0.4` is being ignored; the default 0.2 is in force |

Two settings worth a second look:

- **`CHROMA_COLLECTION=rag_gemini_768`** — the name describes 768-dimensional Gemini
  vectors. The local model (`BAAI/bge-small-en-v1.5`) produces **384** dimensions. The store
  is empty so nothing will break, but rename it to something honest before ingesting.
- **`MONGO_URI` points at MongoDB Atlas**, not localhost. Accounts and chat history are
  therefore still in the cloud. Not a bug, and nothing about your *documents* leaves the
  machine — but "fully local" is not true of the account database. Switch to
  `mongodb://localhost:27017` if you want that too.

---

## 2. What the system can do

Everything below is implemented and wired end to end.

### Documents
- Drop PDFs into `data/`, or upload them from the web UI (multipart `POST /upload`, several
  at once). Uploads are validated: filename scrubbed against traversal, PDF magic bytes
  checked, size cap enforced *while streaming*, per-account quota enforced.
- Content-hash fingerprinting (SHA-256). Re-running ingestion skips unchanged files,
  re-ingests edited ones (deleting the old chunks first, not duplicating), and prunes
  documents whose file is gone.
- Byte-identical duplicates are detected per owner and skipped.
- Text extraction preserves paragraph structure. OCR for scanned PDFs exists behind
  `OCR_ENABLED` (needs `pytesseract`, `pillow`, and the Tesseract binary); with it off,
  image-only PDFs are reported as `skipped`, not failed.
- Delete a document (`DELETE /documents/{name}`) and its vectors, manifest entry and the
  file itself all go.

### Chunking — semantic, and it is the only strategy
Every sentence is embedded together with its neighbours (`SEMANTIC_BUFFER_SIZE`), the cosine
distance between consecutive windows is measured, and a boundary is cut wherever that
distance exceeds the 95th percentile *of that document's own* distribution. Then size bounds
are enforced: over-long groups split at their weakest internal seam, under-sized ones merge
into whichever neighbour they are more similar to. Chunks do not overlap, by design.

Results are memoised per document so a re-run does not re-embed every sentence. If the
embedder ever returns the wrong number of vectors it falls back to a plain size-bounded pack
rather than misaligning every boundary.

### Retrieval
Dense vector search **+** BM25 keyword search → Reciprocal Rank Fusion → cross-encoder
re-rank → relevance floor → neighbour expansion. Each stage is individually switchable, which
is what makes the eval harness able to measure what each is worth.

The relevance floor matters: if nothing clears it, the app answers "not in these documents"
**without calling the LLM at all**.

### Answering
Grounded prompt with a hard word budget (enforced by instruction, never by the token
ceiling), per-sentence citations with document name and page range, follow-up rewriting
("what about the second one?" → a standalone query before retrieval), conversation history
with a character budget, and streaming over SSE. Two providers: Groq (default) and the
Gemini API.

### Accounts and multi-tenancy
Signup/login with JWTs, Argon2id password hashing, revocable tokens (`token_version` — a
password change or "sign out everywhere" invalidates existing tokens immediately), per-user
document ownership enforced at three isolation points plus a defence-in-depth check on the
way out, per-account storage quotas, sliding-window rate limits, an append-only audit log,
and account deletion that removes everything the account owns.

### Chat history
Conversations saved per user, listed in a sidebar, paginated, capped at
`MAX_SESSION_MESSAGES`.

### Web UI
Served at `/` by the same FastAPI process — no build step, no CORS hop. Upload with
per-file progress bars, a live indexing activity trail, document list with per-document
selection for scoped questions, streaming answers, storage quota display.

### Operations
- `scripts/run.py` — starts uvicorn and opens the browser once `/health` actually answers
- `scripts/ingest.py` — build the index without starting the API (`--status`, `--force`)
- `scripts/verify_index.py` — find orphan chunks, ownerless chunks, manifest entries with no
  chunks; `--fix` repairs them
- `scripts/backup.py` — archive `data/`, the index and the database
- `eval/run_eval.py` — hit-rate@k, MRR, refusal rate, optional LLM-as-judge, plus
  `--no-rerank` / `--no-hybrid` / `--no-expand` A/B switches
- `/health`, `/ready`, `/info` for liveness, readiness and "which models am I actually
  running"

---

## 3. Where the code lives

```
src/main.py                 app assembly, lifespan, warm-up, static mount
src/api/       auth chat deps documents sessions system      (thin handlers)
src/core/      config logging ratelimit
src/ml/        embeddings reranker llm gemini
src/services/  pdf chunking vectorstore bm25 retrieval ingestion
               manifest answer_cache uploads ownership database security sessions
src/static/    index.html script.js style.css
```

Data flow, ingest: `pdf.extract_pages` → `pdf.sentences_with_pages` →
`chunking.chunk_pages` → `embeddings.split_to_token_limit` → `vectorstore.add_chunks`.

Data flow, query: `llm.rewrite_question` → `vectorstore.query_chunks` + `bm25.search` →
`retrieval.fuse` → `reranker.rerank` → floor → `retrieval._expand_neighbors` →
`llm.build_context` → `llm.generate_answer` / `stream_answer`.

---

## 4. What was removed (do not go looking for it)

Deleted in this refactor. If a comment, doc or old commit mentions these, they are gone:

- **Cloud deployment**: `api/` (Vercel entrypoint), `vercel.json`, `requirements-cloud.txt`,
  `SWITCHING.md`, the whole `RAG_MODE` switch, the per-request resumable ingestion job, the
  Mongo-backed manifest / answer cache / rate limiter
- **Remote providers**: `src/ml/providers.py` (Pinecone, Cohere, Jina, Gemini embeddings —
  roughly 800 lines of HTTP retry, TLS and backoff machinery), `src/ml/vertex.py`,
  `src/services/vector_pinecone.py`, `src/services/cloudinary_store.py`,
  `src/services/cloud_documents.py`
- **Chunking**: the fixed-size packer (`pdf.chunk_document`, `pdf.units_with_pages`) and the
  hierarchical parent/child chunker, plus `PARENT_CONTEXT` and `retrieval._expand_to_parents`
- **Merged**: `src/services/vector_chroma.py` folded into `src/services/vectorstore.py`; the
  backend-dispatch indirection is gone
- **Scaffold**: `reference/` — a separate agent-style app nothing imported
- **Scripts**: `check_cloud.py`, `diagnose_cloudinary.py`, `check_embeddings.py`,
  `ab_chunking.py`, `draft_golden.py`, `check_golden.py`, `make_test_pdf.py`
- **Tests**: `test_gemini_embeddings_offline.py`, `test_vertex_llm_offline.py`,
  `test_parent_context_offline.py`
- **Docs**: `PLAN.md` (changelog for the removed hierarchical work)

`src/core/config.py` went from 629 to ~330 lines as a result.

---

## 5. What is verified, and what is not

**Verified in this session** (in a Linux container, with `pymupdf`/`chromadb`/`groq` stubbed
where they could not be installed):

- `tests/test_chunking_offline.py` — **50 checks pass**. Sentence splitting, percentile and
  cosine maths, boundary placement against a stub embedder with a known topic structure, size
  bounds, merge direction, page attribution, determinism, the vector-count fallback, and the
  memo cache.
- `tests/test_answer_length_offline.py` — **40 checks pass**. The word-budget contract,
  reminder placement, the token ceiling *not* being the length control, `LLM_PROVIDER`
  validation, and duplicate-key detection in `.env`.
- Every module imports cleanly; every `from src.core.config import X` resolves against what
  config actually defines; no dangling references to any removed symbol; `script.js` parses.

**NOT verified — nobody has run this end to end since the refactor:**

- The server has never been started (blocked on the `.env` fix above)
- No PDF has been ingested through the new pipeline
- No question has been asked, so neither the Groq nor the new Gemini path has made a live
  call. **`src/ml/gemini.py` is brand-new code that has never touched the network.**
- The local embedding model and cross-encoder have not been loaded in this configuration
- The MongoDB connection, signup/login and the web UI have not been exercised
- `scripts/verify_index.py` and `eval/run_eval.py` have not been run against a real index

Treat section 6 as the first real test of the refactor.

---

## 6. Next steps, in order

1. **Fix `.env`** — `LLM_PROVIDER=groq` (or `gemini`). Optionally rename
   `CHROMA_COLLECTION`, rename `GROQ_TEMPERATURE` → `LLM_TEMPERATURE` if you want 0.4 back,
   and delete the 37 dead keys.
2. `pip install -r requirements.txt` — the dependency set changed (Pinecone dropped).
3. **Put at least one PDF in `data/`.** It is currently empty; the two textbooks and all
   per-user uploads were removed during cleanup.
4. `python scripts/ingest.py --force` — first run downloads the embedding model (~130MB).
   Expect this to be slow: semantic chunking embeds *every sentence*, roughly 15× the
   embedding calls of a fixed packer.
5. `python scripts/run.py` — sign up, upload, ask a question. First question downloads the
   cross-encoder (~80MB).
6. Check `GET /info` reports what you expect: `chunking: semantic`,
   `embeddings_provider: local`, `vector_store: chroma`, and your chosen `llm_provider`.
7. Replace `eval/golden_questions.json` with 20–30 questions about your real documents, then
   `python eval/run_eval.py` to get a baseline before tuning anything.

---

## 7. Open questions worth deciding

- **Is semantic chunking actually earning its cost here?** Published comparisons mostly find
  it inside the noise of a well-tuned fixed chunker on well-structured documents, and clearly
  ahead only where formatting carries no signal (transcripts, chat logs, OCR without
  paragraph breaks). It now costs ~15× the embedding calls at ingest. The fixed packer was
  deleted at your request, so answering this means measuring semantic against itself at
  different `SEMANTIC_BREAKPOINT_PERCENTILE` values rather than against a baseline.
- **Accounts are on Atlas, documents are local.** Worth deciding whether that split is
  intended.
- **Single worker only.** The BM25 cache, answer cache, rate limiter and ingestion job are
  all in-process and Chroma's client is single-process. If this ever needs to scale past one
  process, those four are what has to move.
- **The project name.** The folder is "Ecommerence RAG Chatbot" but the app calls itself
  Marginalia and the corpus was AI textbooks. Nothing in the code is e-commerce specific.
