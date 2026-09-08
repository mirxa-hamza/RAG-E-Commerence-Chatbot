# CLAUDE.md — working notes for this codebase

## Current architecture takes precedence (2026-09-07)

The user explicitly switched this project to LangChain. README.md and PLAN.md describe
the current shopping assistant. The PDF-specific notes below are historical; their
no-framework rule and document-owner catalog filters no longer govern the shopping app.
Use `src/agent/` for provider models/prompts/tools, `src/api/shopping.py` for chat,
`src/services/review_store.py` for shared LangChain Chroma reads, and `frontend/` for Next.js.
Accounts and preferences remain private; never let the model choose a user/session ID.
The running Phase 2 ingestion must finish before installing into its `venv`, changing its
embedding/chunking code, or opening the same embedded index in another process.
Application testing uses `.venv-app`. Tests use isolated data and fake models by default.

## Historical PDF-RAG guidance

Guidance for anyone (human or agent) changing this project. `README.md` explains what it
does and how to run it; this file explains the rules that keep it correct.

## What this is

A from-scratch RAG system over PDFs. **Everything except answer generation runs on this
machine**: extraction, semantic chunking, embedding, the vector store, keyword search and
re-ranking. Only the retrieved passages are sent out, to Groq or the Gemini API, to be
written into prose.

The former PDF implementation used plain Python. The current shopping agent uses LangChain
by explicit user decision; preserve visible, testable preprocessing and ranking services.

## Architecture in one pass

```
data/*.pdf
  → pdf.extract_pages()        PyMuPDF, paragraph breaks preserved
  → pdf.sentences_with_pages() sentences, each remembering its page
  → chunking.chunk_pages()     SEMANTIC: embed every sentence + neighbours, cut at the
                               95th-percentile distance spike, then enforce size bounds
  → embeddings.split_to_token_limit()   split anything past the model's token window
  → vectorstore.add_chunks()   embed + store in ChromaDB, batched

question
  → llm.rewrite_question()     only when there is history
  → vectorstore.query_chunks() dense                 ┐
  → bm25.search()              lexical               ┴→ retrieval.fuse()  (RRF)
  → reranker.rerank()          cross-encoder
  → relevance floor            nothing above it = "not in these documents", NO LLM CALL
  → retrieval._expand_neighbors()  chunk_index ± NEIGHBOR_EXPANSION
  → llm.build_context()        under MAX_CONTEXT_CHARS
  → llm.generate_answer() / stream_answer()
```

`src/api/` handlers stay thin: validate, call a service, shape the response. The pipeline
lives in `src/services/` and `src/ml/`.

## The rules that matter

### 1. Isolation — three places, all of them

Documents belong to the account that uploaded them. Three functions reach stored text and
**must** filter by owner:

- `vectorstore.query_chunks(..., user_id=)`
- `vectorstore.get_neighbors_bulk(..., user_id=)` — fetched by index, so it bypasses every
  ranking filter; without the owner clause a hit on your own document pulls in the adjacent
  chunk of someone else's file with the same name
- `vectorstore.all_chunks(user_id=)` — feeds BM25, which ranks in memory and cannot use
  Chroma's where-clause

`user_id=None` means "no filter" and is only ever correct for offline callers (the CLI, the
eval harness). `retrieval.retrieve()` re-asserts ownership on the way out and logs an error
if anything slipped through — that is a net, not the fix.

**Adding a fourth way to reach stored text is the thing to avoid.** Three is already the
number to remember. If you need one, thread `user_id` through it and add it to this list.

### 2. Ingestion is the only entrance

Nothing is indexed from a request body. `POST /upload` writes a validated file into
`data/users/<user_id>/` and the ordinary ingestion job picks it up, so uploaded and
hand-copied PDFs travel identical code paths.

`services/uploads.py` is the boundary between "bytes someone sent over HTTP" and "a file the
pipeline will read". It distrusts the filename (path traversal), the extension and content
type (checks the PDF magic bytes), and the length (enforced while streaming, not after).
Keep it that way.

### 3. Ownership comes from the path, not the manifest

`users/<id>/book.pdf` → `<id>`. Derived, never trusted from stored metadata, so a
hand-edited manifest cannot hand one user another user's document. Files copied into `data/`
by hand belong to the **owner of record** — the first account created — because an ownerless
document is invisible to every filter and would occupy the store forever.

### 4. Single worker, on purpose

The BM25 cache, the answer cache, the rate limiter and the ingestion job are all in-process,
and Chroma's persistent client is single-process. Run `--workers 1`. Running more does not
error; it silently multiplies every rate limit by the worker count and gives each worker its
own stale caches.

### 5. Lazy loading is not optional

uvicorn imports the app **before** it binds the socket. Anything done at import time happens
while the port is closed, and the browser shows ERR_CONNECTION_REFUSED rather than a loading
screen. Importing torch and loading the embedding model at import cost ~18s of that.

So: the embedding model, the cross-encoder, the Chroma client and the Groq client are all
lazy singletons, warmed on a background thread from `main.py`'s lifespan. Do not move any of
them to module scope.

### 6. Stages fail OPEN, and say so

Re-ranking, neighbour expansion and the keyword index all degrade rather than fail the
question: worse ranking beats no answer. But a silent degradation is a bug report six weeks
later, so `/info` reports `reranker_available` and each fallback logs a warning.

### 7. Changing chunking means re-ingesting

Different chunking settings produce different chunk text and therefore different vectors.
Mixing two generations in one collection does not error — it quietly ruins every number
measured afterwards. Change a setting, then:

```bash
python scripts/ingest.py --force
```

and point `CHROMA_COLLECTION` somewhere new if you want to keep the old index around to
compare against.

### 8. Length is enforced by the prompt, never by the token ceiling

`LLM_MAX_TOKENS` does not shorten an answer, it **cuts** it — the model writes the same page
and the transport stops mid-word. A truncated answer is strictly worse than a long one,
because the reader cannot tell which facts were dropped. The word budget is stated in
`SYSTEM_PROMPT` and again in `ANSWER_REMINDER`, which the model reads last; the ceiling stays
generous. `tests/test_answer_length_offline.py` pins this down.

### 9. Retrieved passages are data, not instructions

Anyone who can upload a PDF can write "ignore previous instructions" into it. Chunks are
fenced in `<document>` blocks and the system prompt says to treat their contents as quoted
material. That is a mitigation, not a guarantee — which is why the answer is still built
only from retrieved chunks.

## Configuration

Every setting lives in `src/core/config.py` and nothing else calls `os.getenv()`. Values are
read at **import** time, so a `.env` change needs a restart. `config.py` cannot log
(`core.logging` imports from it), so it collects `CONFIG_WARNINGS` and `main.py` emits them
at startup — including the "this key is set twice in .env" check, which exists because the
symptom is always "I changed the setting and nothing happened".

## Testing and measurement

Two different questions, kept apart on purpose:

- `tests/*.py` — **correctness**. Offline, no model download, no network, no key. The
  embedding model is stubbed with a deterministic fake whose vectors encode a known topic
  structure, so "did it cut in the right place" is checkable rather than a vibe. Every check
  was confirmed to fail when the behaviour it guards was deliberately broken.
- `eval/run_eval.py` — **quality**. hit-rate@k, MRR, refusal rate, optional LLM-as-judge, and
  `--no-rerank` / `--no-hybrid` / `--no-expand` to measure what each stage is actually worth.

A test that cannot fail is decoration. A measurement run against the fixture PDF describes a
fictional document — replace `eval/golden_questions.json` with questions about your real
corpus before believing any of its numbers.

## Style

- Comments explain **why**, not what. A comment restating the line below it is noise; a
  comment recording the bug that motivated the line is the reason the file is readable.
- Keep the honest caveats. Where a docstring says a technique may not beat the simpler
  alternative on this corpus, that is information, not a to-do.
- Prefer one obvious code path over a configurable one. Every branch is a state someone has
  to reason about, and the ones removed from this project were removed because nobody could
  hold all of them in their head at once.
