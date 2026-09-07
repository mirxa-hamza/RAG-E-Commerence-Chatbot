# PLAN — Agentic E-Commerce Chatbot with RAG Memory

Status: **planning document, nothing in this plan has been implemented yet.** This
supersedes the PDF-RAG cleanup described in `STATUS.md` — that cleanup is still real
work (the codebase really is local-only now, tests really do pass), but the product
this repo builds is changing entirely, from "chat with your PDF" to "shop the Amazon
Fashion catalog with an agent that remembers you."

Decisions below were confirmed with the project owner before writing this plan; where
a choice was mine to make (not asked), it's called out explicitly as an **assumption**
so it's easy to challenge later.

## 1. Confirmed decisions

| Question | Decision |
|---|---|
| How much of the dataset to ingest | Curated subset: **~5,000 products**, ranked by review count, plus every review belonging to those products |
| PDF upload/chat feature | **Fully retired** — no parallel PDF mode, no PDF API routes, no PDF UI |
| Tool-calling mechanism | **Native function-calling on both Groq and Gemini** (their own tool-call APIs, not a hand-rolled JSON parser) |
| Frontend | **Rebuilt fresh** — the existing 79KB `script.js` is PDF-shaped and is not adapted |
| MongoDB target | **Local MongoDB** (`mongodb://localhost:27017`) — the live Atlas URI in `.env` gets replaced, not reused |

Confirmed on disk before writing this plan (via the linked machine, `E:\Project\Ecommerence RAG Chatbot`):

- `data\meta_Amazon_Fashion.jsonl` — 1.42 GB (product metadata)
- `data\Amazon_Fashion.jsonl` — 1.05 GB (reviews)

Both are the **full** 2023 Amazon Fashion dump (millions of records) — the 5,000-product
subset in phase 2 is what makes local, CPU-only embedding practical (minutes, not hours).

## 2. What gets kept vs. retired

The PDF-RAG cleanup already did the hard part of stripping this project down to a
plain-Python, local-only pipeline (no LangChain, no cloud vector DB, lazy singleton
loading, single-worker constraint). Almost none of that engineering is wasted — it's
retargeted at a different kind of text.

**Kept and reused (mechanism-wise), retargeted (content-wise):**

- `src/ml/embeddings.py` — same local `BAAI/bge-small-en-v1.5` model. Now embeds review
  text instead of PDF chunks.
- `src/ml/reranker.py` — same local cross-encoder. Now re-ranks review chunks.
- `src/services/vectorstore.py` — same ChromaDB persistent-client wrapper. New
  collection name (`amazon_fashion_reviews_384`, replacing whatever `CHROMA_COLLECTION`
  is set to today — the STATUS.md-flagged `rag_gemini_768` name was already a dimension
  mismatch with the local 384-dim model, so this also fixes that).
- `src/services/bm25.py` — same lexical index, now built over review text.
- `src/services/chunking.py` — same semantic-breakpoint algorithm, but see §3.3: it needs
  a thin new entry point since reviews don't have PDF "pages."
- `src/services/retrieval.py` — the RRF fusion + re-rank + relevance-floor + neighbor
  expansion logic becomes the implementation of the `semantic_review_search` tool
  (§4.2), not a standalone pipeline called on every message.
- `src/ml/llm.py`, `src/ml/gemini.py` — same Groq/Gemini backends, extended with
  tool-calling (§4).
- `src/services/sessions.py`, `src/services/database.py`, `src/services/security.py`,
  `src/core/ratelimit.py` — auth, chat history, rate limiting all stay as-is. Sessions
  gain a `preferences` sub-document (§3).
- `src/core/config.py` — kept, extended with new e-commerce settings, and the dead
  PDF-only settings removed (§5).

**Retired outright (not adapted, deleted):**

- `src/services/pdf.py` — PDF extraction has no place in this product.
- `src/services/uploads.py`, `src/services/manifest.py`, `src/services/ownership.py` —
  these exist to make "a user's uploaded file" a safe, isolated concept. There is no
  per-user upload anymore; the catalog is shared, read-only, identical for every user.
- `src/api/documents.py` — the PDF upload/list/delete routes.
- `data/*.pdf` (if any remain), `eval/` (the golden-question harness is PDF-chunking
  specific — hit-rate@k against PDF passages doesn't transfer to product/review search)
- `report/` — this folder (`RAG-Approaches-Report.{md,pdf,docx}`) is leftover
  documentation about PDF chunking strategies that were already removed in the local-only
  cleanup. It was missed by `cleanup.ps1` (that script only covered code); this plan
  deletes it in phase 1 as part of the same "remove what the product no longer needs" pass.
- The three isolation-point rules in `CLAUDE.md` §1 (`query_chunks`/`get_neighbors_bulk`/
  `all_chunks` filtered by `user_id`) **no longer apply to the product catalog** — every
  user searches the same shared review index, so there is nothing to isolate there. What
  *does* still need per-user isolation is session/preference data, which was already
  scoped by session/user in `sessions.py` and needs no new mechanism.

**Assumption (not asked, flagging it):** the old `answer_cache.py` (repeat-question →
skip retrieval+LLM) is **disabled for the agentic router**, not adapted. A cache keyed
on the raw question text made sense for a linear "always the same pipeline" flow; it's
unsound once the answer depends on which tools the router chose to call *and* on the
caller's stored preferences (two users asking the identical question can legitimately
get different tool calls and different answers). Re-introducing caching later is
possible (e.g. keyed on `(question, resolved tool calls, preference hash)`) but it's out
of scope here — flag if you disagree and want it kept.

## 3. Phase 1 — Environment cleanup

1. **Fix the blocking bug**: `.env` currently has `LLM_PROVIDER=vertex`, which
   `config.py` rejects at import (`ValueError`, confirmed by reproducing it in STATUS.md).
   Set it to `groq` or `gemini`.
2. **Point Mongo at localhost**: replace the Atlas `MONGO_URI` with
   `mongodb://localhost:27017`, `MONGO_DB=ecommerce_agent` (or similar). This plan
   assumes MongoDB Community Server is installed and running locally — if it isn't yet,
   that's a one-time prerequisite before phase 2 can write anything.
3. Rename `CHROMA_COLLECTION` to `amazon_fashion_reviews_384` and delete the existing
   `storage/chroma_db` directory (old vectors are PDF chunks at the wrong dimension
   anyway — nothing there is reusable).
4. Delete `report/`, `eval/`, `src/services/pdf.py`, `src/services/uploads.py`,
   `src/services/manifest.py`, `src/services/ownership.py`, `src/api/documents.py`,
   and whatever's left of `data/*.pdf`.
5. Remove now-dead config keys from `config.py`/`.env.example`: everything under the old
   "STORAGE" section that was PDF/upload-specific (`MAX_UPLOAD_MB`, `MAX_USER_STORAGE_MB`,
   `OCR_ENABLED`/`OCR_DPI`/`OCR_LANG`, `DATA_DIR`'s PDF-only meaning).
6. Add new config keys (all in `config.py`, `.env.example` documented, nothing hardcoded):
   - `AMAZON_META_PATH` (default `data/meta_Amazon_Fashion.jsonl`)
   - `AMAZON_REVIEWS_PATH` (default `data/Amazon_Fashion.jsonl`)
   - `PRODUCT_SUBSET_SIZE` (default `5000`)
   - `MONGO_PRODUCTS_COLLECTION` (default `products`)
   - `CHROMA_COLLECTION` reused, default becomes `amazon_fashion_reviews_384`
   - `AGENT_MAX_TOOL_ROUNDTRIPS` (default `3`) — caps how many tool-call ↔ LLM cycles one
     turn can take, so a confused model can't loop forever
7. Dependencies: no new heavy packages needed. `pymongo`/`motor` are already in
   `requirements.txt`. The JSONL files are one JSON object per line, so streaming them
   is plain `json.loads` per line — no `ijson`/`jsonlines` dependency required. (If
   parse speed on the 1–1.4 GB files turns out to matter, `orjson` is a drop-in faster
   `json.loads` replacement — noted as an optional follow-up, not a day-1 dependency.)

## 4. Phase 2 — Dual-store ingestion pipeline

New module: `src/services/ecommerce_ingest.py` (replaces `pdf.py`'s role;
`ingestion.py`'s background-job/threading scaffolding is reused, its PDF-specific body
is not). New CLI entry point: `scripts/ingest_ecommerce.py` (the existing
`scripts/ingest.py` is PDF-specific and is retired alongside `pdf.py`).

**Three streaming passes over the raw files, never loading either 1GB+ file fully into
memory:**

1. **Pass 1 — pick the subset.** Stream `Amazon_Fashion.jsonl` line by line, count
   reviews per `parent_asin` in a plain dict (≈2M short string keys — comfortably fits
   in RAM, this is the only full pass over the reviews file that doesn't also do
   embedding work). Take the top `PRODUCT_SUBSET_SIZE` (5,000) `parent_asin`s by review
   count. Review count is a reasonable proxy for "real, well-established product" and
   incidentally guarantees every selected product actually has review text to search
   over — a product with metadata but zero reviews would make `semantic_review_search`
   useless for it.
2. **Pass 2 — load and preprocess product metadata.** Stream `meta_Amazon_Fashion.jsonl`,
   keep only records whose `parent_asin` is in the selected set, **clean each record**
   (rules below), then upsert into MongoDB's `products` collection keyed by
   `parent_asin` (`_id`). This collection is what `catalog_search` (§4.1) queries with
   exact filters — it needs to be structured and typed, not embedded, so cleaning
   happens now, not at query time.
3. **Pass 3 — load, preprocess, and embed reviews.** Stream `Amazon_Fashion.jsonl`
   again, keep only reviews whose `parent_asin` is in the selected set, **clean and
   filter each review** (rules below), chunk the survivors with the semantic chunker
   (§4.3), embed each chunk locally, and upsert into the `amazon_fashion_reviews_384`
   Chroma collection with metadata `{parent_asin, review_id, rating, title, timestamp}`.
   This is the collection `semantic_review_search` (§4.2) queries.

### 3.1 Preprocessing rules

Raw Amazon JSONL is not display-ready or embed-ready as-is. This is its own explicit
step inside passes 2 and 3, not an afterthought — bad input here quietly produces
broken product cards or noise-polluted search results three phases later, which is much
harder to debug than catching it at ingestion.

**Product metadata (pass 2), applied before the Mongo upsert:**

- Drop any record missing `title` or `parent_asin` outright — nothing downstream can
  display or key a product without them.
- `price`: the raw field is inconsistently a real number, a numeric string, `"None"`,
  or absent. Parse to `float` where possible (strip `$`/commas, regex out the numeric
  part); store `null` when it can't be parsed. `catalog_search`'s price-range filter
  treats `null` as "excluded from price filtering," never as `0`.
- `images`: the raw field is a list of dicts with several resolutions
  (`thumb`/`large`/`hi_res`). Flatten this to one canonical `image_url` per product
  (prefer `large`, fall back to `hi_res`, then `thumb`, then `null`) so the frontend
  card code never has to know Amazon's nested schema.
- `brand`: the dataset uses `brand` and `store` inconsistently for the same concept —
  fall back `brand → store → null`, never an empty string (empty strings are a
  false-positive match for "no filter selected" in `catalog_search`).
- `categories`: flatten and de-duplicate the list, drop empty/whitespace entries.
- `average_rating`/`rating_number`: coerce to numeric types, default `0`/`null` if
  missing, so `catalog_search`'s "min rating" filter can compare them directly.

**Reviews (pass 3), applied before chunking/embedding — and note the review-count
tally in pass 1 happens *before* this filtering, since a rating-only review is still
real evidence a product is popular even though it won't itself be searchable:**

- Skip reviews with empty/whitespace-only `text` — a 5-star rating with no comment has
  nothing for `semantic_review_search` to retrieve.
- Skip reviews under a minimum word count (default 3 words) — "Great!" or "Love it!"
  embeds to noise, not a retrievable fact about the product.
- HTML-unescape review text (`&#34;`-style entities show up in the raw dataset).
- Drop exact-duplicate review text within the same product (scraped datasets carry
  repeats).
- Normalize `timestamp` (Unix milliseconds) and `rating` (numeric) types.

**Assumption (not asked, flagging it):** no language filtering is applied. The UI and
the local embedding model (`bge-small-en-v1.5`) are both English-tuned, so a non-English
review will embed to a low-quality, roughly-random vector rather than something
genuinely retrievable — it won't crash anything, but it's dead weight in the index and
could very occasionally surface as an irrelevant-looking hit. Adding a real fix (a
language-detection pass, e.g. `langdetect`, dropping or segregating non-English reviews)
means one new lightweight dependency; left out of day-1 scope, but worth revisiting if
search quality looks off on specific products after the full ingest.

**Idempotency:** re-running `scripts/ingest_ecommerce.py` without `--force` should be a
no-op if the subset is unchanged (reuse the existing content-hash-fingerprint pattern
from `manifest.py`'s spirit, applied per-`parent_asin` instead of per-file — skip a
product already present in both stores with a matching hash of its source line).
`--force` wipes both stores and rebuilds from scratch — this is the primary mode you'll
actually use while the pipeline is being built and tuned.

**A thin new chunker entry point (§4.3 detail):** `chunking.chunk_pages()` is written
around PDF pages (a chunk remembers which page it came from). Reviews have no pages.
Rather than bend a review into a fake one-page document, add
`chunking.chunk_review(text: str) -> list[str]` that reuses the same sentence-splitting
and semantic-breakpoint machinery but drops the page bookkeeping — and, since most
Amazon reviews are a few sentences long, short-circuits straight to "one review = one
chunk" below `SEMANTIC_MIN_CHUNK_WORDS`, only running the full breakpoint-detection
logic on genuinely long reviews. This keeps the "cut where meaning changes" idea intact
without wasting embedding calls semantically-chunking two-sentence reviews.

## 5. Phase 3 — Session memory / preferences

Extend the session document in `sessions.py` (MongoDB, already per-user/per-session)
with a `preferences` sub-object:

```
{
  "clothing_size": "M" | null,
  "budget": {"min": 20, "max": 60} | null,
  "color_preference": ["black", "navy"] | [],
  "favorite_brands": ["Nike", ...] | []
}
```

**Assumption (not asked, flagging it):** the spec names a `memory_lookup` tool (read
path) but doesn't specify how preferences get *written*. Two options:

- (a) A lightweight extraction step runs after every user message — a small, cheap
  regex/keyword pass first (sizes, "$", color words, brand names that match known
  brands already in the `products` collection), falling back to nothing fancy if it
  doesn't match. This plan uses **(a)** because it costs no extra LLM call and is easy
  to unit-test deterministically (in the spirit of `tests/test_chunking_offline.py` —
  offline, no network, checkable).
- (b) A 4th tool (`remember_preference`) the LLM calls explicitly when it notices a
  stated preference — more accurate (an LLM understands "nothing too flashy" implies a
  color preference; regex doesn't), but costs router complexity and an extra
  reason for the LLM to go off-script. Noted as a **phase-2 follow-up**, not day-1.

Preferences are injected into the system prompt on every turn (so the model's baseline
tone/suggestions reflect them even when it doesn't explicitly call `memory_lookup`), and
`memory_lookup` lets the agent explicitly re-confirm or quote them back ("you mentioned
you prefer size M — here's...").

## 6. Phase 4 — Agentic tool-calling router

Replaces the linear `retrieve() → build_context() → generate_answer()` pipeline
described in `CLAUDE.md`'s architecture diagram with a tool-calling loop, implemented in
a new `src/services/agent_router.py` (kept separate from `retrieval.py`, which becomes
the internal implementation the `semantic_review_search` tool calls).

**Three tools, one JSON-Schema-shaped definition each, shared source of truth:**

1. **`catalog_search`** — exact/structured filtering against MongoDB `products`
   (category, brand, min/max price, color if present in metadata, min rating). For
   queries like "show me black running shoes under $50."
2. **`semantic_review_search`** — the existing hybrid pipeline (dense + BM25 → RRF →
   cross-encoder rerank → relevance floor → neighbor expansion), now querying the
   `amazon_fashion_reviews_384` collection with no `user_id` filter (shared catalog —
   see §2). For subjective queries like "which of these run small?" or "any complaints
   about durability?"
3. **`memory_lookup`** — reads the calling session's `preferences` sub-document from
   MongoDB. For "recommend something in my size."

**Router loop:**

1. Send the user message + trimmed history + tool schemas + injected preferences to the
   LLM with tool-calling enabled.
2. If the LLM returns tool call(s), execute them (multiple tool calls in one round can
   run concurrently — they're independent reads), append results as tool-role messages,
   and call the LLM again.
3. Repeat up to `AGENT_MAX_TOOL_ROUNDTRIPS` (default 3) — if the model still wants
   another tool call after that, force a final answer from what's been gathered so far
   rather than looping indefinitely.
4. Stream the final answer via SSE as today. When a tool returned specific products,
   attach them as a separate structured SSE event (`event: products`) alongside the
   prose token stream (`event: token`), so the frontend can render cards without
   parsing them out of markdown.

**Provider translation:** Groq's tool-calling API is OpenAI-shaped (`tools` param,
`tool_calls` in the response) — `llm.py` mostly passes the shared schema straight
through. Gemini's is shaped differently (`functionDeclarations`, `functionCall` /
`functionResponse` parts) — `gemini.py` already has a message-translation layer
(`_payload()`) for the plain-chat case; this plan extends that same layer to translate
tool schemas and tool-call/tool-response messages, so `agent_router.py` itself never
branches on provider.

## 7. Phase 5 — API and frontend

**Backend (`src/api/chat.py`):**

- Chat endpoint calls `agent_router` instead of `retrieval.retrieve()` directly.
- SSE event types: `token` (prose), `products` (structured card data), `done`, `error` —
  additive to whatever the endpoint already emits for auth/session-id.
- `src/api/documents.py` removed entirely (§2).
- `src/api/system.py`'s `/info` gains `mongo_products_count`,
  `chroma_reviews_count`, `subset_size`, alongside the existing
  `embeddings_provider`/`llm_provider`/etc. fields.

**Frontend (rebuilt fresh, per the confirmed decision):**

- New minimal `index.html`/`script.js`/`style.css` (or a single Tailwind-CDN-based
  page, consistent with this project's existing "no framework, plain code you can read
  top to bottom" philosophy from `CLAUDE.md`).
- Auth screens (sign in / sign up) rebuilt minimal — same JWT endpoints, new UI.
- Chat view: message stream (SSE `token` events) + a product-card grid rendered from
  `products` events — image, title, price, star rating, and a 1–2 line
  review-summary excerpt (from whichever review chunk the router actually retrieved,
  not a separately-generated summary — keeps the card honest about what was searched).
- No PDF upload UI at all.

## 8. Verification steps

Mirrors the spec's own verification plan, made concrete against this repo's tooling:

1. **Dry run**: `scripts/ingest_ecommerce.py --limit 1000 --force` — confirms the
   3-pass streaming logic, Mongo upserts, and Chroma embedding work end-to-end on a
   fast, disposable slice before committing to the full 5,000-product run.
2. **`/health` and `/info`** after the full ingest — confirms Mongo connectivity,
   Chroma collection populated at the expected count, and the reported product/review
   counts match what was ingested.
3. **Test query 1 (exact catalog filter)**: "show me black dresses under $40" → expects
   a `catalog_search` tool call, product cards, no need for review search.
4. **Test query 2 (subjective review semantics)**: "do these shoes run small?" (on a
   specific product from the results above) → expects `semantic_review_search`, an
   answer grounded in actual review text, not invented.
5. **Test query 3 (memory personalization)**: state a size/brand preference, ask an
   unrelated follow-up ("what do you have for me today?") → expects `memory_lookup`
   firing without being asked, and a response reflecting the stated preference.
6. Offline unit tests for anything deterministic and worth pinning the same way
   `tests/test_chunking_offline.py` does today: the preprocessing rules (§3.1 —
   price parsing, image-URL fallback selection, review min-length/empty/duplicate
   filtering), the preference-extraction heuristics (§5), and the tool-schema
   translation between Groq/Gemini shapes (§6) are all good candidates — no network or
   model download needed to check them.

## 9. Explicit non-goals of this plan

- Not touched: renaming the project/repo folder itself (still `Ecommerence RAG
  Chatbot` on disk) — cosmetic, doesn't block anything, can happen anytime.
- Not touched: rebuilding an e-commerce-flavored eval harness (hit-rate@k against
  product queries) to replace the retired PDF `eval/` — worth doing once the router is
  stable, not a blocker to getting it working first.
- Not in scope: wishlists/favorites, checkout, or any write-path against the product
  catalog — this is a shopping *assistant* (search + recommend + remember), not a
  storefront.

## 10. Implementation order

1. Phase 1 (env cleanup, deletions, new config keys) — everything else depends on the
   app importing cleanly and pointing at local Mongo.
2. Phase 2 (ingestion pipeline) — run the 1,000-record dry run, then the full
   5,000-product ingest. Nothing in phases 3–5 is testable without data in both stores.
3. Phase 3 (session preferences schema + extraction) — small, self-contained, unblocks
   the `memory_lookup` tool.
4. Phase 4 (agentic router + 3 tools) — the core of the product.
5. Phase 5 (API + frontend) — wire the router up to something clickable.
6. Verification (§8) against the real ingested subset.
