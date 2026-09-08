"""
Central place for all configuration. Every other module imports from here instead of
calling os.getenv() directly, so there's exactly one source of truth.

Paths are resolved against the project root (not the current working directory), so the
server and the CLI scripts behave identically no matter where you launch them from.

This build is LOCAL-ONLY by design (see PLAN.md for the full architecture):

  * chunking   - semantic (embedding-breakpoint) chunking, on this machine
  * embeddings - sentence-transformers, on this machine
  * re-ranking - a cross-encoder, on this machine
  * vectors    - ChromaDB, embedded on this machine by default (CHROMA_MODE=embedded);
                 a server-mode client is available for the containerized deployment
  * catalog    - a curated subset of the Amazon Fashion 2023 dataset, ingested from the
                 JSONL dumps in data/ into MongoDB (products) and ChromaDB (reviews)
  * accounts   - MongoDB on this machine

The only thing that leaves the machine is the answer-generation call (Groq or the Gemini
API), which receives the retrieved passages and nothing else.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# This project uses sentence-transformers through PyTorch. If TensorFlow/Keras is also
# installed in the environment, Transformers may try to import it while loading embedding
# helpers and fail on Keras 3 unless `tf-keras` is installed. Disable the TensorFlow path
# early, before any module imports sentence-transformers/transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# src/core/config.py -> src/ -> project root
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
STATIC_DIR = SRC_DIR / "static"

load_dotenv(PROJECT_ROOT / ".env")

# Warnings raised while reading configuration. config.py cannot log - src.core.logging
# imports FROM here, so importing it back would be circular - so they are collected and
# emitted by main.py at startup and by the CLI scripts.
CONFIG_WARNINGS: list = []

_SECRETISH = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASS", "URI", "URL", "DSN", "CREDENTIAL")


def _find_duplicate_env_keys(env_path: Path) -> list:
    """
    Report keys assigned more than once in .env.

    python-dotenv keeps the LAST assignment of a repeated key and says nothing about it, so
    a `.env` that sets a value near the top and a different one two hundred lines down is
    running the second while its author reads the first. Nothing errors and no log line
    mentions it, so it is worth a line at startup.
    """
    try:
        raw = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    seen = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].removeprefix("export ").strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = stripped.split("=", 1)[1].split(" #", 1)[0].strip().strip("\"'")
        seen.setdefault(key, []).append(value)

    warnings = []
    for key, values in seen.items():
        if len(values) < 2:
            continue
        # Never echo a credential back into a log, even one the user set themselves.
        if any(word in key.upper() for word in _SECRETISH):
            warnings.append(
                f".env sets {key} {len(values)} times; the last one wins and the earlier "
                f"{len(values) - 1} are dead. Delete the ones you do not want."
            )
        elif len(set(values)) == 1:
            warnings.append(f".env sets {key} {len(values)} times with the same value.")
        else:
            warnings.append(
                f".env sets {key} {len(values)} times with DIFFERENT values "
                f"({', '.join(repr(v) for v in values)}); python-dotenv keeps the LAST, so "
                f"{key}={values[-1]!r} is what is running. Delete the others."
            )
    return warnings


CONFIG_WARNINGS.extend(_find_duplicate_env_keys(PROJECT_ROOT / ".env"))


def _path_setting(env_var: str, default: Path) -> Path:
    """Reads a path from the environment. Relative values resolve against the project root."""
    raw = os.getenv(env_var)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _flag(env_var: str, default: str) -> bool:
    return os.getenv(env_var, default).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- LLM provider
# Who WRITES the answer: "groq" (the default) or "gemini" (Google's Gemini API with an
# API key from aistudio.google.com). This is the ONE piece of the pipeline that is not
# local; everything else - chunking, embedding, the vector store, re-ranking - runs here.
#
# Both providers are held to the same contract: they get SYSTEM_PROMPT and the CONTEXT
# block and must answer only from the retrieved passages. The facts stay in the documents,
# which is what keeps the citations in the UI meaningful.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
if LLM_PROVIDER not in ("groq", "gemini"):
    raise ValueError(f"LLM_PROVIDER must be 'groq' or 'gemini', got {LLM_PROVIDER!r}")

# ---- Groq (LLM_PROVIDER=groq) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# ---- Gemini (LLM_PROVIDER=gemini) ----
# A free-tier key from https://aistudio.google.com/apikey needs no card.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# Pinned rather than "whatever is current": the URL carries the version, and an app should
# not silently follow an API's moving target.
GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta")
if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    CONFIG_WARNINGS.append(
        "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set - answering will fail. "
        "Get a free key at https://aistudio.google.com/apikey."
    )

# Shared generation settings. Both providers read these.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# The completion budget. On a REASONING model (gpt-oss) this is shared: the model's
# thinking is drawn from the same allowance before it writes a word of the answer, so 800
# was enough for the thinking and nothing else - the request returned HTTP 200 with
# content=None, which reached the user as an empty answer bubble under a populated sources
# list. 2000 leaves room for both. A non-reasoning model never uses the headroom.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))
# How hard a reasoning model thinks: "low" | "medium" | "high", or "" to send nothing.
# Only gpt-oss models accept the field, so ml/llm.py sends it only to those - a llama model
# rejects it outright. "low" is the default because this is document question-answering:
# the answer is meant to come from the retrieved passages, not from extended deliberation,
# and every reasoning token is one the answer does not get.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low").strip().lower()
if GROQ_REASONING_EFFORT not in ("", "low", "medium", "high"):
    raise ValueError("GROQ_REASONING_EFFORT must be '', 'low', 'medium' or 'high', got "
                     f"{GROQ_REASONING_EFFORT!r}")

# Seconds to wait on the Gemini HTTP call before giving up, and how many times a 429/5xx
# is retried. A rate limit on a free tier is a schedule, not a failure.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ---------------------------------------------------------------- answer length
# How long an answer should be. This is enforced by INSTRUCTION, never by the token
# ceiling: LLM_MAX_TOKENS cuts the model off mid-word, which is how a too-long answer
# becomes a truncated one - strictly worse than long. So the budget below is stated to the
# model in the system prompt and again in the reminder it reads last, while the token
# ceiling stays generous enough that a compliant answer is never clipped.
#
# The problem this exists for: retrieval hands the model up to MAX_CONTEXT_CHARS of
# passages, and an open question ("tell me more about X") invites it to summarise every one
# of them - a page of bullets with a citation on each, when three sentences answered the
# question.
ANSWER_STYLES = {
    "brief": 70,       # a couple of sentences; a chat-widget answer
    "concise": 140,    # the default: complete, but no enumeration of the whole context
    "standard": 300,   # room for a genuine multi-part answer
    "detailed": 700,   # ask for it explicitly
}
ANSWER_STYLE = os.getenv("ANSWER_STYLE", "brief").strip().lower()
if ANSWER_STYLE not in ANSWER_STYLES:
    raise ValueError(f"ANSWER_STYLE must be one of {sorted(ANSWER_STYLES)}, "
                     f"got {ANSWER_STYLE!r}")
# The word budget the prompt states. Set it directly to override the style's default.
ANSWER_MAX_WORDS = int(os.getenv("ANSWER_MAX_WORDS", str(ANSWER_STYLES[ANSWER_STYLE])))
if ANSWER_MAX_WORDS < 20:
    raise ValueError(f"ANSWER_MAX_WORDS must be at least 20, got {ANSWER_MAX_WORDS}")
# Most bullets an answer may use. Unbounded bullets are how "summarise the context" gets
# past a word budget: twenty three-word lines read as a dump and technically comply.
ANSWER_MAX_BULLETS = int(os.getenv("ANSWER_MAX_BULLETS", "6"))

# ---------------------------------------------------------------- Embeddings (local)
# sentence-transformers, in this process. bge-small-en-v1.5 has a 512-token window (vs
# all-MiniLM-L6-v2's 256), so a ~300-word chunk fits without being silently truncated at
# embedding time. First run downloads it (~130MB) and then it is cached.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models are trained with an instruction prefix on the QUERY side only; passages are
# embedded bare. Set to "" if you switch to a model that doesn't want one (e.g. MiniLM).
#
# The trailing space is re-added here on purpose: python-dotenv strips trailing whitespace
# from unquoted .env values, which silently glued the prefix to the question
# ("...passages:What is A* search?") and degraded every single query embedding.
_raw_prefix = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages:",
).strip()
EMBEDDING_QUERY_PREFIX = f"{_raw_prefix} " if _raw_prefix else ""
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# ---------------------------------------------------------------- Storage
DATA_DIR = _path_setting("DATA_DIR", PROJECT_ROOT / "data")
# Generated index state lives under storage/, kept out of the source tree and gitignored.
CHROMA_DIR = _path_setting("CHROMA_DIR", PROJECT_ROOT / "storage" / "chroma_db")
# Renamed from the PDF-RAG default (rag_documents) - this collection holds review chunks
# embedded with the local 384-dim bge-small model, not PDF passages. Point it somewhere
# new if you ever want to keep an old generation of vectors around to compare against.
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "amazon_fashion_reviews_384")
# Chroma enforces a max batch size per add() call; stay well under it.
CHROMA_ADD_BATCH = int(os.getenv("CHROMA_ADD_BATCH", "1000"))

# Embedded (PersistentClient, this process, default) vs server (HttpClient, talking to a
# chromadb container) - see PLAN.md Phase 6. HOST/PORT are only read in server mode.
CHROMA_MODE = os.getenv("CHROMA_MODE", "embedded").strip().lower()
if CHROMA_MODE not in ("embedded", "server"):
    raise ValueError(f"CHROMA_MODE must be 'embedded' or 'server', got {CHROMA_MODE!r}")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# ---------------------------------------------------------------- Auth (MongoDB + JWT)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "ecommerce_agent")
USERS_COLLECTION = os.getenv("USERS_COLLECTION", "users")
# Append-only record of logins and account changes. The first thing anyone asks for after
# an incident, and impossible to reconstruct after the fact.
AUDIT_COLLECTION = os.getenv("AUDIT_COLLECTION", "audit")

# ---------------------------------------------------------------- E-commerce ingestion
# The two Amazon Fashion 2023 JSONL dumps (product metadata and reviews) and how much of
# them actually gets ingested. See PLAN.md Phase 2 for the 3-pass streaming pipeline and
# preprocessing rules these feed.
AMAZON_META_PATH = _path_setting("AMAZON_META_PATH", DATA_DIR / "meta_Amazon_Fashion.jsonl")
AMAZON_REVIEWS_PATH = _path_setting("AMAZON_REVIEWS_PATH", DATA_DIR / "Amazon_Fashion.jsonl")
# How many products (ranked by review count) to ingest. The full dataset is millions of
# products; a curated subset is what makes local, CPU-only embedding practical.
PRODUCT_SUBSET_SIZE = int(os.getenv("PRODUCT_SUBSET_SIZE", "5000"))
if PRODUCT_SUBSET_SIZE < 1:
    raise ValueError(f"PRODUCT_SUBSET_SIZE must be at least 1, got {PRODUCT_SUBSET_SIZE}")
MONGO_PRODUCTS_COLLECTION = os.getenv("MONGO_PRODUCTS_COLLECTION", "products")
# Reviews not detected as English are dropped before embedding: the local embedding model
# (bge-small-en-v1.5) is English-tuned, and a foreign-language review embeds to noise
# rather than anything genuinely retrievable.
REVIEW_LANGUAGE_FILTER_ENABLED = _flag("REVIEW_LANGUAGE_FILTER_ENABLED", "true")

# ---------------------------------------------------------------- Agentic router
# Caps how many tool-call <-> LLM round-trips one chat turn may take, so a confused model
# can't loop forever calling tools instead of answering. See PLAN.md Phase 4.
AGENT_MAX_TOOL_ROUNDTRIPS = int(os.getenv("AGENT_MAX_TOOL_ROUNDTRIPS", "3"))
if AGENT_MAX_TOOL_ROUNDTRIPS < 1:
    raise ValueError(
        f"AGENT_MAX_TOOL_ROUNDTRIPS must be at least 1, got {AGENT_MAX_TOOL_ROUNDTRIPS}"
    )
AGENT_CHECKPOINTING_ENABLED = _flag("AGENT_CHECKPOINTING_ENABLED", "true")
AGENT_MODEL_RETRIES = int(os.getenv("AGENT_MODEL_RETRIES", "2"))
AGENT_MODEL_RETRY_INITIAL_DELAY_SECONDS = float(os.getenv("AGENT_MODEL_RETRY_INITIAL_DELAY_SECONDS", "0.5"))
AGENT_MODEL_RETRY_MAX_DELAY_SECONDS = float(os.getenv("AGENT_MODEL_RETRY_MAX_DELAY_SECONDS", "8"))
AGENT_SUMMARY_TRIGGER_FRACTION = float(os.getenv("AGENT_SUMMARY_TRIGGER_FRACTION", "0.75"))
AGENT_SUMMARY_KEEP_MESSAGES = int(os.getenv("AGENT_SUMMARY_KEEP_MESSAGES", "16"))
AGENT_SUMMARY_FALLBACK_TOKENS = int(os.getenv("AGENT_SUMMARY_FALLBACK_TOKENS", "24000"))
AGENT_TOOL_CLEAR_TRIGGER_TOKENS = int(os.getenv("AGENT_TOOL_CLEAR_TRIGGER_TOKENS", "18000"))
AGENT_TOOL_CLEAR_KEEP = int(os.getenv("AGENT_TOOL_CLEAR_KEEP", "2"))
AGENT_TOOL_CLEAR_AT_LEAST_TOKENS = int(os.getenv("AGENT_TOOL_CLEAR_AT_LEAST_TOKENS", "4000"))
if AGENT_MODEL_RETRIES < 0:
    raise ValueError("AGENT_MODEL_RETRIES must not be negative")
if not 0 < AGENT_SUMMARY_TRIGGER_FRACTION < 1:
    raise ValueError("AGENT_SUMMARY_TRIGGER_FRACTION must be between 0 and 1")
if AGENT_SUMMARY_KEEP_MESSAGES < 1:
    raise ValueError("AGENT_SUMMARY_KEEP_MESSAGES must be at least 1")

# ---------------------------------------------------------------- Chat history
SESSIONS_COLLECTION = os.getenv("SESSIONS_COLLECTION", "chat_sessions")
# How many conversations the sidebar asks for at a time. Ten is enough to fill the panel
# without making the first paint wait for a hundred rows.
SESSION_PAGE_SIZE = int(os.getenv("SESSION_PAGE_SIZE", "10"))
# Titles are truncated from the first question; long enough to be recognisable, short
# enough to fit the sidebar without wrapping to three lines.
MAX_SESSION_TITLE = int(os.getenv("MAX_SESSION_TITLE", "48"))
# Messages live inside the session document, so the document has to stay well under
# Mongo's 16MB limit. The oldest are dropped past this.
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", "400"))

# Signing key for JWTs. Generated on first run and written to .env if absent (see
# src/services/security.py) - a key that changed on every restart would silently log
# everyone out, and a hard-coded default would let anyone mint a valid token.
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "12"))

# ---------------------------------------------------------------- Chunking (semantic)
# There is ONE chunker: boundaries fall where the MEANING changes, not where a word count
# runs out. Every sentence is embedded together with its neighbours, the distance between
# consecutive windows is measured, and a cut is made wherever that distance spikes.
# See src/services/chunking.py.
#
# It costs one embedding per sentence at ingest - roughly 15x the embedding calls of a
# fixed packer - which on a local model is CPU minutes rather than money.

# A chunk boundary is placed where the distance between neighbouring sentences lands above
# this percentile of that document's own distance distribution. A percentile, not an
# absolute threshold, because the raw numbers differ per document and per embedding model:
# a fixed 0.3 that splits sensibly in one book cuts every other sentence in the next.
# Lower = more, smaller chunks. 95 is the usual starting point.
SEMANTIC_BREAKPOINT_PERCENTILE = float(os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", "95"))
# Sentences embedded with N neighbours on each side. A lone sentence embeds noisily -
# "It does not." carries no topic at all - and the buffer is what stops that noise being
# read as a topic change. 1 means each embedded window is 3 sentences wide.
SEMANTIC_BUFFER_SIZE = int(os.getenv("SEMANTIC_BUFFER_SIZE", "1"))
# Floor and ceiling in words. The floor merges away one-line chunks (headings, page
# furniture) that would otherwise each occupy a top_k slot; the ceiling stops a passage
# with no detectable topic change from becoming one enormous chunk that the embedding
# model would truncate anyway.
SEMANTIC_MIN_CHUNK_WORDS = int(os.getenv("SEMANTIC_MIN_CHUNK_WORDS", "60"))
SEMANTIC_MAX_CHUNK_WORDS = int(os.getenv("SEMANTIC_MAX_CHUNK_WORDS", "300"))
if SEMANTIC_MIN_CHUNK_WORDS >= SEMANTIC_MAX_CHUNK_WORDS:
    raise ValueError(
        f"SEMANTIC_MIN_CHUNK_WORDS ({SEMANTIC_MIN_CHUNK_WORDS}) must be below "
        f"SEMANTIC_MAX_CHUNK_WORDS ({SEMANTIC_MAX_CHUNK_WORDS})."
    )

# ---------------------------------------------------------------- Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))  # hard ceiling on what a client may request

# Retrieve a wide candidate set, then narrow it with the re-ranker. A bi-encoder is fast
# but coarse; the cross-encoder is the thing that decides what actually reaches the model.
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "10"))

# Hybrid search: BM25 keyword ranking fused with vector ranking (Reciprocal Rank Fusion).
# Dense vectors are weak on exact technical terms ("A* search", "Bayes decision rule");
# BM25 is strong there, and vice versa.
HYBRID_ENABLED = _flag("HYBRID_ENABLED", "true")
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF damping constant; 60 is the published default

# Cross-encoder re-ranking, on this machine. Set RERANK_ENABLED=false to A/B it against
# the eval harness.
RERANK_ENABLED = _flag("RERANK_ENABLED", "true")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Cross-encoder logits are unbounded; ms-marco models put clearly-irrelevant pairs well
# below zero. Chunks scoring under this are dropped, and if nothing survives we answer
# "not in these documents" WITHOUT calling the LLM.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-6.0"))
# Fallback floor used when re-ranking is off (cosine similarity, 0..1).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.15"))

# Context window expansion: also pull the chunks immediately before/after each hit, since
# the sentence that explains an answer often sits in the neighbouring chunk. Semantic
# chunks do not overlap by design - a boundary is the point of the strategy - so this is
# what stops the model reading a passage in isolation.
NEIGHBOR_EXPANSION = int(os.getenv("NEIGHBOR_EXPANSION", "0"))

# Cap on the assembled CONTEXT block, independent of top_k, so a large retrieval can
# never blow past the model's context window.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
REVIEW_EXCERPT_CHARS = int(os.getenv("REVIEW_EXCERPT_CHARS", "700"))
REVIEW_MAX_RESULTS_TO_MODEL = int(os.getenv("REVIEW_MAX_RESULTS_TO_MODEL", "6"))

# Answer cache: repeated questions skip retrieval and the LLM call entirely. Entries are
# per user, die when that user's documents change, and expire after the TTL.
ANSWER_CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", "256"))
ANSWER_CACHE_TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "3600"))
# Set false to measure the pipeline without cache hits confusing the numbers.
ANSWER_CACHE_ENABLED = _flag("ANSWER_CACHE_ENABLED", "true")

# The lexical half of hybrid retrieval. BM25 runs in this process and reads every chunk of
# the corpus once to build its index - a fast disk read against a local Chroma folder.
# "off" disables it; retrieval is then dense search plus the re-ranker.
KEYWORD_SEARCH = os.getenv("KEYWORD_SEARCH", "on").strip().lower()
if KEYWORD_SEARCH not in ("on", "off"):
    raise ValueError(f"KEYWORD_SEARCH must be 'on' or 'off' - got {KEYWORD_SEARCH!r}")

# ---------------------------------------------------------------- Conversation
# How many previous question/answer pairs to carry into the prompt.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "2"))
# Hard ceiling on the conversation carried into the prompt, independent of HISTORY_TURNS.
# The per-field limits in schemas.py stop one enormous turn; this stops several large ones
# adding up.
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "3000"))
# Rewrite a follow-up ("what about the second one?") into a standalone question before
# retrieving - the raw follow-up embeds to nothing useful.
REWRITE_FOLLOWUPS = _flag("REWRITE_FOLLOWUPS", "true")

# Origins allowed to call the API from a browser. The default static frontend is served by
# this same FastAPI process, so local development needs no CORS entry. Set this only if a
# browser calls the API directly from a different origin.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# ---------------------------------------------------------------- Misc
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# "text" for a human at a terminal, "json" for anything that ships logs somewhere.
LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()

# App-only controls; changing these does not change an existing embedding generation.
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))
AGENT_TRACING_ENABLED = _flag("AGENT_TRACING_ENABLED", "false")
APP_WARMUP_ENABLED = _flag("APP_WARMUP_ENABLED", "false")
APP_WARMUP_DELAY_SECONDS = float(os.getenv("APP_WARMUP_DELAY_SECONDS", "20"))
DIRECT_ANSWERS_ENABLED = _flag("DIRECT_ANSWERS_ENABLED", "true")
if AGENT_TIMEOUT_SECONDS <= 0:
    raise ValueError("AGENT_TIMEOUT_SECONDS must be positive")
if APP_WARMUP_DELAY_SECONDS < 0:
    raise ValueError("APP_WARMUP_DELAY_SECONDS cannot be negative")
