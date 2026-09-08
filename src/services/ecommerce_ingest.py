"""
Dual-store ingestion pipeline: the Amazon Fashion 2023 dataset -> MongoDB (products) +
ChromaDB (review chunks). See PLAN.md Phase 2 for the full design and the preprocessing
rules this enforces.

Three streaming passes over the two raw JSONL dumps, never loading either multi-gigabyte
file fully into memory:

  1. _select_subset()   count reviews per parent_asin in AMAZON_REVIEWS_PATH, take the top
                        PRODUCT_SUBSET_SIZE by count.
  2. _ingest_products() stream AMAZON_META_PATH; for each selected product, hash its raw
                        line and compare against the manifest - unchanged products are
                        skipped, new/changed ones are cleaned and upserted into MongoDB.
  3. _ingest_reviews()  stream AMAZON_REVIEWS_PATH again; for the products pass 2 decided
                        were new/changed, clean + filter + chunk + embed their reviews
                        into ChromaDB.

Idempotent by default: a small JSON manifest (ecommerce_manifest.json, beside the Chroma
index) tracks a content hash per product, so re-running without --force only touches what
actually changed. --force wipes both stores and the manifest and starts clean - the mode
you actually use while tuning the pipeline.

Run via `python scripts/ingest_ecommerce.py`, not by importing this module from the
FastAPI app: ingestion is a one-off batch job (see main.py's docstring), not a background
thread kicked off at server startup the way the retired PDF pipeline's ingestion.py was.
"""
import hashlib
import html
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from src.core.config import (
    AMAZON_META_PATH,
    AMAZON_REVIEWS_PATH,
    CHROMA_DIR,
    MONGO_DB,
    MONGO_PRODUCTS_COLLECTION,
    MONGO_URI,
    PRODUCT_SUBSET_SIZE,
    REVIEW_LANGUAGE_FILTER_ENABLED,
)
from src.core.logging import get_logger, timed
from src.services import vectorstore
from src.services.chunking import chunk_review

log = get_logger(__name__)

# Reviews shorter than this contribute nothing retrievable ("Great!", "Love it!") and are
# dropped before chunking/embedding. Not a config setting: this is an ingestion detail,
# not something a deployment needs to tune (see PLAN.md's preprocessing rules).
MIN_REVIEW_WORDS = 3

# How many (id, text, metadata) review chunks to buffer before one embed+add batch.
# Buffering across many reviews - rather than flushing one review at a time - is what
# keeps embed_passages() batching efficiently instead of paying per-call overhead
# thousands of times over for a dataset where most reviews are only 1-2 chunks.
_FLUSH_EVERY = 256

_MANIFEST_PATH = CHROMA_DIR / "ecommerce_manifest.json"


# ==================================================================== langdetect setup

def _configure_langdetect() -> None:
    """
    langdetect's detector is NOT deterministic run-to-run unless seeded - without this, a
    --force re-ingest of the exact same data could classify a handful of borderline
    reviews differently each time, which would be a confusing thing to debug months later.
    """
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0


def _is_english(text: str) -> bool:
    """
    True if the review is (probably) English, or if the language filter is disabled.

    Detection failures (LangDetectException - typically punctuation-only or otherwise
    ambiguous text that slipped past the word-count filter) fail OPEN: keeping a review of
    uncertain language is a smaller loss than dropping one that was actually fine, and a
    genuinely non-English review almost always has enough signal to classify cleanly.
    """
    if not REVIEW_LANGUAGE_FILTER_ENABLED:
        return True
    from langdetect import LangDetectException, detect
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True


# ==================================================================== JSONL streaming

def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Download the Amazon Fashion 2023 dataset and place "
            f"both JSONL files under data/ (see .env.example section 8)."
        )


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    """Yields one dict per line, skipping (and counting) unparseable lines."""
    _require(path)
    bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        log.warning("%s: skipped %d line(s) that were not valid JSON.", path.name, bad)


# ==================================================================== manifest (per-product idempotency)

def _load_manifest() -> Dict[str, Dict]:
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s (%s); treating it as empty.", _MANIFEST_PATH, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_manifest(entries: Dict[str, Dict]) -> None:
    """Atomic write (temp file + os.replace) - same pattern the old PDF manifest used."""
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(_MANIFEST_PATH.parent), prefix=".ecommerce-manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, _MANIFEST_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _content_hash(raw_line: str) -> str:
    return hashlib.sha256(raw_line.encode("utf-8", "replace")).hexdigest()


# ==================================================================== preprocessing (products)

def _parse_price(raw) -> Optional[float]:
    """
    The dataset's price field is inconsistently a real number, a numeric string, "None",
    or absent. Returns None (never 0) when it can't be parsed, so catalog_search's
    price-range filter can treat "no price data" as "excluded from price filtering"
    rather than mistaking it for a free product.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"[\d][\d,]*\.?\d*", str(raw))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def _flatten_image(images) -> Optional[str]:
    """
    One canonical image URL per product, out of the dataset's list of
    {"thumb", "large", "hi_res", "variant"} dicts - the frontend product card (Phase 5)
    should never need to know Amazon's nested image schema.
    """
    if not images:
        return None
    if isinstance(images, dict):
        images = [images]
    for entry in images:
        if not isinstance(entry, dict):
            continue
        for key in ("large", "hi_res", "thumb"):
            url = entry.get(key)
            if url:
                return url
    return None


def _dedupe_strings(values, cap: Optional[int] = None) -> List[str]:
    if not values:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for v in values:
        v = str(v).strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out[:cap] if cap else out


def _clean_product(raw: Dict) -> Optional[Dict]:
    """Returns a Mongo-ready product document, or None if the record is unusable."""
    parent_asin = raw.get("parent_asin")
    title = (raw.get("title") or "").strip()
    if not parent_asin or not title:
        return None

    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    brand = raw.get("brand") or raw.get("store") or details.get("Brand") or None
    if isinstance(brand, str):
        brand = brand.strip() or None

    try:
        rating_number = int(raw.get("rating_number") or 0)
    except (TypeError, ValueError):
        rating_number = 0
    try:
        average_rating = float(raw.get("average_rating") or 0)
    except (TypeError, ValueError):
        average_rating = 0.0

    return {
        "_id": parent_asin,
        "parent_asin": parent_asin,
        "title": title,
        "brand": brand,
        "price": _parse_price(raw.get("price")),
        "image_url": _flatten_image(raw.get("images")),
        "categories": _dedupe_strings(raw.get("categories")),
        "features": _dedupe_strings(raw.get("features"), cap=20),
        "description": _dedupe_strings(raw.get("description"), cap=20),
        "average_rating": average_rating,
        "rating_number": rating_number,
        "main_category": (raw.get("main_category") or "").strip() or None,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ==================================================================== preprocessing (reviews)

# parent_asin -> {sha256(text), ...} seen so far THIS run. Cleared at the start of run().
_seen_review_texts: Dict[str, Set[str]] = {}


def _clean_review_text(raw_text: Optional[str]) -> Optional[str]:
    text = html.unescape((raw_text or "").strip())
    if not text or len(text.split()) < MIN_REVIEW_WORDS:
        return None
    return text


def _is_duplicate(parent_asin: str, text: str) -> bool:
    """Exact-duplicate review text within the same product - scraped datasets carry repeats."""
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    seen = _seen_review_texts.setdefault(parent_asin, set())
    if digest in seen:
        return True
    seen.add(digest)
    return False


# ==================================================================== pass 1

def _select_subset(limit: int) -> Set[str]:
    """Top `limit` parent_asins in AMAZON_REVIEWS_PATH, ranked by review count."""
    counts: Counter = Counter()
    total_lines = 0
    with timed(log, "pass 1: count reviews per product"):
        for rec in _iter_jsonl(AMAZON_REVIEWS_PATH):
            total_lines += 1
            parent_asin = rec.get("parent_asin")
            if parent_asin:
                counts[parent_asin] += 1

    selected = {parent_asin for parent_asin, _ in counts.most_common(limit)}
    log.info(
        "Pass 1: %d review(s) across %d product(s) seen; selected the top %d by review "
        "count.", total_lines, len(counts), len(selected),
    )
    return selected


# ==================================================================== pass 2

def _ingest_products(selected: Set[str], manifest: Dict[str, Dict], products_col) -> Dict[str, str]:
    """
    Cleans and upserts the products in `selected` whose raw line hash differs from what
    `manifest` has on record (or which aren't in the manifest at all).

    Returns {parent_asin: new_content_hash} - exactly the products that were written, and
    exactly what pass 3 needs to know it must (re)process.
    """
    _require(AMAZON_META_PATH)
    changed: Dict[str, str] = {}
    found: Set[str] = set()
    with timed(log, "pass 2: ingest product metadata"):
        with open(AMAZON_META_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    raw = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                parent_asin = raw.get("parent_asin")
                if parent_asin not in selected:
                    continue
                found.add(parent_asin)

                content_hash = _content_hash(raw_line)
                if manifest.get(parent_asin, {}).get("content_hash") == content_hash:
                    continue  # unchanged since the last ingest

                doc = _clean_product(raw)
                if doc is None:
                    log.warning("Skipping %s: unusable product record (missing title).",
                               parent_asin)
                    continue

                products_col.update_one({"_id": parent_asin}, {"$set": doc}, upsert=True)
                changed[parent_asin] = content_hash

    missing = selected - found
    if missing:
        sample = ", ".join(sorted(missing)[:10]) + (" ..." if len(missing) > 10 else "")
        log.warning("%d selected product(s) were not found in %s: %s",
                   len(missing), AMAZON_META_PATH.name, sample)
    log.info("Pass 2: %d/%d selected product(s) found; %d new/changed, %d unchanged.",
             len(found), len(selected), len(changed), len(found) - len(changed))
    return changed


# ==================================================================== pass 3

def _ingest_reviews(to_process: Set[str]) -> Dict[str, Tuple[int, int]]:
    """
    Cleans, filters, chunks and embeds the reviews of every product in `to_process`.

    Returns {parent_asin: (reviews_kept, chunks_written)} for everything actually
    written - what run() records in the manifest once this returns successfully.
    """
    counts: Dict[str, List[int]] = {}
    review_index_by_product: Dict[str, int] = {}
    buffer_ids: List[str] = []
    buffer_texts: List[str] = []
    buffer_meta: List[Dict] = []

    def flush() -> None:
        if not buffer_ids:
            return
        vectorstore.add_review_chunks(list(buffer_ids), list(buffer_texts), list(buffer_meta))
        buffer_ids.clear()
        buffer_texts.clear()
        buffer_meta.clear()

    seen_reviews = 0
    kept_reviews = 0
    with timed(log, "pass 3: filter, chunk and embed reviews"):
        for rec in _iter_jsonl(AMAZON_REVIEWS_PATH):
            parent_asin = rec.get("parent_asin")
            if parent_asin not in to_process:
                continue
            seen_reviews += 1

            text = _clean_review_text(rec.get("text"))
            if text is None:
                continue
            if _is_duplicate(parent_asin, text):
                continue
            if not _is_english(text):
                continue

            chunks = chunk_review(text)
            if not chunks:
                continue

            review_index = review_index_by_product.get(parent_asin, 0)
            review_index_by_product[parent_asin] = review_index + 1
            review_id = f"{parent_asin}::r{review_index}"

            try:
                rating = float(rec.get("rating")) if rec.get("rating") is not None else None
            except (TypeError, ValueError):
                rating = None
            review_title = (rec.get("title") or "").strip() or None
            timestamp = rec.get("timestamp")

            for chunk_index, chunk_text in enumerate(chunks):
                buffer_ids.append(f"{review_id}::c{chunk_index}")
                buffer_texts.append(chunk_text)
                buffer_meta.append({
                    "parent_asin": parent_asin,
                    "review_id": review_id,
                    "rating": rating,
                    "review_title": review_title,
                    "timestamp": timestamp,
                    "chunk_index": chunk_index,
                })

            kept_reviews += 1
            product_counts = counts.setdefault(parent_asin, [0, 0])
            product_counts[0] += 1
            product_counts[1] += len(chunks)

            if len(buffer_ids) >= _FLUSH_EVERY:
                flush()

        flush()

    log.info(
        "Pass 3: %d review(s) seen for the %d product(s) being (re)processed; %d kept "
        "after filtering (empty/too-short/duplicate/non-English dropped).",
        seen_reviews, len(to_process), kept_reviews,
    )
    return {k: (v[0], v[1]) for k, v in counts.items()}


# ==================================================================== orchestration

def run(*, limit: Optional[int] = None, force: bool = False) -> Dict:
    """
    The full pipeline. Returns a summary dict for the CLI to print.

    `limit` overrides PRODUCT_SUBSET_SIZE - used for the dry run (`--limit 1000`)
    PLAN.md recommends before committing to the full ingest.
    `force` wipes the products collection, the vector store and the manifest, then
    re-ingests everything from scratch.
    """
    import pymongo

    _configure_langdetect()
    _seen_review_texts.clear()
    limit = PRODUCT_SUBSET_SIZE if limit is None else limit

    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        client.close()
        raise RuntimeError(
            f"Could not reach MongoDB at {MONGO_URI}: {exc}. Start it before ingesting."
        ) from exc
    products_col = client[MONGO_DB][MONGO_PRODUCTS_COLLECTION]

    manifest = _load_manifest()
    if force:
        log.info("--force: wiping the products collection, the vector store, and the "
                 "ingestion manifest.")
        products_col.delete_many({})
        vectorstore.reset_collection()
        manifest = {}

    selected = _select_subset(limit)
    changed_hashes = _ingest_products(selected, manifest, products_col)

    if not changed_hashes:
        log.info("Nothing new to ingest: all %d selected product(s) are already up to "
                 "date. Pass --force to rebuild anyway.", len(selected))

    for parent_asin in changed_hashes:
        vectorstore.delete_product_chunks(parent_asin)  # replace, don't duplicate

    review_counts = _ingest_reviews(set(changed_hashes.keys()))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for parent_asin, content_hash in changed_hashes.items():
        review_count, chunk_count = review_counts.get(parent_asin, (0, 0))
        manifest[parent_asin] = {
            "content_hash": content_hash,
            "review_count": review_count,
            "chunk_count": chunk_count,
            "ingested_at": now,
        }
    _save_manifest(manifest)
    client.close()

    summary = {
        "selected_products": len(selected),
        "products_processed": len(changed_hashes),
        "products_skipped_unchanged": len(selected) - len(changed_hashes),
        "reviews_ingested": sum(rc for rc, _ in review_counts.values()),
        "chunks_ingested": sum(cc for _, cc in review_counts.values()),
    }
    log.info("Done: %s", summary)
    return summary
