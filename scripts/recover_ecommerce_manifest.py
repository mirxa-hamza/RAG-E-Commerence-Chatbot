"""Recover ecommerce_manifest.json after a late ingestion failure.

Use this only when products and Chroma review chunks were written but the final manifest
was not saved. It does not embed, delete, or rewrite vectors.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymongo  # noqa: E402

from src.core import config  # noqa: E402
from src.services import vectorstore  # noqa: E402
from src.services.ecommerce_ingest import _content_hash, _save_manifest  # noqa: E402


def _product_ids() -> set[str]:
    with pymongo.MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000) as client:
        client.admin.command("ping")
        return {
            doc["parent_asin"]
            for doc in client[config.MONGO_DB][config.MONGO_PRODUCTS_COLLECTION].find(
                {}, {"parent_asin": 1}
            )
            if doc.get("parent_asin")
        }


def _content_hashes(product_ids: set[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with open(config.AMAZON_META_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                raw = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            parent_asin = raw.get("parent_asin")
            if parent_asin in product_ids:
                hashes[parent_asin] = _content_hash(raw_line)
                if len(hashes) == len(product_ids):
                    break
    missing = product_ids - set(hashes)
    if missing:
        sample = ", ".join(sorted(missing)[:10])
        raise RuntimeError(f"Could not recover metadata hashes for {len(missing)} product(s): {sample}")
    return hashes


def _review_counts() -> tuple[dict[str, int], dict[str, int], int]:
    collection = vectorstore._col()  # noqa: SLF001 - recovery script deliberately uses the writer client.
    total = collection.count()
    chunk_counts: dict[str, int] = defaultdict(int)
    review_ids: dict[str, set[str]] = defaultdict(set)
    offset = 0
    while offset < total:
        page = collection.get(limit=config.CHROMA_ADD_BATCH, offset=offset, include=["metadatas"])
        ids = page.get("ids") or []
        if not ids:
            break
        for metadata in page.get("metadatas") or []:
            parent_asin = (metadata or {}).get("parent_asin")
            if not parent_asin:
                continue
            chunk_counts[parent_asin] += 1
            review_id = metadata.get("review_id")
            if review_id:
                review_ids[parent_asin].add(review_id)
        offset += len(ids)
    return dict(chunk_counts), {k: len(v) for k, v in review_ids.items()}, total


def main() -> int:
    manifest_file = config.CHROMA_DIR / "ecommerce_manifest.json"
    if manifest_file.exists():
        raise SystemExit(f"{manifest_file} already exists; refusing to overwrite it.")

    product_ids = _product_ids()
    if not product_ids:
        raise SystemExit("No products found in MongoDB; nothing to recover.")

    content_hashes = _content_hashes(product_ids)
    chunk_counts, review_counts, total_chunks = _review_counts()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        parent_asin: {
            "content_hash": content_hashes[parent_asin],
            "review_count": review_counts.get(parent_asin, 0),
            "chunk_count": chunk_counts.get(parent_asin, 0),
            "ingested_at": now,
            "recovered": True,
        }
        for parent_asin in sorted(product_ids)
    }
    _save_manifest(manifest)
    print(json.dumps({
        "manifest": str(manifest_file),
        "manifest_products": len(manifest),
        "manifest_chunks": sum(item["chunk_count"] for item in manifest.values()),
        "chroma_chunks": total_chunks,
        "products_with_chunks": sum(1 for item in manifest.values() if item["chunk_count"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
