"""
CLI entry point for the e-commerce dual-store ingestion pipeline (PLAN.md Phase 2).

Usage:
    python scripts/ingest_ecommerce.py --limit 1000      # dry-run-ish: small subset first
    python scripts/ingest_ecommerce.py                   # full run (PRODUCT_SUBSET_SIZE)
    python scripts/ingest_ecommerce.py --force            # wipe Mongo + Chroma and rebuild

Run this after Phase 1's cleanup and before starting the API, once the two Amazon Fashion
JSONL files (see .env's AMAZON_META_PATH / AMAZON_REVIEWS_PATH) are in place and MongoDB /
Chroma are reachable.
"""
import argparse
import sys
from pathlib import Path

# When Python runs a file inside scripts/, it puts scripts/ (rather than the project
# root) on sys.path. Add the root explicitly so this command works from the repository
# root and from any other current directory, just like scripts/run.py does.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.ecommerce_ingest import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest the Amazon Fashion dataset into MongoDB (catalog) and "
                    "ChromaDB (review chunks).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of products to select (by review-count popularity). "
             "Defaults to PRODUCT_SUBSET_SIZE from .env (5000). Use a small value "
             "like 1000 first to sanity-check before the full run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe the existing products collection, the Chroma review collection, "
             "and the ingestion manifest before running, instead of incrementally "
             "updating them.",
    )
    args = parser.parse_args()

    try:
        summary = run(limit=args.limit, force=args.force)
    except FileNotFoundError as exc:
        print(f"Dataset file not found: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print("Ingestion complete.")
    print(f"  Selected products:            {summary['selected_products']}")
    print(f"  Products processed (changed): {summary['products_processed']}")
    print(f"  Products skipped (unchanged): {summary['products_skipped_unchanged']}")
    print(f"  Reviews ingested:             {summary['reviews_ingested']}")
    print(f"  Review chunks ingested:       {summary['chunks_ingested']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
