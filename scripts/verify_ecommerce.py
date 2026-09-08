"""Read-only post-ingestion count check. Run only once the ingestion process exits."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import pymongo
    from src.core import config
    from src.services.review_store import count
    manifest_file = config.CHROMA_DIR / "ecommerce_manifest.json"
    if not manifest_file.exists():
        raise SystemExit("No completed ingestion manifest yet. Let the current ingestion finish first.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    with pymongo.MongoClient(config.MONGO_URI,serverSelectionTimeoutMS=5000) as client:
        product_count = client[config.MONGO_DB][config.MONGO_PRODUCTS_COLLECTION].count_documents({})
    expected_chunks = sum(m.get("chunk_count",0) for m in manifest.values())
    actual_chunks = count()
    result = {"products":product_count,"manifest_products":len(manifest),"review_chunks":actual_chunks,
              "manifest_chunks":expected_chunks,"counts_match":product_count == len(manifest) and actual_chunks == expected_chunks}
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["counts_match"] and actual_chunks > 0 and product_count > 0 else 1)
