"""Reuse the ingestion pipeline with a Chroma HTTP client in Docker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import config
from src.services import vectorstore

if __name__ == "__main__":
    if config.CHROMA_MODE != "server":
        raise SystemExit("This entry point requires CHROMA_MODE=server. Use ingest_ecommerce.py locally.")
    import chromadb
    vectorstore._client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    vectorstore._client.heartbeat()
    vectorstore._collection = vectorstore._get_collection()
    from scripts.ingest_ecommerce import main
    raise SystemExit(main())
