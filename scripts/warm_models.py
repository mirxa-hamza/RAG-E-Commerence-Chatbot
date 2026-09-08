"""Explicit model download/warm-up job; do this before serving review questions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    from src.services.review_store import embedding_model
    from src.ml import reranker
    embedding_model().embed_query("warm up")
    if not reranker.available(): raise SystemExit("Re-ranker could not be loaded")
    print("Embedding and reranking models are cached and ready.")
