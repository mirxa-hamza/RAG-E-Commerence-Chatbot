"""LangChain read adapter for the existing review index. Never rewrites ingestion data."""
import threading
from functools import lru_cache
from src.core import config

_lock = threading.Lock()
_store = None


@lru_cache(maxsize=1)
def embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    class BGEEmbeddings(HuggingFaceEmbeddings):
        def embed_query(self, text: str) -> list[float]:
            return super().embed_query(config.EMBEDDING_QUERY_PREFIX + text)

    return BGEEmbeddings(model_name=config.EMBEDDING_MODEL,
                         model_kwargs={"device": "cpu"},
                         encode_kwargs={"normalize_embeddings": True,
                                        "batch_size": config.EMBEDDING_BATCH_SIZE})


def get_store():
    global _store
    with _lock:
        if _store is None:
            from langchain_chroma import Chroma
            client = get_client()
            # Refuse an empty/new collection name: silently creating one hides config errors.
            client.get_collection(config.CHROMA_COLLECTION)
            _store = Chroma(client=client, collection_name=config.CHROMA_COLLECTION,
                            embedding_function=embedding_model(), create_collection_if_not_exists=False)
    return _store


def count() -> int:
    return get_client().get_collection(config.CHROMA_COLLECTION).count()


@lru_cache(maxsize=1)
def get_client():
    import chromadb
    if config.CHROMA_MODE == "embedded" and not (config.CHROMA_DIR / "ecommerce_manifest.json").exists():
        raise RuntimeError("The ingestion manifest is not complete yet. Wait for ingestion before opening the review index.")
    return (chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
            if config.CHROMA_MODE == "server" else
            chromadb.PersistentClient(path=str(config.CHROMA_DIR)))


def is_ready():
    return bool(embedding_model.cache_info().currsize)
