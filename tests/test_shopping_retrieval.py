"""Real Chroma/LangChain adapter with fake vectors; no production index or model."""
import uuid
import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from src.agent.schemas import ReviewQuery
from src.services import review_search


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "shoe" in t or "small" in t else [0.0, 1.0] for t in texts]
    def embed_query(self, text): return [1.0, 0.0]


def test_shared_reviews_retain_identity_and_expand_only_same_review(monkeypatch):
    client = chromadb.EphemeralClient()
    name = "test-" + uuid.uuid4().hex
    client.create_collection(name, metadata={"hnsw:space":"cosine"})
    store = Chroma(client=client, collection_name=name, embedding_function=TinyEmbeddings())
    texts = ["shoe runs small", "try the next size", "a different customer", "dress is blue"]
    meta = [{"parent_asin":"P1","review_id":"P1::r0","chunk_index":0,"rating":4},
            {"parent_asin":"P1","review_id":"P1::r0","chunk_index":1,"rating":4},
            {"parent_asin":"P1","review_id":"P1::r1","chunk_index":1,"rating":1},
            {"parent_asin":"P2","review_id":"P2::r0","chunk_index":0,"rating":5}]
    ids = [f"{m['review_id']}::c{m['chunk_index']}" for m in meta]
    try:
        store.add_texts(texts=texts, metadatas=meta, ids=ids)
        monkeypatch.setattr(review_search, "get_store", lambda:store)
        monkeypatch.setattr(review_search, "_cache", None)
        monkeypatch.setattr(review_search.config, "HYBRID_ENABLED", True)
        monkeypatch.setattr(review_search.config, "RERANK_ENABLED", True)
        monkeypatch.setattr(review_search.config, "NEIGHBOR_EXPANSION", 1)
        monkeypatch.setattr(review_search.reranker, "rerank", lambda question, rows: [(r,10 if "small" in r["text"] else -100) for r in rows])
        result = review_search.search(ReviewQuery(question="small shoe",product_ids=["P1"],min_rating=3,limit=1))
        assert {r["id"] for r in result["results"]} == set(ids[:2])
        monkeypatch.setattr(review_search.reranker, "rerank", lambda question, rows: [(r,-100) for r in rows])
        assert review_search.search(ReviewQuery(question="irrelevant")).get("count") == 0
    finally:
        client.delete_collection(name)


@pytest.mark.parametrize("provider",["groq","gemini"])
def test_provider_binding_uses_typed_tools_without_network(monkeypatch, provider):
    from src.agent.models import get_chat_model, config
    from src.agent.tools import create_tools, Evidence
    monkeypatch.setattr(config,"LLM_PROVIDER",provider)
    monkeypatch.setattr(config,"GROQ_API_KEY","offline-test-key")
    monkeypatch.setattr(config,"GEMINI_API_KEY","offline-test-key")
    get_chat_model.cache_clear()
    try:
        model = get_chat_model()
        bound = model.bind_tools(create_tools("000000000000000000000001",Evidence()))
        assert bound is not None
    finally:
        get_chat_model.cache_clear()
