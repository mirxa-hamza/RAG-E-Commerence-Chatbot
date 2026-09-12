"""Free-form account memory, and the bounded keyword half of review search."""
from unittest.mock import AsyncMock
import mongomock
import pytest
from bson import ObjectId
from pydantic import ValidationError
from src.agent.schemas import PreferenceUpdate, ReviewQuery
from src.services import database, direct_answers, preferences, review_search

UID = "000000000000000000000001"


class AsyncCollection:
    def __init__(self, inner): self.inner = inner
    async def find_one(self, *a, **kw): return self.inner.find_one(*a, **kw)
    async def update_one(self, *a, **kw): return self.inner.update_one(*a, **kw)


@pytest.fixture
def mongo(monkeypatch):
    db = mongomock.MongoClient().db
    db.users.insert_one({"_id": ObjectId(UID)})
    monkeypatch.setattr(database, "users", lambda: AsyncCollection(db.users))
    monkeypatch.setattr(database, "record_audit", AsyncMock())
    return db


def test_memory_accepts_a_sentence_but_not_an_essay():
    PreferenceUpdate(field="memories", action="add", value="Shops for her sister's wedding.")
    with pytest.raises(ValidationError):
        PreferenceUpdate(field="memories", action="add", value="x" * 301)
    with pytest.raises(ValidationError):
        PreferenceUpdate(field="memories", action="set", value="lists use add/remove")


@pytest.mark.asyncio
async def test_memories_evict_oldest_instead_of_refusing_to_learn(mongo, monkeypatch):
    monkeypatch.setattr(preferences.config, "MEMORY_MAX_ENTRIES", 3)
    for i in range(5):
        await preferences.update(UID, PreferenceUpdate(field="memories", action="add", value=f"fact {i}"))
    saved = await preferences.lookup(UID)
    assert saved["memories"] == ["fact 2", "fact 3", "fact 4"], saved["memories"]


@pytest.mark.asyncio
async def test_duplicate_memory_is_not_an_error(mongo):
    await preferences.update(UID, PreferenceUpdate(field="memories", action="add", value="same"))
    await preferences.update(UID, PreferenceUpdate(field="memories", action="add", value="same"))
    assert (await preferences.lookup(UID))["memories"] == ["same"]


@pytest.mark.asyncio
async def test_typed_lists_still_refuse_past_their_cap(mongo):
    for i in range(20):
        await preferences.update(UID, PreferenceUpdate(field="favorite_brands", action="add", value=f"b{i}"))
    with pytest.raises(ValueError):
        await preferences.update(UID, PreferenceUpdate(field="favorite_brands", action="add", value="one too many"))


@pytest.mark.asyncio
async def test_what_do_you_remember_reports_memories(mongo):
    await preferences.update(UID, PreferenceUpdate(field="memories", action="add", value="Buys gifts for her mother"))
    await preferences.update(UID, PreferenceUpdate(field="clothing_size", action="set", value="M"))
    result = await direct_answers._read_simple_preferences(UID, "s", "what do you remember about me?")
    assert "size M" in result.answer
    assert "Buys gifts for her mother." in result.answer


@pytest.mark.asyncio
async def test_no_memories_says_so(mongo):
    result = await direct_answers._read_simple_preferences(UID, "s", "what preferences have you saved?")
    assert "don't have any saved" in result.answer


def test_keyword_terms_drops_stopwords_and_short_words():
    assert review_search.keyword_terms("Are the shoes comfortable for wide feet?") == [
        "shoes", "comfortable", "wide", "feet"]
    assert review_search.keyword_terms("is it a to be?") == []


class FakeStore:
    """Minimal stand-in: records the call and returns two matching chunks."""
    def __init__(self): self.calls = []
    def get(self, **kwargs):
        self.calls.append(kwargs)
        return {"documents": ["runs small on wide feet", "totally unrelated"],
                "metadatas": [{"parent_asin": "P1", "review_id": "P1::r0", "chunk_index": 0, "rating": 4},
                              {"parent_asin": "P1", "review_id": "P1::r1", "chunk_index": 0, "rating": 5}],
                "ids": ["P1::r0::c0", "P1::r1::c0"]}


def test_keyword_candidates_asks_the_store_and_bounds_the_pull():
    store = FakeStore()
    rows = review_search.keyword_candidates(store, ReviewQuery(question="wide feet"))
    call = store.calls[0]
    assert call["limit"] == review_search.config.KEYWORD_CANDIDATE_LIMIT
    assert call["where_document"] == {"$or": [{"$contains": "wide"}, {"$contains": "feet"}]}
    # Only the chunk that actually contains the terms should score above zero.
    assert [r["id"] for r in rows] == ["P1::r0::c0"]


def test_keyword_candidates_skips_the_store_when_nothing_is_searchable():
    store = FakeStore()
    assert review_search.keyword_candidates(store, ReviewQuery(question="is it?")) == []
    assert store.calls == []
