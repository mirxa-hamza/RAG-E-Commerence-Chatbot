"""A follow-up narrows the previous search instead of starting a new one."""
from unittest.mock import AsyncMock
import mongomock
import pytest
from bson import ObjectId
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from src.agent.shopper import answer, history_messages, previous_filters
from src.agent.tools import merge_filters
from src.services import catalog, database

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


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    position: int = 0
    @property
    def _llm_type(self): return "offline-scripted"
    def bind_tools(self, tools, **kwargs): return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = self.responses[min(self.position, len(self.responses)-1)]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=result)])


def call(name, args, identity):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": identity}])


# ----------------------------------------------------------------- merge_filters

def test_bare_constraint_inherits_the_previous_subject():
    merged = merge_filters({"max_price": 40}, {"query": "shoes", "colors": ["black"]})
    assert merged == {"max_price": 40, "query": "shoes", "colors": ["black"]}


def test_naming_a_new_subject_carries_nothing_over():
    merged = merge_filters({"query": "dresses"}, {"query": "shoes", "colors": ["black"], "max_price": 40})
    assert merged == {"query": "dresses"}


def test_a_restated_constraint_wins_over_the_inherited_one():
    merged = merge_filters({"max_price": 25}, {"query": "shoes", "max_price": 40})
    assert merged["max_price"] == 25


def test_first_turn_has_nothing_to_inherit():
    assert merge_filters({"query": "shoes"}, {}) == {"query": "shoes"}


def test_the_three_step_narrowing_accumulates():
    """shoes -> black -> under $40 must end at all three constraints."""
    first = merge_filters({"query": "shoes"}, {})
    second = merge_filters({"colors": ["black"]}, first)
    third = merge_filters({"max_price": 40}, second)
    assert third == {"query": "shoes", "colors": ["black"], "max_price": 40}


# ----------------------------------------------------------------- plumbing

def test_previous_filters_reads_the_latest_search():
    history = [{"role": "assistant", "active_filters": {"query": "shoes"}},
               {"role": "user", "content": "black"},
               {"role": "assistant", "active_filters": {"query": "shoes", "colors": ["black"]}}]
    assert previous_filters(history) == {"query": "shoes", "colors": ["black"]}
    assert previous_filters([]) == {}


def test_history_shows_the_model_what_was_searched():
    turns = history_messages([{"role": "assistant", "content": "Here are some shoes.",
                               "active_filters": {"query": "shoes", "colors": ["black"]}}])
    assert "Search filters used" in turns[0]["content"]
    assert "black" in turns[0]["content"]


@pytest.mark.asyncio
async def test_follow_up_search_is_narrowed_end_to_end(mongo, monkeypatch):
    """The model asks only for a price; the search still keeps the earlier subject."""
    seen = {}

    async def fake_search(query):
        seen["filters"] = query.model_dump(exclude_none=True)
        return {"count": 1, "results": [{"parent_asin": "P1", "title": "Black shoe", "price": 30}],
                "applied_filters": seen["filters"]}

    monkeypatch.setattr(catalog, "search", fake_search)
    history = [{"role": "user", "content": "black shoes"},
               {"role": "assistant", "content": "Here you go.",
                "active_filters": {"query": "shoes", "colors": ["black"]}}]
    model = ScriptedModel(responses=[
        call("catalog_search", {"max_price": 40}, "one"),
        call("AnswerDraft", {"answer": "These are under $40.", "product_ids": ["P1"],
                             "citation_ids": []}, "two")])

    result = await answer(UID, "s", "under $40", history, model=model)

    assert seen["filters"]["query"] == "shoes"
    assert seen["filters"]["colors"] == ["black"]
    assert seen["filters"]["max_price"] == 40
    # and the merged set is handed to the turn after this one
    assert result.active_filters["colors"] == ["black"]
    assert result.active_filters["max_price"] == 40
