"""Offline regression suite: real LangChain graph, fake provider, isolated Mongo fixture."""
import asyncio
from unittest.mock import AsyncMock
import mongomock
import pytest
from bson import ObjectId
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, ValidationError
from src.agent.schemas import AnswerDraft, CatalogQuery, PreferenceUpdate, ReviewQuery
from src.agent.shopper import answer, hydrate, history_messages
from src.agent.tools import Evidence, create_tools
from src.services import catalog, database, preferences, review_search, sessions

UID = "000000000000000000000001"
OTHER = "000000000000000000000002"


class AsyncCollection:
    def __init__(self, inner): self.inner = inner
    async def find_one(self, *a, **kw): return self.inner.find_one(*a, **kw)
    async def update_one(self, *a, **kw): return self.inner.update_one(*a, **kw)
    async def insert_one(self, *a, **kw): return self.inner.insert_one(*a, **kw)
    async def delete_one(self, *a, **kw): return self.inner.delete_one(*a, **kw)


@pytest.fixture
def mongo(monkeypatch):
    db = mongomock.MongoClient().db
    db.users.insert_many([{"_id": ObjectId(UID)}, {"_id": ObjectId(OTHER)}])
    monkeypatch.setattr(database, "users", lambda: AsyncCollection(db.users))
    monkeypatch.setattr(database, "sessions", lambda: AsyncCollection(db.sessions))
    monkeypatch.setattr(database, "record_audit", AsyncMock())
    return db


def test_catalog_constraints_exclude_missing_price():
    col = mongomock.MongoClient().db.products
    col.insert_many([{"title": "Black dress", "price": 30}, {"title": "Black dress", "price": None},
                     {"title": "Black dress", "price": 60}, {"title": "Blue dress", "price": 25}])
    found = list(col.find(catalog.build_filter(CatalogQuery(query="dress", colors=["black"], max_price=40))))
    assert [r["price"] for r in found] == [30]
    assert not list(col.find(catalog.build_filter(CatalogQuery(query=".*"))))


@pytest.mark.parametrize("data", [
    {"field":"budget","action":"set","value":{"min":60,"max":20}},
    {"field":"clothing_size","action":"add","value":"M"},
    {"field":"favorite_brands","action":"set","value":"Nike"},
    {"field":"user_id","action":"set","value":OTHER},
    {"field":"style_notes","action":"clear","value":"x"},
])
def test_invalid_preferences(data):
    with pytest.raises(ValidationError): PreferenceUpdate(**data)


@pytest.mark.asyncio
async def test_memory_is_private_and_list_updates_preserve_values(mongo):
    for color in ("black", "navy", "black"):
        await preferences.update(UID, PreferenceUpdate(field="color_preference", action="add", value=color))
    assert (await preferences.lookup(UID))["color_preference"] == ["black", "navy"]
    assert await preferences.lookup(OTHER) == {}
    await preferences.update(UID, PreferenceUpdate(field="color_preference", action="remove", value="black"))
    assert (await preferences.lookup(UID))["color_preference"] == ["navy"]


@pytest.mark.asyncio
async def test_session_ownership_and_atomic_exchange(mongo):
    session = await sessions.create(UID)
    assert await sessions.get(OTHER, session["id"]) is None
    assert not await sessions.append_exchange(OTHER, session["id"], "hello", {"answer":"no"})
    assert await sessions.append_exchange(UID, session["id"], "Find shoes", {"answer":"Here", "products":[{"parent_asin":"P1"}]})
    found = await sessions.get(UID, session["id"])
    assert found["message_count"] == 2
    assert found["messages"][1]["products"] == [{"parent_asin":"P1"}]


def test_evidence_rejects_fabricated_product():
    with pytest.raises(ValueError):
        hydrate(AnswerDraft(answer="A made up product", product_ids=["fake"]), Evidence(), "session")


def test_zero_results_override_invented_prose():
    result = hydrate(AnswerDraft(answer="A jacket is $1!"), Evidence(searches=[{"count":0,"applied_filters":{"max_price":2}}]), "s")
    assert "$1" not in result.answer and "under $2" in result.answer


def test_product_prose_price_is_grounded_to_selected_card():
    evidence = Evidence(products={"P1": {"parent_asin": "P1", "title": "Grace dress", "price": 28.99}})
    result = hydrate(AnswerDraft(answer="Leadtex dress — $19.99", product_ids=["P1"]), evidence, "s")
    assert "Grace dress" in result.answer
    assert "$28.99" in result.answer
    assert "$19.99" not in result.answer


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
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


@pytest.mark.asyncio
async def test_real_langchain_tool_loop(mongo, monkeypatch):
    monkeypatch.setattr(catalog, "search", AsyncMock(return_value={"count":1, "results":[{"parent_asin":"P1", "title":"Black dress", "price":30}], "applied_filters":{"max_price":40}}))
    model = ScriptedModel(responses=[
        call("remember_preference", {"field":"clothing_size","action":"set","value":"M"}, "one"),
        call("catalog_search", {"query":"dress","max_price":40}, "two"),
        call("AnswerDraft", {"answer":"This black dress is $30.","product_ids":["P1"]}, "three")])
    result = await answer(UID, "s", "I wear M; find a dress under $40", [], model=model)
    assert result.products[0].price == 30
    assert (await preferences.lookup(UID))["clothing_size"] == "M"


@pytest.mark.asyncio
async def test_infinite_tool_loop_is_bounded(mongo):
    model = ScriptedModel(responses=[call("memory_lookup", {}, "loop")])
    result = await answer(UID, "s", "loop", [], model=model)
    assert "search limit" in result.answer
    assert model.position <= 5


def test_tool_identity_is_not_model_argument():
    for tool in create_tools(UID, Evidence()):
        schema = tool.get_input_schema().model_json_schema()
        assert "user_id" not in schema.get("properties", {})


def test_chroma_metadata_sanitizer_drops_none_and_stringifies_complex_values():
    from src.services.vectorstore import _clean_metadata
    assert _clean_metadata({"a": None, "b": "x", "c": 1, "d": ["tag"]}) == {
        "b": "x", "c": 1, "d": "['tag']"
    }


def test_history_retains_product_ids():
    result = history_messages([{"role":"assistant", "content":"Here", "products":[{"parent_asin":"P1"}]}])
    assert "P1" in result[0]["content"]


def test_review_filter_combines_product_and_ratings():
    query = ReviewQuery(question="fit", product_ids=["A"], min_rating=2, max_rating=4)
    assert len(review_search.where_filter(query)["$and"]) == 3


@pytest.mark.asyncio
async def test_agent_timeout(mongo, monkeypatch):
    from src.agent import shopper
    async def slow(_): await asyncio.sleep(2)
    monkeypatch.setattr(preferences, "lookup", slow)
    monkeypatch.setattr(shopper.config, "AGENT_TIMEOUT_SECONDS", .01)
    with pytest.raises(TimeoutError): await answer(UID, "s", "hello", [], model=ScriptedModel())


def test_partial_structured_stream_only_emits_grounded_answer():
    from langchain_core.messages import AIMessageChunk
    from src.agent.shopper import DraftStream
    emitted = []
    stream = DraftStream(Evidence(), "s", emitted.append)
    def feed(text, first=False, name="AnswerDraft", identity="reply"):
        stream.accept(AIMessageChunk(content="", id=identity, tool_call_chunks=[{
            "name":name if first else None, "args":text, "id":"call" if first else None, "index":0
        }]), {"langgraph_node":"model"})
    feed('{"product_ids":[],"citation_ids":[],"answer":"Hel', True)
    feed('lo world"}')
    assert "".join(emitted) == "Hello world"
    feed('{"product_ids":["invented"],"citation_ids":[],"answer":"Fake"}', True, identity="fake")
    assert "".join(emitted) == "Hello world"


@pytest.mark.asyncio
async def test_real_langchain_stream_updates(mongo):
    emitted = []
    model = ScriptedModel(responses=[call("AnswerDraft", {"answer":"Hello", "product_ids":[], "citation_ids":[]}, "final")])
    result = await answer(UID, "s", "hello", [], model=model, on_token=emitted.append)
    assert result.answer == "Hello"
