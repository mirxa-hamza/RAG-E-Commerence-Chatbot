"""Exercise HTTP authentication, persisted replies and SSE envelopes without a model."""
from unittest.mock import AsyncMock
import httpx
import pytest
from fastapi import FastAPI
from src.api import shopping
from src.api.deps import get_current_user
from src.agent.schemas import ShoppingResponse
from src.agent import shopper
from src.services import direct_answers, sessions


@pytest.fixture
def app(monkeypatch):
    app = FastAPI()
    app.include_router(shopping.router)
    monkeypatch.setattr(shopping, "_enforce", lambda *a: None)
    monkeypatch.setattr(sessions, "create", AsyncMock(return_value={"id":"000000000000000000000011", "messages":[]}))
    monkeypatch.setattr(sessions, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(sessions, "append_exchange", AsyncMock(return_value=True))
    monkeypatch.setattr(shopper, "answer", AsyncMock(return_value=ShoppingResponse(session_id="000000000000000000000011", answer="Hello")))
    shopping._busy.clear()
    return app


def signed_in(app):
    app.dependency_overrides[get_current_user] = lambda: {"_id":"000000000000000000000001"}


@pytest.mark.asyncio
async def test_chat_requires_auth(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.post("/chat", json={"question":"hi"})).status_code == 401


@pytest.mark.asyncio
async def test_ownership_and_input_validation(app):
    signed_in(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.post("/chat", json={"question":"hi","session_id":"000000000000000000000099"})).status_code == 404
        assert (await c.post("/chat", json={"question":" "})).status_code == 422
        assert (await c.post("/chat", json={"question":"hi", "history":[]})).status_code == 422


@pytest.mark.asyncio
async def test_sse_order_and_persistence(app):
    signed_in(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/chat/stream", json={"question":"hi"})
    events = [line for line in response.text.splitlines() if line.startswith("event:")]
    assert events == ["event: session", "event: products", "event: citations", "event: token", "event: done"]
    sessions.append_exchange.assert_awaited_once()
    assert not shopping._busy


@pytest.mark.asyncio
async def test_sse_direct_answer_skips_agent(app, monkeypatch):
    signed_in(app)
    direct = ShoppingResponse(session_id="000000000000000000000011", answer="Local answer")
    monkeypatch.setattr(direct_answers, "answer_if_simple", AsyncMock(return_value=direct))
    shopper.answer.reset_mock()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/chat/stream", json={"question":"best rated product"})
    assert "Local answer" in response.text
    shopper.answer.assert_not_awaited()
    sessions.append_exchange.assert_awaited_once()
    assert not shopping._busy


@pytest.mark.asyncio
async def test_direct_preference_answer_saves_memory_without_agent(app, monkeypatch):
    signed_in(app)
    async def local_memory(user_id, session_id, question):
        return ShoppingResponse(session_id=session_id, answer="Saved to your shopping memory: size M.")
    monkeypatch.setattr(direct_answers, "answer_if_simple", local_memory)
    shopper.answer.reset_mock()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/chat/stream", json={"question":"I wear size M and prefer black minimal clothes."})
    assert "Saved to your shopping memory" in response.text
    shopper.answer.assert_not_awaited()
    sessions.append_exchange.assert_awaited_once()


@pytest.mark.asyncio
async def test_sse_error_no_success_and_no_partial_history(app, monkeypatch):
    signed_in(app)
    monkeypatch.setattr(shopper, "answer", AsyncMock(side_effect=TimeoutError))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/chat/stream", json={"question":"hi"})
    assert "event: error" in response.text and "event: done" not in response.text
    sessions.append_exchange.assert_not_awaited()
    assert not shopping._busy


@pytest.mark.asyncio
async def test_sse_provisional_text_reconciles_with_validated_result(app, monkeypatch):
    signed_in(app)
    async def reply(*args, on_token=None, **kwargs):
        on_token("Provisional")
        return ShoppingResponse(session_id="000000000000000000000011", answer="Validated")
    monkeypatch.setattr(shopper, "answer", reply)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/chat/stream", json={"question":"hi"})
    assert response.text.index("event: token") < response.text.index("event: products")
    assert '"answer": "Validated"' in response.text
    sessions.append_exchange.assert_awaited_once()


@pytest.mark.asyncio
async def test_sse_status_events_are_forwarded(app, monkeypatch):
    signed_in(app)
    async def reply(*args, on_status=None, **kwargs):
        on_status("retrieving")
        on_status("generating")
        return ShoppingResponse(session_id="000000000000000000000011", answer="Validated")
    monkeypatch.setattr(shopper, "answer", reply)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/chat/stream", json={"question":"hi"})
    assert '"state": "retrieving"' in response.text
    assert '"state": "generating"' in response.text
