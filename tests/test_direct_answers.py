from unittest.mock import AsyncMock
import pytest
from src.agent.schemas import PreferenceUpdate
from src.services import direct_answers, preferences


@pytest.mark.asyncio
async def test_preference_sentence_is_saved_without_llm(monkeypatch):
    saved = []

    async def update(user_id, change):
        assert user_id == "u1"
        assert isinstance(change, PreferenceUpdate)
        saved.append((change.field, change.action, change.value))
        return {}

    monkeypatch.setattr(preferences, "update", update)
    result = await direct_answers.answer_if_simple("u1", "s1", "I wear size M and prefer black minimal clothes.")
    assert result is not None
    assert "Saved to your shopping memory" in result.answer
    assert ("clothing_size", "set", "M") in saved
    assert ("color_preference", "add", "black") in saved
    assert any(field == "style_notes" and "minimal" in value for field, action, value in saved)


@pytest.mark.asyncio
async def test_non_simple_question_falls_through(monkeypatch):
    monkeypatch.setattr(preferences, "update", AsyncMock())
    assert await direct_answers.answer_if_simple("u1", "s1", "Find dresses under 40 dollars") is None
    preferences.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_constrained_top_rated_request_falls_through(monkeypatch):
    monkeypatch.setattr(preferences, "update", AsyncMock())
    assert await direct_answers.answer_if_simple("u1", "s1", "Show highly rated dresses under $60") is None
    preferences.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_product_color_query_does_not_write_memory(monkeypatch):
    monkeypatch.setattr(preferences, "update", AsyncMock())
    assert await direct_answers.answer_if_simple("u1", "s1", "Find black dresses under $40") is None
    preferences.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_question_reads_preferences_instead_of_saving(monkeypatch):
    monkeypatch.setattr(preferences, "lookup", AsyncMock(return_value={
        "color_preference": ["black"], "clothing_size": "M"
    }))
    monkeypatch.setattr(preferences, "update", AsyncMock())
    result = await direct_answers.answer_if_simple("u1", "s1", "What color and size did I tell you to remember?")
    assert result is not None
    assert "favorite color black" in result.answer
    assert "size M" in result.answer
    preferences.update.assert_not_awaited()
