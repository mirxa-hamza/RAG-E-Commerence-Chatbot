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


@pytest.mark.asyncio
async def test_follow_up_about_shown_products_goes_to_the_agent(monkeypatch):
    """"Which of these is best rated?" after a search must NOT hit the catalog-wide
    shortcut, which would answer with the globally top-rated item instead of the
    products actually under discussion."""
    monkeypatch.setattr(preferences, "update", AsyncMock())
    history = [
        {"role": "user", "content": "Show me shoes under $40"},
        {"role": "assistant", "content": "Here are some shoes.",
         "products": [{"parent_asin": "P1"}]},
    ]
    assert await direct_answers.answer_if_simple(
        "u1", "s1", "Which of these is best rated?", history) is None


@pytest.mark.asyncio
async def test_referential_wording_alone_blocks_the_shortcut(monkeypatch):
    monkeypatch.setattr(preferences, "update", AsyncMock())
    for question in ("Which of these is the best rated?",
                     "which one of those is top rated",
                     "is the second one better rated?",
                     "what is the highest rated of the ones you just showed"):
        assert await direct_answers.answer_if_simple("u1", "s1", question) is None, question


@pytest.mark.asyncio
async def test_standalone_top_rated_still_uses_the_shortcut(monkeypatch):
    """The optimisation must survive for the case it was built for."""
    monkeypatch.setattr(preferences, "update", AsyncMock())
    monkeypatch.setattr(preferences, "lookup", AsyncMock(return_value={}))

    class Cursor:
        def sort(self, *a): return self
        def limit(self, *a): return self
        def max_time_ms(self, *a): return self
        async def to_list(self, *a):
            return [{"parent_asin": "P9", "title": "A well rated thing",
                     "average_rating": 4.9, "rating_number": 120}]

    monkeypatch.setattr(direct_answers, "_collection", lambda: type("C", (), {"find": lambda self, *a: Cursor()})())
    monkeypatch.setattr(direct_answers.database, "_guard", lambda awaitable: awaitable)

    result = await direct_answers.answer_if_simple("u1", "s1", "What is the best rated product?", [])
    assert result is not None
    assert "A well rated thing" in result.answer
