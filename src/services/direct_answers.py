"""Cheap local answers for simple requests that do not need an LLM call."""
import re
from src.agent.schemas import PreferenceUpdate, Product, ShoppingResponse
from src.core import config
from src.services import catalog, database, preferences


_BEST_RATING = re.compile(r"\b(best|top|highest|greatest)\b.*\b(rating|rated|reviewed)\b|\b(rating|rated)\b.*\b(best|top|highest)\b", re.I)
_SIZE = re.compile(r"\b(?:wear|size is|size)\s+(?:a\s+|an\s+)?(?:size\s+)?([a-z0-9][a-z0-9 ._-]{0,12})\b", re.I)
_STYLE = re.compile(r"\b(?:prefer|like|love)\s+([a-z][a-z ,'-]{2,80})\s+(?:clothes|clothing|style|styles|outfits|fashion)\b", re.I)
_COLORS = {"black", "white", "blue", "navy", "red", "green", "yellow", "pink", "purple", "brown", "gray", "grey", "orange", "beige", "cream"}
_PREFERENCE_INTENT = re.compile(
    r"\b(?:remember|save|my favorite|favourite|i wear|my size|i prefer|i like|i love|"
    r"change my|update my|set my)\b", re.I
)
_MEMORY_QUERY = re.compile(
    r"\b(?:what|which|show|tell me)\b.*\b(?:remember|saved|told|preferences?)\b|"
    r"\b(?:remembered|saved)\s+(?:preferences?|details?)\b", re.I
)
_CONSTRAINED_PRODUCT = re.compile(
    r"\b(?:under|below|less than|between|budget|price|color|black|white|blue|navy|"
    r"dress(?:es)?|shoe(?:s)?|shirt(?:s)?|jacket(?:s)?|pants|skirt(?:s)?|size|brand)\b", re.I
)


def _collection():
    return database.get_client()[config.MONGO_DB][config.MONGO_PRODUCTS_COLLECTION]


def _product(doc: dict) -> dict:
    return Product.model_validate(catalog.public_product(doc)).model_dump()


def _clean_size(value: str) -> str | None:
    size = re.split(r"\b(?:and|but|with|for|clothes|clothing|style|styles|outfits)\b", value, maxsplit=1, flags=re.I)[0].strip(" .,-")
    return size[:20].upper() if size else None


async def _read_simple_preferences(user_id: str, session_id: str, question: str) -> ShoppingResponse | None:
    if not _MEMORY_QUERY.search(question):
        return None
    saved = await preferences.lookup(user_id)
    parts = []
    if saved.get("color_preference"):
        colors = saved["color_preference"]
        if isinstance(colors, str):
            colors = [colors]
        parts.append("favorite color" + ("s" if len(colors) != 1 else "") + " " + ", ".join(colors))
    if saved.get("clothing_size"):
        parts.append(f"size {saved['clothing_size']}")
    if saved.get("style_notes"):
        parts.append(f"style {saved['style_notes']}")
    if saved.get("favorite_brands"):
        brands = saved["favorite_brands"]
        if isinstance(brands, str):
            brands = [brands]
        parts.append("favorite brand" + ("s " if len(brands) != 1 else " ") + ", ".join(brands))
    memories = saved.get("memories") or []
    if isinstance(memories, str):
        memories = [memories]
    if not parts and not memories:
        return ShoppingResponse(session_id=session_id, answer="I don't have any saved shopping preferences yet.")
    answer = "Your saved shopping preferences are " + "; ".join(parts) + "." if parts else ""
    if memories:
        # Newest last, in the order they were remembered.
        answer += (" " if answer else "") + "I also remember: " + " ".join(
            note if note.endswith(".") else note + "." for note in memories[-10:])
    return ShoppingResponse(session_id=session_id, answer=answer.strip())


async def _remember_simple_preferences(user_id: str, session_id: str, question: str) -> ShoppingResponse | None:
    q = question.strip()
    lowered = q.lower()
    # A product query can contain colors, sizes, or style words without asking to
    # save them. Only explicit preference language may trigger a memory write.
    if not _PREFERENCE_INTENT.search(q):
        return None
    changes: list[str] = []

    size_match = _SIZE.search(q)
    size = _clean_size(size_match.group(1)) if size_match else None
    if size:
        await preferences.update(user_id, PreferenceUpdate(field="clothing_size", action="set", value=size))
        changes.append(f"size {size}")

    colors = [color for color in sorted(_COLORS) if re.search(rf"\b{re.escape(color)}\b", lowered)]
    for color in colors[:4]:
        await preferences.update(user_id, PreferenceUpdate(field="color_preference", action="add", value=color))
    if colors:
        changes.append("favorite color" + ("s " if len(colors) > 1 else " ") + ", ".join(colors[:4]))

    style_match = _STYLE.search(q)
    style = style_match.group(1).strip(" .,-") if style_match else ""
    style_words = [word for word in re.split(r"[\s,]+", style) if word and word.lower() not in _COLORS and word.lower() not in {"and", "or", "with"}]
    if style_words:
        style_text = " ".join(style_words[:8])
        await preferences.update(user_id, PreferenceUpdate(field="style_notes", action="set", value=style_text))
        changes.append(f"style notes: {style_text}")

    if not changes:
        return None
    return ShoppingResponse(session_id=session_id, answer="Saved to your shopping memory: " + "; ".join(changes) + ".")


async def answer_if_simple(user_id: str, session_id: str, question: str) -> ShoppingResponse | None:
    if not config.DIRECT_ANSWERS_ENABLED:
        return None
    q = question.strip()
    remembered_query = await _read_simple_preferences(user_id, session_id, q)
    if remembered_query is not None:
        return remembered_query
    remembered = await _remember_simple_preferences(user_id, session_id, q)
    if remembered is not None:
        return remembered

    if not _BEST_RATING.search(q) or _CONSTRAINED_PRODUCT.search(q):
        # Constrained requests must use the agent's filtered catalog/review tools.
        # The global top-rated shortcut would otherwise return unrelated products and
        # claim they satisfy a color, category, price, or preference constraint.
        return None

    limit = 5
    rows = await database._guard(_collection().find({
        "average_rating": {"$type": "number"},
        "rating_number": {"$type": "number", "$gte": 5},
    }).sort([("average_rating", -1), ("rating_number", -1), ("_id", 1)]).limit(limit).max_time_ms(3000).to_list(limit))
    products = [_product(row) for row in rows]
    if not products:
        return ShoppingResponse(session_id=session_id, answer="I could not find rated products in the local catalog yet.")

    top = products[0]
    answer = (
        f"The highest-rated product I found in the local Amazon Fashion catalog is "
        f"{top['title']} with {top.get('average_rating', 0):.1f}/5 from "
        f"{top.get('rating_number', 0):,} ratings. I included a few other highly rated options too."
    )
    return ShoppingResponse(session_id=session_id, answer=answer, products=products)
