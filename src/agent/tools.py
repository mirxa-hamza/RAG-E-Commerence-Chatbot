"""Build request-local tools: identity and evidence never enter global agent state."""
import asyncio
from dataclasses import dataclass, field
from langchain.tools import tool
from src.agent.schemas import CatalogQuery, PreferenceUpdate, ReviewQuery
from src.services import catalog, preferences, review_search


@dataclass
class Evidence:
    products: dict = field(default_factory=dict)
    citations: dict = field(default_factory=dict)
    searches: list = field(default_factory=list)


# What the shopper is looking for, as opposed to how it is constrained. A search that
# names one of these is a new search; a search that names none is a follow-up.
_SUBJECT_FIELDS = ("query", "category")
_CARRIED_FIELDS = ("query", "category", "brand", "colors", "min_price", "max_price", "min_rating")


def _blank(value) -> bool:
    return value is None or value == "" or value == []


def merge_filters(new: dict, previous: dict) -> dict:
    """Fill a follow-up search's blanks from the previous search in this conversation.

    "under $40" on its own is a constraint with no subject: searched literally it returns
    anything cheap, which is not what a shopper who just asked about black shoes means.
    The prompt tells the model to restate the subject, but an instruction is not a
    guarantee, so a call that names NO subject inherits the previous one here.

    A call that does name a subject inherits nothing, which is what lets "actually, show
    me dresses" escape the earlier constraints. The known limit is the mirror image: a
    constraint can only be widened by restating it ("under $80"), not dropped by omission,
    because an omitted field is exactly what a follow-up looks like.
    """
    if not previous or any(not _blank(new.get(f)) for f in _SUBJECT_FIELDS):
        return new
    merged = dict(new)
    for name in _CARRIED_FIELDS:
        if _blank(merged.get(name)) and not _blank(previous.get(name)):
            merged[name] = previous[name]
    return merged


def create_tools(user_id: str, evidence: Evidence, previous_filters: dict | None = None):
    @tool(args_schema=CatalogQuery)
    async def catalog_search(**kwargs) -> dict:
        """Find actual catalog products using literal query words and exact budget/brand filters."""
        result = await catalog.search(CatalogQuery(**merge_filters(kwargs, previous_filters or {})))
        evidence.searches.append(result)
        evidence.products.update({p["parent_asin"]: p for p in result["results"]})
        return result

    @tool(args_schema=ReviewQuery)
    async def semantic_review_search(**kwargs) -> dict:
        """Find real customer review evidence about fit, comfort, quality or sizing."""
        result = await asyncio.to_thread(review_search.search, ReviewQuery(**kwargs))
        evidence.citations.update({r["id"]: r for r in result["results"]})
        products = await catalog.by_ids(list(dict.fromkeys(r["parent_asin"] for r in result["results"])))
        evidence.products.update({p["parent_asin"]: p for p in products})
        return {**result, "products": products}

    @tool
    async def memory_lookup() -> dict:
        """Read this authenticated user's saved shopping preferences and remembered facts."""
        return await preferences.lookup(user_id)

    @tool(args_schema=PreferenceUpdate)
    async def remember_preference(**kwargs) -> dict:
        """Save, change or clear something to remember about this user: a size, budget,
        color, brand or style preference, or a free-form fact in 'memories'."""
        return await preferences.update(user_id, PreferenceUpdate(**kwargs))

    return [catalog_search, semantic_review_search, memory_lookup, remember_preference]
