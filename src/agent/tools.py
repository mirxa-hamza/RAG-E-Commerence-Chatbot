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


def create_tools(user_id: str, evidence: Evidence):
    @tool(args_schema=CatalogQuery)
    async def catalog_search(**kwargs) -> dict:
        """Find actual catalog products using literal query words and exact budget/brand filters."""
        result = await catalog.search(CatalogQuery(**kwargs))
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
        """Read this authenticated user's saved shopping preferences."""
        return await preferences.lookup(user_id)

    @tool(args_schema=PreferenceUpdate)
    async def remember_preference(**kwargs) -> dict:
        """Save or clear a size, budget, color, brand or style preference for this user."""
        return await preferences.update(user_id, PreferenceUpdate(**kwargs))

    return [catalog_search, semantic_review_search, memory_lookup, remember_preference]
