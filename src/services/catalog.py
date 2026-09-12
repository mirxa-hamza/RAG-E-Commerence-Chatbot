"""Structured filters over the shared catalog; no generated MongoDB expressions."""
import re
from src.agent.schemas import CatalogQuery, Product
from src.core import config
from src.services import database


def collection():
    return database.get_client()[config.MONGO_DB][config.MONGO_PRODUCTS_COLLECTION]


def build_filter(q: CatalogQuery) -> dict:
    clauses = []
    if q.brand:
        clauses.append({"brand": {"$regex": "^" + re.escape(q.brand) + "$", "$options": "i"}})
    if q.category:
        regex = {"$regex": re.escape(q.category), "$options": "i"}
        clauses.append({"$or": [{"categories": regex}, {"main_category": regex}, {"title": regex}]})
    for term in q.query.split():
        regex = {"$regex": re.escape(term), "$options": "i"}
        clauses.append({"$or": [{"title": regex}, {"features": regex}, {"description": regex}]})
    if q.colors:
        # Amazon does not always provide a normalized colour field. Require literal
        # evidence in the retained metadata and disclose this in applied_filters.
        regex = {"$regex": r"\b(?:" + "|".join(re.escape(c) for c in q.colors) + r")\b", "$options": "i"}
        clauses.append({"$or": [{"title": regex}, {"features": regex}, {"description": regex}]})
    if q.min_price is not None or q.max_price is not None:
        price = {"$type": "number"}
        if q.min_price is not None: price["$gte"] = q.min_price
        if q.max_price is not None: price["$lte"] = q.max_price
        clauses.append({"price": price})
    if q.min_rating is not None:
        clauses.append({"average_rating": {"$gte": q.min_rating}})
    return {"$and": clauses} if clauses else {}


def public_product(doc: dict) -> dict:
    data = {k: doc[k] for k in Product.model_fields if k in doc}
    url = data.get("image_url")
    if url and not url.startswith("https://"):
        data["image_url"] = None
    return Product.model_validate(data).model_dump()


async def search(query: CatalogQuery) -> dict:
    rows = await database._guard(collection().find(build_filter(query)).sort(
        [("rating_number", -1), ("_id", 1)]).limit(query.limit).max_time_ms(5000).to_list(query.limit))
    return {"count": len(rows), "results": [public_product(r) for r in rows],
            "applied_filters": query.model_dump(exclude_none=True),
            "note": "Historical dataset; prices and availability are not live. Colors are matched in text metadata."}


async def by_ids(ids: list[str]) -> list[dict]:
    rows = await database._guard(collection().find({"_id": {"$in": ids[:12]}}).to_list(12))
    return [public_product(r) for r in rows]


async def ensure_indexes():
    for fields in [[("brand", 1), ("price", 1)], [("categories", 1)], [("rating_number", -1)], [("average_rating", -1)]]:
        await database._guard(collection().create_index(fields))
