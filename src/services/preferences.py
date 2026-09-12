"""Account-level durable shopping memory, shared across the owner's conversations."""
from bson import ObjectId
from src.agent.schemas import Budget, PreferenceUpdate
from src.core import config
from src.services import database

# Typed lists stay short because every value there is a filter the catalog can apply;
# free-form memories are the recall surface and get their own, larger budget. Both are
# bounded: an unbounded list is an unbounded document AND an unbounded prompt on every
# question. When the memory budget is full the OLDEST note is dropped, so a long-running
# account keeps remembering recent things instead of freezing at its first twenty.
_TYPED_LIST_LIMIT = 20


def _limit_for(field: str) -> int:
    return config.MEMORY_MAX_ENTRIES if field == "memories" else _TYPED_LIST_LIMIT


async def lookup(user_id: str) -> dict:
    row = await database._guard(database.users().find_one({"_id": ObjectId(user_id)}, {"preferences": 1}))
    if not row:
        raise ValueError("Account no longer exists")
    return row.get("preferences", {})


async def update(user_id: str, change: PreferenceUpdate) -> dict:
    path = "preferences." + change.field
    value = change.value.model_dump() if isinstance(change.value, Budget) else change.value
    if change.action == "clear":
        operation = {"$unset": {path: ""}}
    elif change.action == "set":
        operation = {"$set": {path: value}}
    elif change.action == "add":
        # Bounded lists avoid unbounded documents and ever-growing prompt memory.
        existing = (await lookup(user_id)).get(change.field, [])
        limit = _limit_for(change.field)
        if value in existing:
            return await lookup(user_id)  # already remembered; not an error worth raising
        if change.field == "memories":
            # Keep the newest `limit` notes. Refusing to learn anything new once full is
            # worse memory than forgetting the oldest thing.
            operation = {"$push": {path: {"$each": [value], "$slice": -limit}}}
        else:
            if len(existing) >= limit:
                raise ValueError(f"A preference list supports at most {limit} values")
            operation = {"$addToSet": {path: value}}
    else:
        operation = {"$pull": {path: value}}
    result = await database._guard(database.users().update_one({"_id": ObjectId(user_id)}, operation))
    if not result.matched_count:
        raise ValueError("Account no longer exists")
    await database.record_audit(user_id, None, "preference_" + change.action, detail=change.field)
    return await lookup(user_id)
