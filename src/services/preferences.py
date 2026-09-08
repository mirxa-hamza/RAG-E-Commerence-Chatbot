"""Account-level durable shopping memory, shared across the owner's conversations."""
from bson import ObjectId
from src.agent.schemas import Budget, PreferenceUpdate
from src.services import database


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
        existing = await lookup(user_id)
        if len(existing.get(change.field, [])) >= 20 and value not in existing.get(change.field, []):
            raise ValueError("A preference list supports at most 20 values")
        operation = {"$addToSet": {path: value}}
    else:
        operation = {"$pull": {path: value}}
    result = await database._guard(database.users().update_one({"_id": ObjectId(user_id)}, operation))
    if not result.matched_count:
        raise ValueError("Account no longer exists")
    await database.record_audit(user_id, None, "preference_" + change.action, detail=change.field)
    return await lookup(user_id)
