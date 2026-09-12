"""Authenticated shopping API. One active agent per account bounds writes and cost."""
import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from src.api.auth import _enforce
from src.api.deps import get_current_user, user_id_of
from src.agent.schemas import PreferenceUpdate, ShoppingRequest, ShoppingResponse
from src.core import ratelimit
from src.core.logging import get_logger
from src.services import preferences, sessions

router = APIRouter(tags=["shopping"])
log = get_logger(__name__)
_busy: set[str] = set()


async def prepare(body, user):
    uid = user_id_of(user)
    _enforce(ratelimit.CHAT, uid)
    if uid in _busy:
        raise HTTPException(409, "I'm still finishing your last answer - give it a moment, then try again.")
    _busy.add(uid)
    try:
        session = await sessions.get(uid, body.session_id) if body.session_id else await sessions.create(uid)
        if not session: raise HTTPException(404, "That conversation is no longer available.")
        return uid, session
    except BaseException:
        _busy.discard(uid)
        raise


async def execute(uid, session, question, on_token=None, on_status=None):
    from src.services.direct_answers import answer_if_simple

    # The session's own messages decide whether a shortcut is safe: a follow-up in a
    # conversation that already showed products must go to the agent, not to a
    # whole-catalog lookup that knows nothing about what came before.
    direct = await answer_if_simple(uid, session["id"], question, session.get("messages", []))
    if direct is not None:
        if on_token:
            on_token(direct.answer)
        if not await sessions.append_exchange(uid, session["id"], question, direct.model_dump()):
            raise HTTPException(404, "That conversation was deleted while I was answering.")
        return direct

    from src.agent.shopper import answer
    result = await answer(uid, session["id"], question, session.get("messages", []), on_token=on_token, on_status=on_status)
    if not await sessions.append_exchange(uid, session["id"], question, result.model_dump()):
        raise HTTPException(404, "That conversation was deleted while I was answering.")
    return result


@router.post("/chat", response_model=ShoppingResponse)
async def chat(body: ShoppingRequest, user: dict = Depends(get_current_user)):
    uid, session = await prepare(body, user)
    try:
        return await execute(uid, session, body.question)
    except TimeoutError:
        raise HTTPException(504, "That took longer than expected. Try asking something a little more specific.")
    except HTTPException: raise
    except Exception:
        log.exception("Shopping request failed")
        raise HTTPException(503, "The assistant is having trouble at the moment. Please try again shortly.")
    finally:
        _busy.discard(uid)


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def public_error(exc: Exception) -> str:
    """Turn any failure into something a shopper can actually act on.

    Everything here is read by someone trying to buy clothes, not by whoever runs the
    server: status codes, provider names, exception text and stack fragments are for
    the logs (which already receive the full traceback via log.exception). Anything not
    recognised below becomes a plain apology rather than leaking the raw message.
    """
    lowered = str(exc).lower()
    if any(term in lowered for term in ("connection error", "getaddrinfo", "apiconnectionerror",
                                        "network", "dns", "unreachable")):
        return "I can't reach the assistant right now. Check your internet connection and try again."
    if any(term in lowered for term in ("rate limit", "too many requests", "quota", "429")):
        return "The assistant is handling a lot of requests right now. Please wait a moment and try again."
    if "timeout" in lowered or "timed out" in lowered:
        return "That took longer than expected. Try asking something a little more specific."
    return "The assistant is having trouble at the moment. Please try again shortly."


@router.post("/chat/stream")
async def stream(body: ShoppingRequest, request: Request, user: dict = Depends(get_current_user)):
    uid, session = await prepare(body, user)

    async def events():
        queue = asyncio.Queue()
        task = asyncio.create_task(execute(
            uid, session, body.question,
            on_token=lambda text: queue.put_nowait(("token", {"text": text})),
            on_status=lambda state: queue.put_nowait(("status", {"state": state})),
        ))
        emitted = False
        try:
            yield sse("session", {"session_id": session["id"]})
            while not task.done() or not queue.empty():
                if await request.is_disconnected():
                    return
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=1)
                    if event == "token": emitted = True
                    yield sse(event, payload)
                except TimeoutError:
                    if not task.done(): yield ": searching\n\n"
            result = task.result().model_dump()
            yield sse("products", {"products": result["products"]})
            yield sse("citations", {"citations": result["citations"]})
            if not emitted: yield sse("token", {"text": result["answer"]})
            yield sse("done", {"session_id": session["id"], "answer": result["answer"], "suggested_relaxations": result["suggested_relaxations"]})
        except TimeoutError:
            yield sse("error", {"message": "That took longer than expected. Try asking something a little more specific."})
        except Exception as exc:
            log.exception("Shopping stream failed")
            yield sse("error", {"message": public_error(exc)})
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            _busy.discard(uid)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/api/preferences")
async def get_preferences(user: dict = Depends(get_current_user)):
    return await preferences.lookup(user_id_of(user))


@router.patch("/api/preferences")
async def change_preference(change: PreferenceUpdate, user: dict = Depends(get_current_user)):
    uid = user_id_of(user)
    if uid in _busy: raise HTTPException(409, "I'm still finishing your last answer - give it a moment, then try again.")
    try:
        return await preferences.update(uid, change)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
