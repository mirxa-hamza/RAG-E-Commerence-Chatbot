"""
Gemini answer generation (LLM_PROVIDER=gemini).

Google's Gemini API at `generativelanguage.googleapis.com`, reached with a plain API key
from https://aistudio.google.com/apikey. That is deliberately the simplest of Google's
several front doors: no service account, no OAuth token minting, no project/region
addressing - which is what makes it a drop-in alternative to Groq rather than a second
deployment story.

WHAT THIS DOES NOT CHANGE. Gemini is held to exactly the same contract as Groq: it
receives SYSTEM_PROMPT and the CONTEXT block, and is expected to answer only from the
retrieved passages. The facts stay in the documents, which is what keeps the citations in
the UI meaningful.

The request shape differs from OpenAI's, so `_payload()` translates:
`[{"role": "system"|"user"|"assistant"}]` becomes a `systemInstruction` plus `contents`
whose roles are `user` and `model`.
"""
import json
import time
from typing import Dict, Iterator, List

from src.core.config import (
    GEMINI_API_KEY,
    GEMINI_API_VERSION,
    GEMINI_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
)
from src.core.logging import get_logger, timed

log = get_logger(__name__)

BASE = "https://generativelanguage.googleapis.com"


class GeminiError(RuntimeError):
    """A request that reached Google and came back wrong."""


def is_configured() -> bool:
    """Enough settings to try. Whether the key actually works is a runtime question."""
    return bool(GEMINI_API_KEY and GEMINI_MODEL)


def _url(method: str) -> str:
    return f"{BASE}/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:{method}"


def _headers() -> Dict[str, str]:
    return {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}


def _payload(messages: List[Dict], temperature: float, max_tokens: int) -> Dict:
    """
    OpenAI-style messages -> Gemini's request body.

    Gemini has no "system" role inside `contents`; the system prompt is a separate
    `systemInstruction` field, and the assistant role is spelled "model". Several system
    messages are joined rather than dropped - losing one would quietly remove the grounding
    rules this whole app depends on.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
        if message.get("role") in ("user", "assistant")
    ]

    body: Dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    return body


def _retry_delay_seconds(body: Dict) -> float:
    """
    How long Google asked us to wait, from its RetryInfo detail. Falls back to 0.

    A free-tier 429 is a schedule, not a failure, and Google usually says exactly how long
    the schedule is - honouring that beats guessing.
    """
    for detail in (body.get("error") or {}).get("details") or []:
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                pass
    return 0.0


def _post(method: str, payload: Dict, *, stream: bool = False):
    """
    One POST with a small retry budget for 429/5xx. Returns the httpx.Response.

    httpx is imported here rather than at module scope so that a local-only install that
    never touches Gemini does not pay for the import at startup.
    """
    import httpx

    if not GEMINI_API_KEY:
        raise GeminiError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and put it in .env."
        )

    url = _url(method)
    params = {"alt": "sse"} if stream else None
    last_error = None

    for attempt in range(1, max(1, LLM_MAX_RETRIES) + 1):
        try:
            client = httpx.Client(timeout=LLM_TIMEOUT_SECONDS)
            request = client.build_request("POST", url, json=payload,
                                           headers=_headers(), params=params)
            response = client.send(request, stream=stream)
        except httpx.HTTPError as exc:
            last_error = exc
            log.warning("Gemini %s attempt %d/%d failed to connect (%s).",
                        method, attempt, LLM_MAX_RETRIES, exc)
            time.sleep(min(8.0, 2.0 ** attempt))
            continue

        if response.status_code < 400:
            return response

        # Read the error body before deciding, then close - a streamed response holds the
        # connection open until it is.
        detail = response.read().decode("utf-8", "replace")
        response.close()
        client.close()
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < LLM_MAX_RETRIES:
            try:
                wait = _retry_delay_seconds(json.loads(detail))
            except ValueError:
                wait = 0.0
            wait = wait or min(8.0, 2.0 ** attempt)
            log.warning("Gemini %s returned %d; retrying in %.1fs (%d/%d).",
                        method, response.status_code, wait, attempt, LLM_MAX_RETRIES)
            time.sleep(wait)
            continue
        raise GeminiError(f"Gemini {method} returned HTTP {response.status_code}: "
                          f"{detail[:400]}")

    raise GeminiError(f"Gemini {method} could not be reached: {last_error}")


def _text_of(candidate: Dict) -> str:
    parts = (candidate.get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


def _truncation_note(reason: str) -> str:
    """A finishReason that is not STOP explains itself, rather than leaving a short answer."""
    if reason == "MAX_TOKENS":
        return ("\n\n[The answer was cut off at the LLM_MAX_TOKENS limit. Raise "
                "LLM_MAX_TOKENS in .env to let it finish.]")
    if reason in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
        return f"\n\n[Gemini stopped early: finishReason={reason}.]"
    return ""


def generate(messages: List[Dict], temperature: float, max_tokens: int) -> str:
    """One buffered completion. Raises GeminiError on anything that is not a usable answer."""
    response = _post("generateContent", _payload(messages, temperature, max_tokens))
    try:
        body = response.json()
    finally:
        response.close()

    candidates = body.get("candidates") or []
    if not candidates:
        blocked = (body.get("promptFeedback") or {}).get("blockReason")
        raise GeminiError(
            f"Gemini returned no candidates (blockReason={blocked})." if blocked
            else "Gemini returned no candidates."
        )

    with timed(log, f"Gemini call ({GEMINI_MODEL})"):
        text = _text_of(candidates[0])
    return text + _truncation_note(candidates[0].get("finishReason", ""))


def stream(messages: List[Dict], temperature: float, max_tokens: int) -> Iterator[str]:
    """
    Yields the answer in pieces, using Gemini's SSE endpoint.

    Each `data:` line is one JSON object of the same shape generate() reads, so the text is
    extracted the same way. A finishReason other than STOP is appended as a note at the end
    for exactly the reason it is in generate(): a silently truncated answer is worse than a
    long one.
    """
    response = _post("streamGenerateContent",
                     _payload(messages, temperature, max_tokens), stream=True)
    finish_reason = ""
    try:
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            raw = line[len("data:"):].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                log.debug("Skipping an unparseable Gemini stream line.")
                continue
            for candidate in event.get("candidates") or []:
                finish_reason = candidate.get("finishReason") or finish_reason
                piece = _text_of(candidate)
                if piece:
                    yield piece
    finally:
        response.close()

    note = _truncation_note(finish_reason)
    if note:
        yield note
