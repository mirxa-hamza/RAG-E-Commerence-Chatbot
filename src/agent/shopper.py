"""Bounded LangChain execution, with cards hydrated only from current-turn evidence."""
import asyncio
import json
import re
from langchain.agents import create_agent
from langchain.agents.middleware import (ClearToolUsesEdit, ContextEditingMiddleware,
                                         ModelCallLimitMiddleware, ModelRetryMiddleware,
                                         SummarizationMiddleware)
from langchain.agents.structured_output import ToolStrategy
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.outputs import ChatGeneration
from langsmith import tracing_context
from src.agent.models import get_chat_model
from src.agent.prompts import system_prompt
from src.agent.schemas import AnswerDraft, ShoppingResponse
from src.agent.tools import Evidence, create_tools
from src.core import config
from src.services import preferences

_CHECKPOINTER = InMemorySaver()
_STORE = InMemoryStore()


def hydrate(draft: AnswerDraft, evidence: Evidence, session_id: str) -> ShoppingResponse:
    if any(p not in evidence.products for p in draft.product_ids) or any(c not in evidence.citations for c in draft.citation_ids):
        raise ValueError("The answer referenced evidence that was not retrieved")
    products = [dict(evidence.products[p]) for p in dict.fromkeys(draft.product_ids)]
    citations = [evidence.citations[c] for c in dict.fromkeys(draft.citation_ids)]
    for product in products:
        cited = next((c for c in citations if c["parent_asin"] == product["parent_asin"]), None)
        if cited: product["review_excerpt"] = cited["excerpt"]
    # What this turn actually searched for, so the next turn can keep building on it.
    active = next((s["applied_filters"] for s in reversed(evidence.searches)
                   if s.get("applied_filters")), {})
    if evidence.searches and all(s["count"] == 0 for s in evidence.searches) and not products:
        filters = evidence.searches[-1]["applied_filters"]
        constraints = _human_constraints(filters)
        # Make the all-empty catalog case deterministic; the model cannot fill it with
        # a plausible fictional price, even when its prose ignores the system prompt.
        return ShoppingResponse(session_id=session_id,
            answer=f"No catalog products matched {constraints or 'this search'}. Would you like to broaden the search or relax one constraint?",
            suggested_relaxations=["Try a broader category", "Adjust the price limit"],
            active_filters=active)
    answer = _ground_product_claims(draft.answer, products)
    return ShoppingResponse(session_id=session_id, answer=answer, products=products,
                            citations=citations, suggested_relaxations=draft.suggested_relaxations,
                            active_filters=active)


def _ground_product_claims(answer: str, products: list[dict]) -> str:
    """Prevent prose prices from disagreeing with the product cards we display.

    Models occasionally select a valid product ID but copy a title/price from another
    candidate. The cards are hydrated from the database, so replace an answer containing
    unsupported currency amounts with a deterministic summary of those canonical products.
    """
    if not products:
        return answer
    mentioned = set(re.findall(r"\$\s*\d+(?:\.\d{1,2})?", answer))
    valid = {f"${float(p['price']):.2f}" for p in products if p.get("price") is not None}
    if mentioned and not mentioned.intersection(valid):
        items = "; ".join(
            f"{p.get('title', 'Product')} — ${float(p['price']):.2f}"
            if p.get("price") is not None else str(p.get("title", "Product"))
            for p in products
        )
        return "I found these products in the catalog: " + items + "."
    return answer


def _human_constraints(filters: dict) -> str:
    """Turn internal filter names/arrays into a customer-friendly sentence."""
    parts = []
    query = filters.get("query") or filters.get("category")
    if query:
        parts.append(f"category {query}")
    colors = filters.get("colors") or filters.get("color")
    if colors:
        if isinstance(colors, (list, tuple, set)):
            colors = ", ".join(str(v) for v in colors)
        parts.append(f"color {colors}")
    brand = filters.get("brand")
    if brand:
        parts.append(f"brand {brand}")
    minimum = filters.get("min_price")
    maximum = filters.get("max_price")
    if minimum is not None and maximum is not None:
        parts.append(f"between ${minimum:g} and ${maximum:g}")
    elif maximum is not None:
        parts.append(f"under ${maximum:g}")
    elif minimum is not None:
        parts.append(f"at least ${minimum:g}")
    return ", ".join(parts)


def history_messages(messages: list[dict]) -> list[dict]:
    selected, budget = [], config.MAX_HISTORY_CHARS
    for m in reversed(messages[-2*config.HISTORY_TURNS:]):
        if m.get("role") not in ("user", "assistant"): continue
        content = m.get("content", "")
        if m.get("products"):
            content += "\nPreviously shown product IDs: " + json.dumps([p["parent_asin"] for p in m["products"]])
        if m.get("active_filters"):
            # The prose alone does not say what was actually searched, so a follow-up
            # like "under $40" has nothing to attach itself to without this line.
            content += "\nSearch filters used: " + json.dumps(m["active_filters"], default=str)
        if len(content) > budget: break
        selected.append({"role": m["role"], "content": content})
        budget -= len(content)
    return list(reversed(selected))


def previous_filters(history: list) -> dict:
    """The filters behind the most recent catalog search in this conversation."""
    for message in reversed(history or []):
        if message.get("role") == "assistant" and message.get("active_filters"):
            return message["active_filters"]
    return {}


class DraftStream:
    """Decode only LangChain's structured answer tool, never reasoning or other tools."""
    def __init__(self, evidence, session_id, emit):
        self.evidence, self.session_id, self.emit = evidence, session_id, emit
        self.messages = {}
        self.sent = ""
        self.parser = PydanticToolsParser(tools=[AnswerDraft], first_tool_only=True)

    def accept(self, chunk, metadata):
        if metadata.get("langgraph_node") != "model" or not isinstance(chunk, AIMessageChunk): return
        key = chunk.id
        accumulated = self.messages.get(key)
        accumulated = chunk if accumulated is None else accumulated + chunk
        self.messages[key] = accumulated
        calls = accumulated.tool_calls
        if len(calls) != 1 or calls[0]["name"] != "AnswerDraft": return
        # Wait until identity arrays have arrived; the schema requests these before prose.
        if not {"product_ids", "citation_ids"}.issubset(calls[0]["args"]): return
        draft = self.parser.parse_result([ChatGeneration(message=accumulated)], partial=True)
        if draft is None: return
        try:
            text = hydrate(draft, self.evidence, self.session_id).answer
        except ValueError:
            return  # invented IDs never authorize a provisional answer
        if text.startswith(self.sent):
            delta = text[len(self.sent):]
            if delta: self.emit(delta)
            self.sent = text


def _summary_trigger(model):
    if getattr(model, "profile", None) and model.profile.get("max_input_tokens"):
        return ("fraction", config.AGENT_SUMMARY_TRIGGER_FRACTION)
    return ("tokens", config.AGENT_SUMMARY_FALLBACK_TOKENS)


def _middleware(model):
    return [
        ModelRetryMiddleware(max_retries=config.AGENT_MODEL_RETRIES, on_failure="error",
                             initial_delay=config.AGENT_MODEL_RETRY_INITIAL_DELAY_SECONDS,
                             max_delay=config.AGENT_MODEL_RETRY_MAX_DELAY_SECONDS),
        SummarizationMiddleware(model=model, trigger=_summary_trigger(model),
                                keep=("messages", config.AGENT_SUMMARY_KEEP_MESSAGES)),
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(
            trigger=config.AGENT_TOOL_CLEAR_TRIGGER_TOKENS,
            clear_at_least=config.AGENT_TOOL_CLEAR_AT_LEAST_TOKENS,
            keep=config.AGENT_TOOL_CLEAR_KEEP,
            clear_tool_inputs=True,
            placeholder="[retrieved tool output cleared after summarization]",
        )]),
        ModelCallLimitMiddleware(run_limit=config.AGENT_MAX_TOOL_ROUNDTRIPS+1, exit_behavior="end"),
    ]




def _evidence_for_finalizer(evidence: Evidence) -> str:
    """Keep the second (schema-only) call small and grounded in this turn's evidence."""
    return json.dumps({
        "products": list(evidence.products.values()),
        "citations": list(evidence.citations.values()),
        "searches": evidence.searches,
    }, ensure_ascii=True, default=str)[:config.MAX_CONTEXT_CHARS]


def _conversation_context(history: list) -> str:
    """The recent turns, as text the answer writer can actually read.

    The tool loop is seeded with these messages, but the two_phase finalizer is a
    SEPARATE call with its own prompt - so without this the model that writes the prose
    has no idea what was already said, and every follow-up ("show me cheaper ones",
    "what about the second one") reads as a brand new conversation.
    """
    turns = history_messages(history)
    if not turns:
        return "(this is the first message in the conversation)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in turns)


async def _finalize_real_model(model, question: str, evidence: Evidence, saved: dict,
                               session_id: str, history: list, on_token=None, on_status=None) -> AnswerDraft:
    """Create the structured draft after the tool-enabled agent has finished.

    Groq does not allow response_format=json_* on a request that also contains
    function tools, which is why this is a second call with none bound but AnswerDraft
    itself. Without this streaming, a two_phase turn is a silent wait followed by one
    giant answer chunk - so when a token callback is given, force AnswerDraft as a tool
    call and stream its arguments through the same DraftStream decoder the merged/tool-loop
    path already uses, rather than block on a plain non-streaming structured call.
    """
    if on_status:
        on_status("generating")
    prompt = (
        "Return an AnswerDraft for the user's request. Use only the supplied evidence; "
        "never invent product or citation IDs. Select product_ids and citation_ids from "
        "the evidence, or leave them empty. Keep the answer concise. The answer field must "
        "be polished, human-friendly shopping prose: do not output JSON, Python lists, "
        "internal field names, raw filters, or tool arguments. Use natural phrases such as "
        "'black dresses under $40' instead of repr-style arrays.\n"
        "Read the conversation so far before answering: resolve references like 'those', "
        "'the second one' or 'cheaper' against it, and do not reintroduce what the shopper "
        "already knows.\n"
        f"Conversation so far:\n{_conversation_context(history)}\n"
        f"User request: {question}\n"
        f"Saved preferences and memories: {json.dumps(saved, ensure_ascii=True)}\n"
        f"Evidence: {_evidence_for_finalizer(evidence)}"
    )
    messages = [{"role": "user", "content": prompt}]
    if on_token is None:
        structured = model.with_structured_output(AnswerDraft, method="json_schema")
        return await structured.ainvoke(messages)
    try:
        bound = model.bind_tools([AnswerDraft], tool_choice="AnswerDraft")
    except (TypeError, NotImplementedError):
        # Not every provider accepts a forced tool_choice; fall back to letting it pick
        # among the one tool it was given, which amounts to the same thing in practice.
        bound = model.bind_tools([AnswerDraft])
    stream = DraftStream(evidence, session_id, on_token)
    accumulated = None
    async for chunk in bound.astream(messages):
        accumulated = chunk if accumulated is None else accumulated + chunk
        stream.accept(chunk, {"langgraph_node": "model"})
    if accumulated is not None and len(accumulated.tool_calls) == 1 and accumulated.tool_calls[0]["name"] == "AnswerDraft":
        parser = PydanticToolsParser(tools=[AnswerDraft], first_tool_only=True)
        draft = parser.parse_result([ChatGeneration(message=accumulated)])
        if draft is not None:
            return draft
    # The provider ignored the forced tool call (see langchain-ai/langchain#34155) or
    # streamed something unparseable - fall back to one plain, non-streamed structured
    # call rather than surface a broken or empty answer.
    structured = model.with_structured_output(AnswerDraft, method="json_schema")
    return await structured.ainvoke(messages)


def _use_agent_structured_output(injected_model: bool) -> bool:
    """Whether THIS turn's create_agent() call should bind AnswerDraft as a tool-strategy
    response_format, instead of running the separate two-phase finalize call above.

    Always true for injected/test models (DraftStream already decodes their in-loop
    AnswerDraft tool call). For real providers this is gated by AGENT_RESPONSE_STRATEGY -
    see its definition in src/core/config.py for the tradeoff.
    """
    return injected_model or config.AGENT_RESPONSE_STRATEGY == "merged"


async def _checkpoint_config(user_id: str, session_id: str, history: list[dict], injected_model: bool):
    if injected_model or not config.AGENT_CHECKPOINTING_ENABLED:
        return {}, history_messages(history)
    thread_id = f"shopping:{user_id}:{session_id}"
    options = {"configurable": {"thread_id": thread_id}}
    checkpoint = await _CHECKPOINTER.aget_tuple(options)
    return options, [] if checkpoint else history_messages(history)


async def answer(user_id: str, session_id: str, question: str, history: list[dict], model=None, on_token=None, on_status=None) -> ShoppingResponse:
    evidence = Evidence()
    async with asyncio.timeout(config.AGENT_TIMEOUT_SECONDS):
        saved = await preferences.lookup(user_id)
        active_model = model or get_chat_model()
        injected_model = model is not None
        if config.AGENT_CHECKPOINTING_ENABLED:
            await _STORE.aput(("shopping", user_id), "preferences", saved, index=False)
        checkpoint_options, seed_history = await _checkpoint_config(user_id, session_id, history, injected_model)
        # Hosted Groq models cannot combine native JSON response_format with tools, so the
        # default ('two_phase') runs the tool-enabled agent without a response schema and
        # finalizes separately below. AGENT_RESPONSE_STRATEGY=merged (and every injected
        # test model) binds AnswerDraft as a tool-strategy response_format on THIS same
        # call instead, saving one full model round-trip - see config.py for the tradeoff.
        real_provider = not injected_model
        use_agent_structured_output = _use_agent_structured_output(injected_model)
        agent = create_agent(model=active_model,
                             tools=create_tools(user_id, evidence, previous_filters(history)),
                             system_prompt=system_prompt(saved, structured=use_agent_structured_output),
                             middleware=_middleware(active_model),
                             checkpointer=None if injected_model or not config.AGENT_CHECKPOINTING_ENABLED else _CHECKPOINTER,
                             store=None if not config.AGENT_CHECKPOINTING_ENABLED else _STORE,
                             response_format=ToolStrategy(AnswerDraft, handle_errors=True) if use_agent_structured_output else None)
        # Middleware adds graph nodes. The model-call middleware enforces the actual
        # call budget; the larger graph cap only guards unexpected graph-level cycles.
        stream_from_agent = on_token is not None and use_agent_structured_output
        with tracing_context(enabled=config.AGENT_TRACING_ENABLED):
            try:
                inputs = {"messages": seed_history + [{"role": "user", "content": question}]}
                options = {"recursion_limit": 8*(config.AGENT_MAX_TOOL_ROUNDTRIPS+1)+8, **checkpoint_options}
                if not stream_from_agent:
                    if on_status: on_status("generating" if use_agent_structured_output else "retrieving")
                    result = await agent.ainvoke(inputs, options)
                else:
                    result = {}
                    stream = DraftStream(evidence, session_id, on_token)
                    if on_status: on_status("retrieving")
                    async for mode, data in agent.astream(inputs, options, stream_mode=["messages", "updates"]):
                        if mode == "messages":
                            if on_status: on_status("generating")
                            stream.accept(*data)
                        else:
                            for update in data.values():
                                if isinstance(update, dict) and update.get("messages") and on_status:
                                    last = update["messages"][-1]
                                    if getattr(last, "type", "") == "tool":
                                        on_status("retrieving")
                                if isinstance(update, dict) and update.get("structured_response"):
                                    result["structured_response"] = update["structured_response"]
            except GraphRecursionError:
                return ShoppingResponse(session_id=session_id, answer="I reached the search limit for this turn. Please narrow the category or ask about one product.")
        if not use_agent_structured_output:
            # two_phase: the loop above only gathered evidence. Ask for the validated
            # answer in one more call, streamed through on_token when one was given.
            draft = await _finalize_real_model(active_model, question, evidence, saved,
                                               session_id, history, on_token, on_status)
            return hydrate(draft, evidence, session_id)
        if not result.get("structured_response"):
            return ShoppingResponse(session_id=session_id, answer="I reached the search limit for this turn. Please narrow the category or ask about one product.")
        draft = AnswerDraft.model_validate(result["structured_response"])
        return hydrate(draft, evidence, session_id)
