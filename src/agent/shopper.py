"""Bounded LangChain execution, with cards hydrated only from current-turn evidence."""
import asyncio
import json
import re
from langchain.agents import create_agent
from langchain.agents.middleware import (ClearToolUsesEdit, ContextEditingMiddleware,
                                         ModelCallLimitMiddleware, ModelRetryMiddleware,
                                         SummarizationMiddleware)
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
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
    if evidence.searches and all(s["count"] == 0 for s in evidence.searches) and not products:
        filters = evidence.searches[-1]["applied_filters"]
        constraints = _human_constraints(filters)
        # Make the all-empty catalog case deterministic; the model cannot fill it with
        # a plausible fictional price, even when its prose ignores the system prompt.
        return ShoppingResponse(session_id=session_id,
            answer=f"No catalog products matched {constraints or 'this search'}. Would you like to broaden the search or relax one constraint?",
            suggested_relaxations=["Try a broader category", "Adjust the price limit"])
    answer = _ground_product_claims(draft.answer, products)
    return ShoppingResponse(session_id=session_id, answer=answer, products=products,
                            citations=citations, suggested_relaxations=draft.suggested_relaxations)


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
        if len(content) > budget: break
        selected.append({"role": m["role"], "content": content})
        budget -= len(content)
    return list(reversed(selected))


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


def _structured_response_strategy(injected_model: bool):
    # Fake/scripted test models emit AnswerDraft as a normal tool call. Real hosted
    # providers use the schema-only finalizer below because Groq rejects JSON mode
    # when the same request also contains function tools.
    return ToolStrategy(AnswerDraft, handle_errors=True) if injected_model else ProviderStrategy(AnswerDraft)


def _evidence_for_finalizer(evidence: Evidence) -> str:
    """Keep the second (schema-only) call small and grounded in this turn's evidence."""
    return json.dumps({
        "products": list(evidence.products.values()),
        "citations": list(evidence.citations.values()),
        "searches": evidence.searches,
    }, ensure_ascii=True, default=str)[:config.MAX_CONTEXT_CHARS]


async def _finalize_real_model(model, question: str, evidence: Evidence, saved: dict) -> AnswerDraft:
    """Create the structured draft after the tool-enabled agent has finished.

    Groq does not allow response_format=json_* on a request that also contains
    function tools.  This call has no tools, so native JSON schema is safe.
    """
    structured = model.with_structured_output(AnswerDraft, method="json_schema")
    prompt = (
        "Return an AnswerDraft for the user's request. Use only the supplied evidence; "
        "never invent product or citation IDs. Select product_ids and citation_ids from "
        "the evidence, or leave them empty. Keep the answer concise. The answer field must "
        "be polished, human-friendly shopping prose: do not output JSON, Python lists, "
        "internal field names, raw filters, or tool arguments. Use natural phrases such as "
        "'black dresses under $40' instead of repr-style arrays.\n"
        f"User request: {question}\n"
        f"Saved preferences: {json.dumps(saved, ensure_ascii=True)}\n"
        f"Evidence: {_evidence_for_finalizer(evidence)}"
    )
    return await structured.ainvoke([{"role": "user", "content": prompt}])


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
        # Hosted Groq models cannot combine native JSON response_format with tools.
        # Run the real agent without a response schema, then finalize in a separate
        # schema-only request. Injected test models retain the original tool schema.
        real_provider = not injected_model
        agent = create_agent(model=active_model, tools=create_tools(user_id, evidence),
                             system_prompt=system_prompt(saved, structured=not real_provider),
                             middleware=_middleware(active_model),
                             checkpointer=None if injected_model or not config.AGENT_CHECKPOINTING_ENABLED else _CHECKPOINTER,
                             store=None if not config.AGENT_CHECKPOINTING_ENABLED else _STORE,
                             response_format=_structured_response_strategy(injected_model) if not real_provider else None)
        # Middleware adds graph nodes. The model-call middleware enforces the actual
        # call budget; the larger graph cap only guards unexpected graph-level cycles.
        with tracing_context(enabled=config.AGENT_TRACING_ENABLED):
            try:
                inputs = {"messages": seed_history + [{"role": "user", "content": question}]}
                options = {"recursion_limit": 8*(config.AGENT_MAX_TOOL_ROUNDTRIPS+1)+8, **checkpoint_options}
                if on_token is None or real_provider:
                    if on_status: on_status("generating")
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
        if real_provider:
            draft = await _finalize_real_model(active_model, question, evidence, saved)
            if on_token:
                on_token(draft.answer)
            return hydrate(draft, evidence, session_id)
        if not result.get("structured_response"):
            return ShoppingResponse(session_id=session_id, answer="I reached the search limit for this turn. Please narrow the category or ask about one product.")
        draft = AnswerDraft.model_validate(result["structured_response"])
        return hydrate(draft, evidence, session_id)
