"""Small live tool-calling contract check, using a clearly fictional test catalog."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def check():
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware
    from langchain.agents.structured_output import ToolStrategy
    from langchain.tools import tool
    from langsmith import tracing_context
    from src.agent.models import get_chat_model
    from src.agent.schemas import AnswerDraft
    from src.agent.shopper import DraftStream
    from src.agent.tools import Evidence
    from src.core import config

    calls, tokens = [], []
    evidence = Evidence()
    @tool
    def catalog_search(query: str) -> dict:
        """Search the fictional integration-test catalog."""
        calls.append(query)
        evidence.products["TEST-P1"] = {"parent_asin":"TEST-P1", "title":"Test black dress", "price":30}
        return {"count":1,"results":[{"parent_asin":"TEST-P1","title":"Test black dress","price":30}]}
    agent = create_agent(get_chat_model(), tools=[catalog_search],
                         middleware=[ModelCallLimitMiddleware(run_limit=4)],
                         system_prompt="This is an integration test. Call catalog_search to find a black dress under $40, then return AnswerDraft with the returned product ID.",
                         response_format=ToolStrategy(AnswerDraft))
    with tracing_context(enabled=False):
        async with asyncio.timeout(90):
            result = {}
            stream = DraftStream(evidence, "test-session", tokens.append)
            async for mode, data in agent.astream({"messages":[{"role":"user","content":"Find a black dress under $40 in the test catalog."}]}, {"recursion_limit":40}, stream_mode=["messages", "updates"]):
                if mode == "messages":
                    stream.accept(*data)
                else:
                    for update in data.values():
                        if isinstance(update, dict) and update.get("structured_response"):
                            result["structured_response"] = update["structured_response"]
    draft = result.get("structured_response")
    if not calls or not draft or draft.product_ids != ["TEST-P1"]:
        raise RuntimeError("Provider did not complete the required tool and schema contract")
    print(f"PASS: {config.LLM_PROVIDER} called the tool and returned validated structured output; {len(tokens)} provisional text events.")


if __name__ == "__main__":
    try:
        asyncio.run(check())
    except Exception as exc:
        # Provider exceptions can carry request data. Keep keys and payloads out of CLI output.
        print(f"Provider check failed: {type(exc).__name__}; status={getattr(exc, 'status_code', 'unavailable')}")
        raise SystemExit(1)
