"""Lazy LangChain provider factory; API handlers never speak provider protocols."""
from functools import lru_cache
from src.core import config


@lru_cache(maxsize=1)
def get_chat_model():
    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY: raise RuntimeError("Set GROQ_API_KEY before chatting")
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY,
                        temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS,
                        timeout=config.LLM_TIMEOUT_SECONDS, max_retries=1)
    if not config.GEMINI_API_KEY: raise RuntimeError("Set GEMINI_API_KEY before chatting")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, api_key=config.GEMINI_API_KEY,
                                  vertexai=False, temperature=config.LLM_TEMPERATURE,
                                  max_tokens=config.LLM_MAX_TOKENS, timeout=config.LLM_TIMEOUT_SECONDS,
                                  max_retries=1)
