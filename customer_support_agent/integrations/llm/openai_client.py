from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from customer_support_agent.core.settings import get_settings


def _require_key() -> str:
    key = get_settings().openai_api_key
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")
    return key


@lru_cache
def get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.openai_model,
        api_key=_require_key(),
        temperature=s.llm_temperature,
        timeout=s.llm_timeout,
        max_retries=s.llm_max_retries,
    )


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        api_key=_require_key(),
        dimensions=s.embedding_dims,
        chunk_size=s.embedding_batch_size,
        max_retries=s.llm_max_retries,
    )