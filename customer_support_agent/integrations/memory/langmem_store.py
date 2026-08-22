from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Optional

from langgraph.store.memory import InMemoryStore

from customer_support_agent.core.settings import get_settings

logger = logging.getLogger(__name__)

_store: Optional[InMemoryStore] = None
ROOT = "claims"


# ---------------------------------------------------------------- namespaces
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def company_slug(company: Optional[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (company or "general").strip().lower()).strip("-")
    return f"company::{slug or 'general'}"


def customer_namespace(email: str) -> tuple[str, ...]:
    return (ROOT, "customer", normalize_email(email))


def company_namespace(company: Optional[str]) -> tuple[str, ...]:
    return (ROOT, "company", company_slug(company))


# -------------------------------------------------------------------- store
def get_store() -> InMemoryStore:
    """Singleton store. Vector-indexed when semantic mode is enabled."""
    global _store
    if _store is not None:
        return _store

    s = get_settings()
    if s.semantic_enabled:
        try:
            from customer_support_agent.integrations.llm.openai_client import (
                get_embeddings,
            )

            embedder = get_embeddings()
            _store = InMemoryStore(
                index={
                    "dims": s.embedding_dims,
                    "embed": lambda texts: embedder.embed_documents(list(texts)),
                }
            )
            logger.info("Memory store initialised with semantic index")
            return _store
        except Exception as exc:
            logger.warning("Semantic memory unavailable (%s); using plain store", exc)

    _store = InMemoryStore()
    return _store


def semantic_available() -> bool:
    return getattr(get_store(), "index_config", None) is not None


# -------------------------------------------------------------------- write
def save_memory(
    content: str,
    email: str,
    company: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write one resolution memory into both customer and company scopes."""
    store = get_store()
    payload = {"content": content.strip(), "metadata": metadata or {}}
    written = []

    for namespace in (customer_namespace(email), company_namespace(company)):
        key = str(uuid.uuid4())
        try:
            _write_via_langmem(store, namespace, payload)
        except Exception as exc:
            logger.warning("LangMem tool write failed (%s); using store.put", exc)
            store.put(namespace, key, payload)
        written.append("/".join(namespace))

    return {"saved": True, "namespaces": written}


def _write_via_langmem(store: InMemoryStore, namespace, payload: dict) -> None:
    """Preferred path: LangMem's managed memory tool."""
    from langmem import create_manage_memory_tool

    tool = create_manage_memory_tool(namespace=namespace, store=store)
    tool.invoke({"action": "create", "content": payload["content"]})


# --------------------------------------------------------------------- read
def search_memories(
    query: str,
    email: str,
    company: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Search customer scope then company scope; dedupe; fall back to recent."""
    store = get_store()
    limit = limit or get_settings().memory_top_k
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for scope, namespace in (
        ("customer", customer_namespace(email)),
        ("company", company_namespace(company)),
    ):
        for item in _search_one(store, namespace, query, limit):
            text = _extract_text(item.value)
            fingerprint = text[:160].strip().lower()
            if not text or fingerprint in seen:
                continue
            seen.add(fingerprint)
            results.append(
                {
                    "memory": text,
                    "score": round(float(getattr(item, "score", 0.0) or 0.0), 4),
                    "metadata": {
                        "scope": scope,
                        "namespace": "/".join(namespace),
                        "key": item.key,
                    },
                }
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


def _search_one(store, namespace, query: str, limit: int) -> list:
    """Semantic search, then recent-listing fallback."""
    try:
        hits = store.search(namespace, query=query, limit=limit)
        if hits:
            return hits
    except Exception as exc:
        logger.warning("Semantic memory search failed (%s); listing recent", exc)

    try:
        return store.search(namespace, limit=limit)
    except Exception:
        return []


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "memory", "value"):
            if isinstance(value.get(key), str):
                return value[key]
        return str(value)
    return str(value)

def _slugify(value: str, fallback: str) -> str:
    """LangGraph namespace labels allow no periods, slashes, or spaces."""
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or fallback


def normalize_email(email: str) -> str:
    """Canonical form used as the DB key and for display."""
    return (email or "").strip().lower()


def email_slug(email: str) -> str:
    """asha.rao@example.com -> asha-rao-example-com"""
    return _slugify(normalize_email(email), "unknown-customer")


def company_slug(company: Optional[str]) -> str:
    """Acme Ltd. -> company-acme-ltd"""
    return f"company-{_slugify(company or '', 'general')}"


def customer_namespace(email: str) -> tuple[str, ...]:
    return (ROOT, "customer", email_slug(email))


def company_namespace(company: Optional[str]) -> tuple[str, ...]:
    return (ROOT, "company", company_slug(company))