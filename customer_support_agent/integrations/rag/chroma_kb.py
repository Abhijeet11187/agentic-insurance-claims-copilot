from __future__ import annotations

import logging
from typing import Any, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from customer_support_agent.core.settings import get_settings

logger = logging.getLogger(__name__)
COLLECTION_NAME = "insurance_knowledge"

_client: Optional[chromadb.ClientAPI] = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        s = get_settings()
        _client = chromadb.PersistentClient(path=str(s.chroma_path))
    return _client


def _get_embedder():
    """Return the shared OpenAI embeddings client, or None for Chroma's default."""
    if not get_settings().semantic_enabled:
        return None
    from customer_support_agent.integrations.llm.openai_client import get_embeddings

    return get_embeddings()


def _get_collection():
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def ingest_knowledge_base() -> dict[str, Any]:
    """Read every .md file, chunk it, embed it, and upsert into Chroma."""
    s = get_settings()
    files = sorted(s.knowledge_base_dir.glob("*.md"))
    if not files:
        return {"files": 0, "chunks": 0, "message": "knowledge_base/ is empty"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    ids, docs, metas = [], [], []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{path.stem}::{i}")
            docs.append(chunk)
            metas.append({"source": path.name, "chunk": i})

    collection = _get_collection()
    embedder = _get_embedder()

    BATCH = s.embedding_batch_size
    for start in range(0, len(docs), BATCH):
        b_ids = ids[start : start + BATCH]
        b_docs = docs[start : start + BATCH]
        b_metas = metas[start : start + BATCH]
        kwargs: dict[str, Any] = {
            "ids": b_ids, "documents": b_docs, "metadatas": b_metas
        }
        if embedder is not None:
            kwargs["embeddings"] = embedder.embed_documents(b_docs)
        collection.upsert(**kwargs)

    return {
        "files": len(files),
        "chunks": len(docs),
        "semantic": embedder is not None,
    }


def search_knowledge(query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
    """Return the top-k most relevant knowledge chunks with source metadata."""
    s = get_settings()
    top_k = top_k or s.rag_top_k
    collection = _get_collection()

    if collection.count() == 0:
        return []

    try:
        embedder = _get_embedder()
        if embedder is not None:
            res = collection.query(
                query_embeddings=[embedder.embed_query(query)], n_results=top_k
            )
        else:
            res = collection.query(query_texts=[query], n_results=top_k)
    except Exception as exc:
        logger.warning("Knowledge search failed: %s", exc)
        return []

    hits = []
    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append(
            {
                "text": doc,
                "source": (meta or {}).get("source", "unknown"),
                "score": round(1 - float(dist), 4),
            }
        )
    return hits


def knowledge_stats() -> dict[str, Any]:
    return {
        "collection": COLLECTION_NAME,
        "chunks": _get_collection().count(),
        "semantic": _get_embedder() is not None,
    }