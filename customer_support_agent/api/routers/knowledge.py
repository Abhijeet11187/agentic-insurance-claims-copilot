from fastapi import APIRouter, Query

from customer_support_agent.integrations.rag import chroma_kb

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/ingest")
def ingest():
    return chroma_kb.ingest_knowledge_base()


@router.get("/stats")
def stats():
    return chroma_kb.knowledge_stats()


@router.get("/search")
def search(q: str = Query(..., min_length=3), top_k: int = 4):
    return {"query": q, "hits": chroma_kb.search_knowledge(q, top_k)}