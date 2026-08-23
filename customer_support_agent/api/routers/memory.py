from typing import Optional

from fastapi import APIRouter, Query

from customer_support_agent.integrations.memory import langmem_store as mem

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/search")
def search(
    q: str = Query(..., min_length=3),
    email: str = Query(...),
    company: Optional[str] = None,
):
    return {
        "query": q,
        "semantic": mem.semantic_available(),
        "hits": mem.search_memories(q, email=email, company=company),
    }