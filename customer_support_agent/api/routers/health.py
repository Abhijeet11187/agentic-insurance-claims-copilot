from fastapi import APIRouter

from customer_support_agent.core.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "app": s.app_name,
        "llm_configured": bool(s.openai_api_key),
        "model": s.openai_model,
        "semantic_enabled": s.semantic_enabled,
    }