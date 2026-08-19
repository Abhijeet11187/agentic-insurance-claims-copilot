from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from customer_support_agent.core.settings import get_settings
from customer_support_agent.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from customer_support_agent.api.routers import (
        drafts, health, knowledge, tickets,
    )

    app.include_router(health.router)
    app.include_router(tickets.router)
    app.include_router(drafts.router)
    app.include_router(knowledge.router)
    return app