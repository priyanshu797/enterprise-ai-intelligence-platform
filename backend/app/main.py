"""FastAPI application entrypoint.

Wires routers, configures CORS, initialises the DB on startup, and exposes
auto-generated OpenAPI docs (satisfying the API documentation deliverable).

Run locally:  uvicorn app.main:app --reload
Run production: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import agent, auth, chat, conversations, documents, stats
from .config import settings
from .database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.ENVIRONMENT)
    init_db()
    logger.info("Database initialised at %s", settings.DATABASE_URL)
    # Warm the vector store. If the chroma_db is corrupted, delete it and retry once.
    try:
        from .services.vector_store import get_vector_store
        get_vector_store()
        logger.info("Vector store ready")
    except Exception as exc:
        logger.warning("Vector store failed on first load (%s) — resetting and retrying", exc)
        try:
            import shutil, os
            chroma_path = settings.CHROMA_PERSIST_PATH
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path, ignore_errors=True)
                logger.info("Deleted corrupted chroma_db at %s", chroma_path)
            # Reset singleton so it creates a fresh store
            from .services.vector_store import reset_vector_store
            reset_vector_store()
            from .services.vector_store import get_vector_store
            get_vector_store()
            logger.info("Vector store re-created successfully")
        except Exception as exc2:
            logger.warning("Vector store re-creation also failed: %s — continuing without it", exc2)
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise AI Intelligence Platform",
        description=(
            "Production-ready AI platform: document intelligence, advanced RAG, "
            "multi-provider LLMs with fallback, AI agents with tools, "
            "conversational memory, evaluation, and a dashboard."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    )

    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(o) for o in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    prefix = settings.API_V1_PREFIX
    app.include_router(auth.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(conversations.router, prefix=prefix)
    app.include_router(agent.router, prefix=prefix)
    app.include_router(stats.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    def root():
        return {"app": settings.APP_NAME, "docs": "/docs", "health": f"{prefix}/stats/health"}

    return app


app = create_app()
