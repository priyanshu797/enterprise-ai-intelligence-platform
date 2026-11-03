"""Application configuration.

All settings are loaded from environment variables (see `.env.example`).
Defaults are chosen so the platform runs locally with SQLite + ChromaDB
(embedded) and zero external services. In production, point the variables at
PostgreSQL/pgvector, managed vector DBs, and LLM provider credentials.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ Core
    APP_NAME: str = "Enterprise AI Intelligence Platform"
    ENVIRONMENT: Literal["dev", "prod", "test"] = "dev"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    # Comma-separated list of allowed CORS origins.
    BACKEND_CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # ------------------------------------------------------------- Database
    # SQLite file by default; override with a PostgreSQL URL in production.
    DATABASE_URL: str = "sqlite:///./enterprise_ai.db"

    # ----------------------------------------------------------- Auth / JWT
    # CHANGE THESE in production. 60 * 24 = 1 day access tokens.
    SECRET_KEY: str = "CHANGE-ME-dev-only-not-for-production-use"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --------------------------------------------------------- Vector store
    # "chroma" (embedded file store) | "pgvector" | "pinecone"
    VECTOR_STORE_TYPE: Literal["chroma", "pgvector", "pinecone"] = "chroma"
    CHROMA_PERSIST_PATH: str = "./chroma_db"
    CHROMA_COLLECTION: str = "enterprise_documents"
    PINECONE_API_KEY: str = ""
    PINECONE_ENV: str = ""
    PINECONE_INDEX: str = ""

    # --------------------------------------------------------------- LLM
    # Primary + fallback providers. Fallback is used on errors / rate limits.
    LLM_PRIMARY_PROVIDER: Literal["openai", "anthropic", "groq", "gemini"] = "gemini"
    LLM_FALLBACK_PROVIDER: Literal["openai", "anthropic", "groq", "gemini", "none"] = "groq"

    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_CHAT_MODEL: str = "claude-3-5-haiku-20241022"

    # Groq — free tier at https://console.groq.com
    GROQ_API_KEY: str = ""
    GROQ_CHAT_MODEL: str = "llama-3.3-70b-versatile"

    # Google Gemini — free tier at https://aistudio.google.com/apikey
    # 1500 req/day free, no credit card needed.
    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_MODEL: str = "gemini-flash-lite-latest"
    # Force REST transport to avoid gRPC hang on Windows/restricted networks.
    GOOGLE_API_USE_REST: str = "true"

    # ---------------------------------------------------------- Web Search
    # SerpAPI key for the web_search agent tool.
    # Get a free key at https://serpapi.com/
    SERPAPI_API_KEY: str = ""
    # Base URL for SerpAPI (overridable for testing/proxies).
    SERPAPI_URL: str = "https://serpapi.com/search"

    # If no provider keys are configured, fall back to a local model via Ollama
    # so the platform still runs end-to-end for development/testing.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama3.2"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # Embedding dimensions per model (must match the chosen embedding model).
    EMBEDDING_PROVIDER: Literal["openai", "huggingface", "ollama"] = "huggingface"
    # all-MiniLM-L6-v2: fast, 384-dim, runs on CPU, no API key required.
    HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS: int = 384

    # ----------------------------------------------------------- RAG config
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 6
    # Retrieval strategy: "semantic" | "hybrid" | "rerank" | "rewrite"
    RETRIEVAL_STRATEGY: Literal["semantic", "hybrid", "rerank", "rewrite"] = "hybrid"
    # Minimum similarity score (0-1) to consider a chunk relevant.
    # Lowered to 0.05 — cosine scores from all-MiniLM-L6-v2 via RRF are small.
    MIN_RELEVANCE_SCORE: float = 0.05
    # Confidence below which we refuse to answer (avoid hallucination).
    # Lowered to 0.05 so the model can answer when any relevant chunk is found.
    ANSWER_CONFIDENCE_THRESHOLD: float = 0.05

    # ------------------------------------------------------------- Agent
    AGENT_MAX_ITERATIONS: int = 4   # 4 steps max — keeps total latency under 60s
    AGENT_TOOL_TIMEOUT_SECONDS: int = 20
    AGENT_MAX_TOKENS: int = 4096
    RAG_MAX_TOKENS: int = 4096

    # ------------------------------------------------------------- Memory
    # Max number of past messages retained in a conversation window.
    MEMORY_MAX_MESSAGES: int = 10

    # ------------------------------------------------- Async doc processing
    MAX_CONCURRENT_DOC_JOBS: int = 2

    # ------------------------------------------------------------- Caching
    # Simple in-memory answer cache (cost optimisation).
    ENABLE_RESPONSE_CACHE: bool = True
    CACHE_MAX_ENTRIES: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
