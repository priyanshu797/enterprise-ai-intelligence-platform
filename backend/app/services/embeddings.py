"""Embedding generation with pluggable backends.

Default backend is HuggingFace sentence-transformers (runs locally, no API
key, no network), so the platform works end-to-end out of the box. When an
OpenAI key is present, OpenAI embeddings are used. Ollama is supported for
fully-local deployments.

All backends expose a consistent `.embed_documents()` / `.embed_query()`
interface (mirroring LangChain's `Embeddings` protocol).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Thin wrapper that lazily instantiates the configured embedding backend.

    We hold a single cached instance (backends load a model on first use, which
    is expensive) and expose the LangChain `Embeddings` interface.
    """

    def __init__(self, provider: str | None = None):
        self._provider = provider or settings.EMBEDDING_PROVIDER
        # If OpenAI is configured but no key, fall back gracefully.
        if self._provider == "openai" and not settings.OPENAI_API_KEY:
            logger.warning(
                "OpenAI embeddings requested but OPENAI_API_KEY not set; "
                "falling back to HuggingFace embeddings."
            )
            self._provider = "huggingface"
        self._backend = None

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        if self._provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            self._backend = OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
        elif self._provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            self._backend = OllamaEmbeddings(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_EMBEDDING_MODEL,
            )
        else:  # huggingface — default, local, no key required
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings

            self._backend = HuggingFaceEmbeddings(
                model_name=settings.HUGGINGFACE_EMBEDDING_MODEL,
                model_kwargs={
                    "device": "cpu",
                    # Required for google/embeddinggemma-2 and other models
                    # that ship custom code in their repo.
                    "trust_remote_code": True,
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        logger.info("Embeddings backend initialised: %s", self._provider)
        return self._backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_backend().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_backend().embed_query(text)

    @property
    def provider_name(self) -> str:
        return self._provider


@lru_cache
def get_embeddings_service() -> EmbeddingsService:
    return EmbeddingsService()
