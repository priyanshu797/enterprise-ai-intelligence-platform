"""LLM orchestration service: provider fallback, token tracking, validation.

Implements Module C (LLM integration) and Challenge 1 (intelligent model
fallback). Provider order is resolved from settings: primary, then fallback,
then a local Ollama model if reachable. On a rate-limit or transient error the
orchestrator automatically retries with the next provider and records which
one ultimately served the request.

Honest behaviour note (Rule 6 — No Fabricated Functionality): if NO provider
is configured we raise a clear configuration error instead of returning a
hardcoded response. A clearly-labelled test stub is used in the test suite,
never in the live API.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from ...config import settings
from .base import LLMResponse, LLMUsage, ProviderError, RateLimitError
from .providers import (
    BaseChatProvider,
    make_anthropic_provider,
    make_gemini_provider,
    make_groq_provider,
    make_ollama_provider,
    make_openai_provider,
)

logger = logging.getLogger(__name__)


def _ollama_reachable() -> bool:
    """Quick TCP check — avoids adding Ollama to the chain when it's not running."""
    import socket
    try:
        from urllib.parse import urlparse
        parsed = urlparse(settings.OLLAMA_BASE_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 11434
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class LLMService:
    def __init__(self):
        self._providers: list[BaseChatProvider] = []
        self._names: list[str] = []
        self._build_chain()

    def _build_chain(self) -> None:
        """Resolve the provider chain in priority order."""
        candidates: list[tuple[str, Any]] = []
        for name in (settings.LLM_PRIMARY_PROVIDER, settings.LLM_FALLBACK_PROVIDER):
            if name and name != "none" and name not in [c[0] for c in candidates]:
                candidates.append((name, None))
        # Only add Ollama as a last resort if it's not already in the chain.
        if "ollama" not in [c[0] for c in candidates]:
            candidates.append(("ollama", None))

        factories = {
            "openai": make_openai_provider,
            "anthropic": make_anthropic_provider,
            "groq": make_groq_provider,
            "gemini": make_gemini_provider,
            "ollama": make_ollama_provider,
        }
        for name, _ in candidates:
            factory = factories.get(name)
            if not factory:
                continue
            try:
                provider = factory()
            except Exception as exc:  # noqa: BLE001
                logger.info("Skipping %s provider: %s", name, exc)
                continue
            if provider is not None:
                # Quick connectivity pre-check for Ollama: skip if not reachable
                # so it doesn't silently join the chain and fail on every request.
                if name == "ollama" and not _ollama_reachable():
                    logger.info("Ollama not reachable at %s — skipping", settings.OLLAMA_BASE_URL)
                    continue
                self._providers.append(provider)
                self._names.append(name)
        logger.info("LLM provider chain: %s", self._names or "(none)")

    # ------------------------------------------------------------------ API
    @property
    def available(self) -> bool:
        return len(self._providers) > 0

    @property
    def provider_names(self) -> list[str]:
        return list(self._names)

    @property
    def primary(self) -> str:
        return self._names[0] if self._names else "none"

    @property
    def fallback(self) -> str:
        return self._names[1] if len(self._names) > 1 else "none"

    def _require(self) -> None:
        if not self.available:
            raise ProviderError(
                "No LLM provider is available. Possible causes:\n"
                "1. GEMINI_API_KEY is missing or invalid — real keys start with 'AIza'. "
                "Get one free at https://aistudio.google.com/apikey\n"
                "2. GROQ_API_KEY is empty — get one free at https://console.groq.com\n"
                "3. All configured keys are failing auth (check backend/.env).\n"
                f"Ollama was checked at {settings.OLLAMA_BASE_URL} and was not reachable."
            )

    def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Complete a chat, with automatic provider fallback.

        Tries providers in chain order. `RateLimitError` and transient errors
        move to the next provider; a non-retryable `ProviderError` from the
        last provider surfaces to the caller.
        """
        self._require()
        last_error: Exception | None = None
        for idx, provider in enumerate(self._providers):
            try:
                resp = provider.complete(messages, **kwargs)
                resp.served_by = "primary" if idx == 0 else f"fallback({provider.name})"
                logger.debug("LLM served by %s (%s)", provider.name, resp.served_by)
                return resp
            except RateLimitError as exc:
                last_error = exc
                logger.warning(
                    "Provider %s rate-limited/erred; trying fallback: %s",
                    provider.name, exc,
                )
                continue
            except ProviderError as exc:
                last_error = exc
                # Non-retryable (e.g. auth) — try next provider too, because a
                # different provider may be healthy.
                logger.warning("Provider %s error: %s", provider.name, exc)
                continue
        raise ProviderError(
            f"All LLM providers failed. Last error: {last_error}"
        ) from last_error

    def get_provider(self, name: str) -> BaseChatProvider | None:
        """Fetch a specific provider by name (used by the agent to bind tools)."""
        for p in self._providers:
            if p.name == name:
                return p
        return self._providers[0] if self._providers else None

    def with_structured_output(self, schema: type, provider_name: str | None = None):
        """Return a complete() function that yields parsed structured output,
        with the same fallback semantics."""
        self._require()

        def complete(messages, **kwargs):
            providers = self._providers
            if provider_name:
                p = self.get_provider(provider_name)
                providers = [p] if p else providers
            last_error = None
            for idx, provider in enumerate(providers):
                try:
                    structured = provider.with_structured_output(schema)
                    resp = structured.complete(messages, **kwargs)
                    resp.served_by = "primary" if idx == 0 else f"fallback({provider.name})"
                    return resp
                except RateLimitError as exc:
                    last_error = exc
                    continue
                except ProviderError as exc:
                    last_error = exc
                    continue
            raise ProviderError(f"All providers failed for structured output: {last_error}") from last_error

        return complete

    def estimate_cost(self, provider: str, model: str, usage: LLMUsage) -> float:
        """Rough per-request cost in USD (Module C token tracking, Challenge 5).

        Prices are approximate per-1M-token rates as of late 2024/early 2025;
        update them when models/pricing change.
        """
        # Rates per 1M tokens: (input, output)
        rates = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "gpt-3.5-turbo": (0.50, 1.50),
            "claude-3-5-haiku-20241022": (0.80, 4.00),
            "claude-3-5-sonnet-20241022": (3.00, 15.00),
            # Groq hosted rates
            "llama-3.3-70b-versatile": (0.59, 0.79),
            "llama-3.1-70b-versatile": (0.59, 0.79),
            "llama3-70b-8192": (0.59, 0.79),
            "llama3-8b-8192": (0.05, 0.08),
            "mixtral-8x7b-32768": (0.24, 0.24),
            # Gemini rates (free tier = $0, paid tier approx)
            "gemini-2.0-flash": (0.075, 0.30),
            "gemini-1.5-flash": (0.075, 0.30),
            "gemini-1.5-pro": (1.25, 5.00),
        }
        key = model
        if key not in rates:
            # Unknown model — assume the cheapest known rate as a conservative default.
            rate_in, rate_out = rates["gpt-4o-mini"]
        else:
            rate_in, rate_out = rates[key]
        return round(
            (usage.prompt_tokens / 1_000_000) * rate_in
            + (usage.completion_tokens / 1_000_000) * rate_out,
            6,
        )


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
