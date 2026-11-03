"""Shared LLM primitives: response model, errors, and provider protocol."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """Raised when a provider returns a non-retryable error."""


class RateLimitError(ProviderError):
    """Raised when a provider rate-limits the request (triggers fallback)."""


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class LLMResponse:
    """Normalised response from any provider."""
    content: str
    provider: str
    model: str
    usage: LLMUsage
    latency_ms: int
    # When structured output is requested, the parsed object lives here.
    parsed: Any = None
    # Raw tool-call message dict(s) if tool calling was used.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Which provider produced this response (primary or fallback).
    served_by: str = "primary"


class ChatProvider(Protocol):
    """Backend-agnostic chat model protocol."""

    name: str
    model: str

    def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse: ...

    def with_structured_output(self, schema: type):
        """Return a wrapper whose `.complete()` returns parsed `schema`."""
        ...


def timed(fn):
    """Decorator recording wall-clock latency in ms on LLMResponse."""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = int((time.perf_counter() - start) * 1000)
        if isinstance(result, LLMResponse):
            result.latency_ms = elapsed
        return result

    return wrapper
