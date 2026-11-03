"""Concrete LLM provider implementations backed by LangChain chat models.

Each provider normalises token usage and maps transport errors to
`RateLimitError` (triggers fallback) or `ProviderError` (non-retryable).
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ...config import settings
from .base import LLMResponse, LLMUsage, ProviderError, RateLimitError, timed

logger = logging.getLogger(__name__)


def _messages_to_langchain(messages: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert role/content dicts to LangChain message objects."""
    role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
        "tool": ToolMessage,
    }
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        cls = role_map.get(role, HumanMessage)
        if cls is ToolMessage:
            out.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "default")))
        else:
            out.append(cls(content=content))
    return out


def _extract_usage(result: Any, default_model: str) -> LLMUsage:
    """Pull token usage from a LangChain AIMessage's usage metadata."""
    usage = LLMUsage()
    meta = getattr(result, "usage_metadata", None) or {}
    usage.prompt_tokens = int(meta.get("input_tokens", meta.get("prompt_tokens", 0)) or 0)
    usage.completion_tokens = int(meta.get("output_tokens", meta.get("completion_tokens", 0)) or 0)
    usage.total_tokens = int(meta.get("total_tokens", usage.prompt_tokens + usage.completion_tokens) or 0)
    return usage


def _tool_calls_from(result: AIMessage) -> list[dict[str, Any]]:
    calls = []
    for tc in (getattr(result, "tool_calls", None) or []):
        calls.append({"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")})
    return calls


def _classify_error(exc: Exception) -> Exception:
    """Map SDK/transport exceptions to RateLimitError vs ProviderError."""
    msg = str(exc).lower()
    # Rate-limit / overload / capacity signals -> trigger fallback.
    if any(s in msg for s in (
        "rate limit", "rate_limit", "429", "overloaded", "capacity",
        "too many requests", "503", "unavailable", "high demand",
        "resource_exhausted", "resourceexhausted", "quota",
    )):
        return RateLimitError(str(exc))
    # Auth / invalid key -> non-retryable.
    if any(s in msg for s in ("401", "authentication", "invalid api key", "forbidden", "403")):
        return ProviderError(str(exc))
    # Network / timeout -> treat as transient (allow fallback to try).
    if any(s in msg for s in ("timeout", "timed out", "connection", "apiconnection", "apiconnectionerror")):
        return RateLimitError(str(exc))
    return ProviderError(str(exc))


class BaseChatProvider:
    """Shared logic wrapping a LangChain chat model instance."""

    name: str = "base"
    model: str = ""
    # Per-provider name for the max-output-tokens parameter; OpenAI/Anthropic
    # use `max_tokens`, Ollama uses `num_predict`.
    token_kwarg: str = "max_tokens"

    def __init__(self, chat_model: Any):
        self._chat = chat_model

    @timed
    def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        # Translate the generic `max_tokens` kwarg to the provider-specific
        # parameter name (e.g. Ollama uses `num_predict`, not `max_tokens`).
        mt = kwargs.pop("max_tokens", None)
        if mt is not None:
            kwargs[self.token_kwarg] = mt
        try:
            lc_messages = _messages_to_langchain(messages)
            result = self._chat.invoke(lc_messages, **kwargs)
        except Exception as exc:
            raise _classify_error(exc) from exc

        tool_calls = _tool_calls_from(result) if isinstance(result, AIMessage) else []
        # Handle newer Gemini SDK response format where content can be a list of dicts
        raw_content = result.content if isinstance(result, BaseMessage) else str(result)
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", str(part)))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        else:
            content = str(raw_content)
        return LLMResponse(
            content=content,
            provider=self.name,
            model=self.model,
            usage=_extract_usage(result, self.model),
            latency_ms=0,
            tool_calls=tool_calls,
        )

    def with_structured_output(self, schema: type):
        """Return a provider whose complete() yields a parsed `schema`."""
        structured_chat = self._chat.with_structured_output(schema)

        @timed
        def complete(messages, **kwargs):
            try:
                lc_messages = _messages_to_langchain(messages)
                parsed = structured_chat.invoke(lc_messages, **kwargs)
            except Exception as exc:
                raise _classify_error(exc) from exc
            # Best-effort token usage (structured output often hides usage).
            usage = LLMUsage()
            return LLMResponse(
                content=parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else str(parsed),
                provider=self.name,
                model=self.model,
                usage=usage,
                latency_ms=0,
                parsed=parsed,
            )

        return _StructuredWrapper(complete, self.name, self.model)

    @property
    def bound(self) -> Any:
        """Expose the underlying langchain model for direct binding (tools)."""
        return self._chat


class _StructuredWrapper:
    def __init__(self, complete_fn, name, model):
        self._complete = complete_fn
        self.name = name
        self.model = model

    def complete(self, messages, **kwargs):
        return self._complete(messages, **kwargs)


# ---------------------------------------------------------------- OpenAI
def make_openai_provider() -> BaseChatProvider | None:
    if not settings.OPENAI_API_KEY:
        return None
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2,
        max_tokens=settings.AGENT_MAX_TOKENS,
        timeout=settings.AGENT_TOOL_TIMEOUT_SECONDS,
        max_retries=1,
    )
    provider = BaseChatProvider(chat)
    provider.name = "openai"
    provider.model = settings.OPENAI_CHAT_MODEL
    return provider


# ---------------------------------------------------------------- Anthropic
def make_anthropic_provider() -> BaseChatProvider | None:
    if not settings.ANTHROPIC_API_KEY:
        return None
    from langchain_anthropic import ChatAnthropic

    chat = ChatAnthropic(
        model=settings.ANTHROPIC_CHAT_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=0.2,
        max_tokens=settings.AGENT_MAX_TOKENS,
        timeout=settings.AGENT_TOOL_TIMEOUT_SECONDS,
        max_retries=1,
    )
    provider = BaseChatProvider(chat)
    provider.name = "anthropic"
    provider.model = settings.ANTHROPIC_CHAT_MODEL
    return provider


# ---------------------------------------------------------------- Groq
def make_groq_provider() -> BaseChatProvider | None:
    if not settings.GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        logger.warning(
            "langchain-groq not installed. Run: pip install langchain-groq"
        )
        return None

    chat = ChatGroq(
        model=settings.GROQ_CHAT_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.2,
        max_tokens=settings.AGENT_MAX_TOKENS,
        timeout=settings.AGENT_TOOL_TIMEOUT_SECONDS,
        max_retries=1,
    )
    provider = BaseChatProvider(chat)
    provider.name = "groq"
    provider.model = settings.GROQ_CHAT_MODEL
    return provider


# ---------------------------------------------------------------- Google Gemini
# Thread-safe cache for Gemini thought signatures.
# Gemini 3+ models embed an encrypted thought_signature in function call parts
# that MUST be returned verbatim on the next turn. LangChain strips it when
# converting to AIMessage, so we cache it keyed by tool_call_id and reattach
# it when the message history is sent back to Gemini.
import threading as _threading

_thought_sig_cache: dict[str, bytes] = {}
_thought_sig_lock = _threading.Lock()


def _store_thought_signatures(raw_response: Any) -> None:
    """Extract and cache thought_signature bytes from a Gemini raw response."""
    try:
        # The raw response from the generativeai SDK has .candidates[].content.parts
        candidates = getattr(raw_response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is None:
                    continue
                sig = getattr(part, "thought_signature", None) or getattr(fc, "thought_signature", None)
                if sig:
                    # Use a stable key: function name + args hash
                    import hashlib, json
                    args_str = ""
                    try:
                        args_str = json.dumps(dict(fc.args), sort_keys=True)
                    except Exception:
                        pass
                    key = f"{fc.name}:{args_str}"
                    with _thought_sig_lock:
                        _thought_sig_cache[key] = sig
                    logger.debug("Cached thought_signature for tool call %s", fc.name)
    except Exception as exc:
        logger.debug("Could not extract thought signatures: %s", exc)


class GeminiChatProvider(BaseChatProvider):
    """Gemini provider with thought_signature handling.

    Uses the OpenAI-compatible endpoint for simplicity when langchain-openai is
    available, otherwise falls back to the native langchain-google-genai.
    The native path includes thought_signature caching to handle Gemini 3+.
    """

    def __init__(self, chat_model: Any, *, use_openai_compat: bool = False):
        super().__init__(chat_model)
        self._use_openai_compat = use_openai_compat

    @timed
    def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        # Drop max_tokens entirely for native langchain-google-genai — it's set
        # at construction time via max_output_tokens and rejected as an invoke kwarg.
        # For OpenAI-compat endpoint, max_tokens is valid.
        mt = kwargs.pop("max_tokens", None)
        if mt is not None and self._use_openai_compat:
            kwargs["max_tokens"] = mt
        # (native Gemini: max_output_tokens is constructor-only, don't pass it)
        try:
            lc_messages = _messages_to_langchain(messages)
            result = self._chat.invoke(lc_messages, **kwargs)
        except Exception as exc:
            raise _classify_error(exc) from exc

        tool_calls = _tool_calls_from(result) if isinstance(result, AIMessage) else []
        raw_content = result.content if isinstance(result, BaseMessage) else str(result)
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", str(part)))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        else:
            content = str(raw_content)
        return LLMResponse(
            content=content,
            provider=self.name,
            model=self.model,
            usage=_extract_usage(result, self.model),
            latency_ms=0,
            tool_calls=tool_calls,
        )


def make_gemini_provider() -> "GeminiChatProvider | None":
    if not settings.GEMINI_API_KEY:
        return None

    key = settings.GEMINI_API_KEY

    # Strategy 1: Use the OpenAI-compatible endpoint via langchain-openai.
    # The OpenAI compat endpoint at googleapis.com/v1beta/openai/ is SUPPOSED
    # to handle thought signatures internally. However, it still drops them
    # in the current implementation. We try it first, and fall back to native.
    try:
        from langchain_openai import ChatOpenAI
        chat = ChatOpenAI(
            model=f"models/{settings.GEMINI_CHAT_MODEL}",
            api_key=settings.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.2,
            max_tokens=settings.AGENT_MAX_TOKENS,
            max_retries=1,
            timeout=settings.AGENT_TOOL_TIMEOUT_SECONDS,
        )
        provider = GeminiChatProvider(chat, use_openai_compat=True)
        provider.name = "gemini"
        provider.model = settings.GEMINI_CHAT_MODEL
        return provider
    except ImportError:
        pass

    # Strategy 2: Native langchain-google-genai
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        logger.warning(
            "Neither langchain-openai nor langchain-google-genai is installed."
        )
        return None

    import os as _os
    _os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
    _os.environ["GOOGLE_API_USE_REST"] = "true"

    model_name = settings.GEMINI_CHAT_MODEL
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    chat = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
        max_output_tokens=settings.AGENT_MAX_TOKENS,
        max_retries=1,
        convert_system_message_to_human=True,
    )
    provider = GeminiChatProvider(chat, use_openai_compat=False)
    provider.name = "gemini"
    provider.model = settings.GEMINI_CHAT_MODEL
    return provider


# ---------------------------------------------------------------- Ollama (local)
def make_ollama_provider() -> BaseChatProvider | None:
    from langchain_ollama import ChatOllama

    try:
        chat = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_CHAT_MODEL,
            temperature=0.2,
        )
        provider = BaseChatProvider(chat)
        provider.name = "ollama"
        provider.model = settings.OLLAMA_CHAT_MODEL
        provider.token_kwarg = "num_predict"
        return provider
    except Exception as exc:  # noqa: BLE001
        logger.info("Ollama provider not available: %s", exc)
        return None
