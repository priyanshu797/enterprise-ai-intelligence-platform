"""LLM integration (Module C) with intelligent fallback (Challenge 1).

Public API:
  * `get_llm_service()`  — singleton orchestrator.
  * `LLMResponse`       — normalised result with token usage + provider info.
"""
from __future__ import annotations

from .base import LLMResponse, LLMUsage, ProviderError, RateLimitError
from .service import LLMService, get_llm_service
from .templates import build_rag_prompt, build_agent_system_prompt, build_report_prompt

__all__ = [
    "LLMResponse",
    "LLMUsage",
    "ProviderError",
    "RateLimitError",
    "LLMService",
    "get_llm_service",
    "build_rag_prompt",
    "build_agent_system_prompt",
    "build_report_prompt",
]
