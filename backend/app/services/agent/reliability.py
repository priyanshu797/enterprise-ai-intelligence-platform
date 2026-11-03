"""Agent reliability primitives (Challenge 7).

Wraps tool execution in:
  * a hard timeout (settings.AGENT_TOOL_TIMEOUT_SECONDS),
  * a small retry budget for transient failures,
  * structured error capture so the agent can recover and continue.

Uses a thread-based timeout because LLM/IO calls are blocking.
"""
from __future__ import annotations

import functools
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from ...config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ToolResult:
    output: Any
    success: bool
    error: str = ""
    latency_ms: int = 0


class ToolTimeoutError(RuntimeError):
    pass


def with_timeout_and_retry(func, timeout: float | None = None, retries: int = 1):
    """Run a blocking func with a timeout and retry on transient errors.

    Returns a ToolResult rather than raising, so the agent loop can feed
    errors back to the model as tool messages.
    """
    timeout = timeout or settings.AGENT_TOOL_TIMEOUT_SECONDS
    result: dict[str, Any] = {}

    def _target():
        try:
            result["value"] = func()
        except Exception as exc:  # noqa: BLE001
            result["exc"] = exc

    last_error = ""
    for attempt in range(retries + 1):
        result = {}
        thread = threading.Thread(target=_target, daemon=True)
        start = _perf()
        thread.start()
        thread.join(timeout=timeout)
        latency_ms = int((_perf() - start) * 1000)
        if thread.is_alive():
            # Timed out — the daemon thread is abandoned; record a timeout error.
            last_error = f"Tool timed out after {timeout}s"
            logger.warning("Tool execution timed out (attempt %s)", attempt + 1)
            continue
        if "value" in result:
            return ToolResult(output=result["value"], success=True, latency_ms=latency_ms)
        # Thread died with an exception — capture it.
        exc = result.get("exc")
        last_error = str(exc) if exc else "Unknown tool error"
        logger.warning("Tool failed (attempt %s): %s", attempt + 1, last_error)
    return ToolResult(output=None, success=False, error=last_error, latency_ms=0)


def _perf():
    import time

    return time.perf_counter()
