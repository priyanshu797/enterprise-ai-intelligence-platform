"""Usage logging: persists token consumption, cost, latency, and cache hits.

Backs Module C (token tracking), Module F (token usage / latency monitoring),
and Challenge 5 (cost optimisation). Cheap DB writes; designed to be called
from the RAG/agent services after each LLM call.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from ..models import ActivityLog, UsageLog
from .llm.base import LLMResponse, LLMUsage


def log_usage(
    db: Session,
    *,
    owner_id: str | None,
    endpoint: str,
    response: LLMResponse | None = None,
    usage: LLMUsage | None = None,
    provider: str = "",
    model: str = "",
    latency_ms: int = 0,
    cache_hit: bool = False,
) -> UsageLog:
    """Persist a single usage record."""
    if response is not None:
        usage = response.usage
        provider = provider or response.provider
        model = model or response.model
    elif usage is None:
        usage = LLMUsage()
    record = UsageLog(
        owner_id=owner_id,
        endpoint=endpoint,
        provider=provider,
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=_estimate_cost(provider, model, usage),
        latency_ms=latency_ms,
        cache_hit=cache_hit,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def log_activity(
    db: Session,
    *,
    owner_id: str | None,
    action: str,
    detail: str = "",
    status: str = "success",
    ip_address: str = "",
) -> ActivityLog:
    record = ActivityLog(
        owner_id=owner_id,
        action=action,
        detail=detail[:2000],
        status=status,
        ip_address=ip_address,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _estimate_cost(provider: str, model: str, usage: LLMUsage) -> float:
    rates = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-70b-versatile": (0.59, 0.79),
        "llama3-70b-8192": (0.59, 0.79),
        "llama3-8b-8192": (0.05, 0.08),
    }
    rate_in, rate_out = rates.get(model, rates["gpt-4o-mini"])
    return round(
        (usage.prompt_tokens / 1_000_000) * rate_in
        + (usage.completion_tokens / 1_000_000) * rate_out,
        6,
    )


def aggregate_stats(db: Session, owner_id: str | None = None) -> dict[str, Any]:
    """Compute dashboard usage aggregates."""
    base = select(UsageLog)
    if owner_id is not None:
        base = base.where(UsageLog.owner_id == owner_id)
    rows = db.execute(base).scalars().all()

    total_tokens = sum(r.total_tokens for r in rows)
    prompt_tokens = sum(r.prompt_tokens for r in rows)
    completion_tokens = sum(r.completion_tokens for r in rows)
    total_cost = round(sum(r.cost_usd for r in rows), 4)
    cache_hits = sum(1 for r in rows if r.cache_hit)
    latencies = [r.latency_ms for r in rows if r.latency_ms > 0]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    by_provider: dict[str, int] = {}
    for r in rows:
        by_provider[r.provider] = by_provider.get(r.provider, 0) + r.total_tokens

    return {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost_usd": total_cost,
        "cache_hits": cache_hits,
        "avg_latency_ms": avg_latency,
        "tokens_by_provider": by_provider,
    }
