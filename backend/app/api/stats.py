"""Dashboard, usage, activity, and health endpoints (Module F)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import (
    ActivityLog,
    AgentRun,
    Conversation,
    Document,
    DocumentChunk,
    Message,
    UsageLog,
    User,
)
from ..schemas import ActivityLogOut, DashboardStats, HealthOut, UsageLogOut
from ..security.auth import get_current_active_user
from ..services.usage import aggregate_stats

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/dashboard", response_model=DashboardStats)
def dashboard(db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    owner = user.id
    total_docs = db.execute(select(func.count()).select_from(Document).where(Document.owner_id == owner)).scalar() or 0
    total_chunks = db.execute(
        select(func.count(DocumentChunk.id)).join(Document).where(Document.owner_id == owner)
    ).scalar() or 0
    total_convs = db.execute(select(func.count()).select_from(Conversation).where(Conversation.owner_id == owner)).scalar() or 0
    total_msgs = db.execute(
        select(func.count(Message.id)).join(Conversation).where(Conversation.owner_id == owner)
    ).scalar() or 0
    total_agent = db.execute(select(func.count()).select_from(AgentRun).where(AgentRun.owner_id == owner)).scalar() or 0

    docs_by_status = {}
    rows = db.execute(
        select(Document.status, func.count()).where(Document.owner_id == owner).group_by(Document.status)
    )
    for status_val, count in rows:
        docs_by_status[str(status_val)] = count

    usage = aggregate_stats(db, owner_id=owner)

    recent = db.execute(
        select(ActivityLog).where(ActivityLog.owner_id == owner).order_by(ActivityLog.created_at.desc()).limit(10)
    ).scalars().all()
    recent_activity = [
        {
            "id": a.id,
            "action": a.action,
            "detail": a.detail,
            "status": a.status,
            "created_at": a.created_at.isoformat(),
        }
        for a in recent
    ]
    return DashboardStats(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_conversations=total_convs,
        total_messages=total_msgs,
        total_agent_runs=total_agent,
        total_tokens=usage["total_tokens"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_cost_usd=usage["total_cost_usd"],
        cache_hits=usage["cache_hits"],
        avg_latency_ms=usage["avg_latency_ms"],
        documents_by_status=docs_by_status,
        tokens_by_provider=usage["tokens_by_provider"],
        recent_activity=recent_activity,
    )


@router.get("/usage", response_model=list[UsageLogOut])
def usage_logs(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
    limit: int = Query(50, le=200),
):
    return db.execute(
        select(UsageLog).where(UsageLog.owner_id == user.id).order_by(UsageLog.created_at.desc()).limit(limit)
    ).scalars().all()


@router.get("/activity", response_model=list[ActivityLogOut])
def activity_logs(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
    limit: int = Query(50, le=200),
):
    return db.execute(
        select(ActivityLog).where(ActivityLog.owner_id == user.id).order_by(ActivityLog.created_at.desc()).limit(limit)
    ).scalars().all()


@router.get("/health", response_model=HealthOut)
def health():
    from ..services.llm.service import get_llm_service
    svc = get_llm_service()
    return HealthOut(
        status="ok",
        version="1.0.0",
        environment=settings.ENVIRONMENT,
        database="postgresql" if settings.DATABASE_URL.startswith("postgres") else "sqlite",
        vector_store=settings.VECTOR_STORE_TYPE,
        llm_primary=svc.primary,
        llm_fallback=svc.fallback,
    )
