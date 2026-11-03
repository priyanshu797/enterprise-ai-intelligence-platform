"""Agent execution endpoint (Module D)."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from ..database import get_db
from ..models import AgentRun, AgentStep, User
from ..schemas import AgentRequest, AgentRunOut, AgentStepOut
from ..security.auth import get_current_active_user
from ..services.agent import AgentContext, get_agent_executor
from ..services.llm.base import ProviderError
from ..services.usage import log_activity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def _serialize_run(run: AgentRun, db: Session) -> AgentRunOut:
    """Eagerly load and serialize an AgentRun with all its steps.

    We do this manually to avoid lazy-loading issues when the session closes
    during Pydantic serialization.
    """
    # Reload with steps eagerly loaded so we never hit a detached-instance error.
    fresh = (
        db.query(AgentRun)
        .options(joinedload(AgentRun.steps))
        .filter(AgentRun.id == run.id)
        .first()
    )
    if not fresh:
        fresh = run

    steps = []
    for s in sorted(getattr(fresh, "steps", []) or [], key=lambda x: x.step_index):
        raw_input = s.tool_input or "{}"
        if isinstance(raw_input, str):
            try:
                parsed_input = json.loads(raw_input)
            except Exception:
                parsed_input = {"raw": raw_input}
        else:
            parsed_input = raw_input if isinstance(raw_input, dict) else {}

        steps.append(AgentStepOut(
            step_index=s.step_index,
            tool_name=s.tool_name or "",
            tool_input=parsed_input,
            tool_output=str(s.tool_output or ""),
            success=bool(s.success),
            error_message=str(s.error_message or ""),
            latency_ms=int(s.latency_ms or 0),
        ))

    return AgentRunOut(
        id=str(fresh.id),
        query=str(fresh.query or ""),
        status=str(fresh.status or ""),
        final_answer=str(fresh.final_answer or ""),
        total_tokens=int(fresh.total_tokens or 0),
        latency_ms=int(fresh.latency_ms or 0),
        error_message=str(fresh.error_message or ""),
        created_at=fresh.created_at,
        completed_at=fresh.completed_at,
        steps=steps,
    )


@router.post("/run", response_model=AgentRunOut)
def run_agent(
    payload: AgentRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    from ..services.llm.service import get_llm_service

    if not get_llm_service().available:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider configured. Set GEMINI_API_KEY in backend/.env",
        )

    ctx = AgentContext(db=db, owner_id=user.id)
    try:
        executor = get_agent_executor(ctx)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to build agent executor")
        raise HTTPException(status_code=500, detail=f"Agent init failed: {exc}")

    try:
        run = executor.run(payload.query, conversation_id=payload.conversation_id)
    except Exception as exc:
        logger.exception("Agent execution error")
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")

    try:
        log_activity(
            db, owner_id=user.id, action="agent_run",
            detail=payload.query[:100], status=run.status,
        )
        return _serialize_run(run, db)
    except Exception as exc:
        logger.exception("Failed to serialize agent run")
        raise HTTPException(status_code=500, detail=f"Serialization error: {exc}")


@router.get("/runs", response_model=list[AgentRunOut])
def list_runs(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
    limit: int = 20,
):
    runs = (
        db.query(AgentRun)
        .options(joinedload(AgentRun.steps))
        .filter(AgentRun.owner_id == user.id)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_serialize_run(r, db) for r in runs]


@router.get("/runs/{run_id}", response_model=AgentRunOut)
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    run = (
        db.query(AgentRun)
        .options(joinedload(AgentRun.steps))
        .filter(AgentRun.id == run_id, AgentRun.owner_id == user.id)
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return _serialize_run(run, db)
