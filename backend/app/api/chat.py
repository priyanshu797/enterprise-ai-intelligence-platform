"""Chat / RAG endpoint (Module B)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas import ChatRequest, ChatResponse
from ..security.auth import get_current_active_user
from ..services.llm.base import ProviderError
from ..services.rag import get_rag_service
from ..services.usage import log_activity

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    rag = get_rag_service()
    if not rag.llm.available:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider configured. Set GROQ_API_KEY in backend/.env",
        )
    try:
        response = rag.answer(
            db,
            owner_id=user.id,
            query=payload.query,
            conversation_id=payload.conversation_id,
            filters=payload.filters,
            provider=payload.provider,
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"All LLM providers failed: {exc}. Check your GROQ_API_KEY in backend/.env",
        )
    log_activity(db, owner_id=user.id, action="chat", detail=payload.query[:100], status=response.answer_status)
    return response
