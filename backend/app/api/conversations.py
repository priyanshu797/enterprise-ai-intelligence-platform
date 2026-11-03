"""Conversation history endpoints (Module E + F)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Conversation, User
from ..schemas import ConversationOut, MessageOut
from ..security.auth import get_current_active_user

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    return db.execute(
        select(Conversation).where(Conversation.owner_id == user.id).order_by(Conversation.updated_at.desc())
    ).scalars().all()


@router.get("/{conversation_id}", response_model=list[MessageOut])
def get_messages(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    conv = db.get(Conversation, conversation_id)
    if not conv or conv.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv.messages


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    conv = db.get(Conversation, conversation_id)
    if not conv or conv.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return None
