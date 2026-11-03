"""Conversational memory (Module E).

Responsibilities:
  * Persist user-specific conversation history in the DB (Message table).
  * Provide a sliding window of recent messages to the LLM (don't send the
    entire history — that wastes tokens, violating the "avoid unnecessary
    historical information" requirement).
  * Detect follow-up questions (pronouns/ellipses like "tell me more about
    that", "what about the second one?") and rewrite them into a standalone
    question using the conversation window, so retrieval gets the right query.
  * Summarise long histories into a short context block when needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Conversation, Message, MessageRole, User
from .llm.base import ProviderError
from .llm.service import get_llm_service

logger = logging.getLogger(__name__)

# Simple heuristic indicators that a question depends on prior context.
_FOLLOWUP_HINTS = (
    "it", "that", "this", "those", "the first", "the second", "the last",
    "more about", "elaborate", "what about", "how about", "and the", "summarize it",
    "explain further", "in detail",
)


@dataclass
class ConversationContext:
    messages: list[dict[str, str]]  # recent window for the LLM
    standalone_query: str            # rewritten query for retrieval
    is_followup: bool


def is_followup_question(query: str) -> bool:
    """Heuristic: short questions referencing prior context are follow-ups."""
    q = query.lower().strip()
    if len(q.split()) <= 8:
        return True
    return any(hint in q for hint in _FOLLOWUP_HINTS)


def get_or_create_conversation(
    db: Session, owner_id: str, conversation_id: str | None, title: str | None = None
) -> Conversation:
    if conversation_id:
        conv = db.get(Conversation, conversation_id)
        if conv and conv.owner_id == owner_id:
            return conv
    conv = Conversation(owner_id=owner_id, title=title or _derive_title(None))
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def _derive_title(query: str | None) -> str:
    if query:
        words = query.split()[:6]
        return " ".join(words) + ("..." if len(query.split()) > 6 else "")
    return "New conversation"


def load_window(db: Session, conversation_id: str, max_messages: int | None = None) -> list[Message]:
    """Load the most recent N messages (sliding window)."""
    limit = max_messages or settings.MEMORY_MAX_MESSAGES
    rows = db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    ).scalars().all()
    return list(reversed(rows))


def build_context_window(messages: list[Message]) -> list[dict[str, str]]:
    """Turn recent Message rows into role/content dicts for the LLM."""
    window: list[dict[str, str]] = []
    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        window.append({"role": role, "content": m.content})
    return window


def add_message(
    db: Session,
    conversation_id: str,
    role: MessageRole,
    content: str,
    *,
    sources: list[dict] | None = None,
    provider: str = "",
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
) -> Message:
    import json

    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources_json=json.dumps(sources or []),
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )
    db.add(msg)
    # Bump conversation updated_at + title.
    conv = db.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = _now()
        if role == MessageRole.USER and conv.title in ("New conversation", "New conversation..."):
            conv.title = _derive_title(content)
    db.commit()
    db.refresh(msg)
    return msg


def rewrite_followup_query(
    query: str, window: list[dict[str, str]]
) -> str:
    """Use the LLM to rewrite a follow-up into a standalone question.

    If the LLM is unavailable we fall back to the original query so the system
    never blocks on this optimisation.
    """
    if not window or not is_followup_question(query):
        return query
    llm = get_llm_service()
    if not llm.available:
        return query
    history = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in window[-4:])
    messages = [
        {
            "role": "system",
            "content": (
                "Given the chat history and a follow-up question, rewrite the "
                "follow-up as a complete, standalone question that can be "
                "understood without the history. Do NOT answer the question. "
                "Output ONLY the rewritten question, nothing else."
            ),
        },
        {"role": "user", "content": f"Chat history:\n{history}\n\nFollow-up question: {query}\n\nStandalone question:"},
    ]
    try:
        resp = llm.complete(messages, max_tokens=120)
        rewritten = resp.content.strip().strip('"').strip("'")
        if rewritten and len(rewritten) < len(query) * 4:
            logger.info("Rewrote follow-up: '%s' -> '%s'", query, rewritten)
            return rewritten
    except ProviderError as exc:
        logger.warning("Follow-up rewrite failed, using original query: %s", exc)
    return query


def build_context_block(window: list[dict[str, str]]) -> str:
    """Render a compact summary of the conversation window for the RAG prompt.

    Only prior turns are included so the current question isn't duplicated."""
    if not window:
        return ""
    lines = ["Previous conversation (for context):"]
    for m in window[-6:]:
        lines.append(f"- {m['role']}: {m['content'][:300]}")
    return "\n".join(lines)


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
