"""Pydantic request/response schemas (API contract).

These double as the OpenAPI/Swagger schema that FastAPI auto-generates,
satisfying the API documentation deliverable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------------------------------------------------------------- Auth
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=255)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: EmailStr
    full_name: str
    is_active: bool
    is_superuser: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------- Documents
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    file_type: str
    file_size: int
    title: str
    status: str
    error_message: str
    num_chunks: int
    content_hash: str
    created_at: datetime
    processed_at: datetime | None


class DocumentChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    char_count: int


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    document_id: str
    status: str
    progress: float
    error_message: str
    started_at: datetime | None
    completed_at: datetime | None


# ---------------------------------------------------------------- Chat / RAG
class SourceRef(BaseModel):
    document_id: str
    document_filename: str
    chunk_index: int
    text: str
    score: float


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    # Optional metadata filters, e.g. {"file_type": "pdf"}.
    filters: dict[str, Any] | None = None
    # Optionally select a provider for this request.
    provider: Literal["openai", "anthropic", "auto"] = "auto"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    conversation_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cached: bool = False
    # Whether retrieval had enough signal to answer confidently.
    confidence: float = 0.0
    # "answered" | "insufficient" | "conflicting" | "unsupported"
    answer_status: Literal["answered", "insufficient", "conflicting", "unsupported"] = "answered"


# ---------------------------------------------------------------- Conversations
class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    role: str
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    created_at: datetime


# ---------------------------------------------------------------- Agent
class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    # Force a specific tool sequence (for testing); normally the agent decides.
    force_tools: list[str] | None = None


class AgentStepOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    step_index: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    success: bool
    error_message: str
    latency_ms: int

    @classmethod
    def _parse_tool_input(cls, v: Any) -> dict[str, Any]:
        """Convert JSON string stored in DB back to a dict."""
        if isinstance(v, str):
            import json as _json
            try:
                return _json.loads(v)
            except Exception:
                return {"raw": v}
        if isinstance(v, dict):
            return v
        return {}

    @classmethod
    def from_orm_step(cls, step) -> "AgentStepOut":
        """Build from a SQLAlchemy AgentStep ORM object."""
        import json as _json
        raw_input = step.tool_input or "{}"
        if isinstance(raw_input, str):
            try:
                parsed_input = _json.loads(raw_input)
            except Exception:
                parsed_input = {"raw": raw_input}
        else:
            parsed_input = raw_input if isinstance(raw_input, dict) else {}
        return cls(
            step_index=step.step_index,
            tool_name=step.tool_name,
            tool_input=parsed_input,
            tool_output=step.tool_output or "",
            success=step.success,
            error_message=step.error_message or "",
            latency_ms=step.latency_ms or 0,
        )


class AgentRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    query: str
    status: str
    final_answer: str
    total_tokens: int
    latency_ms: int
    error_message: str
    created_at: datetime
    completed_at: datetime | None
    steps: list[AgentStepOut] = Field(default_factory=list)

    @classmethod
    def from_orm(cls, obj) -> "AgentRunOut":
        """Build from a SQLAlchemy ORM AgentRun, converting JSON-string tool_input in steps."""
        import json as _json
        orm_steps = getattr(obj, "steps", []) or []
        parsed_steps = []
        for s in orm_steps:
            raw_input = s.tool_input or "{}"
            if isinstance(raw_input, str):
                try:
                    parsed_input = _json.loads(raw_input)
                except Exception:
                    parsed_input = {"raw": raw_input}
            else:
                parsed_input = raw_input if isinstance(raw_input, dict) else {}
            parsed_steps.append(AgentStepOut(
                step_index=s.step_index,
                tool_name=s.tool_name,
                tool_input=parsed_input,
                tool_output=s.tool_output or "",
                success=s.success,
                error_message=s.error_message or "",
                latency_ms=s.latency_ms or 0,
            ))
        return cls(
            id=obj.id,
            query=obj.query or "",
            status=obj.status or "",
            final_answer=obj.final_answer or "",
            total_tokens=obj.total_tokens or 0,
            latency_ms=obj.latency_ms or 0,
            error_message=obj.error_message or "",
            created_at=obj.created_at,
            completed_at=obj.completed_at,
            steps=parsed_steps,
        )


# ---------------------------------------------------------------- Stats / Dashboard
class DashboardStats(BaseModel):
    total_documents: int
    total_chunks: int
    total_conversations: int
    total_messages: int
    total_agent_runs: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    cache_hits: int
    avg_latency_ms: float
    documents_by_status: dict[str, int]
    tokens_by_provider: dict[str, int]
    recent_activity: list[dict[str, Any]]


class UsageLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    endpoint: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    cache_hit: bool
    created_at: datetime


class ActivityLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    owner_id: str | None
    action: str
    detail: str
    status: str
    created_at: datetime


class HealthOut(BaseModel):
    status: str
    version: str
    environment: str
    database: str
    vector_store: str
    llm_primary: str
    llm_fallback: str


# ---------------------------------------------------------------- Evaluation
class EvalQuestion(BaseModel):
    id: str
    question: str
    expected_answer: str
    # Reference chunks/text the answer should be grounded in.
    reference_context: str = ""
    # Expected behaviour: answered | insufficient (no relevant docs) | conflicting.
    expected_behaviour: Literal["answered", "insufficient", "conflicting"] = "answered"
    tags: list[str] = Field(default_factory=list)


class EvalRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    num_questions: int
    provider: str
    model: str
    started_at: datetime
    completed_at: datetime | None
    metrics: dict[str, Any] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)


class EvalResultOut(BaseModel):
    question: str
    generated_answer: str
    retrieval_relevance: float
    answer_correctness: float
    source_grounding: float
    hallucination_flag: bool
    latency_ms: int
    tokens: int
    sources: list[dict[str, Any]]
    notes: str
