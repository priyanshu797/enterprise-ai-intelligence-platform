"""Integration tests for the RAG pipeline (Module B + Challenge 4).

Heavy: requires ChromaDB + an embedding backend. Run with the deps installed.
The LLM is stubbed so no real API keys are needed.
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.heavy

from app.services.llm.base import LLMResponse, LLMUsage
from app.services.rag import get_rag_service


class StubLLMService:
    available = True
    primary = "stub"
    fallback = "none"
    provider_names = ["stub"]
    _providers = []

    def complete(self, messages, **kwargs):
        joined = " ".join(m.get("content", "") for m in messages).lower()
        if "fact-checker" in joined or "judge" in joined:
            return LLMResponse(
                content=json.dumps({"grounding_score": 0.9, "unsupported": False, "has_conflict": False, "conflict_summary": ""}),
                provider="stub", model="stub", usage=LLMUsage(5, 5), latency_ms=1,
            )
        return LLMResponse(
            content="Based on the documents [1], the main cause was resource exhaustion.",
            provider="stub", model="stub-model", usage=LLMUsage(20, 15), latency_ms=1,
        )

    def with_structured_output(self, schema):
        def complete(messages, **kwargs):
            return LLMResponse(content="{}", provider="stub", model="stub", usage=LLMUsage(1, 1), latency_ms=1, parsed=None)
        return complete

    def get_provider(self, name):
        return None

    def estimate_cost(self, *a, **k):
        return 0.0


@pytest.fixture()
def rag_with_data(authed_client):
    """Upload a document so the vector store has content, then return the RAG service."""
    import io
    import time
    r = authed_client.post("/api/v1/documents", files={
        "file": ("failures.txt", io.BytesIO(
            b"System failures report.\n"
            b"1. Database connection pool exhaustion caused an outage. Resolution: connection timeout.\n"
            b"2. Memory leaks from an unbounded cache caused OOM crashes. Resolution: bounded the cache."
        ), "text/plain"),
    })
    assert r.status_code == 201
    doc_id = r.json()["id"]
    for _ in range(30):
        job = authed_client.get(f"/api/v1/documents/{doc_id}/job")
        if job.json()["status"] == "completed":
            break
        time.sleep(0.5)
    rag = get_rag_service()
    rag.llm = StubLLMService()
    return rag, doc_id


def test_retrieval_returns_relevant_chunks(rag_with_data):
    rag, _ = rag_with_data
    outcome = rag.retrieve("What caused the database outage?", top_k=3)
    assert len(outcome.chunks) > 0
    assert any("connection pool" in c.text.lower() for c in outcome.chunks)
    assert outcome.confidence > 0


def test_insufficient_information_refuses(authed_client, rag_with_data):
    from app.database import SessionLocal
    from app.models import User
    rag, _ = rag_with_data
    db = SessionLocal()
    try:
        owner = db.query(User).first()
        resp = rag.answer(db, owner_id=owner.id, query="What is the weather in Tokyo?")
        assert resp.answer_status == "insufficient"
        assert "INSUFFICIENT" in resp.answer.upper()
    finally:
        db.close()


def test_answer_status_answered_when_grounded(rag_with_data):
    from app.database import SessionLocal
    from app.models import User
    rag, _ = rag_with_data
    db = SessionLocal()
    try:
        owner = db.query(User).first()
        resp = rag.answer(db, owner_id=owner.id, query="What caused the database outage?")
        assert resp.answer_status == "answered"
        assert len(resp.sources) > 0
        assert resp.provider == "stub"
    finally:
        db.close()


class _ConflictStubLLM(StubLLMService):
    def complete(self, messages, **kwargs):
        joined = " ".join(m.get("content", "") for m in messages).lower()
        if "fact-checker" in joined:
            return LLMResponse(
                content=json.dumps({"grounding_score": 0.7, "unsupported": False, "has_conflict": True, "conflict_summary": "Docs disagree on the limit: 50000 vs 25000."}),
                provider="stub", model="stub", usage=LLMUsage(5, 5), latency_ms=1,
            )
        return LLMResponse(
            content="Sources conflict: one says 50000 USD, another says 25000 USD. The current spec is 50000.",
            provider="stub", model="stub-model", usage=LLMUsage(20, 15), latency_ms=1,
        )


def test_conflicting_status_detected(authed_client):
    """Upload two conflicting docs and verify the conflict is flagged."""
    import io
    import time
    for name, content in [
        ("a.txt", b"The max transaction value is 50000 USD."),
        ("b.txt", b"The max transaction value is 25000 USD per payment."),
    ]:
        r = authed_client.post("/api/v1/documents", files={"file": (name, io.BytesIO(content), "text/plain")})
        assert r.status_code == 201
        doc_id = r.json()["id"]
        for _ in range(20):
            if authed_client.get(f"/api/v1/documents/{doc_id}/job").json()["status"] == "completed":
                break
            time.sleep(0.5)

    from app.database import SessionLocal
    from app.models import User
    rag = get_rag_service()
    rag.llm = _ConflictStubLLM()
    db = SessionLocal()
    try:
        owner = db.query(User).first()
        resp = rag.answer(db, owner_id=owner.id, query="What is the maximum transaction value?")
        assert resp.answer_status == "conflicting"
    finally:
        db.close()
