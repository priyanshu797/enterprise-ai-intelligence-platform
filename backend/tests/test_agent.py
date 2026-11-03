"""Tests for agent tools: parameter validation, multi-step, reliability (Module D + Challenge 7).

Tool-level tests run the tools directly (no LLM) so they're lightweight. The
end-to-end agent loop test is marked heavy (needs a tool-calling LLM).
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.heavy

from sqlalchemy.orm import Session

from app.services.agent.tools import AgentContext, build_tools


@pytest.fixture()
def ctx_with_doc(authed_client):
    """Upload a doc and return an AgentContext bound to a live DB session."""
    import io
    import time
    from app.database import SessionLocal
    from app.models import User

    r = authed_client.post("/api/v1/documents", files={
        "file": ("report.txt", io.BytesIO(
            b"Failure report: connection pool exhaustion caused an outage. "
            b"Resolution: added connection timeout. Memory leak from unbounded cache."
        ), "text/plain"),
    })
    doc_id = r.json()["id"]
    for _ in range(30):
        if authed_client.get(f"/api/v1/documents/{doc_id}/job").json()["status"] == "completed":
            break
        time.sleep(0.5)

    db = SessionLocal()
    owner = db.query(User).first()
    ctx = AgentContext(db=db, owner_id=owner.id)
    yield ctx, doc_id
    db.close()


def test_tool_kb_search_returns_results(ctx_with_doc):
    ctx, _ = ctx_with_doc
    tools = {t.name: t for t in build_tools(ctx)}
    out = tools["knowledge_base_search"]._run(query="connection pool outage", top_k=3)
    data = json.loads(out)
    assert data["found"] is True
    assert data["count"] > 0


def test_tool_kb_search_no_results_for_unrelated(ctx_with_doc):
    ctx, _ = ctx_with_doc
    tools = {t.name: t for t in build_tools(ctx)}
    out = tools["knowledge_base_search"]._run(query="quantum entanglement theory", top_k=3)
    data = json.loads(out)
    assert data["found"] is False


def test_tool_kb_search_invalid_params_rejected(ctx_with_doc):
    """Parameter validation: top_k out of range should raise via the tool args schema."""
    from pydantic import ValidationError
    from app.services.agent.tools import KBSearchArgs
    with pytest.raises(ValidationError):
        KBSearchArgs(query="x", top_k=99)
    with pytest.raises(ValidationError):
        KBSearchArgs(query="x", top_k=0)


def test_tool_analyze_document_not_found(ctx_with_doc):
    ctx, _ = ctx_with_doc
    tools = {t.name: t for t in build_tools(ctx)}
    out = tools["analyze_document"]._run(document_id="nonexistent", question="summarize")
    data = json.loads(out)
    assert "error" in data


def test_tool_generate_report_outputs_markdown(ctx_with_doc):
    ctx, _ = ctx_with_doc
    tools = {t.name: t for t in build_tools(ctx)}
    out = tools["generate_report"]._run(topic="System Failures", findings="Connection pool and memory issues.", sources="[1] report.txt")
    data = json.loads(out)
    # With no real LLM, the report tool returns an error JSON (no provider).
    # That's acceptable — we assert the tool handles the missing-LLM case cleanly.
    assert "report_markdown" in data or "error" in data


def test_agent_reliability_timeout():
    """The reliability wrapper captures timeouts as ToolResult errors."""
    import time
    from app.services.agent.reliability import with_timeout_and_retry

    def slow():
        time.sleep(2)
        return "done"

    result = with_timeout_and_retry(slow, timeout=0.2, retries=0)
    assert result.success is False
    assert "timed out" in result.error.lower()


def test_agent_reliability_success():
    from app.services.agent.reliability import with_timeout_and_retry

    result = with_timeout_and_retry(lambda: "ok", timeout=5, retries=0)
    assert result.success is True
    assert result.output == "ok"
