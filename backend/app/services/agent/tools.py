"""Agent tools (Module D) with parameter validation + timeouts (Challenge 7).

Four tools the agent can choose from based on the user's question:

  1. knowledge_base_search  — semantic/hybrid search over the user's uploaded
                              documents (RAG retrieve step).
  2. web_search             — real-time web search via SerpAPI when the KB has
                              no relevant results or external/current data is
                              needed.
  3. analyze_document       — deep LLM analysis of a single document to answer
                              a focused question.
  4. generate_report        — synthesise gathered findings into a structured
                              Markdown report (always the last step).

Each tool is a LangChain ``BaseTool`` with a Pydantic ``args_schema`` so the
agent's tool calls are validated before execution.  Tools are constructed with
an ``AgentContext`` that carries the DB session, owner id, and shared services,
so they operate within the request's auth scope.

Tool-selection logic is embedded in the agent system prompt (see
``templates.py``), which instructs the LLM to:
  - Use ``knowledge_base_search`` first for any question.
  - Fall back to ``web_search`` when the KB yields no relevant results or the
    question requires real-time / external data.
  - Use ``analyze_document`` after KB search when deeper analysis of a specific
    document is needed.
  - Always finish complex multi-step answers with ``generate_report``.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Type

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...config import settings
from ...models import Document, DocumentStatus
from ..rag import get_rag_service
from .reliability import with_timeout_and_retry

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Runtime context shared by all tools in an agent run."""
    db: Session
    owner_id: str
    run_id: str | None = None  # AgentRun id for step recording


# ─────────────────────────────────────────────── Tool 1: Knowledge Base Search
class KBSearchArgs(BaseModel):
    query: str = Field(..., description="The search query to find relevant knowledge-base passages.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of passages to return.")
    file_type: str | None = Field(default=None, description="Optional filter: pdf, txt, md, csv.")


class KnowledgeBaseSearchTool(BaseTool):
    """Tool 1 — retrieve relevant passages from the user's knowledge base.

    Decision rule: use this tool FIRST for any question.  If it returns
    ``"found": false`` or very low-scoring results, fall back to
    ``web_search``.
    """
    name: str = "knowledge_base_search"
    description: str = (
        "Search the enterprise knowledge base for passages relevant to a "
        "query. Returns ranked source passages with document references. "
        "ALWAYS use this tool first to gather evidence from the user's own "
        "documents before considering other sources."
    )
    args_schema: Type[BaseModel] = KBSearchArgs
    ctx: Any = None

    def _run(self, query: str, top_k: int = 5, file_type: str | None = None) -> str:
        rag = get_rag_service()
        filters = {"file_type": file_type} if file_type else None
        outcome = rag.retrieve(query, top_k=top_k, filters=filters, owner_id=self.ctx.owner_id)
        if not outcome.chunks:
            return json.dumps({"found": False, "message": "No relevant passages found in the knowledge base."})
        passages = [
            {
                "doc": c.document_filename,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "score": round(c.score, 3),
                "text": c.text[:500],
            }
            for c in outcome.chunks
        ]
        return json.dumps({"found": True, "count": len(passages), "passages": passages})


# ──────────────────────────────────────────────────────── Tool 2: Web Search
class WebSearchArgs(BaseModel):
    query: str = Field(..., description="The web search query to find external or real-time information.")
    num_results: int = Field(default=5, ge=1, le=10, description="Number of search results to return.")


class WebSearchTool(BaseTool):
    """Tool 2 — real-time web search via SerpAPI.

    Decision rule: use this tool when:
      - ``knowledge_base_search`` returned no relevant results.
      - The question requires current/external information not in the KB
        (e.g. market data, news, public documentation, current events).
      - The user explicitly asks to search the web.

    Requires ``SERPAPI_API_KEY`` to be set in the backend .env.
    """
    name: str = "web_search"
    description: str = (
        "Search the web for current, real-time, or external information that "
        "is not available in the enterprise knowledge base. Use this tool when "
        "knowledge_base_search returns no relevant results, or when the "
        "question explicitly needs up-to-date external data such as news, "
        "market information, or public documentation."
    )
    args_schema: Type[BaseModel] = WebSearchArgs
    ctx: Any = None

    def _run(self, query: str, num_results: int = 5) -> str:
        api_key = settings.SERPAPI_API_KEY
        if not api_key:
            return json.dumps({
                "found": False,
                "message": "Web search is not configured. Set SERPAPI_API_KEY in backend/.env.",
            })

        params = {
            "q": query,
            "num": num_results,
            "api_key": api_key,
            "engine": "google",
        }

        try:
            # Use httpx (already a project dependency) rather than the
            # google-search-results SDK so we control timeouts.
            with httpx.Client(timeout=20.0) as client:
                response = client.get(settings.SERPAPI_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            return json.dumps({"found": False, "error": f"SerpAPI HTTP error: {exc.response.status_code}"})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"found": False, "error": f"Web search failed: {exc}"})

        organic = data.get("organic_results", [])[:num_results]
        if not organic:
            return json.dumps({"found": False, "message": "No web results found."})

        results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("link", ""),
                "position": r.get("position", i + 1),
            }
            for i, r in enumerate(organic)
        ]
        return json.dumps({"found": True, "count": len(results), "results": results})


# ──────────────────────────────────────────────────── Tool 3: Document Analysis
class AnalyzeDocumentArgs(BaseModel):
    document_id: str = Field(..., description="The id of the document to analyze.")
    question: str = Field(
        ...,
        description="The analytical question to answer about the document, e.g. 'identify the main failure reasons'.",
    )


class DocumentAnalysisTool(BaseTool):
    """Tool 3 — deeper analysis of a single document via the LLM.

    Decision rule: use this tool AFTER ``knowledge_base_search`` when a
    specific document (identified by its ``document_id`` from the KB search
    results) needs deeper investigation to answer the user's question.
    """
    name: str = "analyze_document"
    description: str = (
        "Analyze a specific document to answer a focused question about its "
        "content (e.g. extract failure reasons, summarize risks, find "
        "requirements). Provide a document_id and a question. Use after "
        "knowledge_base_search when you need to dig deeper into a specific source."
    )
    args_schema: Type[BaseModel] = AnalyzeDocumentArgs
    ctx: Any = None

    def _run(self, document_id: str, question: str) -> str:
        from ..llm.service import get_llm_service

        doc = self.ctx.db.get(Document, document_id)
        if not doc or doc.owner_id != self.ctx.owner_id:
            return json.dumps({"error": "Document not found or not accessible."})
        if doc.status != DocumentStatus.COMPLETED:
            return json.dumps({"error": f"Document is not ready (status={doc.status.value})."})

        chunks = doc.chunks
        if not chunks:
            return json.dumps({"error": "Document has no processed content."})
        full_text = "\n\n".join(c.text for c in chunks)[:8000]

        llm = get_llm_service()
        if not llm.available:
            return json.dumps({"error": "No LLM provider available for analysis."})
        messages = [
            {"role": "system", "content": "You analyze documents and extract structured insights. Be concise and cite the document sections you used."},
            {"role": "user", "content": f"Document: {doc.filename}\n\nContent:\n{full_text}\n\nQuestion: {question}"},
        ]
        try:
            resp = llm.complete(messages, max_tokens=settings.AGENT_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"LLM analysis failed: {exc}"})
        return json.dumps({
            "document_id": document_id,
            "filename": doc.filename,
            "analysis": resp.content,
            "tokens": resp.usage.total_tokens,
        })


# ──────────────────────────────────────────────── Tool 4: Report Generation
class GenerateReportArgs(BaseModel):
    topic: str = Field(..., description="The report title/topic.")
    findings: str = Field(..., description="The findings text to turn into a report (from prior tool calls).")
    sources: str = Field(default="", description="Optional source references to cite (document names or URLs).")


class ReportGenerationTool(BaseTool):
    """Tool 4 — synthesise a structured Markdown report from gathered findings.

    Decision rule: use this tool LAST after you have gathered evidence via
    ``knowledge_base_search``, ``web_search``, and/or ``analyze_document``.
    Always use it when the user explicitly asks for a report, summary, or
    structured output.
    """
    name: str = "generate_report"
    description: str = (
        "Generate a well-structured Markdown report from findings gathered by "
        "previous tool calls. Provide a topic, the findings, and optional "
        "sources (document names or URLs). Use this LAST after gathering and "
        "analyzing evidence. This produces the final deliverable for the user."
    )
    args_schema: Type[BaseModel] = GenerateReportArgs
    ctx: Any = None

    def _run(self, topic: str, findings: str, sources: str = "") -> str:
        from ..llm.service import get_llm_service
        from ..llm.templates import build_report_prompt

        llm = get_llm_service()
        if not llm.available:
            return json.dumps({"error": "No LLM provider available for report generation."})
        messages = build_report_prompt(topic, findings, sources or "(no sources provided)")
        try:
            resp = llm.complete(messages, max_tokens=settings.AGENT_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"Report generation failed: {exc}"})
        return json.dumps({
            "topic": topic,
            "report_markdown": resp.content,
            "tokens": resp.usage.total_tokens,
        })


# ───────────────────────────────────────────────────────────────── Factory
def build_tools(ctx: AgentContext) -> list[BaseTool]:
    """Instantiate all 4 tools bound to a runtime context."""
    tools = []
    for cls in (
        KnowledgeBaseSearchTool,
        WebSearchTool,
        DocumentAnalysisTool,
        ReportGenerationTool,
    ):
        tool = cls()
        tool.ctx = ctx
        tools.append(tool)
    return tools


# Used by build_agent_system_prompt() to enumerate tools for the LLM.
TOOL_DESCRIPTIONS = {
    "knowledge_base_search": (
        "Search the user's enterprise knowledge base (uploaded documents) for "
        "relevant passages. Use this FIRST."
    ),
    "web_search": (
        "Search the web via SerpAPI for current or external information. Use "
        "when the knowledge base has no relevant results or the question needs "
        "real-time data."
    ),
    "analyze_document": (
        "Deeply analyze a single document (by document_id) to answer a focused "
        "question. Use AFTER knowledge_base_search when a specific source needs "
        "deeper investigation."
    ),
    "generate_report": (
        "Synthesise gathered evidence into a structured Markdown report. Use "
        "LAST, after all evidence has been gathered and analyzed."
    ),
}
