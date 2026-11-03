"""AI Agent (Module D) with reliability guardrails (Challenge 7).

Tools:
  1. knowledge_base_search   — RAG retrieval over the user's documents.
  2. analyze_document         — deeper analysis of a specific document.
  3. generate_report           — structured Markdown report synthesis.

The agent is implemented as a LangGraph ReAct loop with:
  * per-step parameter validation (Pydantic args schemas),
  * per-tool execution timeouts,
  * a hard iteration cap (settings.AGENT_MAX_ITERATIONS),
  * failure recovery (tool errors are fed back to the model so it can adapt),
  * full step-by-step execution recording (AgentStep rows) for the dashboard.
"""
from __future__ import annotations

from .agent import AgentExecutor, get_agent_executor
from .tools import (
    AgentContext,
    KnowledgeBaseSearchTool,
    DocumentAnalysisTool,
    ReportGenerationTool,
    build_tools,
)

__all__ = [
    "AgentExecutor",
    "get_agent_executor",
    "AgentContext",
    "KnowledgeBaseSearchTool",
    "DocumentAnalysisTool",
    "ReportGenerationTool",
    "build_tools",
]
