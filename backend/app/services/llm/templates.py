"""Prompt templates for RAG, agent, and report generation (Module C).

Centralising templates keeps prompts consistent, reviewable, and tunable.
"""
from __future__ import annotations

from ...security.injection import build_guardrail_instructions


def build_rag_prompt(question: str, context_text: str, conversation_context: str = "") -> list[dict[str, str]]:
    """Build the message list for a grounded RAG answer."""
    system = (
        "You are an enterprise knowledge assistant. Your job is to provide thorough, well-structured answers.\n\n"
        "INSTRUCTIONS:\n"
        "1. Answer ONLY using the information inside <retrieved_context>. Do not add outside knowledge.\n"
        "2. Write a COMPLETE answer — do not cut it short. Cover all relevant aspects found in the context.\n"
        "3. You MUST structure EVERY response using exactly these four sections in order, each on its own line:\n\n"
        "   ## Description\n"
        "   A clear, concise paragraph that introduces and explains the topic or directly answers the question.\n\n"
        "   ## Highlighted Points\n"
        "   The most important facts, findings, or details from the retrieved documents as a bullet list.\n"
        "   - Use **bold** for key terms inside bullets.\n"
        "   - Cite sources inline as [doc: filename] on the relevant bullet.\n\n"
        "   ## Conclusion\n"
        "   A short paragraph (2–4 sentences) that summarises the overall answer and its significance.\n\n"
        "   ## Suggestions\n"
        "   Practical next steps, recommendations, or follow-up actions based on the evidence. Use a numbered list.\n\n"
        "4. Use short paragraphs separated by blank lines for readability.\n"
        "5. If the context is insufficient to answer, say clearly in the Description section: "
        "'The documents do not contain enough information about [topic].' — then leave the other sections minimal.\n"
        "6. Do NOT truncate your answer or add phrases like '...' at the end.\n"
        + build_guardrail_instructions()
    )
    user_parts = []
    if conversation_context:
        user_parts.append(conversation_context)
    user_parts.append(context_text)
    user_parts.append(
        f"\nQuestion: {question}\n\n"
        "Provide a complete answer using the four required sections: "
        "## Description, ## Highlighted Points, ## Conclusion, ## Suggestions"
    )
    user = "\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_agent_system_prompt(tool_descriptions: str) -> list[dict[str, str]]:
    """System prompt for the tool-calling agent.

    Includes explicit tool-selection routing rules so the LLM automatically
    picks the right tool(s) based on the nature of the user's question:

      1. knowledge_base_search  — always try first (user's own documents).
      2. web_search             — use when KB has no results or real-time/
                                  external data is needed.
      3. analyze_document       — use for deep analysis of a specific document
                                  found via KB search.
      4. generate_report        — always the last step when a report/summary
                                  is requested or when multiple tools were used.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are an enterprise intelligence agent with four tools. "
                "You decompose complex user requests into steps, choose the "
                "right tool(s) to gather evidence, and produce a thorough "
                "final answer.\n\n"
                "## Available tools\n"
                f"{tool_descriptions}\n\n"
                "## Tool-selection decision rules\n"
                "Follow these rules strictly to decide which tool(s) to call:\n\n"
                "1. **Always start with `knowledge_base_search`** for ANY question. "
                "The user's uploaded documents are the primary source of truth.\n\n"
                "2. **Use `web_search` when:**\n"
                "   - `knowledge_base_search` returned `\"found\": false` or no "
                "useful passages.\n"
                "   - The question explicitly asks for current news, market data, "
                "external benchmarks, or real-time information.\n"
                "   - The user says 'search the web', 'look online', 'find on the "
                "internet', or similar.\n"
                "   You may call BOTH `knowledge_base_search` AND `web_search` to "
                "combine internal and external evidence.\n\n"
                "3. **Use `analyze_document` when:**\n"
                "   - A specific document_id was returned by `knowledge_base_search` "
                "AND the user needs a deeper analysis of that particular document.\n"
                "   - Only call this tool after `knowledge_base_search`.\n\n"
                "4. **Use `generate_report` when:**\n"
                "   - The user explicitly asks for a report, summary, or structured "
                "output.\n"
                "   - You have called two or more other tools and need to synthesise "
                "the findings into a coherent deliverable.\n"
                "   - Always call this LAST.\n\n"
                + build_guardrail_instructions()
                + "\n\n## General rules\n"
                "- Never answer from your own training data — always use tools.\n"
                "- Validate tool inputs before calling. If a tool fails, note the "
                "failure and decide whether to retry or proceed with available data.\n"
                "- Produce a COMPLETE final answer in clean Markdown with headings, "
                "bullet points, and **bold** key terms. Do not truncate.\n"
                "- Do not call tools unnecessarily — stop once you have sufficient "
                "evidence.\n"
                "- If a tool output contains a JSON `report_markdown` field, include "
                "its content verbatim in your final answer.\n"
            ),
        }
    ]


def build_report_prompt(topic: str, findings: str, sources: str) -> list[dict[str, str]]:
    """Prompt for the structured report generation tool."""
    return [
        {
            "role": "system",
            "content": (
                "You are a technical report generator. Produce a well-structured "
                "Markdown report from the provided findings and sources. Always "
                "include a summary, key findings with evidence, and "
                "recommendations. Cite sources as [n]."
                + "\n" + build_guardrail_instructions()
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n\nFindings:\n{findings}\n\nSources:\n{sources}\n\n"
                "Generate the report."
            ),
        },
    ]
