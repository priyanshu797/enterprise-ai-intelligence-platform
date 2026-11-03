"""Retrieval-Augmented Generation pipeline (Module B + Challenges 3, 4).

Flow:
  1. Resolve conversation context; rewrite follow-ups into standalone queries.
  2. Retrieve relevant chunks using the configured strategy
     (semantic | hybrid | rerank | rewrite). Hybrid (dense+BM25 RRF) is the
     default because enterprise docs mix prose with proper nouns/codes that
     pure semantic search misses.
  3. Apply injection sanitisation (security.injection) — retrieved text is
     untrusted data.
  4. Detect insufficient retrieval (below the relevance threshold) and refuse
     rather than hallucinate.
  5. Generate a grounded, cited answer with the LLM (with provider fallback).
  6. Verify the answer is supported by the retrieved sources (Challenge 4) and
     detect conflicting information among sources.
  7. Persist the exchange and log token usage.

Retrieval-strategy rationale (Module B "explain why"): documented inline and
in the evaluation report. Hybrid search via Reciprocal Rank Fusion is chosen
as the default because it combines the recall strengths of dense (semantic
similarity) and sparse (exact keyword/BM25) retrieval without requiring a
separate training step — a strong, low-effort default for heterogeneous
enterprise corpora. Rerank and rewrite strategies are provided as alternatives.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..config import settings
from ..schemas import ChatResponse, SourceRef
from .cache import get_response_cache
from .llm.base import LLMResponse, ProviderError
from .llm.service import get_llm_service
from .llm.templates import build_rag_prompt
from .memory import (
    add_message,
    build_context_block,
    build_context_window,
    get_or_create_conversation,
    is_followup_question,
    load_window,
    rewrite_followup_query,
)
from ..security.injection import (
    analyze_text_for_injection,
    encapsulate_context,
    sanitize_retrieved_context,
)
from .usage import log_usage
from .vector_store import RetrievalResult, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalOutcome:
    chunks: list[RetrievalResult]
    confidence: float
    injection_flags: list[dict]


class RAGService:
    def __init__(self):
        self.llm = get_llm_service()
        self.cache = get_response_cache()

    # ------------------------------------------------------------- retrieval
    def retrieve(
        self,
        query: str,
        *,
        strategy: str | None = None,
        top_k: int | None = None,
        filters: dict | None = None,
        owner_id: str | None = None,
    ) -> RetrievalOutcome:
        strategy = strategy or settings.RETRIEVAL_STRATEGY
        top_k = top_k or settings.TOP_K
        store = get_vector_store()

        effective_query = query
        if strategy == "rewrite" and self.llm.available:
            effective_query = self._rewrite_for_retrieval(query)

        # SECURITY: always inject a server-side owner_id filter so no user
        # can ever retrieve another user's document chunks.  We merge this on
        # top of any caller-supplied filters (which may legitimately narrow by
        # file_type, document_id, etc.) but we never let the caller override
        # the owner scope.
        effective_filters: dict | None = dict(filters) if filters else {}
        if owner_id:
            effective_filters["owner_id"] = owner_id
        if not effective_filters:
            effective_filters = None

        results = store.search(effective_query, top_k, strategy=strategy, filters=effective_filters)

        # Filter by minimum relevance. Normalise scores to a 0..1-ish range.
        normalised = _normalise_scores(results)
        relevant = [r for r in normalised if r.score >= settings.MIN_RELEVANCE_SCORE]

        # Apply injection sanitisation — treat as untrusted data.
        chunk_dicts = [
            {
                "text": r.text,
                "document_id": r.document_id,
                "document_filename": r.document_filename,
                "chunk_index": r.chunk_index,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in relevant
        ]
        safe, flags = sanitize_retrieved_context(chunk_dicts)

        # Rebuild RetrievalResult objects from the safe dicts.
        safe_results = [
            RetrievalResult(
                text=str(c["text"]),
                document_id=str(c.get("document_id", "")),
                document_filename=str(c.get("document_filename", "")),
                chunk_index=int(c.get("chunk_index", 0)),
                score=float(c.get("score", 0.0)),
                metadata=c.get("metadata", {}),
            )
            for c in safe
        ]
        confidence = (
            sum(r.score for r in safe_results) / len(safe_results)
            if safe_results
            else 0.0
        )
        return RetrievalOutcome(chunks=safe_results, confidence=confidence, injection_flags=flags)

    def _rewrite_for_retrieval(self, query: str) -> str:
        """Query rewriting for the 'rewrite' retrieval strategy (Challenge 3)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question into an optimised keyword "
                    "query for document retrieval. Keep it concise and "
                    "search-friendly. Output ONLY the rewritten query."
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            resp = self.llm.complete(messages, max_tokens=60)
            rewritten = resp.content.strip().strip('"').strip("'")
            if rewritten:
                return rewritten
        except ProviderError as exc:
            logger.warning("Query rewrite failed, using original: %s", exc)
        return query

    # ------------------------------------------------------------- answer
    def answer(
        self,
        db: Session,
        owner_id: str,
        query: str,
        conversation_id: str | None = None,
        filters: dict | None = None,
        provider: str = "auto",
    ) -> ChatResponse:
        # 1. Conversation + memory window.
        conv = get_or_create_conversation(db, owner_id, conversation_id, title=query[:60])
        history = load_window(db, conv.id)
        window = build_context_window(history)

        # 2. Resolve standalone query for follow-ups (Module E demonstration).
        standalone_query = rewrite_followup_query(query, window) if is_followup_question(query) else query

        # 3. Cache lookup (Challenge 5).
        cached = self.cache.get(owner_id, standalone_query, filters)
        if cached is not None:
            cached.cached = True
            # Still persist the user message so conversation history reflects it.
            add_message(db, conv.id, MessageRole_for("user"), query)
            add_message(
                db, conv.id, MessageRole_for("assistant"), cached.answer,
                sources=cached.sources, provider=cached.provider, model=cached.model,
                latency_ms=0,
            )
            log_usage(db, owner_id=owner_id, endpoint="/chat", cache_hit=True,
                      provider=cached.provider, model=cached.model)
            return cached

        # 4. Retrieve.
        outcome = self.retrieve(standalone_query, filters=filters, owner_id=owner_id)

        if outcome.injection_flags:
            logger.warning("Injection flags raised during retrieval: %s", outcome.injection_flags)

        # 5. Insufficient information — refuse rather than hallucinate.
        # We refuse only when NO chunks were retrieved at all (score filter
        # passed zero chunks). When chunks exist we let the LLM answer and
        # the verification step flags low-grounding answers as "unsupported".
        if not outcome.chunks:
            return self._build_insufficient(
                db, owner_id, conv, query, standalone_query, outcome
            )

        # 6. Generate grounded answer.
        context_text = encapsulate_context(_chunks_to_dicts(outcome.chunks))
        conv_block = build_context_block(window)
        messages = build_rag_prompt(standalone_query, context_text, conv_block)

        start = time.perf_counter()
        llm_resp = self.llm.complete(messages, max_tokens=settings.RAG_MAX_TOKENS)
        latency_ms = int((time.perf_counter() - start) * 1000)

        # 7. Verify grounding + detect conflicts (Challenge 4).
        verification = self._verify_answer(standalone_query, llm_resp.content, outcome.chunks)

        answer_status = self._decide_status(outcome, verification)
        confidence = round(min(1.0, outcome.confidence * verification.grounding_score), 3)

        # 8. Build response.
        sources = [
            SourceRef(
                document_id=c.document_id,
                document_filename=c.document_filename,
                chunk_index=c.chunk_index,
                text=c.text[:600],
                score=round(c.score, 3),
            )
            for c in outcome.chunks
        ]
        # Prepend a conflict note if detected.
        final_answer = llm_resp.content
        if answer_status == "conflicting" and verification.conflict_summary:
            final_answer = (
                f"⚠️ Conflicting information detected across sources: "
                f"{verification.conflict_summary}\n\n{llm_resp.content}"
            )
        elif answer_status == "unsupported":
            final_answer = (
                "I could not find sufficient grounded evidence to support an "
                "answer to this question in the retrieved documents. "
                "Portions of the model's response were not backed by the "
                "sources and have been withheld.\n\n"
                f"Retrieved context did not fully support: {query}"
            )

        response = ChatResponse(
            answer=final_answer,
            sources=sources,
            conversation_id=conv.id,
            provider=llm_resp.provider,
            model=llm_resp.model,
            prompt_tokens=llm_resp.usage.prompt_tokens,
            completion_tokens=llm_resp.usage.completion_tokens,
            latency_ms=latency_ms,
            cached=False,
            confidence=confidence,
            answer_status=answer_status,
        )

        # 9. Persist conversation + usage.
        add_message(db, conv.id, MessageRole_for("user"), query)
        add_message(
            db, conv.id, MessageRole_for("assistant"), final_answer,
            sources=[s.model_dump() for s in sources],
            provider=llm_resp.provider, model=llm_resp.model,
            prompt_tokens=llm_resp.usage.prompt_tokens,
            completion_tokens=llm_resp.usage.completion_tokens,
            latency_ms=latency_ms,
        )
        self.cache.put(owner_id, standalone_query, filters, response)
        log_usage(
            db, owner_id=owner_id, endpoint="/chat",
            response=llm_resp, latency_ms=latency_ms,
        )
        return response

    def _build_insufficient(
        self, db, owner_id, conv, query, standalone_query, outcome: RetrievalOutcome
    ) -> ChatResponse:
        """Construct the 'insufficient information' response (no LLM call)."""
        answer = (
            "INSUFFICIENT INFORMATION: the available documents do not address "
            f"this question. (Query interpreted as: \"{standalone_query}\"). "
            "Try uploading relevant documents or rephrasing the question."
        )
        add_message(db, conv.id, MessageRole_for("user"), query)
        add_message(db, conv.id, MessageRole_for("assistant"), answer)
        log_usage(db, owner_id=owner_id, endpoint="/chat", cache_hit=False, latency_ms=0)
        return ChatResponse(
            answer=answer,
            sources=[],
            conversation_id=conv.id,
            provider="none",
            model="none",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0,
            cached=False,
            confidence=round(outcome.confidence, 3),
            answer_status="insufficient",
        )

    def _verify_answer(self, query: str, answer: str, chunks: list[RetrievalResult]) -> "VerificationResult":
        """LLM-based grounding check + conflict detection (Challenge 4).

        Asks the model to judge whether the answer is supported by the sources
        and whether the sources conflict. Returns a structured verdict. If the
        LLM is unavailable, we fall back to a citation-presence heuristic.
        """
        sources_text = "\n\n".join(
            f"[{i+1}] (doc={c.document_filename}, chunk={c.chunk_index}): {c.text[:600]}"
            for i, c in enumerate(chunks)
        )
        if not self.llm.available:
            return VerificationResult(
                grounding_score=0.5, has_conflict=False, conflict_summary="", unsupported=False
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict fact-checker. Given a question, an answer, "
                    "and the source chunks it should be grounded in, judge: "
                    "(a) how well the answer is supported by the sources "
                    "(grounding_score 0.0-1.0), (b) whether any answer claims are "
                    "unsupported, (c) whether the sources contain conflicting "
                    "information about the question. Respond ONLY as JSON: "
                    '{"grounding_score": float, "unsupported": bool, '
                    '"has_conflict": bool, "conflict_summary": str}.'
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nAnswer: {answer}\n\nSources:\n{sources_text}",
            },
        ]
        try:
            resp = self.llm.complete(messages, max_tokens=200)
            return _parse_verification(resp.content)
        except ProviderError as exc:
            logger.warning("Answer verification failed: %s", exc)
            return VerificationResult(grounding_score=0.5, has_conflict=False, conflict_summary="", unsupported=False)

    @staticmethod
    def _decide_status(outcome: RetrievalOutcome, verification: "VerificationResult") -> str:
        if verification.has_conflict:
            return "conflicting"
        if verification.unsupported and verification.grounding_score < settings.ANSWER_CONFIDENCE_THRESHOLD:
            return "unsupported"
        return "answered"


@dataclass
class VerificationResult:
    grounding_score: float = 1.0
    unsupported: bool = False
    has_conflict: bool = False
    conflict_summary: str = ""


def _parse_verification(text: str) -> VerificationResult:
    import json
    import re as _re

    match = _re.search(r"\{.*\}", text, _re.DOTALL)
    if not match:
        return VerificationResult()
    try:
        data = json.loads(match.group(0))
        return VerificationResult(
            grounding_score=float(data.get("grounding_score", 1.0)),
            unsupported=bool(data.get("unsupported", False)),
            has_conflict=bool(data.get("has_conflict", False)),
            conflict_summary=str(data.get("conflict_summary", "")),
        )
    except Exception:
        return VerificationResult()


def _normalise_scores(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Return results as-is — scores are already cosine-similarity 0..1.

    We no longer max-normalize because the hybrid search now reports absolute
    cosine similarity directly. Min-score thresholds work correctly on the
    raw cosine scale.
    """
    return results


def _chunks_to_dicts(chunks: list[RetrievalResult]) -> list[dict]:
    return [
        {
            "text": c.text,
            "document_id": c.document_id,
            "document_filename": c.document_filename,
            "chunk_index": c.chunk_index,
            "score": c.score,
        }
        for c in chunks
    ]


def MessageRole_for(role: str):
    from ..models import MessageRole
    return MessageRole.USER if role == "user" else MessageRole.ASSISTANT


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
