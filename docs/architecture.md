# System Architecture

This document describes the components of the Enterprise AI Intelligence Platform, how they interact, and the rationale behind the architecture.

## High-level architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         Next.js Frontend                          │
│  Auth · Dashboard · Documents · Chat · Agent · Logs               │
│                         (TypeScript, Tailwind)                    │
└───────────────────────────┬───────────────────────────────────────┘
                            │ HTTPS / JWT bearer
┌───────────────────────────▼───────────────────────────────────────┐
│                        FastAPI Backend                            │
│  Routers (thin): auth · documents · chat · conversations · agent ·│
│                  stats                                            │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                      Services Layer                          │  │
│  │  RAGService ── LLMService (primary→fallback chain)          │  │
│  │     └── VectorStore ── EmbeddingsService                    │  │
│  │     └── injection defense (sanitize + encapsulate)         │  │
│  │  AgentExecutor (LangGraph) ── tools (KB search, analyze,   │  │
│  │     report) ── reliability (timeout/retry/iter cap)         │  │
│  │  MemoryService (window + follow-up rewrite)                │  │
│  │  DocumentJobs (async thread pool) ── processing + chunking  │  │
│  │  ResponseCache · UsageLogger · EvaluationRunner            │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───┬───────────────────────────┬───────────────────────┬───────────┘
    │                           │                       │
┌───▼────────┐         ┌────────▼─────────┐      ┌───────▼────────┐
│ PostgreSQL │         │   ChromaDB       │      │  LLM Providers │
│ / SQLite   │         │  (vector store)  │      │ OpenAI/Anthropic│
│ metadata   │         │  (embedded/dev)  │      │   /Ollama      │
└────────────┘         └──────────────────┘      └────────────────┘
```

## Component responsibilities

### Routers (`app/api/`)
Thin HTTP adapters. Validate input via Pydantic schemas, enforce auth via `get_current_active_user`, delegate to services, log usage/activity. No business logic.

### Services (`app/services/`)
All domain logic, deliberately framework-agnostic and unit-testable.

**RAGService (`rag.py`)** — orchestrates the answer flow:
1. Resolve conversation context; rewrite follow-ups into a standalone query (`memory.py`).
2. Check the response cache (`cache.py`).
3. Retrieve chunks via `VectorStore` with the configured strategy.
4. Filter by `MIN_RELEVANCE_SCORE`; if nothing relevant, return "insufficient" without an LLM call.
5. Sanitize retrieved context through the injection defense (`security/injection.py`).
6. Build the prompt (`llm/templates.py`) and call the LLM with fallback (`llm/service.py`).
7. Verify grounding + detect conflicts via a fact-checker LLM call (`_verify_answer`).
8. Decide `answer_status` (answered/insufficient/conflicting/unsupported).
9. Persist the message and log token usage.

**LLMService (`llm/service.py`)** — provider chain (`primary → fallback → ollama`). `complete()` tries each provider; `RateLimitError`/transient errors move to the next; records which provider served the request. Token usage + cost estimation per call.

**AgentExecutor (`agent/agent.py`)** — a LangGraph-style ReAct loop (implemented manually for control over step recording, timeouts, and the iteration cap). Binds tools to the model, loops: model picks a tool → validate args (Pydantic) → execute with timeout/retry → feed result back. Records every step to `agent_step`.

**VectorStore (`vector_store.py`)** — ChromaDB backend with semantic/hybrid/rerank search. Hybrid = Reciprocal Rank Fusion of dense (cosine) and sparse (BM25) scores.

**DocumentJobs (`document_jobs.py`)** — async pipeline: register (dedup by hash) → thread-pool worker extracts → chunks → embeds → stores vectors → updates status. Per-job DB session.

**MemoryService (`memory.py`)** — sliding-window of recent messages; detects follow-ups and rewrites them into standalone queries for retrieval.

## Data flow: a chat question

```
User question ──▶ Chat router (auth)
   └─▶ RAGService.answer
        ├─ MemoryService: load window, rewrite follow-up
        ├─ ResponseCache: lookup
        ├─ VectorStore.search (hybrid: dense + BM25 → RRF)
        ├─ security.injection.sanitize_retrieved_context
        ├─ [if no chunks / low score] → return "insufficient" (no LLM)
        ├─ LLMService.complete (primary → fallback)  ──▶ provider API
        ├─ RAGService._verify_answer (grounding + conflict LLM call)
        ├─ decide answer_status
        ├─ persist Message + UsageLog
        └─ return ChatResponse (answer, sources, tokens, latency, status)
```

## Data flow: an agent task

```
Agent query ──▶ Agent router (auth)
   └─▶ AgentExecutor.run
        ├─ create AgentRun row
        ├─ loop (until no tool calls OR iteration cap):
        │    ├─ model.invoke(messages + tools bound) → AIMessage with tool_calls
        │    ├─ for each tool_call:
        │    │    ├─ validate args (Pydantic args_schema)
        │    │    ├─ execute with timeout/retry (reliability.with_timeout_and_retry)
        │    │    └─ append ToolMessage(result or error) + record AgentStep
        │    └─ (tool output informs the next model call)
        ├─ model produces final answer
        └─ persist steps + final answer + tokens
```

Example multi-step chain: "Find the common system failures and generate a report" →
`knowledge_base_search` (finds failure docs) → `analyze_document` (extracts reasons) → `generate_report` (synthesizes Markdown).

## Security layers

1. **Auth** — bcrypt password hashing, JWT access/refresh tokens, route guards via dependency.
2. **Prompt injection defense** — retrieved text is untrusted data:
   - Detection: pattern matching for instruction-override / role-hijack / system-prefix / control chars / invisible unicode.
   - Neutralization: strip dangerous patterns from flagged chunks.
   - Encapsulation: wrap chunks in `<retrieved_context>` delimiters with a "reference data only" header.
   - Guardrail: system prompts instruct the model to treat context as evidence and never reveal instructions.
3. **Response verification** — a fact-checker call grounds answers in sources and flags conflicts/hallucination.

## Persistence

- **Relational (SQLAlchemy)**: users, documents, chunks metadata, jobs, conversations, messages, agent runs/steps, usage logs, activity logs, evaluation runs/results. Dev = SQLite; prod = PostgreSQL.
- **Vector (ChromaDB)**: chunk embeddings + metadata, one collection, cosine space. Vectors are decoupled from chunk text (kept in the DB for citation without round-trips).

## Scalability notes

- The in-process thread pool + job store is the async-processing MVP. Moving to Celery/RQ only requires replacing `_get_executor().submit` with a queue enqueue; the per-job logic is unchanged.
- The vector store abstraction lets production move to a managed vector DB (pgvector/Pinecone) without touching the RAG service.
- LLM fallback is per-request; the provider chain is built once and reused.
