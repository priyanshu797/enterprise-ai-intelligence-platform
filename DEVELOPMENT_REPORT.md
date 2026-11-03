# Development Report — AI Tool Usage & Engineering Decisions

This report discloses AI tool usage (per Rule 2 of the assessment) and documents the independent engineering decisions made.

## 1. AI tools used

| Tool | Role | What it was used for |
|---|---|---|
| Kiro (AI coding agent) | Implementation assistant | Scaffolding the project, generating boilerplate, writing initial service/API/frontend code, iterating on structure. |
| My own engineering judgment | Architect + reviewer | Every architectural decision below was made independently; generated code was reviewed, corrected, and integrated against the requirements. |

### Tasks delegated to the AI tool
- Generating repetitive boilerplate (ORM models, Pydantic schemas, FastAPI routers, React pages, config files).
- Drafting initial implementations of standard algorithms (BM25 scoring, JWT auth, recursive chunking).
- Writing documentation drafts (README sections, this report's structure).

### Decisions made independently
- The overall architecture (FastAPI + LangChain/LangGraph + ChromaDB + Next.js).
- The hybrid RRF retrieval strategy as the default.
- The defence-in-depth prompt-injection approach and its honest limitations.
- The "refuse rather than hallucinate" insufficient-information path.
- LLM-based answer verification + conflict detection.
- The async document-processing job model and per-job DB sessions.
- The evaluation methodology and the 22-question dataset design (including the injection test case).

### Errors identified in generated code and how they were validated
- Import-order / circular-import risks between `rag.py` and `memory.py` — resolved by keeping `MessageRole_for` local and importing models lazily.
- LangChain chat-model API differences across versions (`usage_metadata` key names) — handled defensively in `_extract_usage`.
- Chroma score semantics (cosine distance vs similarity, small RRF scores) — normalized in `_normalise_scores` before the threshold check.
- Frontend dev-proxy vs production direct-call routing — fixed by making `API_BASE` read `NEXT_PUBLIC_API_URL`.
- All Python files were syntax-checked with `py_compile`; tier-1 tests run without heavy dependencies.

### Validation approach
- `python -m py_compile` across the whole backend (syntax + import-time safety).
- Tier-1 unit/integration tests with a stubbed LLM (no API keys needed).
- Manual reasoning over each module against the assessment requirements matrix.

## 2. Engineering decisions and rationale

### Backend: FastAPI over Rails
Async-native, ideal for LLM streaming/long calls, auto-generates OpenAPI (API-doc deliverable), and the Python ecosystem is where LangChain/LangGraph and the embedding/eval libraries live.

### Retrieval: hybrid (RRF) by default
Enterprise documents mix prose with proper nouns, codes, and identifiers. Pure dense (semantic) retrieval misses exact terms; pure BM25 misses paraphrases. Reciprocal Rank Fusion combines both with no training step — a strong, low-effort default. `rerank` and `rewrite` are offered as alternatives and selected via config.

### Vector store: ChromaDB embedded, behind an abstraction
No external server for dev → the platform runs out of the box. The `VectorStore` protocol means production can swap to pgvector/Pinecone without touching the RAG service.

### Embeddings: HuggingFace local default
`all-MiniLM-L6-v2` runs on CPU with no API key, so the platform is fully functional locally. When an OpenAI key is present, OpenAI embeddings are used automatically. This avoids a hard dependency on a paid API to run.

### LLM: provider chain with automatic fallback
Primary (OpenAI) → fallback (Anthropic) → local (Ollama). `RateLimitError` and transient errors move to the next provider; which provider served each request is recorded. This is both a resilience feature (Module C) and Challenge 1.

### Refuse-rather-than-hallucinate
Below `ANSWER_CONFIDENCE_THRESHOLD`, the system returns an explicit "insufficient information" message without an LLM call. This is a deliberate anti-hallucination design: an LLM cannot confabulate if it is never asked. A follow-up LLM verification step checks grounding and flags conflicts before returning.

### Agent: manual ReAct loop over the prebuilt graph
LangGraph's prebuilt react agent is convenient, but a manual loop gives explicit control over per-tool timeouts, parameter validation, the hard iteration cap, failure recovery, and per-step DB recording (Challenge 7). The trade-off is more code for more control.

### Async document processing: in-process thread pool
Avoids blocking the upload request and avoids a Redis/Celery dependency for a self-contained app. Per-job DB sessions avoid the threading pitfall of sharing a request session. The job logic is isolated so it can move to Celery for horizontal scale with no changes.

### Prompt injection: defence in depth, honestly limited
Detection (patterns) + neutralization (strip) + encapsulation (delimiters) + guardrails (system prompt). No static pattern matcher is complete; the design raises the bar and flags suspicious content, and the evaluation dataset includes an injection test case. The limitations are stated explicitly in the methodology and in the code comments.

### Cost optimization: response cache + tracking
A bounded, TTL'd in-memory cache keyed by (normalized query, filters, owner) skips the LLM call entirely on a hit. Every LLM call logs prompt/completion tokens and an estimated cost. (Challenge 5.)

## 3. Honest limitations

- A live cloud URL is not hosted from this environment; the repository provides Docker/Render configs and instructions. Standing up the URL requires a cloud account, API keys, and billing.
- Real LLM quality depends on the configured provider/model; the default `gpt-4o-mini`/`claude-3-5-haiku` are cost-friendly but less capable than larger models.
- The injection defense is heuristic, not a complete block.
- The evaluation uses LLM-as-judge which itself is approximate (see methodology).
