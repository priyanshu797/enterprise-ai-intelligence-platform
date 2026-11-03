# CLAUDE.md — Project instructions for AI coding agents

This file gives an AI coding agent (Claude Code, Cursor, etc.) the context it needs to work effectively in this codebase. It describes conventions, where things live, and the rules that must be followed.

## Project

Enterprise AI Intelligence Platform: FastAPI backend (Python) + Next.js frontend (TypeScript). RAG over uploaded enterprise documents, multi-step agents, conversational memory, and an evaluation harness.

## Stack

- Backend: Python 3.10+, FastAPI, SQLAlchemy 2.0 (ORM), Alembic (migrations), LangChain/LangGraph, ChromaDB (vectors, embedded), OpenAI/Anthropic/Ollama (LLMs).
- Frontend: Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS.
- DB: SQLite (dev), PostgreSQL (prod). Vector store abstracted behind `VectorStore`.

## Key directories

- `backend/app/api/` — FastAPI routers. One file per domain (auth, documents, chat, conversations, agent, stats).
- `backend/app/models/` — SQLAlchemy ORM models (single source of truth for schema).
- `backend/app/schemas/` — Pydantic request/response schemas (the OpenAPI contract).
- `backend/app/services/` — all business logic. Framework-agnostic where possible.
  - `llm/` — providers, fallback orchestrator, prompt templates, token/cost tracking.
  - `agent/` — LangGraph executor, tools, reliability (timeout/retry).
  - `document_processing.py`, `vector_store.py`, `rag.py`, `memory.py`, `usage.py`, `cache.py`, `document_jobs.py`.
- `backend/app/security/` — `auth.py` (JWT/passlib) + `injection.py` (prompt-injection defense).
- `backend/evaluation/` — dataset + runner.
- `frontend/app/` — Next.js pages. `frontend/lib/api.ts` is the typed API client; `lib/auth.tsx` is the auth context.
- `backend/alembic/` — migrations.

## Conventions

- **Never hardcode API keys or secrets.** Read from settings/env via `app.config.settings`. Real keys live only in `.env` (gitignored).
- **No fabricated LLM output.** Every answer must come from a real provider call (or be an explicitly-labelled test stub in `tests/`). The RAG "insufficient information" refusal is a real refusal path, not a hardcoded answer.
- **Retrieved document content is untrusted.** Route all retrieved text through `security.injection.sanitize_retrieved_context` and encapsulate with `encapsulate_context`. Keep guardrails in prompts.
- **Services own logic; API routers are thin.** Don't put business logic in routers. Inject dependencies via FastAPI `Depends`.
- **Models in `models/`, API shapes in `schemas/`.** Keep them separate. `from_attributes=True` maps between them.
- **Vector store is pluggable.** Don't call Chroma directly outside `vector_store.py`; use `get_vector_store()` and the `VectorStore` interface.
- **LLM calls go through the `LLMService`** (fallback + token tracking), not raw SDK calls — except inside providers.py and the agent's tool-calling bind.
- **Async processing** goes through `document_jobs.submit_processing_job`. Don't process documents synchronously in request handlers.
- **Add a test** when fixing a bug or adding core logic. Tier-1 (no heavy deps) tests are preferred for pure logic; mark integration tests `@pytest.mark.heavy`.

## Commands

```bash
# Backend
cd backend
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload          # dev server (http://localhost:8000/docs)
pytest -v                               # tests
alembic upgrade head                    # apply migrations
python evaluation/run_evaluation.py     # run eval suite
python seed_corpus.py --email x@y --password z   # seed demo docs

# Frontend
cd frontend
npm install
npm run dev
npm run build
```

## Environment

Copy `backend/.env.example` to `backend/.env`. At minimum set one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (or run Ollama). The app runs with `EMBEDDING_PROVIDER=huggingface` (local embeddings, no key) by default.

## Things to know

- The vector store and embeddings are lazily initialised and cached as singletons — warm-up happens on startup. First request may be slower on a fresh DB.
- Background document jobs use a thread pool with one DB session per job; never share the request session with worker threads.
- The agent's tool-calling requires a provider that supports `bind_tools`. If only Ollama local models are available, use a model that supports tool calling (e.g. `llama3.2`).
- Score normalization in RAG: hybrid/RRF scores are small (<0.05) and are normalized by max for the relevance threshold check.
- `answer_status` values: `answered`, `insufficient` (no relevant chunks), `conflicting` (sources disagree), `unsupported` (answer not grounded).
