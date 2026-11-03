# Enterprise AI Intelligence Platform

An end-to-end, production-oriented AI platform that ingests enterprise documents, answers complex questions with source citations via Retrieval-Augmented Generation (RAG), runs multi-step AI agents, retains conversational memory, and is evaluated against a 22-question dataset covering answered, insufficient, and conflicting scenarios.

Built as the AI Engineer assessment project. Python/FastAPI backend, Next.js frontend, pluggable LLM providers with automatic fallback, embedded vector store, and cloud-ready Docker deployment.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Quick start (local)](#quick-start-local)
4. [Configuration](#configuration)
5. [Running the evaluation](#running-the-evaluation)
6. [Tests](#tests)
7. [API reference](#api-reference)
8. [Database schema](#database-schema)
9. [Deployment](#deployment)
10. [Design decisions](#design-decisions)
11. [Project structure](#project-structure)

---

## What it does

Implements every required module:

| Module | What's implemented |
|---|---|
| **A — Document Intelligence** | Upload PDF/TXT/MD/CSV, validate + extract, chunk, embed, store vectors, track processing status, handle duplicates (content-hash dedup), record failures. Async background processing (non-blocking uploads). |
| **B — Advanced RAG** | Embed query → retrieve chunks → cited, context-aware answers. Detects insufficient info and refuses rather than hallucinates. Detects conflicting sources. Retrieval strategy is **hybrid** (dense + BM25 via Reciprocal Rank Fusion) by default, with `rerank` and `rewrite` alternatives and metadata filtering. |
| **C — LLM Integration** | Two providers (OpenAI + Anthropic, with Ollama as a local option). Structured output, prompt templates, model selection, token tracking, rate-limit/error handling, automatic fallback, response validation. |
| **D — AI Agents** | LangGraph ReAct agent with 3 tools: knowledge-base search, document analysis, structured report generation. Multi-step chaining (search → analyze → report), parameter validation, per-tool timeouts, iteration cap, failure recovery, full step recording. |
| **E — Conversational Memory** | Multiple conversations per user, sliding-window context, follow-up detection + standalone-query rewriting, no unnecessary history sent to the LLM. |
| **F — Dashboard** | JWT auth + protected routes, dashboard stats, document management, chat UI with source citations, conversation history, agent execution visibility, token usage, latency, error tracking, activity logs. |
| **G — AI Evaluation** | 22-question dataset, custom framework scoring retrieval relevance, answer correctness, source grounding, hallucination handling, latency, and tokens. RAGAS/DeepEval optional. |

**Advanced engineering challenges completed (5 of 7):**

1. **Intelligent Model Fallback** — automatic provider switch on rate-limit/transient errors.
2. **Prompt Injection Protection** — retrieved content treated as untrusted data (detection, neutralization, encapsulation, system-prompt guardrails).
3. **Retrieval Optimization** — hybrid search (RRF of dense + BM25), plus rerank and rewrite strategies.
4. **AI Response Verification** — LLM-based grounding check + conflict detection before returning answers.
5. **Cost Optimization** — response cache (bounded, TTL'd), token + cost tracking per request.

(Asynchronous document processing and agent reliability are implemented as core features of Modules A/D rather than as separate challenge entries.)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Next.js Frontend                        │
│  Auth · Dashboard · Documents · Chat · Agent · Logs          │
└────────────────────────────┬────────────────────────────────┘
                             │ REST / JWT
┌────────────────────────────▼────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │
│  │  Auth     │ │ Documents│ │  Chat  │ │ Agent  │ │ Stats  │  │
│  │ (JWT)     │ │ (async)  │ │ (RAG)  │ │(LangGr)│ │(usage) │  │
│  └────┬─────┘ └────┬─────┘ └───┬────┘ └───┬────┘ └───┬────┘  │
│       │            │           │           │           │       │
│  ┌────▼────────────▼───────────▼───────────▼───────────▼────┐ │
│  │                  Services Layer                          │ │
│  │  RAG · LLM(fallback) · Agent/Tools · Memory · Cache ·    │ │
│  │  Embeddings · VectorStore · DocumentProcessing · Eval    │ │
│  └────┬──────────────┬───────────────┬─────────────────────┘ │
│       │              │               │                       │
│  ┌────▼────┐  ┌──────▼──────┐  ┌─────▼─────┐                  │
│  │PostgreSQL│ │  ChromaDB   │  │ OpenAI /  │                  │
│  │ /SQLite  │ │ (vectors)   │  │ Anthropic │                  │
│  │(metadata)│ │             │  │  / Ollama │                  │
│  └─────────┘  └─────────────┘  └───────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

Component responsibilities and data flow are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick start (local)

### Prerequisites

- Python 3.10+
- Node.js 20+
- An LLM API key (OpenAI **or** Anthropic). Without one the platform runs but chat/agent endpoints return a clear "no provider configured" error. For a fully local LLM, run [Ollama](https://ollama.com) and set `EMBEDDING_PROVIDER=huggingface` (default).

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: add at least OPENAI_API_KEY or ANTHROPIC_API_KEY

# Run migrations (creates SQLite DB + tables)
alembic upgrade head
# (Or rely on auto-create_all on startup.)

# Start the API
uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# App at http://localhost:3000
```

Register a user, sign in, upload a document (try `evaluation/dataset.json` seed docs or any PDF/TXT/CSV), wait for it to process, then ask questions in Chat.

### Seed the demo corpus

```bash
cd backend
python seed_corpus.py --email you@example.com --password yourpass
```

This uploads the evaluation seed documents so you can immediately ask "What are the most common reasons for system failures?"

---

## Configuration

All config is environment-driven (see `backend/.env.example`). Key knobs:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./enterprise_ai.db` | SQLite dev; PostgreSQL prod |
| `VECTOR_STORE_TYPE` | `chroma` | Embedded; switch to `pgvector`/`pinecone` |
| `EMBEDDING_PROVIDER` | `huggingface` | Local, no API key; or `openai`/`ollama` |
| `LLM_PRIMARY_PROVIDER` | `openai` | `openai`/`anthropic` |
| `LLM_FALLBACK_PROVIDER` | `anthropic` | used on rate-limit/error |
| `RETRIEVAL_STRATEGY` | `hybrid` | `semantic`/`hybrid`/`rerank`/`rewrite` |
| `TOP_K` | `6` | chunks retrieved |
| `MIN_RELEVANCE_SCORE` | `0.25` | below this = insufficient |
| `ANSWER_CONFIDENCE_THRESHOLD` | `0.35` | refuse below this |
| `AGENT_MAX_ITERATIONS` | `8` | hard step cap |
| `ENABLE_RESPONSE_CACHE` | `true` | cost optimisation |

---

## Running the evaluation

```bash
cd backend
python evaluation/run_evaluation.py --out evaluation/evaluation_report.json
```

This loads `evaluation/dataset.json` (22 questions), runs each through the RAG pipeline, scores retrieval relevance / answer correctness / source grounding / hallucination handling / latency / tokens, and writes a JSON report + a DB `evaluation_run` row.

The dataset first needs the seed corpus uploaded (use `seed_corpus.py`). Methodology and limitations are in [`evaluation/METHODOLOGY.md`](evaluation/METHODOLOGY.md).

---

## Tests

```bash
cd backend
pip install pytest pytest-asyncio httpx
pytest -v
```

- **Tier-1** tests (security, document processing, auth API) run with lightweight deps only.
- **Tier-2** tests (`@pytest.mark.heavy` — vector store, RAG, agent, document upload) require ChromaDB + an embedding backend and are auto-skipped if those imports are missing. They stub the LLM so no API keys are needed.

---

## API reference

Full OpenAPI/Swagger at `/docs` when the backend is running. Summary (all under `/api/v1`):

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/register` | — | Create account |
| POST | `/auth/login` | — | Get JWT tokens |
| POST | `/auth/refresh` | refresh token | Refresh access token |
| GET | `/auth/me` | ✓ | Current user |
| POST | `/documents` | ✓ | Upload (async processing) |
| GET | `/documents` | ✓ | List documents |
| GET | `/documents/{id}` | ✓ | Document detail |
| GET | `/documents/{id}/chunks` | ✓ | Processed chunks |
| GET | `/documents/{id}/job` | ✓ | Processing job status |
| DELETE | `/documents/{id}` | ✓ | Delete + remove vectors |
| POST | `/chat` | ✓ | RAG answer with citations |
| GET | `/conversations` | ✓ | List conversations |
| GET | `/conversations/{id}` | ✓ | Messages in a conversation |
| DELETE | `/conversations/{id}` | ✓ | Delete conversation |
| POST | `/agent/run` | ✓ | Run multi-step agent |
| GET | `/agent/runs` | ✓ | Recent agent runs |
| GET | `/agent/runs/{id}` | ✓ | Agent run detail + steps |
| GET | `/stats/dashboard` | ✓ | Aggregated stats |
| GET | `/stats/usage` | ✓ | Token usage logs |
| GET | `/stats/activity` | ✓ | Activity logs |
| GET | `/stats/health` | — | Service health |

---

## Database schema

Managed via SQLAlchemy ORM + Alembic migrations (`backend/alembic/`). Tables:

`user`, `document`, `document_chunk`, `processing_job`, `conversation`, `message`, `agent_run`, `agent_step`, `usage_log`, `activity_log`, `evaluation_run`, `evaluation_result`.

Create/apply:

```bash
cd backend
alembic upgrade head     # apply migrations
alembic revision --autogenerate -m "describe change"  # new migration
```

The app also auto-creates tables on startup via `init_db()` for convenience in dev.

---

## Deployment

The app is containerised and cloud-ready.

### Live Cloud Deployment

> **Live URL:** https://enterprise-ai-platform.onrender.com  
> **Frontend:** https://enterprise-ai-platform-frontend.onrender.com  
> *(Deployed via Render using the `render.yaml` blueprint in the repository root.)*

If the live URL is temporarily unavailable (Render free-tier instances spin down after inactivity), the platform is fully runnable locally in under 5 minutes via the Quick Start steps below.

### Docker Compose (full stack locally)

```bash
cp .env.example backend/.env  # set API keys
export GEMINI_API_KEY=your-gemini-api-key-here
docker compose up --build
# backend :8000, frontend :3000, postgres :5432
```

### Render (blueprint included)

`render.yaml` defines two web services (backend Docker, frontend Docker). Set `DATABASE_URL` (managed Postgres), `SECRET_KEY`, `GEMINI_API_KEY`/`GROQ_API_KEY`, `NEXT_PUBLIC_API_URL`, and `BACKEND_CORS_ORIGINS` (your frontend origin) in the Render dashboard.

### Other platforms

The backend Dockerfile runs `uvicorn` directly; deploy to any container host (Fly.io, Railway, AWS ECS, Google Cloud Run). Point `DATABASE_URL` at a managed Postgres, set CORS origins to the frontend URL, and set at least one LLM key.

---

## Design decisions

A few key choices and why:

- **FastAPI** over Rails: async-native, first-class for AI/LLM streaming workloads, and auto-generates the OpenAPI docs (satisfying the API-doc deliverable).
- **Hybrid search (RRF)** as the default retrieval strategy: enterprise documents mix prose with proper nouns, codes, and identifiers. Pure semantic (dense) retrieval misses exact-match terms; pure keyword (BM25) misses paraphrases. RRF combines both with no training step — a strong, low-effort default. `rerank` and `rewrite` are provided as alternatives.
- **ChromaDB embedded** for dev (no external server, runs in-process), behind a `VectorStore` abstraction so production can swap to pgvector/Pinecone without touching RAG code.
- **HuggingFace `all-MiniLM-L6-v2`** default embeddings: runs locally on CPU with no API key, so the platform works end-to-end out of the box. OpenAI embeddings are used automatically when an OpenAI key is set.
- **In-process thread-pool** for async document processing: keeps the system self-contained (no Redis/Celery) while not blocking uploads. The job logic is isolated so it can be moved to Celery for horizontal scale without changes.
- **LLM-based answer verification + conflict detection**: a dedicated fact-checker call scores grounding and flags conflicting sources before returning an answer, reducing unsupported/hallucinated responses.
- **Prompt-injection defence in depth**: pattern detection + neutralization + context encapsulation + system-prompt guardrails. Documented honestly as a heuristic high-recall filter, not a complete block.

---

## Project structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routers (auth, documents, chat, agent, stats)
│   │   ├── models/         # SQLAlchemy ORM models
│   │   ├── schemas/        # Pydantic request/response schemas
│   │   ├── security/       # auth (JWT) + injection defense
│   │   ├── services/
│   │   │   ├── llm/        # providers, fallback, templates, token tracking
│   │   │   ├── agent/      # LangGraph executor + tools + reliability
│   │   │   ├── document_processing.py, vector_store.py, rag.py,
│   │   │   ├── memory.py, usage.py, cache.py, document_jobs.py
│   │   ├── config.py, database.py, main.py
│   ├── alembic/            # migrations
│   ├── evaluation/         # dataset + runner + methodology
│   ├── tests/
│   ├── Dockerfile, requirements.txt, .env.example
├── frontend/               # Next.js app
│   ├── app/                 # pages (login, dashboard, documents, chat, agent, logs)
│   ├── components/, lib/
│   ├── Dockerfile, package.json
├── docs/ARCHITECTURE.md
├── CLAUDE.md
├── DEVELOPMENT_REPORT.md
├── docker-compose.yml, render.yaml
└── README.md
```
