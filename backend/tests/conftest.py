"""Shared pytest fixtures.

The `client`/`authed_client` fixtures give each test an isolated SQLite
database by reassigning the module-level engine + session factory in
`app.database` to a fresh temp-file DB, then creating the schema. This is more
robust than reloading modules (which detaches ORM models from the metadata).

Tier-1 tests (security, document processing, auth API) run with lightweight
deps only (FastAPI/SQLAlchemy/passlib/jose). Tier-2 tests (`@pytest.mark.heavy`
— vector store, RAG, agent, document upload) require ChromaDB + an embedding
backend and are skipped automatically if those imports fail.
"""
from __future__ import annotations

import os
import sys

# Force a test environment BEFORE any app module is imported.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("ANTHROPIC_API_KEY", "")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient against an isolated temp-file SQLite database."""
    db_file = tmp_path / "test.db"
    url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", url)

    import app.database as dbmod
    import app.config as configmod

    # Clear the settings cache so get_settings picks up the new DATABASE_URL.
    configmod.get_settings.cache_clear()
    # Reuse a single connection for SQLite so background threads (doc jobs) and
    # the main thread share the same in-memory-ish file state.
    test_engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    test_session = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Reassign the module-level globals so get_db / init_db use the test engine.
    monkeypatch.setattr(dbmod, "engine", test_engine)
    monkeypatch.setattr(dbmod, "SessionLocal", test_session)

    # Import models so they register with Base.metadata, then create the schema.
    import app.models  # noqa: F401
    dbmod.Base.metadata.create_all(bind=test_engine)

    import app.main as mainmod
    with TestClient(mainmod.app) as c:
        yield c

    test_engine.dispose()
    configmod.get_settings.cache_clear()


@pytest.fixture()
def authed_client(client):
    """A client with a registered + logged-in user (auth header set)."""
    client.post("/api/v1/auth/register", json={
        "email": "tester@example.com", "password": "test-password", "full_name": "Tester",
    })
    resp = client.post("/api/v1/auth/login", json={
        "email": "tester@example.com", "password": "test-password",
    })
    token = resp.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


def pytest_collection_modifyitems(config, items):
    """Auto-skip heavy tests if their deps aren't importable."""
    skip_heavy = pytest.mark.skip(reason="heavy dependencies (chromadb/langchain) not installed")
    try:
        import chromadb  # noqa: F401
        import langchain  # noqa: F401
    except Exception:
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
