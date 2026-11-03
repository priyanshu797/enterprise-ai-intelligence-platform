"""Integration tests for the documents API + async processing (Challenge 6)."""
from __future__ import annotations

import io
import time

import pytest

pytestmark = pytest.mark.heavy


def test_upload_processes_async(authed_client):
    r = authed_client.post("/api/v1/documents", files={
        "file": ("notes.txt", io.BytesIO(b"Important note about server failures and resolutions."), "text/plain"),
    })
    assert r.status_code == 201
    data = r.json()
    assert data["status"] in ("pending", "processing", "completed")
    doc_id = data["id"]

    # Poll the job status until completed (async background processing).
    for _ in range(40):
        job = authed_client.get(f"/api/v1/documents/{doc_id}/job")
        assert job.status_code == 200
        status = job.json()["status"]
        if status == "completed":
            break
        time.sleep(0.5)
    assert job.json()["status"] == "completed"

    # Document should now reflect completed status with chunks.
    doc = authed_client.get(f"/api/v1/documents/{doc_id}").json()
    assert doc["status"] == "completed"
    assert doc["num_chunks"] > 0


def test_duplicate_upload_detected(authed_client):
    payload = b"Identical content for dedup testing."
    authed_client.post("/api/v1/documents", files={"file": ("d1.txt", io.BytesIO(payload), "text/plain")})
    time.sleep(1)
    r2 = authed_client.post("/api/v1/documents", files={"file": ("d2.txt", io.BytesIO(payload), "text/plain")})
    assert r2.status_code == 201
    assert r2.json()["status"] == "duplicate"


def test_list_documents(authed_client):
    authed_client.post("/api/v1/documents", files={
        "file": ("a.txt", io.BytesIO(b"Content A."), "text/plain"),
    })
    r = authed_client.get("/api/v1/documents")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_unsupported_file_type_rejected(authed_client):
    r = authed_client.post("/api/v1/documents", files={
        "file": ("bad.exe", io.BytesIO(b"\x00\x01\x02"), "application/octet-stream"),
    })
    assert r.status_code == 422


def test_delete_document(authed_client):
    r = authed_client.post("/api/v1/documents", files={
        "file": ("del.txt", io.BytesIO(b"To be deleted."), "text/plain"),
    })
    doc_id = r.json()["id"]
    time.sleep(1)
    r2 = authed_client.delete(f"/api/v1/documents/{doc_id}")
    assert r2.status_code == 204
    # Listing should not include the deleted doc.
    docs = authed_client.get("/api/v1/documents").json()
    assert all(d["id"] != doc_id for d in docs)
