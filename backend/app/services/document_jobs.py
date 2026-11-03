"""Asynchronous document processing (Challenge 6).

Document ingestion (extraction, chunking, embedding, vector storage) is
offloaded to a background worker so the upload API returns immediately with a
queued job. Status is polled via the job/document status endpoints.

Implementation: a bounded `ThreadPoolExecutor` (process documents
concurrently up to MAX_CONCURRENT_DOC_JOBS) fed from an in-process queue. Each
job owns its own DB session (background threads cannot share the request
session). This keeps the system self-contained without an external worker
(Redis/Celery); for horizontal scale, swap `_submit` to enqueue onto a
Celery queue — the job logic below is unchanged.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Document, DocumentStatus, JobStatus, ProcessingJob
from .document_processing import (
    build_chunk_metadata,
    chunk_text,
    content_hash,
    count_tokens,
    extract_text,
    validate_file,
)
from .embeddings import get_embeddings_service
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=settings.MAX_CONCURRENT_DOC_JOBS,
                    thread_name_prefix="doc-worker",
                )
                logger.info("Document worker pool started (workers=%s)", settings.MAX_CONCURRENT_DOC_JOBS)
    return _executor


def submit_processing_job(
    db: Session,
    document: Document,
    file_data: bytes,
) -> ProcessingJob:
    """Queue a document for background processing, returning the job record."""
    job = ProcessingJob(document_id=document.id, status=JobStatus.QUEUED)
    db.add(job)
    db.commit()
    db.refresh(job)
    # Store file data is the caller's responsibility; here we hand bytes to the
    # worker via the future so we don't need to persist raw files.
    _get_executor().submit(_process, job.id, document.id, file_data)
    return job


def _process(job_id: str, document_id: str, file_data: bytes) -> None:
    """Background worker: extract -> chunk -> embed -> store -> update status."""
    # Resolve the session factory dynamically so background threads use the
    # currently-configured engine (important for tests that swap the engine).
    from ..database import SessionLocal

    db = SessionLocal()
    started = time.perf_counter()
    try:
        job = db.get(ProcessingJob, job_id)
        document = db.get(Document, document_id)
        if not job or not document:
            logger.error("Job/doc not found: job=%s doc=%s", job_id, document_id)
            return
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        document.status = DocumentStatus.PROCESSING
        db.commit()

        # 1. Extract.
        _update_progress(db, job, 0.1)
        extracted = extract_text(file_data, document.file_type, document.filename)
        document.title = extracted.title
        _update_progress(db, job, 0.3)

        # 2. Chunk.
        chunks = chunk_text(extracted.text)
        if not chunks:
            raise ValueError("No text content could be extracted from the document.")
        _update_progress(db, job, 0.5)

        # 3. Embed + store vectors.
        store = get_vector_store()
        embeddings_svc = get_embeddings_service()
        chunk_records = []
        metadatas = []
        vector_ids = []
        for i, ctext in enumerate(chunks):
            meta = build_chunk_metadata(ctext, i, document.id, document.filename, document.file_type)
            # SECURITY: always stamp owner_id into vector metadata so retrieval
            # can be scoped server-side — never rely on caller-supplied filters.
            meta["owner_id"] = document.owner_id
            vid = f"{document.id}_chunk_{i}"
            vector_ids.append(vid)
            metadatas.append(meta)
            chunk_records.append((ctext, i, vid, meta))
        store.add(texts=[c[0] for c in chunk_records], metadatas=metadatas, ids=vector_ids)
        _update_progress(db, job, 0.8)

        # 4. Persist chunks.
        from ..models import DocumentChunk

        for ctext, i, vid, meta in chunk_records:
            db.add(DocumentChunk(
                document_id=document.id,
                chunk_index=i,
                text=ctext,
                vector_id=vid,
                token_count=meta["token_count"],
                char_count=meta["char_count"],
            ))

        document.num_chunks = len(chunks)
        document.status = DocumentStatus.COMPLETED
        document.processed_at = datetime.now(timezone.utc)
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Document %s processed (%s chunks in %.2fs)",
                    document.id, len(chunks), time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document processing failed for %s", document_id)
        try:
            job = db.get(ProcessingJob, job_id)
            document = db.get(Document, document_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)[:2000]
                job.completed_at = datetime.now(timezone.utc)
            if document:
                document.status = DocumentStatus.FAILED
                document.error_message = str(exc)[:2000]
            db.commit()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to record processing failure")
    finally:
        db.close()


def _update_progress(db: Session, job: ProcessingJob, progress: float) -> None:
    job.progress = progress
    db.commit()


def is_duplicate(db: Session, owner_id: str, data: bytes) -> Document | None:
    """Return an existing completed document with the same content hash, if any."""
    h = content_hash(data)
    existing = db.query(Document).filter(
        Document.owner_id == owner_id,
        Document.content_hash == h,
    ).first()
    return existing


def register_upload(
    db: Session,
    owner_id: str,
    filename: str,
    file_data: bytes,
) -> tuple[Document, str]:
    """Validate + register a document, handling duplicates.

    Returns (document, status) where status is "created" or "duplicate".
    """
    file_type = validate_file(filename, len(file_data))
    h = content_hash(file_data)
    existing = db.query(Document).filter(
        Document.owner_id == owner_id,
        Document.content_hash == h,
    ).first()
    if existing:
        existing.status = DocumentStatus.DUPLICATE
        db.commit()
        return existing, "duplicate"
    doc = Document(
        owner_id=owner_id,
        filename=filename,
        file_type=file_type,
        file_size=len(file_data),
        content_hash=h,
        title=filename,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc, "created"
