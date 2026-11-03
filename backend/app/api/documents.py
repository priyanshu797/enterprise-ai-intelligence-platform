"""Document management endpoints (Module A + Challenge 6 async processing).

POST /documents           — upload (async processing queued, returns immediately)
GET  /documents           — list the user's documents
GET  /documents/{id}       — document detail
GET  /documents/{id}/chunks — view processed chunks
DELETE /documents/{id}     — remove document + its vectors
GET  /documents/{id}/job   — processing job status
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Document, DocumentChunk, ProcessingJob, User
from ..schemas import DocumentChunkOut, DocumentOut, JobOut
from ..security.auth import get_current_active_user
from ..services.cache import get_response_cache
from ..services.document_jobs import register_upload, submit_processing_job
from ..services.usage import log_activity
from ..services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    raw = await file.read()
    try:
        doc, status = register_upload(db, user.id, file.filename or "upload.bin", raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if title:
        doc.title = title
        db.commit()
        db.refresh(doc)
    if status == "duplicate":
        log_activity(db, owner_id=user.id, action="upload", detail=f"duplicate: {doc.filename}", status="duplicate")
        return doc
    # Queue async processing.
    submit_processing_job(db, doc, raw)
    log_activity(db, owner_id=user.id, action="upload", detail=f"{doc.filename} ({doc.file_type})")
    # Invalidate cache since the knowledge base changed.
    get_response_cache().invalidate()
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
    status_filter: str | None = None,
):
    q = select(Document).where(Document.owner_id == user.id)
    if status_filter:
        q = q.where(Document.status == status_filter)
    q = q.order_by(Document.created_at.desc())
    return db.execute(q).scalars().all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    doc = db.get(Document, document_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/{document_id}/chunks", response_model=list[DocumentChunkOut])
def get_chunks(document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    doc = db.get(Document, document_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.chunks


@router.get("/{document_id}/job", response_model=JobOut)
def get_job(document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    doc = db.get(Document, document_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    job = db.execute(
        select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc()).limit(1)
    ).scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="No processing job found for this document")
    return job


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    doc = db.get(Document, document_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    # Remove vectors from the store.
    try:
        get_vector_store().delete_document(document_id)
    except Exception:
        pass
    db.delete(doc)
    db.commit()
    get_response_cache().invalidate()
    log_activity(db, owner_id=user.id, action="delete", detail=document_id)
    return None
