"""Document ingestion pipeline: validation, extraction, chunking, hashing.

Supports PDF, TXT, Markdown, and CSV (Module A). Chunking uses a recursive
character splitter with overlap. Content is SHA-256 hashed for duplicate
detection (Challenge 4 at live eval, plus dedup handling here).
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"pdf", "txt", "md", "csv", "markdown"}
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".csv"}

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB safety cap


@dataclass
class ExtractedDocument:
    text: str
    file_type: str
    num_pages_or_rows: int
    title: str


def detect_file_type(filename: str) -> str:
    lower = filename.lower()
    for ext in (".pdf", ".csv"):
        if lower.endswith(ext):
            return ext.lstrip(".")
    if lower.endswith(".markdown"):
        return "md"
    if lower.endswith(".md"):
        return "md"
    if lower.endswith(".txt"):
        return "txt"
    raise ValueError(
        f"Unsupported file type for '{filename}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
    )


def validate_file(filename: str, file_size: int) -> str:
    """Validate filename + size, returning the detected file type."""
    if file_size <= 0:
        raise ValueError("Uploaded file is empty.")
    if file_size > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large ({file_size} bytes). Maximum is {MAX_FILE_SIZE} bytes."
        )
    return detect_file_type(filename)


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- extraction
def extract_text(data: bytes, file_type: str, filename: str) -> ExtractedDocument:
    if file_type == "pdf":
        return _extract_pdf(data, filename)
    if file_type == "csv":
        return _extract_csv(data, filename)
    # txt / md / markdown
    text = data.decode("utf-8", errors="replace")
    title = _derive_title(filename, text)
    return ExtractedDocument(text=text, file_type=file_type, num_pages_or_rows=text.count("\n") + 1, title=title)


def _extract_pdf(data: bytes, filename: str) -> ExtractedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF processing") from exc

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n\n".join(pages).strip()
    title = _derive_title(filename, text)
    return ExtractedDocument(text=text, file_type="pdf", num_pages_or_rows=len(reader.pages), title=title)


def _extract_csv(data: bytes, filename: str) -> ExtractedDocument:
    text_content = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text_content))
    rows = list(reader)
    if not rows:
        return ExtractedDocument(text="", file_type="csv", num_pages_or_rows=0, title=filename)
    # Render CSV rows as readable text so the RAG pipeline can reason over it.
    # Each row becomes a line: "col1: val1 | col2: val2 ..." using the header row.
    header = rows[0]
    lines = []
    for row in rows[1:]:
        fields = [f"{h}: {v}" for h, v in zip(header, row) if v]
        if fields:
            lines.append(" | ".join(fields))
    text = "\n".join(lines)
    title = _derive_title(filename, text_content)
    return ExtractedDocument(text=text, file_type="csv", num_pages_or_rows=len(rows), title=title)


def _derive_title(filename: str, text: str) -> str:
    # Use the first non-empty, reasonably short line as a title fallback.
    for line in text.splitlines():
        line = line.strip()
        if 3 <= len(line) <= 120:
            return line
    return filename


# ---------------------------------------------------------------- chunking
def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Recursive character chunker that prefers splitting on paragraph /
    sentence / word boundaries.

    A recursive splitter is appropriate for heterogeneous enterprise docs
    because it keeps semantic units (sentences/paragraphs) intact when possible
    instead of cutting mid-word.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )
        return splitter.split_text(text)
    except Exception:
        # Fallback simple chunker if langchain_text_splitters is unavailable.
        return _simple_chunk(text, chunk_size, overlap)


def _simple_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Best-effort token count. Uses tiktoken when available, else heuristic."""
    if not text:
        return 0
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Rough estimate: ~4 chars per token for English text.
        return max(1, len(text) // 4)


def build_chunk_metadata(
    chunk_text: str,
    chunk_index: int,
    document_id: str,
    document_filename: str,
    file_type: str,
) -> dict[str, str | int]:
    return {
        "document_id": document_id,
        "document_filename": document_filename,
        "chunk_index": chunk_index,
        "file_type": file_type,
        "char_count": len(chunk_text),
        "token_count": count_tokens(chunk_text),
    }
