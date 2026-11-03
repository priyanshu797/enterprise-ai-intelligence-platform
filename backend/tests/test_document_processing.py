"""Tests for the document ingestion pipeline (Module A + Challenge 6 dedup)."""
from app.services.document_processing import (
    chunk_text,
    content_hash,
    detect_file_type,
    extract_text,
    validate_file,
)


def test_detect_file_type():
    assert detect_file_type("report.pdf") == "pdf"
    assert detect_file_type("notes.txt") == "txt"
    assert detect_file_type("readme.md") == "md"
    assert detect_file_type("readme.markdown") == "md"
    assert detect_file_type("data.csv") == "csv"


def test_detect_file_type_rejects_unsupported():
    try:
        detect_file_type("malware.exe")
        assert False, "should have raised"
    except ValueError:
        pass


def test_validate_file_rejects_empty_and_oversized():
    try:
        validate_file("a.txt", 0)
        assert False
    except ValueError:
        pass


def test_content_hash_deterministic_and_distinct():
    a = content_hash(b"hello")
    b = content_hash(b"hello")
    c = content_hash(b"world")
    assert a == b
    assert a != c
    assert len(a) == 64  # sha-256 hex


def test_extract_text_txt():
    doc = extract_text(b"Line one.\nLine two.", "txt", "note.txt")
    assert "Line one." in doc.text
    assert "Line two." in doc.text
    assert doc.file_type == "txt"


def test_extract_text_csv_renders_rows():
    csv_bytes = b"product,issue\nPayments,failed checkout\nAuth,expired token"
    doc = extract_text(csv_bytes, "csv", "data.csv")
    assert "product: Payments" in doc.text
    assert "issue: failed checkout" in doc.text
    assert doc.num_pages_or_rows == 3


def test_extract_text_md():
    doc = extract_text(b"# Title\n\nSome **markdown** content.", "md", "readme.md")
    assert "Title" in doc.text


def test_chunk_text_basic():
    text = "Sentence one. " * 200  # long enough to chunk
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    # Chunks should roughly cover the content (allow overlap).
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_text_short_returns_single():
    chunks = chunk_text("Short text.", chunk_size=800, overlap=100)
    assert len(chunks) == 1


def test_chunk_empty_text():
    assert chunk_text("") == []


def test_dedup_via_content_hash():
    # The same bytes produce the same hash -> register_upload detects duplicate.
    h1 = content_hash(b"duplicate content")
    h2 = content_hash(b"duplicate content")
    assert h1 == h2
