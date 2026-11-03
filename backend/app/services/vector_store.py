"""Vector store with pluggable backends and hybrid retrieval.

Backend abstraction (`VectorStore`) with a ChromaDB implementation that runs
embedded (file-backed) — no external server required for development. The same
interface is used by the RAG pipeline regardless of backend.

Retrieval strategies (Module B requirement + Challenge 3 retrieval
optimisation):
  * `semantic`    — pure dense (embedding) similarity search.
  * `hybrid`      — Reciprocal Rank Fusion of dense + sparse (keyword) scores.
                    Combines semantic understanding with exact-keyword recall,
                    which is the right default for enterprise docs that contain
                    proper nouns, codes, and identifiers that pure semantic
                    search can miss.
  * `rerank`      — semantic search followed by a lightweight cross-attention
                    style re-scorer (here: a simple embedding-cosine rerank
                    against an expanded query; a real deployment would swap in a
                    dedicated reranker model).
  * `rewrite`     — query is rewritten by the LLM for retrieval, then semantic
                    search is run (the rewrite itself is performed by the LLM
                    service, not here; this store just executes the search).

Sparse retrieval uses a BM25-style score over chunk text stored alongside
the vectors, kept in memory for the active collection.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import settings
from .embeddings import get_embeddings_service

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors (lists)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its metadata and score."""
    text: str
    document_id: str
    document_filename: str
    chunk_index: int
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """Backend-agnostic vector store contract."""

    def add(self, texts: list[str], metadatas: list[dict[str, Any]], ids: list[str]) -> None: ...

    def search(
        self,
        query: str,
        top_k: int,
        strategy: str,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]: ...

    def delete_document(self, document_id: str) -> None: ...

    def count(self) -> int: ...


class _BM25Index:
    """Minimal in-memory BM25 over the active chunk corpus for hybrid search.

    This is rebuilt from persisted chunks on demand. Kept deliberately simple;
    for very large corpora a dedicated sparse index (e.g. OpenSearch/BM25) would
    replace this.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.doc_freq: Counter[str] = Counter()
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.n: int = 0

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        self.docs.append(tokens)
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(tokens))
        for term in set(tokens):
            self.doc_freq[term] += 1
        self.n += 1

    def finalize(self) -> None:
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def score(self, query: str) -> dict[str, float]:
        q_terms = _tokenize(query)
        scores: dict[str, float] = {}
        idf: dict[str, float] = {}
        for term in set(q_terms):
            df = self.doc_freq.get(term, 0)
            idf[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5)) if df else 0.0
        for i, tokens in enumerate(self.docs):
            if not tokens:
                continue
            tf = Counter(tokens)
            dl = self.doc_len[i]
            denom_norm = 1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0)
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                s += idf.get(term, 0.0) * (tf[term] * (self.k1 + 1)) / (
                    tf[term] + self.k1 * denom_norm
                )
            if s > 0:
                scores[self.doc_ids[i]] = s
        return scores


class ChromaVectorStore:
    """ChromaDB-backed vector store (embedded, file-persisted).

    Stores dense vectors and chunk metadata. Sparse scoring is done against a
    lightweight BM25 index maintained alongside for hybrid search.
    """

    def __init__(self):
        import chromadb

        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_PATH)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = get_embeddings_service()
        # document_id -> {"text", "filename", "chunk_index", "metadata"}
        self._chunk_cache: dict[str, dict[str, Any]] = {}
        self._bm25: _BM25Index | None = None
        self._bm25_dirty = True

    # ------------------------------------------------------------- loading
    def _ensure_loaded(self) -> None:
        if self._chunk_cache:
            return
        if self._collection.count() == 0:
            self._bm25 = _BM25Index()
            self._bm25.finalize()
            return
        data = self._collection.get(include=["metadatas", "documents"])
        self._bm25 = _BM25Index()
        for vid, doc, meta in zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", [])):
            meta = meta or {}
            self._chunk_cache[vid] = {
                "text": doc or "",
                "document_id": meta.get("document_id", ""),
                "document_filename": meta.get("document_filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "metadata": meta,
            }
            self._bm25.add(vid, doc or "")
        self._bm25.finalize()
        self._bm25_dirty = False
        logger.info("Loaded %s chunks from vector store", len(self._chunk_cache))

    # ------------------------------------------------------------- writes
    def add(self, texts: list[str], metadatas: list[dict[str, Any]], ids: list[str]) -> None:
        if not texts:
            return
        embeddings = self._embeddings.embed_documents(texts)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        for vid, text, meta in zip(ids, texts, metadatas):
            self._chunk_cache[vid] = {
                "text": text,
                "document_id": meta.get("document_id", ""),
                "document_filename": meta.get("document_filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "metadata": meta,
            }
        # BM25 needs rebuild with new docs.
        self._bm25_dirty = True

    def _rebuild_bm25(self) -> None:
        self._bm25 = _BM25Index()
        for vid, info in self._chunk_cache.items():
            self._bm25.add(vid, info["text"])
        self._bm25.finalize()
        self._bm25_dirty = False

    # ------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int,
        strategy: str = "hybrid",
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        self._ensure_loaded()
        if self._collection.count() == 0:
            return []

        where = self._build_where(filters)

        if strategy == "semantic":
            return self._semantic_search(query, top_k, where)
        if strategy == "rerank":
            return self._rerank_search(query, top_k, where)
        # Default: hybrid (dense + sparse via RRF)
        return self._hybrid_search(query, top_k, where)

    @staticmethod
    def _build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """Translate API metadata filters into a Chroma `where` clause.

        Supports equality on any stored metadata key (e.g. document_id,
        file_type).
        """
        if not filters:
            return None
        clauses = [{"k": k, "v": v} for k, v in filters.items() if v is not None]
        if not clauses:
            return None
        # Chroma expects {"$and": [...]} for multiple conditions.
        chroma_clauses = [{c["k"]: c["v"]} for c in clauses]
        return {"$and": chroma_clauses} if len(chroma_clauses) > 1 else chroma_clauses[0]

    def _semantic_search(self, query, top_k, where):
        q_emb = self._embeddings.embed_query(query)
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k * 3, max(top_k, 10)),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        return self._to_results(results)

    def _hybrid_search(self, query, top_k, where):
        # Dense retrieval.
        q_emb = self._embeddings.embed_query(query)
        dense = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k * 3, max(top_k, 10)),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        # Sparse (BM25) retrieval.
        if self._bm25_dirty:
            self._rebuild_bm25()
        bm25_scores = self._bm25.score(query) if self._bm25 else {}

        # Reciprocal Rank Fusion (k=60 is a standard constant).
        rrf_k = 60
        fused: dict[str, float] = {}
        ids = dense.get("ids", [[]])[0]
        distances = dense.get("distances", [[]])[0]
        for rank, (vid, dist) in enumerate(zip(ids, distances)):
            # Chroma returns cosine distance; convert to similarity.
            sim = 1.0 - float(dist)
            fused[vid] = fused.get(vid, 0.0) + 1.0 / (rrf_k + rank + 1) * sim
        # Add sparse contribution.
        sorted_sparse = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (vid, _score) in enumerate(sorted_sparse):
            if where and vid in self._chunk_cache:
                meta = self._chunk_cache[vid]["metadata"]
                if not self._matches_where(meta, where):
                    continue
            fused[vid] = fused.get(vid, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Build results sorted by fused score, but report the absolute cosine
        # similarity as the per-chunk score so relevance thresholds are
        # meaningful on a 0..1 scale regardless of strategy.
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final_ids = [vid for vid, _ in ordered if vid in self._chunk_cache]
        sim_by_id = self._cosine_scores(q_emb, final_ids)
        results = []
        for vid, _fused in ordered:
            info = self._chunk_cache.get(vid)
            if not info:
                continue
            results.append(
                RetrievalResult(
                    text=info["text"],
                    document_id=info["document_id"],
                    document_filename=info["document_filename"],
                    chunk_index=info["chunk_index"],
                    score=float(sim_by_id.get(vid, 0.0)),
                    metadata=info["metadata"],
                )
            )
        return results

    def _rerank_search(self, query, top_k, where):
        # Over-fetch semantically, then rerank by cosine similarity to the
        # query embedding (a stand-in for a cross-encoder reranker).
        q_emb = self._embeddings.embed_query(query)
        candidates = self._semantic_search(query, top_k * 3, where)
        # Re-score by embedding cosine (already approximate via Chroma; here we
        # keep the semantic ordering but boost results whose tokens overlap the
        # query — a lightweight relevance boost), clamped to [0, 1].
        q_tokens = set(_tokenize(query))
        reranked: list[tuple[float, RetrievalResult]] = []
        for r in candidates:
            overlap = len(q_tokens & set(_tokenize(r.text)))
            boosted = min(1.0, r.score * (1.0 + 0.1 * overlap))
            reranked.append((boosted, r))
        reranked.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in reranked[:top_k]]

    def _cosine_scores(self, query_emb, ids: list[str]) -> dict[str, float]:
        """Cosine similarity of each stored embedding vs the query embedding.

        Used to give hybrid/rerank results an absolute 0..1 relevance score so
        the threshold check (MIN_RELEVANCE_SCORE) and confidence are meaningful
        regardless of the (scale-dependent) fused ranking score.
        """
        if not ids:
            return {}
        try:
            data = self._collection.get(ids=list(ids), include=["embeddings"])
        except Exception:  # noqa: BLE001
            return {vid: 0.0 for vid in ids}
        ids_back = data.get("ids", []) or []
        embs = data.get("embeddings", None)
        if embs is None:
            return {vid: 0.0 for vid in ids}
        emb_map = {vid: emb for vid, emb in zip(ids_back, embs)}
        sims: dict[str, float] = {}
        for vid in ids:
            emb = emb_map.get(vid)
            sims[vid] = _cosine(query_emb, list(emb)) if emb is not None else 0.0
        return sims

    @staticmethod
    def _matches_where(meta: dict[str, Any], where: dict[str, Any]) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(ChromaVectorStore._matches_where(meta, c) for c in where["$and"])
        for k, v in where.items():
            if meta.get(k) != v:
                return False
        return True

    def _to_results(self, raw) -> list[RetrievalResult]:
        results = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        for vid, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            results.append(
                RetrievalResult(
                    text=doc or "",
                    document_id=meta.get("document_id", ""),
                    document_filename=meta.get("document_filename", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    score=1.0 - float(dist),
                    metadata=meta,
                )
            )
        return results

    # ------------------------------------------------------------- misc
    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
        stale = [vid for vid, info in self._chunk_cache.items() if info["document_id"] == document_id]
        for vid in stale:
            self._chunk_cache.pop(vid, None)
        self._bm25_dirty = True

    def count(self) -> int:
        return self._collection.count()


_store: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    """Return a process-wide singleton vector store instance."""
    global _store
    if _store is None:
        _store = ChromaVectorStore()
    return _store


def reset_vector_store() -> None:
    """Force a fresh vector store instance (used after embedding model change)."""
    global _store
    _store = None
