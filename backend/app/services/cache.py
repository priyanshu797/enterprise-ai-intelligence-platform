"""Response cache for cost optimisation (Advanced Challenge 5).

A bounded in-memory cache keyed by a hash of (normalised query, filters,
owner). On a cache hit we return the stored answer without invoking the LLM,
saving tokens and latency. The cache stores the full ChatResponse payload so
sources and metadata are returned verbatim.

This is a deliberate, disclosed cost-optimisation layer, not a substitute for
LLM calls: only answers that were actually generated are cached, and cache
entries expire after `ttl_seconds` to avoid stale answers after new documents
are uploaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


class ResponseCache:
    def __init__(self, max_entries: int | None = None, ttl_seconds: int = 600):
        self.enabled = settings.ENABLE_RESPONSE_CACHE
        self.max_entries = max_entries or settings.CACHE_MAX_ENTRIES
        self.ttl = ttl_seconds
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

    @staticmethod
    def _key(owner_id: str | None, query: str, filters: dict | None) -> str:
        norm = " ".join(query.lower().split())
        raw = json.dumps({"q": norm, "f": filters or {}, "o": owner_id or ""}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, owner_id: str | None, query: str, filters: dict | None) -> Any | None:
        if not self.enabled:
            return None
        key = self._key(owner_id, query, filters)
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        logger.info("Response cache HIT (key=%s...)", key[:8])
        return value

    def put(self, owner_id: str | None, query: str, filters: dict | None, value: Any) -> None:
        if not self.enabled:
            return
        key = self._key(owner_id, query, filters)
        self._store[key] = (time.time(), value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def invalidate(self) -> None:
        """Clear the cache (e.g. after a new document upload)."""
        self._store.clear()
        logger.info("Response cache invalidated")

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "entries": len(self._store), "max_entries": self.max_entries}


_response_cache = ResponseCache()


def get_response_cache() -> ResponseCache:
    return _response_cache
