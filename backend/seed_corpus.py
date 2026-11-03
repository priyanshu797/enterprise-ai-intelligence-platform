"""Seed the evaluation corpus into the knowledge base.

Usage:
    python seed_corpus.py --email you@example.com --password yourpass

Uploads the documents described in evaluation/dataset.json under
`corpus_seed_documents` so the RAG pipeline has content to retrieve against
during evaluation and demos. Idempotent: re-running skips already-uploaded
documents (detected via content hash).
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API = sys.argv and "http://localhost:8000/api/v1"


def load_corpus() -> list[dict]:
    import json
    p = Path(__file__).resolve().parent / "evaluation" / "dataset.json"
    return json.loads(p.read_text(encoding="utf-8"))["corpus_seed_documents"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--base", default="http://localhost:8000/api/v1")
    args = parser.parse_args()

    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    tok = s.post(f"{args.base}/auth/login", json={"email": args.email, "password": args.password})
    tok.raise_for_status()
    s.headers.update({"Authorization": f"Bearer {tok.json()['access_token']}"})

    for doc in load_corpus():
        files = {"file": (doc["filename"], io.BytesIO(doc["content"].encode("utf-8")), "text/plain")}
        r = s.post(f"{args.base}/documents", files=files)
        if r.status_code == 201:
            data = r.json()
            logger.info("Uploaded %s -> status=%s", doc["filename"], data["status"])
            if data["status"] == "duplicate":
                continue
            # Poll until processed.
            for _ in range(40):
                j = s.get(f"{args.base}/documents/{data['id']}/job")
                if j.status_code == 200 and j.json()["status"] in ("completed", "failed"):
                    logger.info("  -> %s", j.json()["status"])
                    break
                time.sleep(0.5)
        else:
            logger.warning("Upload failed for %s: %s", doc["filename"], r.text)
    logger.info("Corpus seed complete.")


if __name__ == "__main__":
    main()
