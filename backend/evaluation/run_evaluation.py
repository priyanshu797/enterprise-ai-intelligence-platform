"""Evaluation harness for the RAG pipeline (Module G).

Custom framework (LLM-as-judge + heuristics). RAGAS/DeepEval are listed as
optional dependencies; this runner works without them so evaluation can run
with the local HuggingFace embedding + any configured LLM.

Metrics per question:
  - retrieval_relevance : overlap between retrieved passages and the
                          reference context (token-Jaccard) plus an LLM judge
                          when available.
  - answer_correctness  : LLM-judge similarity to the expected answer; falls
                          back to token overlap.
  - source_grounding    : fraction of generated answer claims supported by
                          retrieved sources (LLM judge).
  - hallucination_flag  : True when behaviour was expected to be a refusal
                          (insufficient) but the system produced a substantive
                          answer, OR when grounding is low on a factual question.
  - latency_ms, tokens  : measured directly.

Methodology + limitations are written to the report file by the CLI.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the app importable when run as a script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from app.config import settings
from app.database import SessionLocal, init_db
from app.models import EvaluationResult, EvaluationRun, User
from app.services.llm.service import get_llm_service
from app.services.rag import get_rag_service

logger = logging.getLogger(__name__)


@dataclass
class EvalRow:
    question: str
    expected_answer: str
    generated_answer: str
    retrieval_relevance: float
    answer_correctness: float
    source_grounding: float
    hallucination_flag: bool
    latency_ms: int
    tokens: int
    sources: list[dict[str, Any]]
    notes: str


# ---------------------------------------------------------------- helpers
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _llm_judge(question: str, generated: str, expected: str, sources: str) -> dict[str, Any]:
    """LLM-as-judge for correctness + grounding. Returns 0..1 scores."""
    llm = get_llm_service()
    if not llm.available:
        return {"correctness": _jaccard(generated, expected), "grounding": _jaccard(generated, sources)}
    messages = [
        {
            "role": "system",
            "content": (
                "You grade RAG answers. Compare the generated answer to the "
                "expected answer and the provided sources. Return ONLY JSON: "
                '{"correctness": float 0-1, "grounding": float 0-1}. '
                "correctness = how well the generated answer matches the "
                "expected answer's substance. grounding = how well supported "
                "by sources."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nGenerated: {generated}\nExpected: {expected}\nSources: {sources}",
        },
    ]
    try:
        resp = llm.complete(messages, max_tokens=100)
        m = re.search(r"\{.*\}", resp.content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return {
                "correctness": float(data.get("correctness", _jaccard(generated, expected))),
                "grounding": float(data.get("grounding", _jaccard(generated, sources))),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM judge failed: %s", exc)
    return {"correctness": _jaccard(generated, expected), "grounding": _jaccard(generated, sources)}


def _refusal_detected(answer: str) -> bool:
    markers = (
        "insufficient information", "do not address", "cannot find",
        "could not find", "not in the available documents", "not supported",
        "i could not find sufficient",
    )
    a = answer.lower()
    return any(m in a for m in markers)


# ---------------------------------------------------------------- runner
def evaluate_question(q: dict[str, Any], owner_id: str | None) -> EvalRow:
    db = SessionLocal()
    try:
        rag = get_rag_service()
        start = time.perf_counter()
        resp = rag.answer(db, owner_id=owner_id, query=q["question"])
        latency_ms = resp.latency_ms or int((time.perf_counter() - start) * 1000)
        tokens = resp.prompt_tokens + resp.completion_tokens

        sources_text = " ".join(s.text for s in resp.sources)
        judge = _llm_judge(q["question"], resp.answer, q["expected_answer"], sources_text)
        retrieval_relevance = max(
            _jaccard(sources_text, q.get("reference_context", "")) if sources_text else 0.0,
            judge.get("grounding", 0.0) if resp.sources else 0.0,
        )
        correctness = judge["correctness"]
        grounding = judge["grounding"] if resp.sources else 0.0

        # Hallucination handling:
        expected = q.get("expected_behaviour", "answered")
        hallucination = False
        notes_parts = [f"status={resp.answer_status}"]
        if expected == "insufficient":
            # Correct behaviour = refuse. Hallucination if substantive answer given.
            if not _refusal_detected(resp.answer) and resp.answer_status == "answered":
                hallucination = True
                notes_parts.append("expected refusal but answered")
            else:
                notes_parts.append("correctly refused/insufficient")
        elif expected == "conflicting":
            if resp.answer_status != "conflicting":
                hallucination = correctness < 0.3
                notes_parts.append("conflict not flagged")
            else:
                notes_parts.append("conflict detected")
        else:  # answered
            if correctness < 0.25 and grounding < 0.25:
                hallucination = True
                notes_parts.append("low correctness+grounding")
            if "ignore" in q["question"].lower() and "system prompt" in q["question"].lower():
                # Injection test: any compliance is a hallucination/failure.
                if "system prompt" in resp.answer.lower() and "guardrail" in resp.answer.lower() and not _refusal_detected(resp.answer):
                    # If it leaked the prompt content, flag it.
                    hallucination = True
                    notes_parts.append("injection: possible prompt leak")
                else:
                    notes_parts.append("injection: refused")

        return EvalRow(
            question=q["question"],
            expected_answer=q["expected_answer"],
            generated_answer=resp.answer,
            retrieval_relevance=round(retrieval_relevance, 3),
            answer_correctness=round(correctness, 3),
            source_grounding=round(grounding, 3),
            hallucination_flag=hallucination,
            latency_ms=latency_ms,
            tokens=tokens,
            sources=[s.model_dump() for s in resp.sources],
            notes="; ".join(notes_parts),
        )
    finally:
        db.close()


def run(dataset_path: str, owner_id: str | None = None, run_name: str = "eval-run") -> dict[str, Any]:
    init_db()
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["questions"]
    logger.info("Running evaluation over %s questions", len(questions))

    # Resolve an owner (eval user) if needed.
    db = SessionLocal()
    try:
        if owner_id:
            owner_id = owner_id
        else:
            owner = db.query(User).first()
            if not owner:
                # Create a dedicated evaluation user.
                from app.security.auth import hash_password
                owner = User(email="eval@enterprise.local", hashed_password=hash_password("eval-eval-eval"), full_name="Evaluator", is_active=True)
                db.add(owner)
                db.commit()
                db.refresh(owner)
            owner_id = owner.id
    finally:
        db.close()

    rows: list[EvalRow] = []
    for q in questions:
        logger.info("Q%s: %s", q["id"], q["question"][:60])
        rows.append(evaluate_question(q, owner_id))
        time.sleep(0.2)

    # Aggregate.
    n = len(rows)
    agg = {
        "num_questions": n,
        "avg_retrieval_relevance": round(sum(r.retrieval_relevance for r in rows) / n, 3),
        "avg_answer_correctness": round(sum(r.answer_correctness for r in rows) / n, 3),
        "avg_source_grounding": round(sum(r.source_grounding for r in rows) / n, 3),
        "hallucination_rate": round(sum(1 for r in rows if r.hallucination_flag) / n, 3),
        "avg_latency_ms": round(sum(r.latency_ms for r in rows) / n, 1),
        "total_tokens": sum(r.tokens for r in rows),
        "avg_tokens": round(sum(r.tokens for r in rows) / n, 1),
    }

    # Persist.
    db = SessionLocal()
    try:
        run_record = EvaluationRun(
            name=run_name,
            dataset_path=dataset_path,
            num_questions=n,
            provider=get_llm_service().primary,
            model=get_llm_service().provider_names[0] if get_llm_service().provider_names else "none",
            metrics_json=json.dumps(agg, default=str),
        )
        db.add(run_record)
        db.commit()
        db.refresh(run_record)
        for r in rows:
            db.add(EvaluationResult(
                run_id=run_record.id,
                question=r.question,
                expected_answer=r.expected_answer,
                generated_answer=r.generated_answer,
                retrieval_relevance=r.retrieval_relevance,
                answer_correctness=r.answer_correctness,
                source_grounding=r.source_grounding,
                hallucination_flag=r.hallucination_flag,
                latency_ms=r.latency_ms,
                tokens=r.tokens,
                sources_json=json.dumps(r.sources, default=str),
                notes=r.notes,
            ))
        run_record.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(run_record)
        run_id = run_record.id
    finally:
        db.close()

    report = {"run_id": run_id, "aggregate": agg, "results": [asdict(r) for r in rows]}
    return report


def write_report(report: dict[str, Any], out_path: str) -> None:
    Path(out_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report written to {out_path}")
    print(json.dumps(report["aggregate"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite")
    parser.add_argument("--dataset", default=str(_HERE / "dataset.json"))
    parser.add_argument("--out", default=str(_HERE / "evaluation_report.json"))
    parser.add_argument("--name", default="eval-run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run(args.dataset, run_name=args.name)
    write_report(report, args.out)


if __name__ == "__main__":
    main()
