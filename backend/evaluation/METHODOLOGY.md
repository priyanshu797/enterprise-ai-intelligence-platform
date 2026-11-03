# Evaluation Methodology & Limitations

## Methodology

The evaluation suite (`evaluation/run_evaluation.py`) measures the RAG pipeline against a 22-question dataset (`evaluation/dataset.json`) covering three behaviours: `answered` (should retrieve + answer correctly), `insufficient` (no relevant docs; should refuse), and `conflicting` (sources disagree; should flag the conflict). The dataset also includes a prompt-injection probe.

### Metrics per question

| Metric | How it's measured |
|---|---|
| **Retrieval relevance** (0–1) | Token-Jaccard overlap between retrieved passage text and the question's `reference_context`, combined with the LLM-judge grounding score when sources exist. |
| **Answer correctness** (0–1) | LLM-as-judge comparing the generated answer to the expected answer's substance; falls back to token-Jaccard when no LLM is available. |
| **Source grounding** (0–1) | LLM-judge scoring of how well the answer is supported by the retrieved sources; token-Jaccard fallback. |
| **Hallucination flag** (bool) | Behavioural: for `insufficient` questions, true if the system produced a substantive answer instead of refusing. For `answered` questions, true if correctness AND grounding are both very low. The injection case is flagged if the model appears to comply/leak. |
| **Latency (ms)** | Wall-clock of the `rag.answer()` call. |
| **Tokens** | prompt + completion tokens from the LLM response. |

### Aggregates
Average of each metric across all questions, plus `hallucination_rate` (fraction flagged) and total/average tokens.

### Why a custom framework
RAGAS/DeepEval are listed as optional dependencies and would require a fully-connected eval graph; this custom runner works offline with the local HuggingFace embeddings and any configured LLM, so evaluation can run without paid APIs. The metrics align with the assessment's required axes (retrieval relevance, answer correctness, source grounding, hallucination handling, latency, token consumption).

### Running
```bash
# 1. Seed the corpus first (creates the documents the questions reference)
python seed_corpus.py --email you@example.com --password yourpass

# 2. Run the suite
python evaluation/run_evaluation.py --out evaluation/evaluation_report.json
```

Results are written to JSON and persisted as an `evaluation_run` (with one `evaluation_result` row per question) for the dashboard.

## Limitations (stated honestly)

1. **LLM-as-judge is approximate.** Correctness/grounding scores depend on the judge model's judgement; a different model may score differently. The token-Jaccard fallback is a coarse proxy.
2. **Retrieval relevance via Jaccard** rewards lexical overlap and may undervalue semantically-equivalent paraphrases. The LLM-judge component mitigates this but does not eliminate it.
3. **The dataset is small and hand-authored** (22 questions) around a synthetic corpus. It is a targeted regression suite, not a statistically representative benchmark. It is easy to overfit to.
4. **Hallucination detection is behavioural, not semantic.** We check whether the system *refused* for insufficient questions and whether answers are low-quality for answered ones; we do not do full claim-level entailment checking.
5. **The injection test is a single probe.** Passing it does not prove robustness to novel injection; the production defense is defence-in-depth, not a guarantee.
6. **Latency includes everything** the RAG flow does (retrieval + injection sanitization + LLM + verification), so it is representative of real user-facing latency but not decomposed per stage in this report.
7. **Cost/token estimates** use approximate per-1M-token rates that change over time.
8. **Results depend on the configured LLM.** With no provider configured, only the "insufficient" path can be exercised; full correctness scores require a real LLM.

## Interpretation guidance

Read the aggregate `hallucination_rate` together with the per-question `notes` field, which records the behavioural judgement (e.g. "correctly refused", "conflict not flagged", "low correctness+grounding"). A low hallucination rate + high correctness on the `answered` subset + correct refusals on the `insufficient` subset is the target profile.
