"""
evaluate.py — Phase 6: CI Evaluation Pipeline with Ragas
---------------------------------------------------------
What this does:
  Runs your 10 golden test questions through the full RAG pipeline,
  scores each answer on 3 Ragas metrics, and fails with exit code 1
  if any score drops below the threshold.

Ragas metrics:
  - Faithfulness:        Does the answer only contain facts from the context?
                         (catches hallucination)
  - Answer Relevance:    Does the answer actually address the question?
                         (catches tangential responses)
  - Context Precision:   Are the retrieved chunks actually useful?
                         (catches retrieval quality issues)

Target: All 3 metrics >= 0.70
Exit code 0 = pass (GitHub Actions green check)
Exit code 1 = fail (GitHub Actions blocks the deploy)

Run manually:
  python evaluate.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Thresholds (adjust after calibration) ─────────────────────────────────────
FAITHFULNESS_THRESHOLD      = 0.70
ANSWER_RELEVANCE_THRESHOLD  = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.70

RESULTS_FILE = "eval_results.json"


def load_test_dataset() -> list[dict]:
    """Load the golden Q&A pairs from eval_dataset.json."""
    dataset_path = Path("eval_dataset.json")
    if not dataset_path.exists():
        print("ERROR: eval_dataset.json not found.")
        print("Create it with at least 10 question/ground_truth pairs.")
        sys.exit(1)

    with open(dataset_path) as f:
        data = json.load(f)

    if len(data) < 5:
        print(f"WARNING: Only {len(data)} test cases found. Recommend at least 10.")
    return data


def run_pipeline_on_dataset(dataset: list[dict]) -> dict:
    """
    Run each question through the RAG pipeline.
    Returns a dict of lists in Ragas-compatible format.
    """
    from rag_pipeline import query as rag_query

    questions       = []
    answers         = []
    contexts        = []
    ground_truths   = []

    for i, item in enumerate(dataset, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i}/{len(dataset)}] {q[:70]}...")

        try:
            result = rag_query(q)
            answer = result["answer"]
            ctx    = [chunk.text for chunk in result["chunks"]]
        except Exception as e:
            print(f"    ERROR: {e}")
            answer = ""
            ctx    = []

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    }


def evaluate_with_ragas(pipeline_data: dict) -> dict:
    """Score the results using Ragas metrics."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

    print("\n  Scoring with Ragas (this calls the LLM — takes ~2 min)...")

    dataset = Dataset.from_dict(pipeline_data)

    # Ragas uses an LLM internally to score — we point it at Groq
    # via the standard OpenAI-compatible env vars
    os.environ["OPENAI_API_BASE"]   = "https://api.groq.com/openai/v1"
    os.environ["OPENAI_API_KEY"]    = os.environ.get("GROQ_API_KEY", "")
    os.environ["OPENAI_MODEL_NAME"] = "llama-3.1-70b-versatile"

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
        ],
    )

    scores = {
        "faithfulness":      float(result["faithfulness"]),
        "answer_relevancy":  float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
    }
```
    return scores


def print_report(scores: dict, passed: bool, n_questions: int):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 55)
    print("  Ragas Evaluation Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print(f"  Questions evaluated: {n_questions}")
    print()

    checks = [
        ("Faithfulness",      scores["faithfulness"],      FAITHFULNESS_THRESHOLD),
        ("Answer Relevance",  scores["answer_relevancy"],  ANSWER_RELEVANCE_THRESHOLD),
        ("Context Precision", scores["context_precision"], CONTEXT_PRECISION_THRESHOLD),
    ]

    for name, score, threshold in checks:
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {status}  {name:<22} {score:.3f}  [{bar}]  (min: {threshold})")

    print()
    if passed:
        print("  ✅ ALL CHECKS PASSED — deploy is approved")
    else:
        print("  ❌ CHECKS FAILED — fix retrieval quality before deploying")
    print("=" * 55)


def save_results(scores: dict, passed: bool, n_questions: int):
    """Save results to JSON for GitHub Actions artifacts."""
    results = {
        "timestamp":   datetime.now().isoformat(),
        "n_questions": n_questions,
        "scores":      scores,
        "thresholds": {
            "faithfulness":      FAITHFULNESS_THRESHOLD,
            "answer_relevancy":  ANSWER_RELEVANCE_THRESHOLD,
            "context_precision": CONTEXT_PRECISION_THRESHOLD,
        },
        "passed": passed,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {RESULTS_FILE}")


def main():
    print("=" * 55)
    print("  Ask My Docs — Evaluation Pipeline")
    print("=" * 55)

    print("\n[1/3] Loading test dataset...")
    dataset = load_test_dataset()
    print(f"  Found {len(dataset)} test cases")

    print("\n[2/3] Running RAG pipeline on test questions...")
    pipeline_data = run_pipeline_on_dataset(dataset)

    print("\n[3/3] Scoring with Ragas metrics...")
    scores = evaluate_with_ragas(pipeline_data)

    # Determine pass/fail
    passed = (
        scores["faithfulness"]      >= FAITHFULNESS_THRESHOLD
        and scores["answer_relevancy"]  >= ANSWER_RELEVANCE_THRESHOLD
        and scores["context_precision"] >= CONTEXT_PRECISION_THRESHOLD
    )

    # Report
    print_report(scores, passed, len(dataset))
    save_results(scores, passed, len(dataset))

    # Exit code used by GitHub Actions
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
