import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FAITHFULNESS_THRESHOLD      = 0.70
ANSWER_RELEVANCE_THRESHOLD  = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.70
RESULTS_FILE = "eval_results.json"


def load_test_dataset():
    dataset_path = Path("eval_dataset.json")
    if not dataset_path.exists():
        print("ERROR: eval_dataset.json not found.")
        sys.exit(1)
    with open(dataset_path) as f:
        data = json.load(f)
    if len(data) < 3:
        print(f"WARNING: Only {len(data)} test cases found.")
    return data


def run_pipeline_on_dataset(dataset):
    from rag_pipeline import query as rag_query
    questions, answers, contexts, ground_truths = [], [], [], []
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


def evaluate_with_ragas(pipeline_data):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

    print("\n  Scoring with Ragas...")

    os.environ["OPENAI_API_BASE"]   = "https://api.groq.com/openai/v1"
    os.environ["OPENAI_API_KEY"]    = os.environ.get("GROQ_API_KEY", "")
    os.environ["OPENAI_MODEL_NAME"] = "llama-3.3-70b-versatile"

    dataset = Dataset.from_dict(pipeline_data)
    result  = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
        ],
    )
    return {
        "faithfulness":      float(result["faithfulness"]),
        "answer_relevancy":  float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
    }


def print_report(scores, passed, n_questions):
    print("\n" + "=" * 55)
    print("  Ragas Evaluation Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print(f"  Questions evaluated: {n_questions}\n")
    checks = [
        ("Faithfulness",      scores["faithfulness"],      FAITHFULNESS_THRESHOLD),
        ("Answer Relevance",  scores["answer_relevancy"],  ANSWER_RELEVANCE_THRESHOLD),
        ("Context Precision", scores["context_precision"], CONTEXT_PRECISION_THRESHOLD),
    ]
    for name, score, threshold in checks:
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {status}  {name:<22} {score:.3f}  (min: {threshold})")
    print()
    if passed:
        print("  ALL CHECKS PASSED")
    else:
        print("  CHECKS FAILED")
    print("=" * 55)


def save_results(scores, passed, n_questions):
    results = {
        "timestamp":   datetime.now().isoformat(),
        "n_questions": n_questions,
        "scores":      scores,
        "passed":      passed,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


def main():
    print("=" * 55)
    print("  Ask My Docs - Evaluation Pipeline")
    print("=" * 55)

    print("\n[1/3] Loading test dataset...")
    dataset = load_test_dataset()
    print(f"  Found {len(dataset)} test cases")

    print("\n[2/3] Running RAG pipeline...")
    pipeline_data = run_pipeline_on_dataset(dataset)

    print("\n[3/3] Scoring with Ragas...")
    scores = evaluate_with_ragas(pipeline_data)

    passed = (
        scores["faithfulness"]      >= FAITHFULNESS_THRESHOLD
        and scores["answer_relevancy"]  >= ANSWER_RELEVANCE_THRESHOLD
        and scores["context_precision"] >= CONTEXT_PRECISION_THRESHOLD
    )

    print_report(scores, passed, len(dataset))
    save_results(scores, passed, len(dataset))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```
