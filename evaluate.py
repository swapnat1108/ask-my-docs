import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULTS_FILE = "eval_results.json"

def load_test_dataset():
    dataset_path = Path("eval_dataset.json")
    if not dataset_path.exists():
        print("ERROR: eval_dataset.json not found.")
        sys.exit(1)
    with open(dataset_path) as f:
        return json.load(f)

def main():
    print("=" * 55)
    print("  Ask My Docs - Smoke Test CI")
    print("=" * 55)

    dataset = load_test_dataset()
    print(f"  Found {len(dataset)} test cases")

    from rag_pipeline import query as rag_query

    passed_count = 0
    results_detail = []

    for i, item in enumerate(dataset, 1):
        q = item["question"]
        print(f"  [{i}/{len(dataset)}] {q[:60]}...")
        try:
            result = rag_query(q)
            answer = result["answer"]
            chunks = result["chunks"]
            has_answer = len(answer.strip()) > 20
            has_chunks = len(chunks) > 0
            passed = has_answer and has_chunks
            if passed:
                passed_count += 1
            status = "PASS" if passed else "FAIL"
            print(f"    {status} — answer length: {len(answer)}, chunks: {len(chunks)}")
            results_detail.append({"question": q, "passed": passed, "answer_length": len(answer), "chunks_retrieved": len(chunks)})
        except Exception as e:
            print(f"    FAIL — error: {e}")
            results_detail.append({"question": q, "passed": False, "error": str(e)})

    pass_rate = passed_count / len(dataset)
    passed = pass_rate >= 0.8

    print()
    print(f"  Results: {passed_count}/{len(dataset)} passed ({pass_rate*100:.0f}%)")
    print("  PASSED" if passed else "  FAILED — less than 80% of queries returned answers")
    print("=" * 55)

    results = {
        "timestamp": datetime.now().isoformat(),
        "type": "smoke_test",
        "passed_count": passed_count,
        "total": len(dataset),
        "pass_rate": pass_rate,
        "passed": passed,
        "details": results_detail
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
