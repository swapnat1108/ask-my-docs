import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FAITHFULNESS_THRESHOLD = 0.70
ANSWER_RELEVANCE_THRESHOLD = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.70
RESULTS_FILE = "eval_results.json"

def load_test_dataset():
    dataset_path = Path("eval_dataset.json")
    if not dataset_path.exists():
        print("ERROR: eval_dataset.json not found.")
        sys.exit(1)
    with open(dataset_path) as f:
        data = json.load(f)
    return data

def run_pipeline_on_dataset(dataset):
    from rag_pipeline import query as rag_query
    questions, answers, contexts, ground_truths = [], [], [], []
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{len(dataset)}] {q[:70]}...")
        try:
            result = rag_query(q)
            answer = result["answer"]
            ctx = [chunk.text for chunk in result["chunks"]]
        except Exception as e:
            print(f"    ERROR: {e}")
            answer = ""
            ctx = []
        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)
    return {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}

def get_score(result, key):
    val = result[key]
    if isinstance(val, list):
        valid = [v for v in val if v is not None]
        return sum(valid) / len(valid) if valid else 0.0
    return float(val)

def evaluate_with_ragas(pipeline_data):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.chat_models import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    groq_key = os.environ.get("GROQ_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = groq_key

    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_key=groq_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )
    ragas_llm = LangchainLLMWrapper(llm)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    dataset = Dataset.from_dict(pipeline_data)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    print("Raw scores:", dict(result))
    return {
        "faithfulness": get_score(result, "faithfulness"),
        "answer_relevancy": get_score(result, "answer_relevancy"),
        "context_precision": get_score(result, "context_precision"),
    }

def main():
    print("=" * 55)
    print("  Ask My Docs - Evaluation Pipeline")
    print("=" * 55)
    dataset = load_test_dataset()
    print(f"  Found {len(dataset)} test cases")
    pipeline_data = run_pipeline_on_dataset(dataset)
    scores = evaluate_with_ragas(pipeline_data)
    passed = (
        scores["faithfulness"] >= FAITHFULNESS_THRESHOLD
        and scores["answer_relevancy"] >= ANSWER_RELEVANCE_THRESHOLD
        and scores["context_precision"] >= CONTEXT_PRECISION_THRESHOLD
    )
    print(scores)
    print("PASSED" if passed else "FAILED")
    results = {"timestamp": datetime.now().isoformat(), "scores": scores, "passed": passed}
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
