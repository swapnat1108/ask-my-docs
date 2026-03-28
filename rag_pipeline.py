"""
rag_pipeline.py — Phase 4: Hybrid Retrieval + Reranking + LLM
--------------------------------------------------------------
Pipeline flow for every user question:

  User query
      │
      ├── BM25 keyword search (rank_bm25)     → top 20 chunks
      ├── Vector similarity search (ChromaDB) → top 20 chunks
      │
      └── Reciprocal Rank Fusion (RRF)        → merged top 20
              │
              └── Cohere Rerank v3            → top 5 chunks
                      │
                      └── Groq Llama 3.1 70B  → cited answer
"""

import os
import json
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import cohere
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR    = "chroma_db"
BM25_INDEX    = "bm25_index.pkl"
CHUNKS_STORE  = "chunks_store.json"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
GROQ_MODEL    = "llama-3.3-70b-versatile"
RETRIEVE_N    = 20   # candidates from each retriever
RERANK_TOP_N  = 5    # final chunks sent to LLM
RRF_K         = 60   # RRF constant (standard value, do not change)


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    text:   str
    source: str
    page:   int
    score:  float


# ── Singleton loaders (cached across requests) ────────────────────────────────
_embeddings   = None
_vectorstore  = None
_bm25         = None
_chunks       = None
_groq_client  = None
_cohere_client = None


def _load_resources():
    """Load all resources once and cache them."""
    global _embeddings, _vectorstore, _bm25, _chunks, _groq_client, _cohere_client

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_embeddings,
        )

    if _bm25 is None:
        with open(BM25_INDEX, "rb") as f:
            _bm25 = pickle.load(f)
        with open(CHUNKS_STORE, "r") as f:
            _chunks = json.load(f)

    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    if _cohere_client is None:
        _cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])


# ── Step 1: BM25 keyword retrieval ────────────────────────────────────────────
def _bm25_search(query: str, n: int = RETRIEVE_N) -> List[Tuple[int, float]]:
    """Return list of (chunk_index, bm25_score) sorted descending."""
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:n]


# ── Step 2: Vector similarity retrieval ───────────────────────────────────────
def _vector_search(query: str, n: int = RETRIEVE_N) -> List[Tuple[int, float]]:
    """Return list of (chunk_index, similarity_score) sorted descending."""
    results = _vectorstore.similarity_search_with_relevance_scores(query, k=n)
    index_scores = []
    for doc, score in results:
        # Match back to chunks_store by text content
        for i, chunk in enumerate(_chunks):
            if chunk["text"] == doc.page_content:
                index_scores.append((i, score))
                break
    return index_scores


# ── Step 3: Reciprocal Rank Fusion ────────────────────────────────────────────
def _reciprocal_rank_fusion(
    bm25_results:   List[Tuple[int, float]],
    vector_results: List[Tuple[int, float]],
    k: int = RRF_K,
) -> List[Tuple[int, float]]:
    """
    Merge two ranked lists using RRF.
    RRF score = Σ 1/(k + rank_in_list)
    Higher is better. k=60 is the standard constant from the original paper.
    """
    rrf_scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(bm25_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

    for rank, (idx, _) in enumerate(vector_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:RETRIEVE_N]


# ── Step 4: Cohere cross-encoder reranking ────────────────────────────────────
def _rerank(query: str, candidates: List[Tuple[int, float]]) -> List[RetrievedChunk]:
    """
    Send top candidates to Cohere Rerank v3.
    Cross-encoders score (query, document) jointly — far more accurate than
    bi-encoder cosine similarity. Uses ~1 API call = 1 of your 1000 free/month.
    """
    docs = [_chunks[idx]["text"] for idx, _ in candidates]

    response = _cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=RERANK_TOP_N,
    )

    results = []
    for hit in response.results:
        original_idx = candidates[hit.index][0]
        chunk = _chunks[original_idx]
        results.append(RetrievedChunk(
            text=chunk["text"],
            source=chunk["source"],
            page=int(chunk.get("page", 0)),
            score=hit.relevance_score,
        ))
    return results


# ── Step 5: LLM answer generation with citation enforcement ───────────────────
SYSTEM_PROMPT = """You are a precise document assistant. Your ONLY job is to answer
questions using the provided document excerpts below.

CITATION RULES (non-negotiable):
- Every factual sentence in your answer MUST end with [Source: {source}, p.{page}]
- Use the exact source name and page number from the context provided
- If the answer to a question is NOT in the provided excerpts, say exactly:
  "I could not find information about this in the provided documents."
- Never invent facts, never use your general knowledge, never guess

FORMAT:
- Answer in clear paragraphs
- After your answer, list all sources used under a "## Sources" heading
- Keep answers concise — 3 to 5 sentences per question is ideal

The document excerpts follow:
---
{context}
---"""


def _build_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Excerpt {i}] Source: {chunk.source}, Page: {chunk.page}\n"
            f"Relevance score: {chunk.score:.3f}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def _generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    """Send context + query to Groq Llama 3.1 70B and return the answer."""
    context = _build_context(chunks)
    system  = SYSTEM_PROMPT.format(
        context=context,
        source="{source}",   # kept as literal placeholders for the LLM
        page="{page}",
    )

    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": query},
        ],
        temperature=0.1,     # low temp = more faithful, less creative
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Public API ────────────────────────────────────────────────────────────────
def query(user_question: str) -> dict:
    """
    Full RAG pipeline. Returns a dict with keys:
      answer   – LLM response with inline citations
      chunks   – list of RetrievedChunk used as context
      steps    – dict of intermediate results for debugging
    """
    _load_resources()

    # Step 1 & 2: Parallel retrieval
    bm25_hits   = _bm25_search(user_question)
    vector_hits = _vector_search(user_question)

    # Step 3: Merge with RRF
    merged = _reciprocal_rank_fusion(bm25_hits, vector_hits)

    # Step 4: Rerank with Cohere
    reranked = _rerank(user_question, merged)

    # Step 5: Generate cited answer
    answer = _generate_answer(user_question, reranked)

    return {
        "answer":  answer,
        "chunks":  reranked,
        "steps": {
            "bm25_hits":   len(bm25_hits),
            "vector_hits": len(vector_hits),
            "merged":      len(merged),
            "reranked":    len(reranked),
        },
    }


# ── CLI test mode ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Ask My Docs — Pipeline test")
    print("Type a question (or 'quit' to exit)\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        result = query(q)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nDebug: {result['steps']}\n{'─'*50}")
