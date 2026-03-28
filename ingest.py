"""
ingest.py — Phase 3: Document Ingestion & Indexing
----------------------------------------------------
What this does:
  1. Reads every PDF, TXT, and Markdown file from your docs/ folder
  2. Splits them into overlapping 512-token chunks
  3. Embeds each chunk using BAAI/bge-small-en (free, local model)
  4. Stores embeddings in ChromaDB (your local vector database)
  5. Also stores raw text for BM25 keyword search

Run once before starting the app:
  python ingest.py
"""

import os
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR       = Path("docs")
CHROMA_DIR     = "chroma_db"
BM25_INDEX     = "bm25_index.pkl"
CHUNKS_STORE   = "chunks_store.json"
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 50

# ── Supported file types ──────────────────────────────────────────────────────
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md":  UnstructuredMarkdownLoader,
}

def load_documents():
    """Load all supported documents from the docs/ directory."""
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir()
        print(f"Created docs/ folder. Add your PDF, TXT, or MD files there, then re-run.")
        return []

    docs = []
    files_found = list(DOCS_DIR.rglob("*"))
    supported = [f for f in files_found if f.suffix.lower() in LOADERS]

    if not supported:
        print("No supported files found in docs/. Add PDF, TXT, or MD files and re-run.")
        return []

    for file_path in supported:
        try:
            loader_cls = LOADERS[file_path.suffix.lower()]
            loader = loader_cls(str(file_path))
            loaded = loader.load()
            # Attach source metadata to every page
            for doc in loaded:
                doc.metadata["source"]    = file_path.name
                doc.metadata["file_path"] = str(file_path)
            docs.extend(loaded)
            print(f"  Loaded: {file_path.name}  ({len(loaded)} page(s))")
        except Exception as e:
            print(f"  WARNING: Could not load {file_path.name}: {e}")

    return docs


def split_documents(docs):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"\n  Split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks, embeddings):
    """Embed chunks and store in ChromaDB."""
    print("\n  Building ChromaDB vector store...")
    print("  (First run downloads the embedding model ~130MB — be patient)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Stored {vectorstore._collection.count()} vectors in {CHROMA_DIR}/")
    return vectorstore


def build_bm25_index(chunks):
    """Build a BM25 keyword search index over all chunks."""
    print("\n  Building BM25 keyword index...")
    texts = [chunk.page_content for chunk in chunks]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    # Save BM25 index
    with open(BM25_INDEX, "wb") as f:
        pickle.dump(bm25, f)

    # Save raw chunks as JSON for retrieval at query time
    chunks_data = [
        {
            "text":     chunk.page_content,
            "source":   chunk.metadata.get("source", "unknown"),
            "page":     chunk.metadata.get("page", 0),
            "start":    chunk.metadata.get("start_index", 0),
        }
        for chunk in chunks
    ]
    with open(CHUNKS_STORE, "w") as f:
        json.dump(chunks_data, f, indent=2)

    print(f"  BM25 index saved → {BM25_INDEX}")
    print(f"  Chunk store saved → {CHUNKS_STORE}  ({len(chunks_data)} chunks)")
    return bm25


def main():
    print("=" * 55)
    print("  Ask My Docs — Ingestion Pipeline")
    print("=" * 55)

    print("\n[1/4] Loading documents from docs/...")
    docs = load_documents()
    if not docs:
        return

    print(f"\n[2/4] Splitting {len(docs)} document page(s) into chunks...")
    chunks = split_documents(docs)

    print("\n[3/4] Loading embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    build_vector_store(chunks, embeddings)

    print("\n[4/4] Building BM25 keyword index...")
    build_bm25_index(chunks)

    print("\n" + "=" * 55)
    print("  Ingestion complete! You are ready to run:")
    print("  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
