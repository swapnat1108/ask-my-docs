---
title: Ask My Docs
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# Ask My Docs

A production-grade RAG (Retrieval-Augmented Generation) system with:

- **Hybrid retrieval**: BM25 keyword search + ChromaDB vector search
- **Cross-encoder reranking**: Cohere Rerank v3 for precision
- **Citation enforcement**: Every answer cites its exact source document and page
- **CI evaluation gate**: Ragas scores checked on every deploy

## Setup

1. Fork this Space
2. Add your secrets in Settings → Repository Secrets:
   - `GROQ_API_KEY` (free at console.groq.com)
   - `COHERE_API_KEY` (free at cohere.com)
3. Add your PDF/TXT/MD files to the `docs/` folder
4. The Space will auto-run `ingest.py` on first boot
