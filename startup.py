"""
startup.py — HuggingFace Spaces bootstrap
------------------------------------------
HuggingFace Spaces can run a startup script before the main app.
Add this to your Space by setting the entry point in README.md or
by calling it from a shell script.

This script:
  1. Checks if the index already exists (skips re-ingestion if so)
  2. Runs ingest.py if first boot or if new docs detected
  3. Then hands off to Streamlit

Usage (add to your Space as an alternative app_file, or call from a
custom CMD in a Docker Space):
  python startup.py
"""

import os
import subprocess
import sys
from pathlib import Path


def needs_ingestion() -> bool:
    """Return True if we need to (re)build the index."""
    chroma_exists = Path("chroma_db").exists()
    bm25_exists   = Path("bm25_index.pkl").exists()
    docs_exist    = any(Path("docs").rglob("*.pdf")) if Path("docs").exists() else False

    if not chroma_exists or not bm25_exists:
        return True

    # Re-ingest if docs were updated more recently than the index
    if docs_exist:
        index_mtime = Path("bm25_index.pkl").stat().st_mtime
        for doc in Path("docs").rglob("*"):
            if doc.suffix.lower() in (".pdf", ".txt", ".md"):
                if doc.stat().st_mtime > index_mtime:
                    print(f"  New/updated doc detected: {doc.name}")
                    return True
    return False


def main():
    print("=" * 50)
    print("  Ask My Docs — Startup")
    print("=" * 50)

    # Validate API keys
    missing = []
    for key in ("GROQ_API_KEY", "COHERE_API_KEY"):
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print(f"\nWARNING: Missing API keys: {', '.join(missing)}")
        print("Add them in HuggingFace Space Settings → Repository Secrets\n")

    # Run ingestion if needed
    if needs_ingestion():
        print("\nRunning ingestion pipeline...")
        result = subprocess.run([sys.executable, "ingest.py"], check=False)
        if result.returncode != 0:
            print("WARNING: Ingestion had errors. App may not work correctly.")
    else:
        print("\nIndex already exists — skipping ingestion.")

    # Launch Streamlit
    print("\nLaunching Streamlit app...")
    os.execvp("streamlit", [
        "streamlit", "run", "app.py",
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ])


if __name__ == "__main__":
    main()
