"""
Standalone script to index insurance_policies.txt into Endee.
Run this once after starting the API: python scripts/ingest.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag_pipeline import RAGPipeline
from app.config import settings


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "insurance_policies.txt")
    with open(data_path, "r") as f:
        content = f.read()

    # Split on double newlines to get logical sections
    sections = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 50]

    print(f"Loaded {len(sections)} sections from insurance_policies.txt")

    pipeline = RAGPipeline()
    pipeline.initialize_index()
    count = pipeline.index_documents(sections)

    print(f"✅ Successfully indexed {count} chunks into Endee index '{settings.index_name}'")


if __name__ == "__main__":
    main()
