"""
Ingestion script — reads data/insurance_policies.txt, chunks by section,
and upserts all chunks into the Endee hybrid index.

Run:
    python -m scripts.ingest

Or triggered via POST /index endpoint.
"""
import sys
import re
from pathlib import Path
from loguru import logger

# Make sure app/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.endee_store import EndeeHybridStore

DATA_FILE = Path(__file__).parent.parent / "data" / "insurance_policies.txt"


def _parse_policy_file(filepath: Path) -> list[dict]:
    """
    Parse the policy text file into labelled chunks.

    Each chunk follows the format:
        ===SECTION: <section_name>===
        <chunk text>

    Returns:
        List of dicts: {id, text, section, source}
    """
    raw = filepath.read_text(encoding="utf-8")
    chunks = []
    # Split on section markers
    parts = re.split(r"===SECTION:\s*(\w+)===", raw)

    # parts[0] is preamble (empty), then alternating section_name, text
    i = 1
    while i < len(parts) - 1:
        section = parts[i].strip()
        text = parts[i + 1].strip()
        if text:
            section_count = sum(1 for c in chunks if c["section"] == section)
            chunks.append({
                "id": f"{section}_{section_count:03d}",
                "text": text,
                "section": section,
                "source": "insurance_policies.txt",
            })
        i += 2

    return chunks


def run_ingest() -> int:
    """
    Full ingestion pipeline:
      1. Parse data file
      2. Create (or recreate) Endee hybrid index
      3. Upsert all chunks

    Returns:
        Number of chunks indexed.
    """
    logger.info(f"Reading policy data from: {DATA_FILE}")
    chunks = _parse_policy_file(DATA_FILE)
    logger.info(f"Parsed {len(chunks)} chunks across sections: "
                f"{set(c['section'] for c in chunks)}")

    store = EndeeHybridStore()

    logger.info("Creating Endee hybrid index (dense + BM25)...")
    store.create_index()

    logger.info("Upserting chunks...")
    n = store.upsert_chunks(chunks)

    logger.success(f"Ingestion complete — {n} chunks indexed")
    return n


if __name__ == "__main__":
    run_ingest()
