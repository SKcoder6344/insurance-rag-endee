"""
EndeeHybridStore — wrapper around Endee SDK for hybrid (dense + BM25) search.

Key design decisions:
- Uses endee_bm25 sparse model for keyword precision
- cosine similarity + INT8 quantization for memory efficiency
- Metadata filtering by policy section for targeted agent steps
"""
from typing import List, Optional, Dict, Any
from loguru import logger

from endee import Endee, Precision

from app.config import settings
from app.embedder import HybridEmbedder
from app.schemas import RetrievedChunk


class EndeeHybridStore:
    """
    Manages a single Endee hybrid index for insurance policy documents.

    Supports:
    - create_index()    → initialise / recreate hybrid index
    - upsert_chunks()   → embed + insert document chunks
    - hybrid_search()   → dense + sparse query with optional section filter
    """

    def __init__(self) -> None:
        self._embedder = HybridEmbedder()

        # Initialise Endee client
        self._client = Endee(settings.ENDEE_AUTH_TOKEN or "")
        if settings.ENDEE_URL != "http://localhost:8080":
            self._client.set_base_url(f"{settings.ENDEE_URL}/api/v1")

        self._index = None
        logger.info(f"EndeeHybridStore initialised → {settings.ENDEE_URL}")

    # ── Index management ──────────────────────────────────────────────────────

    def create_index(self) -> None:
        """Drop and recreate the hybrid index (cosine + INT8 + BM25 sparse)."""
        logger.info(f"Creating hybrid index: {settings.INDEX_NAME}")
        try:
            self._client.delete_index(settings.INDEX_NAME)
            logger.debug("Existing index deleted")
        except Exception:
            pass  # Index didn't exist yet — that's fine

        self._client.create_index(
            name=settings.INDEX_NAME,
            dimension=settings.DENSE_DIM,
            sparse_model="endee_bm25",   # ← enables hybrid search
            space_type="cosine",
            precision=Precision.INT8,    # memory-efficient quantization
        )
        self._index = self._client.get_index(name=settings.INDEX_NAME)
        logger.success(f"Hybrid index ready: dim={settings.DENSE_DIM}, sparse=BM25, precision=INT8")

    def _get_index(self):
        """Lazy-load index reference."""
        if self._index is None:
            self._index = self._client.get_index(name=settings.INDEX_NAME)
        return self._index

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and upsert document chunks into Endee.

        Args:
            chunks: List of dicts with keys:
                      id       — unique chunk identifier
                      text     — raw text to embed
                      section  — policy section tag (coverage, exclusions, …)
                      source   — document source label

        Returns:
            Number of chunks successfully upserted.
        """
        texts = [c["text"] for c in chunks]
        dense_vecs, sparse_embs = self._embedder.embed_documents(texts)

        vectors = []
        for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_embs):
            vectors.append({
                "id": chunk["id"],
                "vector": dense.tolist(),
                "sparse_indices": sparse.indices.tolist(),
                "sparse_values": sparse.values.tolist(),
                "meta": {
                    "text": chunk["text"],
                    "section": chunk.get("section", "general"),
                    "source": chunk.get("source", "policy_doc"),
                },
                "filter": {
                    "section": chunk.get("section", "general"),
                },
            })

        self._get_index().upsert(vectors)
        logger.success(f"Upserted {len(vectors)} chunks into Endee")
        return len(vectors)

    # ── Search ────────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        section: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Run a hybrid (dense + BM25) similarity search.

        Args:
            query:   Natural language query string.
            top_k:   Number of results to return.
            section: Optional section filter (e.g. 'exclusions').

        Returns:
            List of RetrievedChunk objects sorted by similarity desc.
        """
        dense_vec, sparse_emb = self._embedder.embed_query(query)

        query_params: Dict[str, Any] = {
            "vector": dense_vec,
            "sparse_indices": sparse_emb.indices.tolist(),
            "sparse_values": sparse_emb.values.tolist(),
            "top_k": top_k,
            "ef": 128,
        }

        if section:
            query_params["filter"] = [{"section": {"$eq": section}}]
            query_params["filter_boost_percentage"] = 20  # compensate for filter narrowing

        raw_results = self._get_index().query(**query_params)

        chunks = []
        for r in raw_results:
            meta = r.get("meta", {})
            chunks.append(
                RetrievedChunk(
                    text=meta.get("text", ""),
                    section=meta.get("section", ""),
                    similarity=round(r.get("similarity", 0.0), 4),
                    chunk_id=r.get("id", ""),
                )
            )

        logger.debug(f"hybrid_search '{query[:40]}...' → {len(chunks)} results")
        return chunks
