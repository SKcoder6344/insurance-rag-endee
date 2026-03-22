"""
HybridEmbedder — combines dense (SentenceTransformers) and sparse (Endee BM25)
embeddings for hybrid vector search.

Dense:  all-MiniLM-L6-v2  → 384-dim semantic vectors
Sparse: endee/bm25         → BM25 keyword-weighted sparse vectors
"""
from typing import List, Tuple
from loguru import logger

from sentence_transformers import SentenceTransformer
from endee_model import SparseModel

from app.config import settings


class HybridEmbedder:
    """
    Wraps both dense and sparse embedding models.

    Use embed_documents() for corpus text (TF+IDF+length-norm).
    Use embed_query()     for search queries (IDF-only — asymmetric BM25).
    """

    def __init__(self) -> None:
        logger.info(f"Loading dense model: {settings.DENSE_MODEL}")
        self._dense = SentenceTransformer(settings.DENSE_MODEL)

        logger.info("Loading sparse model: endee/bm25")
        self._sparse = SparseModel(model_name="endee/bm25")

    # ── Document embedding (for ingestion) ───────────────────────────────────

    def embed_documents(self, texts: List[str]):
        """
        Generate dense + sparse embeddings for a list of documents.

        Args:
            texts: List of document strings to embed.

        Returns:
            (dense_vecs, sparse_embs) — parallel lists
            dense_vecs  : np.ndarray of shape (N, 384)
            sparse_embs : list of SparseEmbedding objects
        """
        logger.debug(f"Embedding {len(texts)} documents")
        dense_vecs = self._dense.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sparse_embs = list(self._sparse.embed(texts))
        return dense_vecs, sparse_embs

    # ── Query embedding (for search) ─────────────────────────────────────────

    def embed_query(self, query: str) -> Tuple[List[float], object]:
        """
        Generate dense + sparse embeddings for a single query string.

        Args:
            query: User search query.

        Returns:
            (dense_vec, sparse_emb)
            dense_vec  : list[float] of length 384
            sparse_emb : SparseEmbedding with .indices and .values
        """
        dense_vec = self._dense.encode([query], normalize_embeddings=True)[0].tolist()
        sparse_emb = next(self._sparse.query_embed(query))  # IDF-only for queries
        return dense_vec, sparse_emb
