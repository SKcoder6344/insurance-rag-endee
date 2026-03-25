"""
HybridEmbedder — uses fastembed (ONNX-based, ~80MB RAM) instead of
sentence-transformers (PyTorch, ~900MB RAM) — required for Render free tier.

Dense:  BAAI/bge-small-en-v1.5 via fastembed → 384-dim
Sparse: endee/bm25 → keyword-weighted sparse vectors
"""
from typing import List, Tuple
from loguru import logger
from fastembed import TextEmbedding
from endee_model import SparseModel
from app.config import settings


class HybridEmbedder:
    """
    Wraps fastembed (dense) and endee BM25 (sparse) embedding models.

    Use embed_documents() for corpus text.
    Use embed_query()     for search queries.
    """

    def __init__(self) -> None:
        logger.info("Loading dense model via fastembed ONNX (lightweight)")
        self._dense = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("Loading sparse model: endee/bm25")
        self._sparse = SparseModel(model_name="endee/bm25")

    def embed_documents(self, texts: List[str]):
        """
        Generate dense + sparse embeddings for a list of documents.

        Returns:
            (dense_vecs, sparse_embs) — parallel lists
        """
        logger.debug(f"Embedding {len(texts)} documents")
        dense_vecs = list(self._dense.embed(texts))
        sparse_embs = list(self._sparse.embed(texts))
        return dense_vecs, sparse_embs

    def embed_query(self, query: str) -> Tuple[List[float], object]:
        """
        Generate dense + sparse embeddings for a single query.

        Returns:
            (dense_vec, sparse_emb)
        """
        dense_vec = list(self._dense.embed([query]))[0].tolist()
        sparse_emb = next(self._sparse.query_embed(query))
        return dense_vec, sparse_emb
