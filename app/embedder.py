from typing import List, Tuple
from loguru import logger
from fastembed import TextEmbedding
from endee_model import SparseModel
from app.config import settings

class HybridEmbedder:
    def __init__(self) -> None:
        logger.info("Loading dense model via fastembed (ONNX - lightweight)")
        self._dense = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("Loading sparse model: endee/bm25")
        self._sparse = SparseModel(model_name="endee/bm25")

    def embed_documents(self, texts: List[str]):
        dense_vecs = list(self._dense.embed(texts))
        sparse_embs = list(self._sparse.embed(texts))
        return dense_vecs, sparse_embs

    def embed_query(self, query: str) -> Tuple[List[float], object]:
        dense_vec = list(self._dense.embed([query]))[0].tolist()
        sparse_emb = next(self._sparse.query_embed(query))
        return dense_vec, sparse_emb
