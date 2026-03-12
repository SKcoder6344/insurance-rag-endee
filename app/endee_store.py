"""
Endee Vector Store Wrapper
Handles index creation, upsert, and semantic search via Endee Python SDK.
"""

import logging
import uuid
from typing import List, Dict, Any

from endee import Endee, Precision
from app.config import settings

logger = logging.getLogger(__name__)


class EndeeVectorStore:
    def __init__(self):
        token = settings.endee_auth_token or None
        self.client = Endee(token) if token else Endee()
        self.client.set_base_url(f"{settings.endee_host}/api/v1")
        self.index_name = settings.index_name
        self.index = None

    def initialize(self):
        """Create index if it doesn't exist, then load it."""
        existing = [idx.name for idx in self.client.list_indexes()]
        if self.index_name not in existing:
            logger.info(f"Creating Endee index '{self.index_name}' (dim={settings.embedding_dimension})")
            self.client.create_index(
                name=self.index_name,
                dimension=settings.embedding_dimension,
                space_type="cosine",
                precision=Precision.INT8,
            )
        self.index = self.client.get_index(name=self.index_name)
        logger.info(f"Endee index '{self.index_name}' ready.")

    def upsert(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> int:
        """Insert or update vectors with metadata."""
        items = [
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "meta": meta,
            }
            for vec, meta in zip(vectors, metadata)
        ]
        self.index.upsert(items)
        logger.info(f"Upserted {len(items)} vectors into Endee.")
        return len(items)

    def query(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k similar chunks for a query vector."""
        results = self.index.query(vector=query_vector, top_k=top_k)
        return [
            {
                "text": r.meta.get("text", ""),
                "source": r.meta.get("source", "unknown"),
                "similarity": round(r.similarity, 4),
            }
            for r in results
        ]

    def clear(self):
        """Delete and recreate the index."""
        try:
            self.client.delete_index(name=self.index_name)
            logger.info(f"Deleted index '{self.index_name}'.")
        except Exception as e:
            logger.warning(f"Delete index warning: {e}")
        self.initialize()
