"""
Insurance Policy Q&A RAG API
Built on Endee Vector Database + OpenAI
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.rag_pipeline import RAGPipeline
from app.schemas import QueryRequest, QueryResponse, IndexRequest, HealthResponse
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

rag: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logger.info("Initializing RAG pipeline with Endee vector DB...")
    rag = RAGPipeline()
    rag.initialize_index()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Insurance Policy Q&A — Powered by Endee Vector DB",
    description="RAG-based chatbot that answers insurance queries using semantic search over policy documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="ok", version="1.0.0", vector_db="endee")


@app.post("/index", tags=["Indexing"])
def index_documents(req: IndexRequest):
    """Ingest and index insurance policy documents into Endee."""
    try:
        count = rag.index_documents(req.documents)
        return {"message": f"Successfully indexed {count} chunks into Endee.", "chunks": count}
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(req: QueryRequest):
    """Ask a question. Returns answer + retrieved context chunks."""
    start = time.time()
    try:
        result = rag.answer(req.question, top_k=req.top_k)
        result["latency_ms"] = round((time.time() - start) * 1000, 2)
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index", tags=["Indexing"])
def clear_index():
    """Clear all vectors from Endee index (use with caution)."""
    try:
        rag.clear_index()
        return {"message": "Index cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
