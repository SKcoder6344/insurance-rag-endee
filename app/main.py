"""
PolicyGuard AI — FastAPI application entry point.

Endpoints:
  GET  /health          → system health check
  POST /analyze         → 3-step agentic claim analysis (main endpoint)
  POST /search          → direct hybrid search (for debugging / exploration)
  POST /index           → trigger re-ingestion from data/insurance_policies.txt
  DELETE /index         → clear and reset the Endee index
"""
import time
import os
from pathlib import Path
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.schemas import (
    ClaimRequest, ClaimVerdict,
    SearchRequest, SearchResponse,
    HealthResponse, IngestResponse,
)
from app.agent import InsuranceClaimsAgent
from app.endee_store import EndeeHybridStore

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Auto-ingest on startup ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-ingest policy data on startup — handles Render cold starts."""
    logger.info("Startup: checking Endee index...")
    try:
        from scripts.ingest import run_ingest
        run_ingest()
        logger.success("Startup ingest complete")
    except Exception as e:
        logger.warning(f"Startup ingest failed (Endee may not be ready yet): {e}")
    yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="PolicyGuard AI",
    description=(
        "**Agentic Insurance Claims Analyzer** powered by Endee Hybrid Search.\n\n"
        "Uses a 3-step reasoning pipeline (coverage → exclusions → procedure) "
        "with Endee dense+BM25 hybrid vectors and Groq LLaMA3 synthesis.\n\n"
        "No OpenAI required — fully free-tier compatible."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # open for deployment — Streamlit Cloud URL varies
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Singletons (created once at startup) ─────────────────────────────────────
store = EndeeHybridStore()
agent = InsuranceClaimsAgent()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — returns server status and active index."""
    return HealthResponse(
        status="ok",
        index=settings.INDEX_NAME,
    )


@app.post(
    "/analyze",
    response_model=ClaimVerdict,
    tags=["Core"],
    summary="Analyze an insurance claim (3-step agentic pipeline)",
)
@limiter.limit("20/minute")
def analyze_claim(request: Request, body: ClaimRequest):
    """
    Submit a natural-language claim description.

    The agent will:
    1. Search Endee for relevant coverage terms
    2. Search Endee for exclusions and waiting periods
    3. Search Endee for the claim filing procedure

    Then synthesise a structured verdict: COVERED / NOT_COVERED / PARTIAL.
    """
    try:
        return agent.analyze_claim(body.claim)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"/analyze error: {e}")
        raise HTTPException(status_code=500, detail="Internal error — check server logs")


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Exploration"],
    summary="Direct hybrid search on policy documents",
)
@limiter.limit("30/minute")
def search(request: Request, body: SearchRequest):
    """
    Run a direct hybrid search against Endee. Useful for exploring the index
    or debugging retrieval quality before running the full agent.
    """
    try:
        results = store.hybrid_search(body.query, body.top_k, body.section)
        return SearchResponse(results=results, count=len(results), query=body.query)
    except Exception as e:
        logger.error(f"/search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IngestResponse, tags=["Admin"])
def trigger_ingest():
    """
    Re-run ingestion from data/insurance_policies.txt.
    This endpoint is idempotent — safe to call multiple times.
    """
    from scripts.ingest import run_ingest
    try:
        n = run_ingest()
        return IngestResponse(message="Ingestion complete", chunks_indexed=n)
    except Exception as e:
        logger.error(f"/index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index", tags=["Admin"])
def reset_index():
    """Drop and recreate the Endee hybrid index (all vectors deleted)."""
    try:
        store.create_index()
        return {"message": "Index reset", "index": settings.INDEX_NAME}
    except Exception as e:
        logger.error(f"DELETE /index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
