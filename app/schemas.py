"""
Pydantic schemas for request/response validation.
Every API endpoint uses typed models — no raw dicts.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


# ── Inbound ──────────────────────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        example="My father has diabetes and needs knee replacement surgery. Is it covered?",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    section: Optional[str] = Field(
        default=None,
        example="exclusions",
        description="Filter by policy section: coverage, exclusions, waiting_period, claim_procedure, emergency",
    )


# ── Agent internals ───────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    text: str
    section: str
    similarity: float
    chunk_id: str


class AgentStep(BaseModel):
    step: int
    action: str
    query: str
    results: List[RetrievedChunk]


# ── Outbound ─────────────────────────────────────────────────────────────────

class ClaimVerdict(BaseModel):
    verdict: str = Field(..., example="COVERED", description="COVERED | NOT_COVERED | PARTIAL")
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    coverage_points: List[str] = []
    exclusion_points: List[str] = []
    waiting_period: str = "None"
    claim_steps: List[str] = []
    recommendation: str = ""
    steps: List[AgentStep] = []
    latency_ms: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[RetrievedChunk]
    count: int
    query: str


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    index: str
    mode: str = "Hybrid Search (Dense + BM25)"


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int
