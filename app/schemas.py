from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, example="What does my health insurance cover?")
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    text: str
    source: str
    similarity: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    latency_ms: Optional[float] = None


class IndexRequest(BaseModel):
    documents: List[str] = Field(
        ...,
        description="List of insurance policy text passages to index.",
        example=["Health insurance covers hospitalization up to Rs.5 lakh per year."]
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_db: str
