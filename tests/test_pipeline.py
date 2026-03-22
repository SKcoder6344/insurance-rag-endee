"""
Unit tests for PolicyGuard AI pipeline.

All external dependencies (Endee, Groq) are mocked — tests run without
any running services. This is what CI/CD executes on every push.
"""
import pytest
from unittest.mock import MagicMock, patch


# ── Test: schemas validation ──────────────────────────────────────────────────

def test_claim_request_min_length():
    from app.schemas import ClaimRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ClaimRequest(claim="hi")  # too short (< 5 chars)


def test_claim_request_valid():
    from app.schemas import ClaimRequest
    req = ClaimRequest(claim="Is knee surgery covered under my policy?")
    assert req.claim == "Is knee surgery covered under my policy?"


def test_search_request_top_k_bounds():
    from app.schemas import SearchRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SearchRequest(query="diabetes coverage", top_k=0)  # min is 1
    with pytest.raises(ValidationError):
        SearchRequest(query="diabetes coverage", top_k=99)  # max is 20


# ── Test: ingestion parser ────────────────────────────────────────────────────

def test_parse_policy_file(tmp_path):
    from scripts.ingest import _parse_policy_file
    sample = """===SECTION: coverage===
Knee replacement is covered after the waiting period.

===SECTION: exclusions===
Cosmetic surgery is not covered.
"""
    f = tmp_path / "test_policy.txt"
    f.write_text(sample)

    chunks = _parse_policy_file(f)
    assert len(chunks) == 2
    assert chunks[0]["section"] == "coverage"
    assert chunks[1]["section"] == "exclusions"
    assert "Knee replacement" in chunks[0]["text"]


# ── Test: verdict JSON parsing ────────────────────────────────────────────────

def test_verdict_json_parsing():
    """Agent must parse clean LLM JSON response into ClaimVerdict."""
    from app.schemas import ClaimVerdict
    import json

    raw = json.dumps({
        "verdict": "COVERED",
        "confidence": 0.87,
        "summary": "Knee surgery is covered after the 2-year waiting period.",
        "coverage_points": ["Knee replacement is a listed covered procedure"],
        "exclusion_points": [],
        "waiting_period": "2 years",
        "claim_steps": ["Step 1: Contact TPA", "Step 2: Submit documents"],
        "recommendation": "Check if your 2-year period has elapsed.",
    })

    data = json.loads(raw)
    verdict = ClaimVerdict(**data)
    assert verdict.verdict == "COVERED"
    assert verdict.confidence == pytest.approx(0.87)
    assert len(verdict.claim_steps) == 2


# ── Test: context character cap ───────────────────────────────────────────────

def test_context_cap():
    """Agent must not send more than _MAX_CONTEXT_CHARS of context to LLM."""
    from app.agent import InsuranceClaimsAgent, _MAX_CONTEXT_CHARS
    from app.schemas import RetrievedChunk

    agent = InsuranceClaimsAgent.__new__(InsuranceClaimsAgent)  # skip __init__

    long_chunks = [
        RetrievedChunk(
            text="x" * 400,
            section="coverage",
            similarity=0.9,
            chunk_id=f"c{i}",
        )
        for i in range(10)
    ]

    ctx = agent._build_context_block(long_chunks)
    assert len(ctx) <= _MAX_CONTEXT_CHARS + 100  # small buffer for metadata


# ── Test: API health endpoint ─────────────────────────────────────────────────

def test_health_endpoint():
    from fastapi.testclient import TestClient
    from unittest.mock import patch, MagicMock

    with patch("app.main.EndeeHybridStore"), patch("app.main.InsuranceClaimsAgent"):
        from app.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
