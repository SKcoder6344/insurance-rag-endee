"""
InsuranceClaimsAgent — multi-step agentic pipeline.

Instead of a single RAG query, this agent reasons in 3 targeted steps:

  Step 1 → Search Endee for COVERAGE terms related to the claim
  Step 2 → Search Endee for EXCLUSIONS and WAITING PERIODS
  Step 3 → Search Endee for CLAIM PROCEDURE and required documents

Then synthesises all 3 contexts into a structured verdict via Groq LLaMA3.

This pattern (plan → retrieve → retrieve → retrieve → synthesise) is what
production AI systems (Perplexity, Bing AI, Gemini) actually use — and what
sets this project apart from basic single-pass RAG.
"""
import json
import time
from loguru import logger
from groq import Groq

from app.config import settings
from app.endee_store import EndeeHybridStore
from app.schemas import AgentStep, ClaimVerdict, RetrievedChunk

# Maximum input characters sent to the LLM per context block
_MAX_CONTEXT_CHARS = 1200

# Verdict prompt — structured JSON output, low temperature for determinism
_VERDICT_PROMPT_TEMPLATE = """You are a precise insurance policy analyst. A customer has submitted a claim. Analyse it using the policy context below and return a structured verdict.

CUSTOMER CLAIM:
{claim}

POLICY CONTEXT (retrieved from policy documents via hybrid semantic + keyword search):

[COVERAGE TERMS]
{coverage_ctx}

[EXCLUSIONS & WAITING PERIODS]
{exclusion_ctx}

[CLAIM PROCEDURE]
{procedure_ctx}

Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON:
{{
  "verdict": "COVERED" | "NOT_COVERED" | "PARTIAL",
  "confidence": 0.0–1.0,
  "summary": "One clear sentence verdict for the customer",
  "coverage_points": ["What is covered related to this claim (bullet points)"],
  "exclusion_points": ["Any exclusions or conditions that apply"],
  "waiting_period": "None" | "X months" | "X years",
  "claim_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "recommendation": "Concrete next action the customer should take"
}}"""


class InsuranceClaimsAgent:
    """
    3-step agentic pipeline for insurance claim analysis.

    Each step fires a targeted hybrid search against Endee, then
    the LLM synthesises all three retrieved contexts into a verdict.
    """

    def __init__(self) -> None:
        self._store = EndeeHybridStore()
        self._llm = Groq(api_key=settings.GROQ_API_KEY)
        logger.info("InsuranceClaimsAgent ready")

    def analyze_claim(self, claim: str) -> ClaimVerdict:
        """
        Run the full 3-step agentic analysis on an insurance claim.

        Args:
            claim: Natural language description of what the customer wants covered.

        Returns:
            ClaimVerdict with structured verdict, agent steps, and latency.
        """
        t_start = time.perf_counter()
        steps: list[AgentStep] = []

        # ── Step 1: Coverage retrieval ────────────────────────────────────────
        logger.info("[Agent] Step 1 — coverage search")
        coverage_results = self._store.hybrid_search(
            query=f"coverage benefits included policy covers {claim}",
            top_k=settings.TOP_K,
        )
        steps.append(AgentStep(
            step=1,
            action="Searching coverage & benefits",
            query=f"coverage benefits: {claim}",
            results=coverage_results,
        ))

        # ── Step 2: Exclusions + waiting periods ──────────────────────────────
        logger.info("[Agent] Step 2 — exclusions + waiting period search")
        exclusion_results = self._store.hybrid_search(
            query=f"exclusions not covered waiting period pre-existing condition {claim}",
            top_k=settings.TOP_K,
        )
        steps.append(AgentStep(
            step=2,
            action="Checking exclusions & waiting periods",
            query=f"exclusions + waiting period: {claim}",
            results=exclusion_results,
        ))

        # ── Step 3: Claim procedure ───────────────────────────────────────────
        logger.info("[Agent] Step 3 — claim procedure search")
        procedure_results = self._store.hybrid_search(
            query=f"how to file claim procedure documents required hospital network {claim}",
            top_k=settings.TOP_K,
        )
        steps.append(AgentStep(
            step=3,
            action="Retrieving claim procedure",
            query=f"claim filing process: {claim}",
            results=procedure_results,
        ))

        # ── LLM Synthesis ─────────────────────────────────────────────────────
        logger.info("[Agent] Synthesising verdict via Groq LLaMA3")
        verdict = self._synthesise(claim, steps)
        verdict.steps = steps
        verdict.latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

        logger.success(
            f"[Agent] Done — verdict={verdict.verdict} "
            f"confidence={verdict.confidence} "
            f"latency={verdict.latency_ms}ms"
        )
        return verdict

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """Concatenate retrieved chunks into a context string, respecting char limit."""
        lines = []
        total = 0
        for chunk in chunks:
            snippet = f"[{chunk.section}] (sim={chunk.similarity}) {chunk.text}"
            if total + len(snippet) > _MAX_CONTEXT_CHARS:
                break
            lines.append(snippet)
            total += len(snippet)
        return "\n".join(lines) if lines else "No relevant policy text found."

    def _synthesise(self, claim: str, steps: list[AgentStep]) -> ClaimVerdict:
        """
        Call Groq LLaMA3 to synthesise a structured verdict from 3 context blocks.

        Returns a ClaimVerdict (without steps field — caller fills that in).
        Raises ValueError if LLM response cannot be parsed as valid JSON.
        """
        coverage_ctx = self._build_context_block(steps[0].results)
        exclusion_ctx = self._build_context_block(steps[1].results)
        procedure_ctx = self._build_context_block(steps[2].results)

        prompt = _VERDICT_PROMPT_TEMPLATE.format(
            claim=claim[:500],  # cap claim length — never let user input bloat token cost
            coverage_ctx=coverage_ctx,
            exclusion_ctx=exclusion_ctx,
            procedure_ctx=procedure_ctx,
        )

        response = self._llm.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned non-JSON: {raw[:200]}")
            raise ValueError(f"LLM synthesis failed — invalid JSON: {e}") from e

        return ClaimVerdict(**data)
