# 🛡️ PolicyGuard AI — Agentic Insurance Claims Analyzer

> **Multi-step agentic RAG system** that reasons through insurance claims in 3 targeted retrieval steps using **Endee Hybrid Vector Search (Dense + BM25)** and **Groq LLaMA3**. Returns structured verdicts: `COVERED` | `NOT_COVERED` | `PARTIAL`.

[![CI](https://github.com/SKcoder6344/insurance-rag-endee/actions/workflows/ci.yml/badge.svg)](https://github.com/SKcoder6344/insurance-rag-endee/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Endee](https://img.shields.io/badge/VectorDB-Endee_Hybrid-orange)](https://endee.io)
[![Groq](https://img.shields.io/badge/LLM-Groq_LLaMA3-purple)](https://console.groq.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)

---

## 🔍 What it does

Insurance claims are complex — a single claim touches **coverage terms**, **exclusion clauses**, **waiting periods**, and **filing procedures** simultaneously. A basic RAG query retrieves random policy text and misses this nuance.

**PolicyGuard AI uses an agentic 3-step pipeline** — each step fires a targeted hybrid search against Endee, then Groq LLaMA3 synthesises all three retrieved contexts into a structured, explainable verdict.

**Example:**

> **Claim:** *"My father has Type 2 diabetes and needs a kidney transplant. Is it covered?"*
>
> **Agent Step 1** → searches `coverage benefits organ transplant diabetes`
> **Agent Step 2** → searches `exclusions waiting period pre-existing diabetes`
> **Agent Step 3** → searches `claim procedure documents transplant network hospital`
>
> **Verdict:** `PARTIAL` — Kidney transplant is covered, but pre-existing diabetes triggers a **4-year waiting period**. If policy is > 4 years old, full coverage applies. Donor expenses covered up to 50% of sum insured.

---

## 🏗️ Architecture

```
User Claim (Streamlit UI)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│               InsuranceClaimsAgent                      │
│                                                         │
│  Step 1: hybrid_search("coverage benefits [claim]")     │
│       │                                                 │
│       ▼                                                 │
│  Step 2: hybrid_search("exclusions waiting period")     │
│       │                                                 │
│       ▼                                                 │
│  Step 3: hybrid_search("claim procedure documents")     │
│       │                                                 │
│       ▼                                                 │
│  Groq LLaMA3-70B  ──►  Structured JSON Verdict         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│           Endee Hybrid Search (per step)                │
│                                                         │
│  Dense Query (all-MiniLM-L6-v2, 384-dim)               │
│       +                                                 │
│  Sparse Query (Endee BM25 — keyword weights)           │
│       │                                                 │
│  HNSW Approximate Nearest Neighbor Search               │
│  cosine similarity | INT8 precision                     │
│       │                                                 │
│  → Top-K policy chunks with similarity scores          │
└─────────────────────────────────────────────────────────┘
```

### Why Hybrid Search?

| Search Type | Finds |
|---|---|
| Dense only | *"renal failure requiring dialysis"* matches *"kidney transplant"* (semantics) |
| BM25 only | Exact terms — *"waiting period"*, *"4 years"*, *"pre-existing"* |
| **Hybrid (Endee)** | **Both** — best recall + best precision |

Endee's `endee_bm25` sparse model combined with sentence-transformer dense vectors means the agent retrieves the right clause even when user phrasing differs from policy language.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Vector DB | **Endee** (Hybrid: Dense + BM25) | Native hybrid search, HNSW, INT8, Docker-ready |
| Dense Embeddings | `all-MiniLM-L6-v2` | 384-dim, free, runs locally, high semantic quality |
| Sparse Embeddings | `endee-model / endee_bm25` | Endee-native BM25 — asymmetric doc/query weighting |
| LLM (Synthesis) | **Groq LLaMA3-70B** | Free tier, 500ms latency, excellent instruction following |
| API | FastAPI + SlowAPI | Async, rate limiting, Swagger/ReDoc auto-docs |
| UI | Streamlit | Interactive chat, agent step trace, live similarity scores |
| Testing | Pytest | 5 unit tests, fully mocked — no live services needed |
| CI/CD | GitHub Actions | Auto-test on every push |
| Containerisation | Docker + Compose | One-command startup: Endee + API + Streamlit |

**No OpenAI. No paid APIs.** Fully free-tier compatible.

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- [Groq API Key](https://console.groq.com) (free, no credit card)

### 1. Clone and configure

```bash
git clone https://github.com/SKcoder6344/insurance-rag-endee
cd insurance-rag-endee
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 2. Start all services

```bash
docker compose up -d
```

This starts:
- **Endee** vector DB at `localhost:8080`
- **FastAPI** backend at `localhost:8000`
- **Streamlit** UI at `localhost:8501`

### 3. Index policy documents into Endee

```bash
curl -X POST http://localhost:8000/index
```

This chunks `data/insurance_policies.txt` by section, generates hybrid embeddings (dense + BM25), and upserts all vectors into Endee.

### 4. Open the UI

Visit **http://localhost:8501** and submit a claim.

### 5. Or use the API directly

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"claim": "I need knee replacement surgery. My policy is 3 years old."}'
```

---

## 📁 Folder Structure

```
insurance-rag-endee/
├── app/
│   ├── main.py           # FastAPI app + endpoints + rate limiting
│   ├── agent.py          # 3-step InsuranceClaimsAgent
│   ├── endee_store.py    # Endee hybrid search wrapper
│   ├── embedder.py       # HybridEmbedder (dense + BM25)
│   ├── schemas.py        # Pydantic request/response models
│   └── config.py         # Settings from .env (pydantic-settings)
├── data/
│   └── insurance_policies.txt   # Sectioned policy corpus
├── scripts/
│   └── ingest.py         # Parse → embed → upsert to Endee
├── tests/
│   └── test_pipeline.py  # 5 unit tests (fully mocked)
├── streamlit_app.py      # Chat UI with agent step trace
├── .github/workflows/
│   └── ci.yml            # GitHub Actions CI
├── docker-compose.yml    # Endee + API + Streamlit
├── Dockerfile
├── requirements.txt      # Pinned deps
└── .env.example
```

---

## 📈 Key Metrics

| Metric | Value |
|---|---|
| Search type | Hybrid (Dense HNSW + BM25 sparse) |
| Dense model | `all-MiniLM-L6-v2` (384 dimensions) |
| Vector similarity | Cosine, INT8 precision |
| LLM | Groq LLaMA3-70B |
| Agent steps | 3 targeted retrievals per claim |
| Policy corpus | 8 sections, 30+ chunks |
| API rate limit | 20 req/min (configurable) |
| Avg total latency | ~1.5–3s (retrieval + LLM) |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

All tests run without live Endee or Groq — fully mocked.

---

## 🔮 What makes this different

| Feature | Basic RAG (competitors) | PolicyGuard AI |
|---|---|---|
| Search | Single dense query | 3-step targeted hybrid search |
| Vector type | Dense only | **Dense + BM25 (Endee native)** |
| LLM call | One generic prompt | Structured synthesis from 3 contexts |
| Output | Plain text answer | Typed JSON: verdict, confidence, steps |
| UI | None / basic | Chat UI with retrieval trace |
| Infrastructure | Script-only | Docker + FastAPI + Streamlit + CI/CD |
| Free-tier | ❌ Needs OpenAI | ✅ Groq + SentenceTransformers |

---

## 📬 Author

**Sujal Kumar Nayak**
- GitHub: [@SKcoder6344](https://github.com/SKcoder6344)
- LinkedIn: [linkedin.com/in/sujal-kumar-nayak](https://linkedin.com/in/sujal-kumar-nayak)
