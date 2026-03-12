# Insurance Policy Q&A — RAG System Powered by Endee Vector Database

> Retrieval-Augmented Generation (RAG) chatbot that answers insurance queries using semantic search over policy documents — built on top of the **Endee high-performance vector database**.

![CI](https://github.com/SKcoder6344/insurance-rag-endee/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Endee](https://img.shields.io/badge/VectorDB-Endee-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)

---

## What It Does

Insurance policy documents are dense, jargon-heavy, and hard to navigate. This system ingests raw policy documents, chunks them into semantically meaningful passages, indexes them into **Endee** (a high-performance open-source vector database), and answers natural language queries using OpenAI GPT — returning both the answer and the exact source passages retrieved.

**Example:**
> Q: *"Is pre-existing diabetes covered?"*
> A: *"Pre-existing diseases including diabetes are covered after a 4-year waiting period from the policy inception date..."* `[Source: document_1 | Similarity: 0.94]`

---

## Architecture

```
User Query
    │
    ▼
[FastAPI /query endpoint]
    │
    ▼
[OpenAI text-embedding-3-small]  ←── Embed the question
    │
    ▼
[Endee Vector DB]  ←── Semantic search (cosine similarity, top-k)
    │
    ▼
[Retrieved Policy Chunks]
    │
    ▼
[OpenAI GPT-3.5-turbo]  ←── Generate grounded answer
    │
    ▼
[Response: answer + sources + latency]
```

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Vector Database | **Endee** | High-performance, open-source, 1B vector support |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim, fast, high-quality |
| LLM | OpenAI `gpt-3.5-turbo` | Accurate, grounded responses |
| API | FastAPI | Async, auto-swagger, production-ready |
| Containerization | Docker + Docker Compose | One-command deployment |
| Testing | Pytest | 5 unit tests with mocked dependencies |
| CI/CD | GitHub Actions | Auto-test on every push |

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key

### 1. Clone and configure
```bash
git clone https://github.com/SKcoder6344/insurance-rag-endee
cd insurance-rag-endee
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Start Endee vector DB + API
```bash
docker compose up -d
```

### 3. Index the sample insurance documents
```bash
python scripts/ingest.py
```

### 4. Query via API (Swagger UI)
Open **http://localhost:8000/docs** and try the `/query` endpoint.

Or via curl:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does health insurance cover?", "top_k": 3}'
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/index` | Ingest policy documents into Endee |
| `POST` | `/query` | Ask a question (RAG pipeline) |
| `DELETE` | `/index` | Clear all vectors |

Full interactive docs: **http://localhost:8000/docs**

---

## Project Structure

```
insurance-rag-endee/
├── app/
│   ├── main.py           # FastAPI app + endpoints
│   ├── rag_pipeline.py   # Core RAG: chunk → embed → retrieve → generate
│   ├── endee_store.py    # Endee vector DB wrapper
│   ├── schemas.py        # Pydantic request/response models
│   └── config.py         # Settings from .env
├── data/
│   └── insurance_policies.txt   # Sample health + motor policy documents
├── scripts/
│   └── ingest.py         # CLI to index documents
├── tests/
│   └── test_pipeline.py  # 5 unit tests (mocked Endee + OpenAI)
├── .github/workflows/
│   └── ci.yml            # GitHub Actions CI
├── docker-compose.yml    # Endee server + API
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Performance

| Metric | Value |
|---|---|
| Embedding Model | `text-embedding-3-small` (1536 dims) |
| Vector Similarity | Cosine (INT8 precision) |
| Avg Query Latency | ~800ms (embedding + retrieval + generation) |
| Documents Indexed | Health + Motor policy (8+ sections, 30+ chunks) |
| Top-k Retrieval | Configurable (default: 3) |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Built With Endee

This project uses [Endee](https://endee.io) — an open-source, high-performance vector database capable of handling up to 1 billion vectors on a single node. Endee was selected for its:
- Native Python SDK
- Cosine similarity with INT8 precision
- Docker-first deployment
- Significantly lower memory footprint vs. alternatives like Pinecone or Weaviate

---

## Author

**Sujal Kumar Nayak**
- GitHub: [@SKcoder6344](https://github.com/SKcoder6344)
- LinkedIn: [linkedin.com/in/sujal-kumar-nayak](https://linkedin.com/in/sujal-kumar-nayak)
