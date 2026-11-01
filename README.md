# RAG Finance Chatbot 🤖💰

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-orange?logo=google)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

A production-style **Agentic RAG (Retrieval-Augmented Generation)** chatbot for finance document QA. Upload financial documents (invoices, AR reports, transaction logs), ask questions in natural language, and get grounded answers — with built-in hallucination detection.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Retrieval Agent   │  ← Query expansion via Gemini LLM
│  (retrieval_agent)  │  ← Semantic search over FAISS vector store
└────────┬────────────┘
         │  top-k relevant chunks
         ▼
┌─────────────────────┐
│   Reasoning Agent   │  ← Chain-of-thought answer generation
│  (reasoning_agent)  │  ← Grounded strictly to retrieved context
└────────┬────────────┘
         │  raw grounded answer
         ▼
┌─────────────────────┐
│   Response Agent    │  ← Formats answer for business users
│  (response_agent)   │
└────────┬────────────┘
         │  formatted answer
         ▼
┌─────────────────────┐
│ Hallucination Check │  ← Self-reflection grounding score (0–1)
│ (hallucination_     │  ← Warns if answer drifts from context
│   check.py)         │
└────────┬────────────┘
         │
         ▼
    Final Response (JSON)
```

---

## Features

- **Hierarchical 3-agent pipeline** — Retrieval → Reasoning → Response
- **Semantic search** via FAISS + Google Generative AI embeddings
- **Hallucination detection** — self-reflection loop scores answer grounding
- **Prompt versioning** — A/B test prompt templates without code changes
- **Finance domain** — sample data includes invoices, AR reports, payments
- **REST API** — clean FastAPI endpoints with full Swagger docs
- **PDF + TXT ingestion** — upload documents via API
- **Modular** — swap any agent or model independently

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | Google `embedding-001` |
| Vector Store | FAISS (local) |
| Orchestration | LangChain |
| API Server | FastAPI + Uvicorn |
| Language | Python 3.10+ |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/zaidrazavi/rag-finance-chatbot.git
cd rag-finance-chatbot
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

API docs available at: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ingest/text` | Ingest raw text |
| `POST` | `/ingest/file` | Upload PDF or TXT |
| `POST` | `/query` | Ask a question |
| `GET` | `/prompts` | List active prompt versions |
| `DELETE` | `/vectorstore` | Clear vector store |

### Example: Ingest sample data

```bash
curl -X POST "http://localhost:8000/ingest/text" \
  -H "Content-Type: application/json" \
  -d '["Invoice #001 from TechSupply Ltd for $12,500 is overdue by 18 days."]'
```

### Example: Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Which invoices are overdue and by how much?"}'
```

### Example Response

```json
{
  "answer": "Based on the financial records, two invoices are currently overdue:\n1. Invoice #INV-2024-001 (TechSupply Ltd) — $12,500, overdue by 18 days\n2. Invoice #INV-2024-003 (OfficeWorld) — $3,400, overdue by 8 days\n\nTotal overdue amount: $15,900",
  "grounding_score": 0.95,
  "grounding_passed": true,
  "warning": null,
  "expanded_query": "overdue invoices outstanding payments finance accounts receivable",
  "sources": [
    {
      "source": "sample_finance.txt",
      "preview": "Invoice #INV-2024-001 | Vendor: TechSupply Ltd | Amount: $12,500...",
      "similarity_score": 0.182
    }
  ],
  "prompt_versions": {
    "retrieval": "retrieval-v2.0",
    "reasoning": "reasoning-v2.0",
    "response": "response-v1.0"
  }
}
```

---

## Prompt Versioning

Prompts are versioned in `app/prompts.py`. Switch versions by editing `ACTIVE_PROMPTS`:

```python
ACTIVE_PROMPTS = {
    "retrieval": RETRIEVAL_PROMPT_V2,   # swap to V1 for A/B test
    "reasoning": REASONING_PROMPT_V2,
    ...
}
```

Every response includes `prompt_versions` so you can track which version produced each answer.

---

## Project Structure

```
rag-finance-chatbot/
├── app/
│   ├── main.py                  # FastAPI server & endpoints
│   ├── rag_pipeline.py          # Pipeline orchestrator
│   ├── embeddings.py            # FAISS vector store
│   ├── prompts.py               # Versioned prompt templates
│   ├── hallucination_check.py   # Self-reflection grounding scorer
│   └── agents/
│       ├── retrieval_agent.py   # Query expansion + semantic search
│       ├── reasoning_agent.py   # Grounded answer generation
│       └── response_agent.py    # Response formatting
├── data/
│   └── sample_finance.txt       # Sample AR/invoice data
├── tests/
│   └── test_pipeline.py         # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Author

**Mohd Zayed Kazim Shalil**
- GitHub: [@zaidrazavi](https://github.com/zaidrazavi)
- LinkedIn: [linkedin.com/in/mohdZayed](https://linkedin.com/in/mohdZayed)
- Email: khanzaid1375@gmail.com

---

## License

MIT License — free to use and modify.
