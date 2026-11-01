"""
FastAPI server — REST API for the RAG Finance Chatbot.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_pipeline import RAGPipeline
from app.prompts import list_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initialising RAG pipeline...")
    pipeline = RAGPipeline()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="RAG Finance Chatbot",
    description=(
        "Agentic RAG chatbot for finance document QA. "
        "Upload documents, ask questions, get grounded answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, example="What is the total outstanding invoice amount?")


class SourceItem(BaseModel):
    source: str
    preview: str
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    grounding_score: float
    grounding_passed: bool
    warning: Optional[str]
    expanded_query: str
    sources: List[SourceItem]
    prompt_versions: dict


class IngestResponse(BaseModel):
    message: str
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    version: str
    pipeline_ready: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "version": "1.0.0",
        "pipeline_ready": pipeline is not None,
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "pipeline_ready": pipeline is not None,
    }


@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
def ingest_text(texts: List[str]):
    """
    Ingest raw text strings into the vector store.
    Useful for quick testing without file uploads.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided.")
    try:
        chunks = pipeline.ingest(texts, [{"source": f"text_input_{i}"} for i in range(len(texts))])
        return {"message": f"Successfully ingested {len(texts)} document(s).", "chunks_created": chunks}
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a .txt or .pdf file to ingest into the vector store.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    filename = file.filename or "uploaded_file"

    if filename.endswith(".txt"):
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        texts = [text]
        metadatas = [{"source": filename}]

    elif filename.endswith(".pdf"):
        import tempfile, pypdf
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        reader = pypdf.PdfReader(tmp_path)
        texts = [page.extract_text() or "" for page in reader.pages]
        metadatas = [{"source": f"{filename}_page_{i+1}"} for i in range(len(texts))]
        os.unlink(tmp_path)

    else:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    try:
        chunks = pipeline.ingest(texts, metadatas)
        return {"message": f"'{filename}' ingested successfully.", "chunks_created": chunks}
    except Exception as e:
        logger.error(f"File ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest):
    """
    Ask a question. The pipeline retrieves relevant context,
    reasons over it, and returns a grounded answer with hallucination score.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        result = pipeline.query(request.query)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please ingest documents first via /ingest/text or /ingest/file."
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prompts", tags=["Meta"])
def get_prompts():
    """List all active prompt versions — useful for A/B testing audit."""
    return list_prompts()


@app.delete("/vectorstore", tags=["Meta"])
def clear_vectorstore():
    """Clear the vector store. Use with caution."""
    import shutil
    from pathlib import Path
    vs_path = Path("data/vectorstore")
    if vs_path.exists():
        shutil.rmtree(vs_path)
        return {"message": "Vector store cleared."}
    return {"message": "Vector store was already empty."}


# ── Serve UI ─────────────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

_static = Path(__file__).parent.parent / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")

    @app.get("/ui", include_in_schema=False)
    def serve_ui():
        return FileResponse(str(_static / "index.html"))
