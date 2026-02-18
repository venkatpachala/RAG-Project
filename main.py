# """
# main.py — FastAPI Application Entry Point

# This file:
#   1. Creates the FastAPI app instance
#   2. Attaches CORS middleware (so the React frontend can talk to it)
#   3. Registers all API route groups
#   4. Defines startup/shutdown lifecycle hooks
#   5. Runs the server via uvicorn

# Run with:  uvicorn main:app --reload --port 8000
# """

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings
from api import documents, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 NotebookLM Clone starting up...")

    settings.ensure_directories()
    logger.info(f"Upload dir ready: {settings.UPLOAD_DIR}")
    logger.info(f"ChromaDB dir ready: {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"LLM Model: {settings.NVIDIA_LLM_MODEL}")
    logger.info(f"Embedding Model: {settings.NVIDIA_EMBEDDING_MODEL}")
    logger.info(f"Retrieval Top-K: {settings.RETRIEVAL_TOP_K}")

    yield  # App runs here

    # ── SHUTDOWN ──
    logger.info("👋 NotebookLM Clone shutting down...")


# ── App Instance ─────────────────────────────────────────────────
# `lifespan` wires in our startup/shutdown hooks above
app = FastAPI(
    title="NotebookLM Clone API",
    description="AI-powered document Q&A using NVIDIA open-source models",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc"     # ReDoc UI at /redoc
)


# ── CORS Middleware ───────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing.
# Browsers BLOCK requests from one origin (localhost:5173) to another
# (localhost:8000) unless the server explicitly allows it.
# We MUST add this or the React frontend cannot talk to FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,  # ["http://localhost:3000", ...]
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, DELETE, OPTIONS, etc.
    allow_headers=["*"],        # Authorization, Content-Type, etc.
)


# ── Route Registration ────────────────────────────────────────────
# Each router is a separate file with its own group of endpoints.
# `prefix` adds a URL prefix to every route in that router.
# `tags` groups endpoints in the Swagger /docs UI.

app.include_router(
    documents.router,
    prefix="/api/documents",
    tags=["Documents"]
)
# Gives us: POST /api/documents/upload
#           GET  /api/documents/
#           DELETE /api/documents/{doc_id}

app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["Chat"]
)
# Gives us: POST /api/chat/query
#           GET  /api/chat/history/{session_id}
#           DELETE /api/chat/history/{session_id}


# ── Health Check ─────────────────────────────────────────────────
# Simple endpoint to verify the server is alive.
# Used by Docker health checks, load balancers, and monitoring tools.
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "llm_model": settings.NVIDIA_LLM_MODEL,
        "embedding_model": settings.NVIDIA_EMBEDDING_MODEL,
    }


# ── Root ─────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "NotebookLM Clone API",
        "docs": "/docs",
        "health": "/health"
    }


# ── Dev Server Entry Point ────────────────────────────────────────
# Only runs when you execute `python main.py` directly.
# In production, we use: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True,    # Auto-restart on file changes (dev only)
        log_level="info"
    )