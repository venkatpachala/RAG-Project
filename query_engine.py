"""
RAG Query Engine
================
Thin coordinator that wires together all modular components from src/.

Ingest pipeline:  PDF → Loader → Chunker → Embedder → VectorStore
Query pipeline:   Question → Retriever → Generator → Answer + Citations
"""

import os
import shutil
from pathlib import Path

from src.core.config import settings
from src.core.logger import get_logger
from src.rag.loader import SimplePDFLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.services.generator import GeminiGenerator

logger = get_logger(__name__)


class RAGQueryEngine:
    """
    Central coordinator for the RAG pipeline.
    All heavy lifting is delegated to modular components in src/.
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("🚀 Initializing RAG Query Engine")
        logger.info("=" * 60)

        # --- Components ---
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        self.embedder = EmbeddingGenerator(
            model_name=settings.EMBEDDING_MODEL,
        )

        self.vector_store = VectorStore(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME,
        )
        self.vector_store.connect()

        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

        self.generator = GeminiGenerator(
            api_key=settings.GOOGLE_API_KEY,
            model_name=settings.GEMINI_MODEL,
        )

        logger.info("✅ All components initialized")

    # ------------------------------------------------------------------
    # INGEST PIPELINE: PDF → Chunks → Embeddings → VectorStore
    # ------------------------------------------------------------------
    def add_document(self, file_path: str) -> dict:
        """
        Ingest a single PDF into the RAG pipeline.

        Returns:
            dict with ingest stats (num_chunks, filename, etc.)
        """
        path = Path(file_path)
        logger.info(f"📥 Ingesting: {path.name}")

        # 1. Load PDF
        loader = SimplePDFLoader(data_folder=path.parent)
        pdf_data = loader.load_pdf(path.name)
        logger.info(f"📄 Loaded {pdf_data['metadata']['total_pages']} pages")

        # 2. Chunk
        chunks = self.chunker.chunk_pdf_data(pdf_data)
        if not chunks:
            logger.warning("⚠️ No chunks created from document")
            return {"filename": path.name, "num_chunks": 0}

        # 3. Save chunks to data/chunks/
        self.chunker.save_chunks(
            {path.name: chunks},
            output_folder=settings.chunks_dir,
        )

        # 4. Embed
        embedded_data = {path.name: self.embedder.embed_chunks(chunks)}

        # 5. Save embeddings to data/embeddings/
        self.embedder.save_embeddings(embedded_data, settings.embeddings_dir)

        # 6. Store in ChromaDB
        self.vector_store.store_embedded_data(embedded_data)

        stats = {"filename": path.name, "num_chunks": len(chunks)}
        logger.info(f"✅ Ingested {path.name}: {len(chunks)} chunks stored")
        return stats

    # ------------------------------------------------------------------
    # QUERY PIPELINE: Question → Retriever → Generator → Answer
    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = None,
              filter_source: str = None) -> dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            filter_source: Optional source file filter

        Returns:
            dict with 'answer' and 'sources' keys
        """
        k = top_k or settings.RETRIEVAL_TOP_K

        # 1. Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=question, top_k=k, filter_source=filter_source
        )

        # 2. Generate answer with citations
        answer = self.generator.generate(question, chunks)

        # 3. Build sources list
        sources = []
        seen = set()
        for c in chunks:
            key = f"{c['source_file']}|{c['page_number']}"
            if key not in seen:
                sources.append({
                    "file": c["source_file"],
                    "page": c["page_number"],
                    "score": c["score"],
                })
                seen.add(key)

        return {"answer": answer, "sources": sources}

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return self.vector_store.get_stats()

    def get_source_files(self) -> list:
        """Get list of ingested source files."""
        return self.retriever.get_available_sources()

    def clear_all(self):
        """Clear vector store and all generated data."""
        logger.info("🗑️ Clearing all data...")

        # Clear ChromaDB collection
        self.vector_store.clear_collection()

        # Clear generated data directories
        for d in [settings.chunks_dir, settings.embeddings_dir]:
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

        logger.info("✅ All data cleared")