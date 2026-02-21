"""
Retriever Module
================
Embeds a user query and searches ChromaDB for the most similar chunks.
Returns documents with full metadata (source file, page number) for citations.
"""

from src.core.logger import get_logger
from src.rag.embedder import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from typing import List, Dict, Optional

logger = get_logger(__name__)


class Retriever:
    """Retrieves relevant chunks from the vector store for a given query."""

    def __init__(self, embedder: EmbeddingGenerator, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5,
                 filter_source: Optional[str] = None) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question
            top_k: Number of results to return
            filter_source: Optional filename to filter results by

        Returns:
            List of dicts, each with 'text', 'source_file', 'page_number', 'score'
        """
        logger.info(f"🔍 Retrieving top {top_k} chunks for query: '{query[:80]}...'")

        # 1. Embed the query
        query_embedding = self.embedder.embed_texts([query], show_progress=False)[0]

        # 2. Search vector store
        raw_results = self.vector_store.search(
            query_embedding, n_results=top_k, filter_source=filter_source
        )

        # 3. Format results with metadata
        results = []
        documents = raw_results.get("documents", [])
        metadatas = raw_results.get("metadatas", [])
        distances = raw_results.get("distances", [])

        for i, doc_text in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0.0

            results.append({
                "text": doc_text,
                "source_file": meta.get("source_file", "Unknown"),
                "page_number": meta.get("page_number", "N/A"),
                "score": round(1 - distance, 4),  # cosine similarity
                "chunk_index": meta.get("chunk_index", i),
            })

        logger.info(f"✅ Retrieved {len(results)} chunks")
        return results

    def get_available_sources(self) -> List[str]:
        """Get list of unique source files in the vector store."""
        try:
            stats = self.vector_store.get_stats()
            return stats.get("sample_sources", [])
        except Exception:
            return []
