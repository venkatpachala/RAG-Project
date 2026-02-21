"""
RAG Module
==========
Modular components for the RAG pipeline:
  - loader: PDF document loading
  - chunker: Text splitting into chunks
  - embedder: Vector embedding generation
  - vector_store: ChromaDB storage and retrieval
  - retriever: Query → Embed → Search
  - visualizer: PCA / heatmap visualizations
"""

from src.rag.loader import SimplePDFLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever

__all__ = [
    "SimplePDFLoader",
    "TextChunker",
    "EmbeddingGenerator",
    "VectorStore",
    "Retriever",
]