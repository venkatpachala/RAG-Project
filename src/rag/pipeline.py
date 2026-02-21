"""
PDF Pipeline Module
====================
Handles: PDF → Text → Chunks → Embeddings → ChromaDB
"""

from pathlib import Path
import logging
import time
from datetime import datetime

from src.rag.loader import SimplePDFLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import EmbeddingGenerator
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class PDFPipeline:
    """Complete RAG pipeline: PDF → Text → Chunks → Embeddings → ChromaDB"""

    def __init__(self, data_folder, chunk_size=512, chunk_overlap=50,
                 embedding_model='fast', generate_embeddings=True,
                 store_vectors=True, collection_name='knowledge_base'):
        """
        Initialize the pipeline

        Args:
            data_folder: Path to folder containing PDFs
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            embedding_model: 'fast', 'balanced', or 'best'
            generate_embeddings: Whether to generate embeddings
            store_vectors: Whether to store in ChromaDB
            collection_name: ChromaDB collection name
        """
        self.data_folder = Path(data_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.generate_embeddings = generate_embeddings
        self.store_vectors = store_vectors
        self.collection_name = collection_name

        # Output folders
        self.chunks_folder = self.data_folder / "chunks"
        self.embeddings_folder = self.data_folder / "embeddings"
        self.vectordb_folder = self.data_folder / "chroma_db"

        logger.info(f"📋 Pipeline Configuration:")
        logger.info(f"   📂 Data folder:    {self.data_folder}")
        logger.info(f"   ✂️  Chunk size:     {chunk_size}")
        logger.info(f"   🔗 Chunk overlap:  {chunk_overlap}")
        logger.info(f"   🧠 Embeddings:     {embedding_model if generate_embeddings else 'Disabled'}")
        logger.info(f"   🗄️  Vector Store:   {'Enabled' if store_vectors else 'Disabled'}")

    def run(self):
        """Run the complete pipeline"""
        pipeline_start = time.time()

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🚀 STARTING RAG PIPELINE")
        logger.info(f"{'='*60}")

        results = {
            'chunks': {},
            'embeddings': {},
            'vector_store': {},
            'stats': {},
            'chunks_folder': str(self.chunks_folder),
            'embeddings_folder': None,
            'vectordb_folder': None,
        }

        # ================================================================
        # STAGE 1: PDF EXTRACTION
        # ================================================================
        logger.info(f"\n{'─'*60}")
        logger.info(f"📖 STAGE 1: PDF EXTRACTION")
        logger.info(f"{'─'*60}")

        loader = SimplePDFLoader(data_folder=self.data_folder)
        all_pdf_data = loader.load_all_pdfs()

        if not all_pdf_data:
            logger.error("❌ No PDFs loaded. Stopping pipeline.")
            return None

        # Calculate total pages
        total_pages = sum(
            d['metadata']['total_pages'] for d in all_pdf_data
        )
        logger.info(f"✅ Loaded {len(all_pdf_data)} PDF(s), {total_pages} pages total")

        # ================================================================
        # STAGE 2: CHUNKING
        # ================================================================
        logger.info(f"\n{'─'*60}")
        logger.info(f"✂️  STAGE 2: CHUNKING")
        logger.info(f"{'─'*60}")

        chunker = TextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # chunk_multiple_pdfs expects list of pdf_data dicts
        all_chunks = chunker.chunk_multiple_pdfs(all_pdf_data)

        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        total_characters = sum(
            sum(c['length'] for c in chunks)
            for chunks in all_chunks.values()
        )

        results['chunks'] = all_chunks

        # Save chunks to disk
        chunker.save_chunks(all_chunks, self.chunks_folder)
        logger.info(f"✅ {total_chunks} chunks created and saved")

        # ================================================================
        # STAGE 3: EMBEDDINGS
        # ================================================================
        embedded_data = {}

        if self.generate_embeddings:
            logger.info(f"\n{'─'*60}")
            logger.info(f"🧠 STAGE 3: EMBEDDING GENERATION")
            logger.info(f"{'─'*60}")

            embedder = EmbeddingGenerator(
                model_name=self.embedding_model,
                batch_size=32
            )
            embedder.load_model()

            # embed_all_chunks expects Dict[str, List[Dict]]
            # all_chunks is already in that format: {filename: [chunks]}
            embedded_data = embedder.embed_all_chunks(all_chunks)

            # Save embeddings to disk
            embedder.save_embeddings(embedded_data, self.embeddings_folder)

            results['embeddings'] = embedded_data
            results['embeddings_folder'] = str(self.embeddings_folder)

            logger.info(f"✅ Embeddings generated and saved")

        else:
            logger.info(f"\n⏭️  STAGE 3: SKIPPED (embeddings disabled)")

        # ================================================================
        # STAGE 4: VECTOR STORE (ChromaDB)
        # ================================================================
        if self.store_vectors and embedded_data:
            logger.info(f"\n{'─'*60}")
            logger.info(f"🗄️  STAGE 4: VECTOR STORAGE (ChromaDB)")
            logger.info(f"{'─'*60}")

            try:
                vector_store = VectorStore(
                    persist_directory=str(self.vectordb_folder),
                    collection_name=self.collection_name
                )
                vector_store.connect()

                # Store all embedded data
                store_stats = vector_store.store_embedded_data(embedded_data)

                results['vector_store'] = store_stats
                results['vectordb_folder'] = str(self.vectordb_folder)

                logger.info(f"✅ Vectors stored in ChromaDB")

            except Exception as e:
                logger.error(f"❌ Vector storage failed: {e}", exc_info=True)
                results['vector_store'] = {'error': str(e)}

        elif self.store_vectors and not embedded_data:
            logger.warning(
                "⚠️  Skipping vector storage — no embeddings generated"
            )
            logger.warning(
                "   Enable embeddings with generate_embeddings=True"
            )

        else:
            logger.info(f"\n⏭️  STAGE 4: SKIPPED (vector store disabled)")

        # ================================================================
        # COMPILE STATISTICS
        # ================================================================
        pipeline_time = time.time() - pipeline_start

        results['stats'] = {
            'total_pdfs': len(all_pdf_data),
            'total_pages': total_pages,
            'total_chunks': total_chunks,
            'total_characters': total_characters,
            'embeddings_generated': bool(embedded_data),
            'vectors_stored': bool(
                results.get('vector_store', {}).get('total_stored')
            ),
            'vector_count': results.get(
                'vector_store', {}
            ).get('total_in_collection', 0),
            'pipeline_time': pipeline_time,
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ PIPELINE COMPLETE in {pipeline_time:.2f}s")
        logger.info(f"{'='*60}")
        logger.info(f"   📚 PDFs:       {len(all_pdf_data)}")
        logger.info(f"   📄 Pages:      {total_pages}")
        logger.info(f"   ✂️  Chunks:     {total_chunks}")
        logger.info(f"   🧠 Embeddings: {'✅' if embedded_data else '⏭️'}")
        logger.info(f"   🗄️  Vectors:    {'✅' if results['stats']['vectors_stored'] else '⏭️'}")
        logger.info(f"{'='*60}")

        return results