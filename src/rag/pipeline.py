"""
PDF Processing Pipeline with Embeddings
========================================
Complete pipeline: Load → Chunk → Embed → Save
"""

from pathlib import Path
import logging

# Absolute imports
from src.rag.loader import SimplePDFLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import EmbeddingGenerator

logger = logging.getLogger(__name__)


class PDFPipeline:
    """Complete PDF processing pipeline: Load → Chunk → Embed → Save"""
    
    def __init__(
        self,
        data_folder,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model='fast',
        generate_embeddings=True
    ):
        """
        Initialize pipeline
        
        Args:
            data_folder: Folder containing PDFs
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            embedding_model: 'fast', 'balanced', or 'best'
            generate_embeddings: Whether to generate embeddings
        """
        self.data_folder = Path(data_folder)
        self.chunks_folder = self.data_folder / "chunks"
        self.embeddings_folder = self.data_folder / "embeddings"
        self.logs_folder = self.data_folder / "logs"
        
        # Create folders
        self.chunks_folder.mkdir(parents=True, exist_ok=True)
        self.embeddings_folder.mkdir(parents=True, exist_ok=True)
        self.logs_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = SimplePDFLoader(self.data_folder)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        
        # Initialize embedder if needed
        self.generate_embeddings = generate_embeddings
        self.embedder = None
        
        if generate_embeddings:
            logger.info(f"🧠 Initializing embedder with model: {embedding_model}")
            self.embedder = EmbeddingGenerator(model_name=embedding_model)
        
        logger.info(f"🔧 Pipeline initialized")
        logger.info(f"   📁 Data folder: {self.data_folder}")
        logger.info(f"   📁 Chunks folder: {self.chunks_folder}")
        if generate_embeddings:
            logger.info(f"   📁 Embeddings folder: {self.embeddings_folder}")
        logger.info(f"   🧠 Generate embeddings: {generate_embeddings}")
    
    def run(self):
        """
        Run the complete pipeline
        
        Returns:
            Dictionary with results or None if failed
        """
        logger.info("="*60)
        logger.info("🚀 STARTING PIPELINE")
        logger.info("="*60)
        
        # ================================================================
        # STEP 1: Load PDFs
        # ================================================================
        logger.info("\n📥 STEP 1: Loading PDFs...")
        all_pdf_data = self.loader.load_all_pdfs()
        
        if not all_pdf_data:
            logger.error("❌ No PDFs loaded!")
            return None
        
        logger.info(f"✅ Loaded {len(all_pdf_data)} PDF(s)")
        
        # ================================================================
        # STEP 2: Chunk PDFs
        # ================================================================
        logger.info("\n✂️ STEP 2: Chunking text...")
        all_chunks = self.chunker.chunk_multiple_pdfs(all_pdf_data)
        
        total_chunks = sum(len(c) for c in all_chunks.values())
        logger.info(f"✅ Created {total_chunks} chunks from {len(all_chunks)} PDF(s)")
        
        # ================================================================
        # STEP 3: Save chunks
        # ================================================================
        logger.info("\n💾 STEP 3: Saving chunks...")
        self.chunker.save_chunks(all_chunks, self.chunks_folder)
        
        logger.info(f"✅ Chunks saved to: {self.chunks_folder}")
        
        # ================================================================
        # STEP 4: Generate embeddings (if enabled)
        # ================================================================
        embedded_data = None
        
        if self.generate_embeddings and self.embedder:
            logger.info("\n🧠 STEP 4: Generating embeddings...")
            
            try:
                # Load model first
                logger.info("   Loading embedding model...")
                self.embedder.load_model()
                
                # Generate embeddings for all chunks
                embedded_data = self.embedder.embed_all_chunks(all_chunks)
                
                # Save embeddings to disk
                logger.info("\n   Saving embeddings...")
                self.embedder.save_embeddings(embedded_data, self.embeddings_folder)
                
                logger.info(f"✅ Embeddings saved to: {self.embeddings_folder}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate embeddings: {e}")
                logger.exception("Full error traceback:")
                logger.warning("⚠️  Continuing without embeddings...")
                embedded_data = None
        else:
            logger.info("\n⏭️  STEP 4: Skipping embeddings (disabled)")
        
        # ================================================================
        # Calculate statistics
        # ================================================================
        total_pages = sum(d['metadata']['total_pages'] for d in all_pdf_data)
        total_chars = sum(d['metadata']['total_characters'] for d in all_pdf_data)
        total_chunks = sum(len(c) for c in all_chunks.values())
        
        logger.info("\n" + "="*60)
        logger.info("📊 PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"   📚 PDFs processed: {len(all_pdf_data)}")
        logger.info(f"   📄 Total pages: {total_pages}")
        logger.info(f"   ✂️  Total chunks: {total_chunks}")
        logger.info(f"   🔤 Total characters: {total_chars:,}")
        
        if embedded_data:
            total_embeddings = sum(len(d['chunks']) for d in embedded_data.values())
            logger.info(f"   🧠 Embeddings: ✅ Generated ({total_embeddings} vectors)")
        else:
            logger.info(f"   🧠 Embeddings: ⏭️  Skipped")
        
        logger.info("="*60)
        
        # ================================================================
        # Return results
        # ================================================================
        return {
            'pdf_data': all_pdf_data,
            'chunks': all_chunks,
            'embeddings': embedded_data,
            'chunks_folder': self.chunks_folder,
            'embeddings_folder': self.embeddings_folder if self.generate_embeddings else None,
            'stats': {
                'total_pdfs': len(all_pdf_data),
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'total_characters': total_chars,
                'embeddings_generated': embedded_data is not None
            }
        }