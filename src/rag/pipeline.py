"""PDF Processing Pipeline"""

from pathlib import Path
import logging

# Use absolute imports
from src.rag.loader import SimplePDFLoader
from src.rag.chunker import TextChunker

logger = logging.getLogger(__name__)


class PDFPipeline:
    """Complete PDF processing pipeline"""
    
    def __init__(self, data_folder, chunk_size=1000, chunk_overlap=200):
        self.data_folder = Path(data_folder)
        self.chunks_folder = self.data_folder / "chunks"
        self.logs_folder = self.data_folder / "logs"
        
        # Create folders
        self.chunks_folder.mkdir(parents=True, exist_ok=True)
        self.logs_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = SimplePDFLoader(self.data_folder)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        
        logger.info(f"🔧 Pipeline initialized")
    
    def run(self):
        """Run the complete pipeline"""
        logger.info("="*60)
        logger.info("🚀 STARTING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load PDFs
        logger.info("\n📥 STEP 1: Loading PDFs...")
        all_pdf_data = self.loader.load_all_pdfs()
        
        if not all_pdf_data:
            logger.error("❌ No PDFs loaded!")
            return None
        
        logger.info(f"✅ Loaded {len(all_pdf_data)} PDF(s)")
        
        # Step 2: Chunk PDFs
        logger.info("\n✂️ STEP 2: Chunking text...")
        all_chunks = self.chunker.chunk_multiple_pdfs(all_pdf_data)
        
        logger.info(f"✅ Created chunks for {len(all_chunks)} PDF(s)")
        
        # Step 3: Save chunks
        logger.info("\n💾 STEP 3: Saving chunks...")
        self.chunker.save_chunks(all_chunks, self.chunks_folder)
        
        logger.info(f"✅ Chunks saved to: {self.chunks_folder}")
        
        # Stats
        total_pages = sum(d['metadata']['total_pages'] for d in all_pdf_data)
        total_chars = sum(d['metadata']['total_characters'] for d in all_pdf_data)
        total_chunks = sum(len(c) for c in all_chunks.values())
        
        logger.info("\n" + "="*60)
        logger.info("📊 PIPELINE COMPLETE")
        logger.info("="*60)
        
        return {
            'pdf_data': all_pdf_data,
            'chunks': all_chunks,
            'chunks_folder': self.chunks_folder,
            'stats': {
                'total_pdfs': len(all_pdf_data),
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'total_characters': total_chars
            }
        }