"""
Main Entry Point with Embeddings
=================================
Run: python main.py
"""

from pathlib import Path
from datetime import datetime
import logging

from src.rag.pipeline import PDFPipeline

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FOLDER = r"C:\Users\pritam\Desktop\RAG-Project\data"

# Embedding settings
EMBEDDING_CONFIG = {
    'model': 'fast',           # Options: 'fast', 'balanced', 'best'
    'batch_size': 32,          # Batch size for embedding generation
    'generate': True           # Set to False to skip embeddings
}

# Chunking settings
CHUNK_CONFIG = {
    'size': 512,              # Characters per chunk
    'overlap': 50          # Overlap between chunks
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(data_folder):
    """Setup logging configuration"""
    logs_folder = Path(data_folder) / "logs"
    logs_folder.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_folder / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function - Run the complete RAG pipeline"""
    
    # Setup logging
    log_file = setup_logging(DATA_FOLDER)
    logger = logging.getLogger(__name__)
    
    # Print header
    print("\n" + "="*70)
    print("📚 RAG PDF PROCESSING PIPELINE")
    print("="*70)
    print(f"📁 Data folder: {DATA_FOLDER}")
    print(f"📝 Log file: {log_file}")
    print(f"🧠 Embeddings: {'Enabled' if EMBEDDING_CONFIG['generate'] else 'Disabled'}")
    if EMBEDDING_CONFIG['generate']:
        print(f"🤖 Model: {EMBEDDING_CONFIG['model']}")
    print("="*70 + "\n")
    
    # Check for PDFs
    data_path = Path(DATA_FOLDER)
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found!")
        print(f"   Please add PDFs to: {DATA_FOLDER}\n")
        logger.warning("No PDF files found in data folder")
        return
    
    # Display found PDFs
    print(f"📄 Found {len(pdf_files)} PDF(s):")
    total_size_mb = 0
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / 1024 / 1024
        total_size_mb += size_mb
        print(f"   • {pdf.name} ({size_mb:.2f} MB)")
    print(f"   📊 Total size: {total_size_mb:.2f} MB\n")
    
    # Confirm before processing
    try:
        user_input = input("Press ENTER to start processing (or Ctrl+C to cancel)... ")
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user\n")
        logger.info("Processing cancelled by user")
        return
    
    print()
    logger.info("Starting pipeline processing")
    
    # ========================================================================
    # RUN PIPELINE
    # ========================================================================
    
    try:
        # Create pipeline instance
        pipeline = PDFPipeline(
            data_folder=DATA_FOLDER,
            chunk_size=CHUNK_CONFIG['size'],
            chunk_overlap=CHUNK_CONFIG['overlap'],
            embedding_model=EMBEDDING_CONFIG['model'],
            generate_embeddings=EMBEDDING_CONFIG['generate']
        )
        
        # Run the pipeline
        results = pipeline.run()
        
        if not results:
            print("\n❌ Pipeline failed! Check logs for details.\n")
            logger.error("Pipeline returned no results")
            return
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        
        print("\n" + "="*70)
        print("📊 PROCESSING RESULTS")
        print("="*70)
        
        # Show chunks per file
        print("\n📦 CHUNKS PER FILE:")
        print("─"*70)
        for idx, (filename, chunks) in enumerate(results['chunks'].items(), 1):
            avg_size = sum(c['length'] for c in chunks) / len(chunks) if chunks else 0
            print(f"\n{idx}. {filename}")
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Average chunk size: {avg_size:.0f} characters")
            print(f"   • Min size: {min(c['length'] for c in chunks) if chunks else 0} chars")
            print(f"   • Max size: {max(c['length'] for c in chunks) if chunks else 0} chars")
        
        # Show embeddings if generated
        if results.get('embeddings'):
            print(f"\n{'─'*70}")
            print("🧠 EMBEDDINGS:")
            print("─"*70)
            for filename, data in results['embeddings'].items():
                embedding_shape = data['embeddings'].shape
                print(f"\n   • {filename}")
                print(f"     Shape: {embedding_shape}")
                print(f"     Dimension: {embedding_shape[1]}")
                print(f"     Model: {data['metadata']['model_name']}")
        
        # Show overall statistics
        stats = results['stats']
        print(f"\n{'='*70}")
        print("📈 OVERALL STATISTICS")
        print("="*70)
        print(f"   📚 Total PDFs processed: {stats['total_pdfs']}")
        print(f"   📄 Total pages: {stats['total_pages']}")
        print(f"   ✂️  Total chunks created: {stats['total_chunks']}")
        print(f"   🔤 Total characters: {stats['total_characters']:,}")
        print(f"   📊 Average chunks per PDF: {stats['total_chunks']/stats['total_pdfs']:.1f}")
        print(f"   📊 Average chars per chunk: {stats['total_characters']/stats['total_chunks']:.0f}")
        
        if stats.get('embeddings_generated'):
            print(f"   🧠 Embeddings: ✅ Generated")
        else:
            print(f"   🧠 Embeddings: ⏭️  Skipped")
        
        # Show output locations
        print(f"\n{'─'*70}")
        print("📁 OUTPUT LOCATIONS:")
        print("─"*70)
        print(f"   • Chunks: {results['chunks_folder']}")
        if results.get('embeddings_folder'):
            print(f"   • Embeddings: {results['embeddings_folder']}")
        print(f"   • Logs: {log_file.parent}")
        
        print("="*70)
        
        # Success message
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📝 Full log available at: {log_file}")
        print("="*70 + "\n")
        
        logger.info("Pipeline completed successfully")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user\n")
        logger.warning("Processing interrupted by user")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        print(f"Check log file for details: {log_file}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()