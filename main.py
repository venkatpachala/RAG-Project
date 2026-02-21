"""
Main Entry Point - Complete RAG Pipeline
==========================================
Stages: PDF → Chunks → Embeddings → ChromaDB
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
    'model': 'fast',
    'batch_size': 32,
    'generate': True
}

# Chunking settings
CHUNK_CONFIG = {
    'size': 512,
    'overlap': 50
}

# Vector Store settings  ← NEW
VECTOR_STORE_CONFIG = {
    'store': True,                          # Set to False to skip ChromaDB
    'collection_name': 'knowledge_base',    # ChromaDB collection name
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

    # ========================================================================
    # HEADER
    # ========================================================================
    print("\n" + "="*70)
    print("📚 RAG PDF PROCESSING PIPELINE")
    print("="*70)
    print(f"📁 Data folder:   {DATA_FOLDER}")
    print(f"📝 Log file:      {log_file}")
    print(f"✂️  Chunking:      size={CHUNK_CONFIG['size']}, overlap={CHUNK_CONFIG['overlap']}")
    print(f"🧠 Embeddings:    {'Enabled (' + EMBEDDING_CONFIG['model'] + ')' if EMBEDDING_CONFIG['generate'] else 'Disabled'}")
    print(f"🗄️  Vector Store:  {'Enabled (' + VECTOR_STORE_CONFIG['collection_name'] + ')' if VECTOR_STORE_CONFIG['store'] else 'Disabled'}")
    print("="*70 + "\n")

    # ========================================================================
    # CHECK FOR PDFs
    # ========================================================================
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

    # Confirm
    try:
        input("Press ENTER to start processing (or Ctrl+C to cancel)... ")
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
        pipeline = PDFPipeline(
            data_folder=DATA_FOLDER,
            chunk_size=CHUNK_CONFIG['size'],
            chunk_overlap=CHUNK_CONFIG['overlap'],
            embedding_model=EMBEDDING_CONFIG['model'],
            generate_embeddings=EMBEDDING_CONFIG['generate'],
            store_vectors=VECTOR_STORE_CONFIG['store'],              # ← NEW
            collection_name=VECTOR_STORE_CONFIG['collection_name'],  # ← NEW
        )

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

        # ── Chunks per file ──
        print("\n📦 CHUNKS PER FILE:")
        print("─"*70)
        for idx, (filename, chunks) in enumerate(results['chunks'].items(), 1):
            avg_size = sum(c['length'] for c in chunks) / len(chunks) if chunks else 0
            print(f"\n{idx}. {filename}")
            print(f"   • Total chunks:       {len(chunks)}")
            print(f"   • Average chunk size:  {avg_size:.0f} characters")
            print(f"   • Min size:            {min(c['length'] for c in chunks) if chunks else 0} chars")
            print(f"   • Max size:            {max(c['length'] for c in chunks) if chunks else 0} chars")

        # ── Embeddings ──
        if results.get('embeddings'):
            print(f"\n{'─'*70}")
            print("🧠 EMBEDDINGS:")
            print("─"*70)
            for filename, data in results['embeddings'].items():
                shape = data['embeddings'].shape
                print(f"\n   • {filename}")
                print(f"     Shape:     {shape}")
                print(f"     Dimension: {shape[1]}")
                print(f"     Model:     {data['metadata']['model_name']}")

        # ── Vector Store ──                                        ← NEW
        if results.get('vector_store') and not results['vector_store'].get('error'):
            vs = results['vector_store']
            print(f"\n{'─'*70}")
            print("🗄️  VECTOR STORE (ChromaDB):")
            print("─"*70)
            print(f"   • Collection:      {vs.get('collection_name', 'N/A')}")
            print(f"   • Chunks stored:   {vs.get('total_stored', 0)}")
            print(f"   • Total in DB:     {vs.get('total_in_collection', 0)}")
            print(f"   • Storage time:    {vs.get('storage_time', 0):.2f}s")
            print(f"   • Location:        {vs.get('persist_directory', 'N/A')}")
        elif results.get('vector_store', {}).get('error'):
            print(f"\n{'─'*70}")
            print("🗄️  VECTOR STORE: ❌ Failed")
            print(f"   Error: {results['vector_store']['error']}")

        # ── Overall Statistics ──
        stats = results['stats']
        print(f"\n{'='*70}")
        print("📈 OVERALL STATISTICS")
        print("="*70)
        print(f"   📚 PDFs processed:        {stats['total_pdfs']}")
        print(f"   📄 Total pages:           {stats['total_pages']}")
        print(f"   ✂️  Total chunks:          {stats['total_chunks']}")
        print(f"   🔤 Total characters:      {stats['total_characters']:,}")
        print(f"   📊 Avg chunks per PDF:    {stats['total_chunks']/max(stats['total_pdfs'],1):.1f}")
        print(f"   📊 Avg chars per chunk:   {stats['total_characters']/max(stats['total_chunks'],1):.0f}")
        print(f"   🧠 Embeddings:            {'✅ Generated' if stats.get('embeddings_generated') else '⏭️  Skipped'}")
        print(f"   🗄️  Vector Store:          {'✅ Stored (' + str(stats.get('vector_count', 0)) + ' docs)' if stats.get('vectors_stored') else '⏭️  Skipped'}")
        print(f"   ⏱️  Total pipeline time:   {stats.get('pipeline_time', 0):.2f}s")

        # ── Output Locations ──
        print(f"\n{'─'*70}")
        print("📁 OUTPUT LOCATIONS:")
        print("─"*70)
        print(f"   • Chunks:       {results['chunks_folder']}")
        if results.get('embeddings_folder'):
            print(f"   • Embeddings:   {results['embeddings_folder']}")
        if results.get('vectordb_folder'):
            print(f"   • Vector DB:    {results['vectordb_folder']}")
        print(f"   • Logs:         {log_file.parent}")

        print(f"\n{'='*70}")
        print(f"✅ Pipeline completed successfully!")
        print(f"📝 Full log: {log_file}")
        print(f"\n💡 Next step: Build the query/retrieval module!")
        print("="*70 + "\n")

        logger.info("Pipeline completed successfully")

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user\n")
        logger.warning("Processing interrupted by user")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"Check log file: {log_file}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()