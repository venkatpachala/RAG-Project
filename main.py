"""
Main Entry Point
Run: python main.py
"""

from pathlib import Path
from datetime import datetime
import logging

from src.rag.pipeline import PDFPipeline

# Configuration
DATA_FOLDER = r"C:\Users\pritam\Desktop\RAG-Project\data"


def setup_logging(data_folder):
    """Setup logging"""
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


def main():
    """Main function"""
    
    log_file = setup_logging(DATA_FOLDER)
    
    print("\n" + "="*70)
    print("📚 RAG PDF PROCESSING PIPELINE")
    print("="*70)
    print(f"📁 Data folder: {DATA_FOLDER}")
    print(f"📝 Log file: {log_file}")
    print("="*70 + "\n")
    
    # Check for PDFs
    data_path = Path(DATA_FOLDER)
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found!")
        print(f"   Please add PDFs to: {DATA_FOLDER}\n")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF(s):")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / 1024 / 1024
        print(f"   • {pdf.name} ({size_mb:.2f} MB)")
    print()
    
    # Run pipeline
    pipeline = PDFPipeline(
        data_folder=DATA_FOLDER,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    results = pipeline.run()
    
    if results:
        print("\n" + "="*70)
        print("📊 RESULTS")
        print("="*70)
        
        for idx, (filename, chunks) in enumerate(results['chunks'].items(), 1):
            avg_size = sum(c['length'] for c in chunks) / len(chunks) if chunks else 0
            print(f"\n{idx}. {filename}")
            print(f"   • Chunks: {len(chunks)}")
            print(f"   • Avg size: {avg_size:.0f} chars")
        
        stats = results['stats']
        print(f"\n{'─'*70}")
        print(f"📊 TOTALS: {stats['total_pdfs']} PDFs | {stats['total_pages']} pages | {stats['total_chunks']} chunks")
        print(f"📁 Output: {results['chunks_folder']}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()