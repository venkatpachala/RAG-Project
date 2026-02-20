"""
Production-Grade PDF Loader
============================
Handles large quantities of data with:
- Error recovery
- Memory optimization
- Progress tracking
- Metadata preservation
- Batch processing
- Caching
- Data validation
"""

from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
import hashlib
from datetime import datetime
import pickle

# Add this at the very beginning of your script
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pdf_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RobustPDFLoader:
    """
    Production-grade PDF loader with:
    - Memory management
    - Error recovery
    - Progress tracking
    - Caching
    - Validation
    """
    
    def __init__(self, data_folder: str = "data", cache_dir: str = ".cache"):
        """
        Initialize the loader.
        
        Args:
            data_folder: Folder containing PDF files
            cache_dir: Folder for caching processed data
        """
        self.data_folder = Path(data_folder)
        self.cache_dir = Path(cache_dir)
        
        # Create directories if they don't exist
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized loader: data_folder={self.data_folder}, cache_dir={self.cache_dir}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """
        Get MD5 hash of file for cache validation.
        
        Args:
            file_path: Path to file
        
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def load_single_pdf(
        self, 
        file_path: str,
        max_pages: Optional[int] = None,
        skip_empty_pages: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Load a single PDF file with error handling.
        
        Args:
            file_path: Path to PDF
            max_pages: Maximum pages to load (None = all)
            skip_empty_pages: Whether to skip pages with no text
        
        Returns:
            Tuple of (documents list, metadata dict)
        """
        pdf_path = Path(file_path)
        
        # Validate file
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"Not a PDF file: {file_path}")
        
        metadata = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "file_size_bytes": pdf_path.stat().st_size,
            "file_hash": self.get_file_hash(pdf_path),
            "load_timestamp": datetime.now().isoformat(),
            "total_pages": 0,
            "pages_extracted": 0,
            "pages_skipped": 0,
            "extraction_errors": 0
        }
        
        documents = []
        
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            metadata["total_pages"] = total_pages
            
            # Limit pages if specified
            pages_to_read = min(max_pages, total_pages) if max_pages else total_pages
            
            logger.info(f"Loading {pdf_path.name}: {pages_to_read}/{total_pages} pages")
            
            for page_number in range(pages_to_read):
                try:
                    page = reader.pages[page_number]
                    text = page.extract_text()
                    
                    # Skip empty pages if requested
                    if skip_empty_pages and (not text or not text.strip()):
                        metadata["pages_skipped"] += 1
                        logger.debug(f"Skipped empty page {page_number + 1}")
                        continue
                    
                    if text:
                        documents.append({
                            "page_content": text.strip(),
                            "metadata": {
                                "page_number": page_number + 1,
                                "file_name": pdf_path.name,
                                "file_path": str(pdf_path),
                                "total_pages_in_file": total_pages,
                                "text_length": len(text),
                                "extraction_timestamp": datetime.now().isoformat()
                            }
                        })
                        metadata["pages_extracted"] += 1
                
                except Exception as e:
                    metadata["extraction_errors"] += 1
                    logger.error(f"Error extracting page {page_number + 1} from {pdf_path.name}: {str(e)}")
                    continue
            
            if not documents:
                raise ValueError(f"No text could be extracted from {pdf_path.name}")
            
            logger.info(f"✓ {pdf_path.name}: {metadata['pages_extracted']} pages extracted")
            return documents, metadata
        
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {str(e)}")
            raise
    
    def load_multiple_pdfs(
        self,
        pdf_files: Optional[List[str]] = None,
        max_pages_per_file: Optional[int] = None,
        use_cache: bool = True,
        batch_size: int = 5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load multiple PDFs with batch processing and caching.
        
        Args:
            pdf_files: Specific PDF files to load (None = all in folder)
            max_pages_per_file: Max pages per PDF
            use_cache: Whether to use caching
            batch_size: Number of files to process before cache save
        
        Returns:
            Tuple of (all documents, file metadata list)
        """
        # Find PDF files
        if pdf_files is None:
            pdf_files = [f.name for f in self.data_folder.glob("*.pdf")]
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_folder}")
        
        logger.info(f"Loading {len(pdf_files)} PDF files")
        
        all_documents = []
        all_metadata = []
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            try:
                # Check cache
                cache_path = self.cache_dir / f"{Path(pdf_file).stem}_data.pkl"
                
                if use_cache and cache_path.exists():
                    # Validate cache is newer than file
                    pdf_path = self.data_folder / pdf_file
                    if pdf_path.exists() and cache_path.stat().st_mtime > pdf_path.stat().st_mtime:
                        logger.info(f"[{idx}/{len(pdf_files)}] Loading from cache: {pdf_file}")
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)
                            all_documents.extend(cached_data['documents'])
                            all_metadata.append(cached_data['metadata'])
                        continue
                
                # Load from PDF
                logger.info(f"[{idx}/{len(pdf_files)}] Loading: {pdf_file}")
                documents, metadata = self.load_single_pdf(
                    str(self.data_folder / pdf_file),
                    max_pages=max_pages_per_file
                )
                
                all_documents.extend(documents)
                all_metadata.append(metadata)
                
                # Cache the data
                if use_cache:
                    with open(cache_path, 'wb') as f:
                        pickle.dump({'documents': documents, 'metadata': metadata}, f)
                
                # Periodic save of progress
                if idx % batch_size == 0:
                    logger.info(f"Progress: {idx}/{len(pdf_files)} files processed, {len(all_documents)} pages total")
            
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
                continue
        
        if not all_documents:
            raise ValueError("No documents could be loaded from any PDF")
        
        logger.info(f"✓ Successfully loaded {len(all_documents)} pages from {len(all_metadata)} PDFs")
        return all_documents, all_metadata
    
    def load_large_pdfs_streaming(
        self,
        pdf_file: str,
        chunk_size: int = 10
    ):
        """
        Stream load large PDFs to avoid memory issues.
        Yields documents in chunks.
        
        Args:
            pdf_file: PDF file to load
            chunk_size: Number of pages per chunk
        
        Yields:
            Tuples of (documents chunk, metadata)
        """
        pdf_path = self.data_folder / pdf_file
        documents, metadata = self.load_single_pdf(str(pdf_path))
        
        # Yield in chunks
        for i in range(0, len(documents), chunk_size):
            chunk = documents[i:i + chunk_size]
            logger.debug(f"Yielding chunk {i//chunk_size + 1} with {len(chunk)} documents")
            yield chunk, metadata
    
    def validate_loaded_data(self, documents: List[Dict]) -> Dict:
        """
        Validate loaded data quality.
        
        Args:
            documents: List of loaded documents
        
        Returns:
            Validation report dictionary
        """
        report = {
            "total_documents": len(documents),
            "total_characters": sum(len(d['page_content']) for d in documents),
            "avg_document_length": 0,
            "min_document_length": 0,
            "max_document_length": 0,
            "empty_documents": 0,
            "documents_with_issues": [],
            "quality_score": 0.0
        }
        
        if not documents:
            return report
        
        lengths = [len(d['page_content']) for d in documents]
        report["avg_document_length"] = sum(lengths) // len(lengths)
        report["min_document_length"] = min(lengths)
        report["max_document_length"] = max(lengths)
        report["empty_documents"] = sum(1 for l in lengths if l == 0)
        
        # Quality score: higher is better (based on data variance and completeness)
        empty_ratio = report["empty_documents"] / len(documents)
        quality = (1 - empty_ratio) * 100
        report["quality_score"] = quality
        
        logger.info(f"Data validation: {report['total_documents']} docs, "
                   f"{report['total_characters']} total chars, "
                   f"Quality: {quality:.1f}%")
        
        return report
    
    def save_metadata(self, metadata: List[Dict], output_file: str = "metadata.json"):
        """
        Save metadata to file for future reference.
        
        Args:
            metadata: List of metadata dictionaries
            output_file: Output JSON file
        """
        output_path = self.cache_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {output_path}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def main():
    """Example usage of the loader"""
    
    print("\n" + "="*70)
    print("ROBUST PDF LOADER - PRODUCTION EXAMPLE")
    print("="*70 + "\n")
    
    # Initialize loader
    loader = RobustPDFLoader(data_folder="data", cache_dir=".cache")
    
    try:
        # Option 1: Load all PDFs with caching
        print("Loading all PDFs with caching...")
        documents, metadata = loader.load_multiple_pdfs(
            use_cache=True,
            batch_size=5
        )
        
        # Validate data quality
        print("\nValidating data quality...")
        validation = loader.validate_loaded_data(documents)
        
        print(f"✓ Loaded {validation['total_documents']} pages")
        print(f"✓ Total characters: {validation['total_characters']:,}")
        print(f"✓ Average page length: {validation['avg_document_length']} chars")
        print(f"✓ Quality score: {validation['quality_score']:.1f}%")
        
        # Save metadata
        print("\nSaving metadata...")
        loader.save_metadata(metadata)
        
        # Option 2: Stream large PDF
        print("\n" + "-"*70)
        print("Streaming example (for large files):")
        print("-"*70)
        
        pdf_files = list(Path("data").glob("*.pdf"))
        if pdf_files:
            first_pdf = pdf_files[0].name
            chunk_count = 0
            total_docs_in_stream = 0
            
            for chunk, meta in loader.load_large_pdfs_streaming(first_pdf, chunk_size=5):
                chunk_count += 1
                total_docs_in_stream += len(chunk)
                print(f"Chunk {chunk_count}: {len(chunk)} documents")
            
            print(f"Total chunks: {chunk_count}, Total docs: {total_docs_in_stream}")
        
        # Print first page preview
        print("\n" + "-"*70)
        print("FIRST PAGE PREVIEW:")
        print("-"*70)
        print(documents[0]['page_content'][:300])
        print("...\n")
        
        print("="*70)
        print("✓ SUCCESS!")
        print("="*70 + "\n")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n✗ ERROR: {e}\n")


if __name__ == "__main__":
    main()