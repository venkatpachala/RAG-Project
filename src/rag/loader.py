"""PDF Loader Module"""

from pypdf import PdfReader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SimplePDFLoader:
    """Simple PDF loader"""
    
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        logger.info(f"📁 Loader initialized: {self.data_folder}")
    
    def load_pdf(self, pdf_file):
        """Load a single PDF"""
        pdf_path = self.data_folder / pdf_file
        logger.info(f"📄 Loading: {pdf_file}")
        
        reader = PdfReader(pdf_path)
        pages_text = []
        
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()
            if text and text.strip():
                pages_text.append(text.strip())
                logger.info(f"   ✓ Page {page_num + 1}: {len(text)} chars")
            else:
                pages_text.append("")
        
        return {
            'filename': pdf_file,
            'pages': pages_text,
            'metadata': {
                'total_pages': len(pages_text),
                'pages_with_text': sum(1 for p in pages_text if p),
                'total_characters': sum(len(p) for p in pages_text)
            }
        }
    
    def load_all_pdfs(self):
        """Load all PDFs"""
        pdf_files = sorted(list(self.data_folder.glob("*.pdf")))
        
        if not pdf_files:
            logger.warning("No PDF files found")
            return []
        
        logger.info(f"📚 Found {len(pdf_files)} PDF(s)")
        
        all_data = []
        for pdf_path in pdf_files:
            try:
                data = self.load_pdf(pdf_path.name)
                all_data.append(data)
            except Exception as e:
                logger.error(f"Failed to load {pdf_path.name}: {e}")
        
        return all_data