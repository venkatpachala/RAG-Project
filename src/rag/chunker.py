"""Text Chunker Module"""

from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunks text into smaller pieces"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"✂️ Chunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text, source_info=None):
        """Split text into chunks"""
        if not text or not text.strip():
            return []
        
        chunks = []
        text = text.strip()
        
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'chunk_index': 0,
                'length': len(text),
                'source': source_info or {}
            }]
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            if end < len(text):
                for punct in ['. ', '! ', '? ', '\n\n', '\n', ' ']:
                    pos = text.rfind(punct, start, end)
                    if pos != -1 and pos > start + self.chunk_size // 2:
                        end = pos + len(punct)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'start_char': start,
                    'end_char': end,
                    'length': len(chunk_text),
                    'source': source_info or {}
                })
                chunk_index += 1
            
            start = end - self.chunk_overlap
            if start <= (chunks[-1]['start_char'] if chunks else 0):
                start = end
        
        return chunks
    
    def chunk_pdf_data(self, pdf_data):
        """Chunk all pages from a PDF"""
        filename = pdf_data['filename']
        pages = pdf_data['pages']
        
        logger.info(f"✂️ Chunking: {filename}")
        
        all_chunks = []
        
        for page_num, page_text in enumerate(pages, 1):
            if page_text and page_text.strip():
                source_info = {
                    'filename': filename,
                    'page_number': page_num
                }
                chunks = self.chunk_text(page_text, source_info)
                all_chunks.extend(chunks)
                logger.info(f"   ✓ Page {page_num}: {len(chunks)} chunk(s)")
        
        logger.info(f"   📊 Total chunks: {len(all_chunks)}")
        return all_chunks
    
    def chunk_multiple_pdfs(self, all_pdf_data):
        """Chunk multiple PDFs"""
        logger.info(f"✂️ Chunking {len(all_pdf_data)} PDF(s)")
        
        all_chunks = {}
        
        for pdf_data in all_pdf_data:
            chunks = self.chunk_pdf_data(pdf_data)
            all_chunks[pdf_data['filename']] = chunks
        
        total = sum(len(c) for c in all_chunks.values())
        logger.info(f"✅ Total chunks created: {total}")
        
        return all_chunks
    
    def save_chunks(self, all_chunks, output_folder):
        """Save all chunks to JSON files"""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        for filename, chunks in all_chunks.items():
            output_file = output_folder / f"{Path(filename).stem}_chunks.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved: {output_file}")