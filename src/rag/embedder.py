"""
Embedding Generator Module
==========================
Converts text chunks into vector embeddings using sentence transformers.
"""

from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import json
import numpy as np
import time
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings from text chunks using sentence transformers"""
    
    # Available models (sorted by speed/quality)
    MODELS = {
        'fast': 'all-MiniLM-L6-v2',           # Fast, 384 dims
        'balanced': 'all-mpnet-base-v2',      # Balanced, 768 dims
        'best': 'all-distilroberta-v1',       # Best quality, 768 dims
    }
    
    def __init__(self, model_name='best', batch_size=32):
        """
        Initialize embedding generator
        
        Args:
            model_name: 'fast', 'balanced', or 'best' (or custom model name)
            batch_size: Number of texts to embed at once
        """
        # Get model name
        if model_name in self.MODELS:
            self.model_name = self.MODELS[model_name]
            logger.info(f"🤖 Using preset: {model_name} → {self.model_name}")
        else:
            self.model_name = model_name
            logger.info(f"🤖 Using custom model: {model_name}")
        
        self.batch_size = batch_size
        self.model = None
        self.embedding_dim = None
        
        logger.info(f"⚙️  Batch size: {batch_size}")
    
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is not None:
            logger.info("✅ Model already loaded")
            return
        
        logger.info(f"⏳ Loading model: {self.model_name}...")
        logger.info("   (This may take a few seconds on first run)")
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f}s")
            logger.info(f"📊 Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def embed_texts(self, texts: List[str], show_progress=True) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
        
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"🔄 Generating embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better similarity
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Generated {len(texts)} embeddings in {elapsed:.2f}s")
        logger.info(f"⚡ Speed: {len(texts)/elapsed:.1f} embeddings/sec")
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Generate embeddings for chunks from chunker
        
        Args:
            chunks: List of chunk dictionaries from chunker
        
        Returns:
            Dictionary with chunks and their embeddings
        """
        logger.info(f"📦 Processing {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Combine chunks with embeddings
        result = {
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': {
                'num_chunks': len(chunks),
                'embedding_dim': self.embedding_dim,
                'model_name': self.model_name,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        logger.info(f"✅ Embedded {len(chunks)} chunks")
        logger.info(f"📊 Embedding matrix shape: {embeddings.shape}")
        
        return result
    
    def embed_all_chunks(self, all_chunks: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Generate embeddings for all chunks from multiple PDFs
        
        Args:
            all_chunks: Dictionary mapping filenames to chunk lists
        
        Returns:
            Dictionary mapping filenames to embedded chunk data
        """
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🚀 EMBEDDING GENERATION")
        logger.info(f"{'='*60}")
        
        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        logger.info(f"📚 Total files: {len(all_chunks)}")
        logger.info(f"📄 Total chunks: {total_chunks}")
        
        embedded_data = {}
        processed_chunks = 0
        
        start_time = time.time()
        
        for idx, (filename, chunks) in enumerate(all_chunks.items(), 1):
            logger.info(f"")
            logger.info(f"⏳ [{idx}/{len(all_chunks)}] Processing: {filename}")
            
            try:
                result = self.embed_chunks(chunks)
                embedded_data[filename] = result
                processed_chunks += len(chunks)
                
                logger.info(f"✅ Completed: {filename}")
                
            except Exception as e:
                logger.error(f"❌ Failed to embed {filename}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🎉 EMBEDDING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Files processed: {len(embedded_data)}/{len(all_chunks)}")
        logger.info(f"✅ Chunks embedded: {processed_chunks}/{total_chunks}")
        logger.info(f"⏱️  Total time: {total_time:.2f}s")
        logger.info(f"⚡ Average speed: {processed_chunks/total_time:.1f} chunks/sec")
        logger.info(f"{'='*60}")
        
        return embedded_data
    
    def save_embeddings(self, embedded_data: Dict, output_folder: Path):
        """
        Save embeddings to disk
        
        Args:
            embedded_data: Dictionary with embedded chunks
            output_folder: Folder to save embeddings
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving embeddings to: {output_folder}")
        
        for filename, data in embedded_data.items():
            base_name = Path(filename).stem
            
            # Save embeddings as numpy array
            embeddings_file = output_folder / f"{base_name}_embeddings.npy"
            np.save(embeddings_file, data['embeddings'])
            logger.info(f"   💾 Saved embeddings: {embeddings_file.name}")
            
            # Save metadata and chunks
            metadata_file = output_folder / f"{base_name}_metadata.json"
            metadata = {
                'chunks': data['chunks'],
                'metadata': data['metadata'],
                'embedding_file': str(embeddings_file.name)
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"   📄 Saved metadata: {metadata_file.name}")
        
        logger.info(f"✅ All embeddings saved")
