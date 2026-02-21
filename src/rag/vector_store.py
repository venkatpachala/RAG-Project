"""
Vector Store Module
===================
Stores and retrieves embeddings using ChromaDB.
Integrates with EmbeddingGenerator output.
"""

import chromadb
from pathlib import Path
import logging
import time
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStore:
    """Stores and retrieves embeddings using ChromaDB"""

    def __init__(self, persist_directory: str = "./chroma_db",
                 collection_name: str = "knowledge_base"):
        """
        Initialize ChromaDB vector store

        Args:
            persist_directory: Path to store ChromaDB data on disk
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        logger.info(f"🗄️  Vector Store config:")
        logger.info(f"   📂 Directory: {self.persist_directory}")
        logger.info(f"   📦 Collection: {self.collection_name}")

    def connect(self):
        """Connect to ChromaDB and get/create collection"""
        if self.client is not None:
            logger.info("✅ Already connected to ChromaDB")
            return

        logger.info(f"⏳ Connecting to ChromaDB...")
        start_time = time.time()

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            elapsed = time.time() - start_time
            logger.info(f"✅ Connected to ChromaDB in {elapsed:.2f}s")
            logger.info(f"📊 Existing documents: {self.collection.count()}")

        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            raise

    def store_embedded_data(self, embedded_data: Dict[str, Dict],
                            batch_size: int = 100) -> Dict:
        """
        Store embedded data from EmbeddingGenerator into ChromaDB

        Args:
            embedded_data: Output from EmbeddingGenerator.embed_all_chunks()
            batch_size: Number of documents to add per batch

        Returns:
            Dictionary with storage statistics
        """
        if self.collection is None:
            self.connect()

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🚀 STORING EMBEDDINGS IN CHROMADB")
        logger.info(f"{'='*60}")

        total_files = len(embedded_data)
        total_chunks = sum(len(d['chunks']) for d in embedded_data.values())
        logger.info(f"📚 Files to store: {total_files}")
        logger.info(f"📄 Chunks to store: {total_chunks}")

        stored_count = 0
        skipped_count = 0
        start_time = time.time()

        for idx, (filename, data) in enumerate(embedded_data.items(), 1):
            logger.info(f"")
            logger.info(f"⏳ [{idx}/{total_files}] Storing: {filename}")

            try:
                chunks = data['chunks']
                embeddings = data['embeddings']
                model_metadata = data.get('metadata', {})

                ids = []
                documents = []
                metadatas = []
                embedding_list = []

                for chunk_idx, chunk in enumerate(chunks):
                    # Create unique ID
                    doc_id = f"{Path(filename).stem}_chunk_{chunk_idx}"

                    # Skip if already exists
                    # (prevents duplicates on re-run)
                    ids.append(doc_id)
                    documents.append(chunk['text'])

                    # Build metadata
                    meta = {
                        "source_file": str(filename),
                        "chunk_index": chunk_idx,
                        "embedding_model": model_metadata.get(
                            'model_name', 'unknown'
                        ),
                        "stored_at": datetime.now().isoformat(),
                    }

                    # Add chunk metadata (flatten safely)
                    for key, value in chunk.items():
                        if key == 'text':
                            continue
                        if isinstance(value, (str, int, float, bool)):
                            meta[key] = value
                        elif value is not None:
                            meta[key] = str(value)

                    metadatas.append(meta)

                    # Convert numpy to list
                    if isinstance(embeddings[chunk_idx], np.ndarray):
                        embedding_list.append(
                            embeddings[chunk_idx].tolist()
                        )
                    else:
                        embedding_list.append(embeddings[chunk_idx])

                # Add in batches
                for i in range(0, len(ids), batch_size):
                    batch_end = min(i + batch_size, len(ids))

                    self.collection.upsert(  # upsert to handle re-runs
                        ids=ids[i:batch_end],
                        embeddings=embedding_list[i:batch_end],
                        documents=documents[i:batch_end],
                        metadatas=metadatas[i:batch_end]
                    )

                    batch_num = (i // batch_size) + 1
                    total_batches = (len(ids) + batch_size - 1) // batch_size
                    logger.info(
                        f"   ✅ Batch {batch_num}/{total_batches} "
                        f"stored ({batch_end - i} chunks)"
                    )

                stored_count += len(chunks)
                logger.info(f"✅ Stored {len(chunks)} chunks from: {filename}")

            except Exception as e:
                logger.error(f"❌ Failed to store {filename}: {e}")
                continue

        total_time = time.time() - start_time

        stats = {
            'total_stored': stored_count,
            'total_in_collection': self.collection.count(),
            'storage_time': total_time,
            'files_processed': total_files,
            'persist_directory': str(self.persist_directory),
            'collection_name': self.collection_name,
        }

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"🎉 VECTOR STORAGE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Chunks stored: {stored_count}/{total_chunks}")
        logger.info(f"📊 Total in collection: {self.collection.count()}")
        logger.info(f"⏱️  Storage time: {total_time:.2f}s")
        logger.info(f"💾 Saved to: {self.persist_directory}")
        logger.info(f"{'='*60}")

        return stats

    def search(self, query_embedding, n_results: int = 5,
               filter_source: Optional[str] = None) -> Dict:
        """
        Search for similar chunks

        Args:
            query_embedding: Query vector (numpy array or list)
            n_results: Number of results to return
            filter_source: Optional source file filter

        Returns:
            Dict with documents, metadatas, distances, ids
        """
        if self.collection is None:
            self.connect()

        logger.info(f"🔍 Searching for top {n_results} results...")
        start_time = time.time()

        # Convert numpy to list
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.collection.count()),
        }

        if filter_source:
            query_params["where"] = {"source_file": filter_source}

        results = self.collection.query(**query_params)

        elapsed = time.time() - start_time
        logger.info(f"✅ Search completed in {elapsed:.4f}s")

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
        }

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        if self.collection is None:
            self.connect()

        count = self.collection.count()
        
        stats = {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }

        # Get unique sources
        if count > 0:
            peek = self.collection.peek(limit=min(count, 10))
            sources = set()
            for meta in peek.get("metadatas", []):
                if meta and "source_file" in meta:
                    sources.add(meta["source_file"])
            stats["sample_sources"] = list(sources)

        return stats

    def clear_collection(self):
        """Clear and recreate the collection"""
        if self.client is None:
            self.connect()

        logger.info(f"🗑️  Clearing collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✅ Collection cleared. Count: {self.collection.count()}")