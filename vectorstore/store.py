# vectorstore/store.py

import faiss
import numpy as np
from typing import List


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []

    def add(self, vectors: List[List[float]], chunks: List[dict]):
        vectors_np = np.array(vectors).astype("float32")
        self.index.add(vectors_np)
        self.text_chunks.extend(chunks)

    def search(self, query_vector: List[float], top_k: int = 3):
        query_np = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_np, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.text_chunks[idx])

        return results
