# embeddings/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Converts list of texts into embedding vectors.
        """
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
