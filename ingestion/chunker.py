from typing import List, Dict

def chunk_text(
        documents: List[Dict],
        chunk_size: int=500,
        overlap: int=100,
)-> List[Dict]:
    """
    Splits Documents into overlapping chunks
    Args: 
        documents: Output from loader
        chunk_size: Approx number of words per chunk
        overlap: overlap between chunks

    Returns:
        List of chunk dictionaries
    """
    chunks=[]
    chunk_id=0

    for doc in documents:
        words=doc["page_content"].split()

        start=0
        while start < len(words):
            end=start +chunk_size
            chunk_words = words[start:end]
            chunk_text=" ".join(chunk_words)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": doc["metadata"],
                }
            )

            chunk_id +=1
            start +=chunk_size -overlap

    return chunks