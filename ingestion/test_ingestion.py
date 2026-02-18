# test_ingestion.py

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

if __name__ == "__main__":
    file_path = "sample-1.pdf" 

    print("Loading PDF...")
    documents = load_pdf(file_path)
    print(f"Pages loaded: {len(documents)}")

    print("Chunking text...")
    chunks = chunk_text(documents)

    print(f"Total chunks created: {len(chunks)}")

    print("\nFirst chunk preview:")
    print(chunks[0]["text"][:300])

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

documents = load_pdf("sample.pdf")
chunks = chunk_text(documents)

print("Total chunks:", len(chunks))
print("Example chunk:")
print(chunks[0])
