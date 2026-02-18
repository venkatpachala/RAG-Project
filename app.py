import streamlit as st
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from vectorstore.store import VectorStore
from llm.generator import LLMGenerator

# -------------------------
# INITIALIZATION
# -------------------------

st.set_page_config(page_title="NotebookLM Clone", layout="wide")

st.title("📘 NotebookLM Clone (RAG Engine)")

if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.embedder = Embedder()
    st.session_state.llm = LLMGenerator()
    st.session_state.chunks = []

# -------------------------
# PDF UPLOAD
# -------------------------

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open(f"temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully.")

    documents = load_pdf("temp.pdf")
    chunks = chunk_text(documents)

    texts = [chunk["text"] for chunk in chunks]
    vectors = st.session_state.embedder.embed_texts(texts)

    dimension = len(vectors[0])
    store = VectorStore(dimension)
    store.add(vectors, chunks)

    st.session_state.store = store
    st.session_state.chunks = chunks

    st.success("Document processed and indexed.")

# -------------------------
# MODE SELECTION
# -------------------------

mode = st.selectbox(
    "Select Mode",
    ["qa", "summary", "study_guide", "faq", "flashcards"]
)

question = None
if mode == "qa":
    question = st.text_input("Ask a question about the document")

# -------------------------
# GENERATION
# -------------------------

if st.button("Generate"):
    if st.session_state.store is None:
        st.warning("Upload and process a document first.")
    else:
        query = question if question else "Summarize the document"

        query_vector = st.session_state.embedder.embed_texts([query])[0]
        retrieved = st.session_state.store.search(query_vector, top_k=5)

        context = "\n\n---\n\n".join([chunk["text"] for chunk in retrieved])

        prompt = f"""
You are a document assistant.

Mode: {mode}

Use ONLY the context below.

Context:
{context}

"""

        if question:
            prompt += f"\nQuestion:\n{question}\n"

        response = st.session_state.llm.generate(prompt)

        st.subheader("📄 Output")
        st.write(response)

        st.subheader("📑 Citations")
        for chunk in retrieved:
            st.markdown(
                f"- Page {chunk['metadata']['page']} "
                f"({chunk['metadata']['source']})"
            )
