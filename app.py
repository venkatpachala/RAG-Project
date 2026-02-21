"""
RAG Chatbot - Streamlit Frontend
=================================
A clean, multi-tab interface for the RAG pipeline.

Tabs:
  💬 Chat      — Ask questions, get answers with citations
  📁 Documents — Upload PDFs, manage knowledge base
  📊 Visualize — Explore embeddings with PCA plots
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

# Must be first Streamlit call
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for a polished look ---
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { 
        color: white; 
        margin: 0; 
        font-size: 1.8rem; 
    }
    .main-header p { 
        color: rgba(255,255,255,0.85); 
        margin: 0.3rem 0 0 0; 
        font-size: 0.95rem; 
    }
    
    /* Citation cards */
    .citation-card {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-card h3 { 
        margin: 0; 
        font-size: 1.8rem; 
        color: #667eea; 
    }
    .stat-card p { 
        margin: 0.2rem 0 0 0; 
        color: #666; 
        font-size: 0.85rem; 
    }
    
    /* Document list items */
    .doc-item {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# SESSION STATE INITIALIZATION
# ======================================================================
def init_session():
    if "engine" not in st.session_state:
        try:
            from query_engine import RAGQueryEngine
            st.session_state.engine = RAGQueryEngine()
        except Exception as e:
            st.error(f"❌ Failed to initialize engine: {str(e)}")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_source" not in st.session_state:
        st.session_state.selected_source = "All Documents"


init_session()
engine = st.session_state.engine


# ======================================================================
# HEADER
# ======================================================================
st.markdown("""
<div class="main-header">
    <h1>📚 RAG Chatbot</h1>
    <p>Upload PDFs and chat with your documents — powered by Gemini AI</p>
</div>
""", unsafe_allow_html=True)


# ======================================================================
# TABS
# ======================================================================
tab_chat, tab_docs, tab_viz = st.tabs(["💬 Chat", "📁 Documents", "📊 Visualize"])


# ======================================================================
# TAB 1: CHAT
# ======================================================================
with tab_chat:
    # --- Sidebar: retrieval settings ---
    with st.sidebar:
        st.header("⚙️ Settings")

        retrieval_k = st.slider(
            "Chunks to retrieve:",
            min_value=1, max_value=10, value=5,
            help="Number of document chunks used as context"
        )

        # Selective document filter
        st.subheader("📂 Filter by Document")
        available_sources = engine.get_source_files()
        source_options = ["All Documents"] + available_sources
        st.session_state.selected_source = st.selectbox(
            "Query against:",
            options=source_options,
            help="Choose a specific document or search all"
        )

        st.divider()

        # Quick stats
        try:
            stats = engine.get_stats()
            doc_count = stats.get("total_documents", 0)
        except Exception:
            doc_count = 0

        st.metric("📄 Chunks in DB", doc_count)
        st.metric("💬 Messages", len(st.session_state.messages))

    # --- Chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents & generating answer..."):
                # Determine filter
                filter_src = None
                if st.session_state.selected_source != "All Documents":
                    filter_src = st.session_state.selected_source

                try:
                    result = engine.query(
                        question=prompt,
                        top_k=retrieval_k,
                        filter_source=filter_src,
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    # Display answer
                    st.markdown(answer)

                    # Display citation cards
                    if sources:
                        st.markdown("---")
                        st.caption("📎 **Retrieved Sources:**")
                        for src in sources:
                            score_pct = f"{src['score']*100:.1f}%" if src['score'] else "N/A"
                            st.markdown(
                                f'<div class="citation-card">'
                                f'📄 <strong>{src["file"]}</strong> · '
                                f'Page {src["page"]} · '
                                f'Relevance: {score_pct}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# ======================================================================
# TAB 2: DOCUMENTS
# ======================================================================
with tab_docs:
    col_upload, col_manage = st.columns([2, 1])

    with col_upload:
        st.subheader("📤 Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Drop your PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Only PDF files are supported"
        )

        if uploaded_files:
            if st.button("📥 Ingest All Documents", use_container_width=True, type="primary"):
                progress = st.progress(0, text="Starting ingestion...")

                for i, uploaded_file in enumerate(uploaded_files):
                    progress.progress(
                        (i) / len(uploaded_files),
                        text=f"Processing {uploaded_file.name}..."
                    )

                    # Save to secure temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f"_{uploaded_file.name}"
                    ) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    try:
                        result = engine.add_document(tmp_path)
                        st.success(
                            f"✅ **{uploaded_file.name}** — "
                            f"{result['num_chunks']} chunks created"
                        )
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                progress.progress(1.0, text="✅ All documents processed!")

    with col_manage:
        st.subheader("📊 Knowledge Base")

        try:
            stats = engine.get_stats()
            total = stats.get("total_documents", 0)
        except Exception:
            total = 0

        st.markdown(
            f'<div class="stat-card"><h3>{total}</h3>'
            f'<p>Total Chunks</p></div>',
            unsafe_allow_html=True,
        )

        st.markdown("")  # spacer

        sources = engine.get_source_files()
        if sources:
            st.caption("📁 **Ingested Files:**")
            for src in sources:
                st.markdown(f"- 📄 `{src}`")
        else:
            st.info("No documents ingested yet. Upload some PDFs to get started!")

        st.markdown("")

        if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
            engine.clear_all()
            st.session_state.messages = []
            st.success("All data cleared!")
            st.rerun()


# ======================================================================
# TAB 3: VISUALIZE
# ======================================================================
with tab_viz:
    st.subheader("📊 Embedding Visualizations")

    try:
        stats = engine.get_stats()
        total = stats.get("total_documents", 0)
    except Exception:
        total = 0

    if total == 0:
        st.info(
            "📭 **No data to visualize yet.**\n\n"
            "Upload some PDFs in the Documents tab first, then come back here "
            "to explore your embeddings!"
        )
    else:
        st.markdown(f"Visualizing **{total}** chunks in the vector store.")

        viz_type = st.selectbox(
            "Choose visualization:",
            ["2D PCA Plot", "3D PCA Plot", "Similarity Heatmap", "Statistics Dashboard"]
        )

        if st.button("🎨 Generate Visualization", type="primary"):
            with st.spinner("Generating visualization..."):
                try:
                    from src.rag.visualizer import load_embeddings
                    from src.core.config import settings
                    from sklearn.decomposition import PCA
                    import plotly.graph_objects as go
                    import numpy as np

                    embeddings_data = load_embeddings(str(settings.embeddings_dir))

                    if not embeddings_data:
                        st.warning(
                            "No saved embeddings found. "
                            "Try uploading a document first."
                        )
                    else:
                        for doc_name, data in embeddings_data.items():
                            embeds = data['embeddings']
                            chunks = data['chunks']

                            st.markdown(f"#### 📄 {doc_name}")

                            if viz_type == "2D PCA Plot":
                                pca = PCA(n_components=2, random_state=42)
                                coords = pca.fit_transform(embeds)
                                colors = [c.get('source', {}).get('page_number', 0) for c in chunks]
                                hover = [f"Chunk {c.get('chunk_index',i)} | Page {c.get('source',{}).get('page_number','?')}<br>{c['text'][:80]}..." for i, c in enumerate(chunks)]

                                fig = go.Figure(go.Scatter(
                                    x=coords[:, 0], y=coords[:, 1], mode='markers',
                                    marker=dict(size=6, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(title='Page')),
                                    text=hover, hovertemplate='%{text}<extra></extra>'
                                ))
                                variance = pca.explained_variance_ratio_
                                fig.update_layout(
                                    title=f"{doc_name} — PCA 2D ({variance.sum():.1%} variance)",
                                    xaxis_title=f"PC1 ({variance[0]:.1%})", yaxis_title=f"PC2 ({variance[1]:.1%})",
                                    height=600, template='plotly_white'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            elif viz_type == "3D PCA Plot":
                                pca = PCA(n_components=3, random_state=42)
                                coords = pca.fit_transform(embeds)
                                colors = [c.get('source', {}).get('page_number', 0) for c in chunks]

                                fig = go.Figure(go.Scatter3d(
                                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode='markers',
                                    marker=dict(size=3, color=colors, colorscale='Viridis', showscale=True)
                                ))
                                fig.update_layout(title=f"{doc_name} — PCA 3D", height=700)
                                st.plotly_chart(fig, use_container_width=True)

                            elif viz_type == "Similarity Heatmap":
                                max_n = min(50, len(embeds))
                                sample = embeds[:max_n]
                                similarity = np.dot(sample, sample.T)

                                fig = go.Figure(go.Heatmap(
                                    z=similarity, colorscale='RdBu_r',
                                    zmin=-1, zmax=1
                                ))
                                fig.update_layout(title=f"{doc_name} — Similarity (first {max_n} chunks)", height=600)
                                st.plotly_chart(fig, use_container_width=True)

                            elif viz_type == "Statistics Dashboard":
                                lengths = [c.get('length', len(c.get('text', ''))) for c in chunks]
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(x=lengths, nbinsx=30, name="Chunk Lengths"))
                                fig.update_layout(title=f"{doc_name} — Chunk Length Distribution", xaxis_title="Length (chars)", yaxis_title="Count", height=500, template='plotly_white')
                                st.plotly_chart(fig, use_container_width=True)

                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("Total Chunks", len(chunks))
                                col_b.metric("Avg Length", f"{np.mean(lengths):.0f}")
                                col_c.metric("Embedding Dim", embeds.shape[1])

                except ImportError as e:
                    st.warning(
                        f"Visualization dependencies missing: {e}\n\n"
                        "Install with: `pip install plotly scikit-learn`"
                    )
                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")