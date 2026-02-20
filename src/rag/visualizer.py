"""
Fast Standalone Visualization Script
====================================
Run: python visualize.py
"""

from pathlib import Path
import logging
import time
import numpy as np
import json

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_FOLDER = r"C:\Users\pritam\Desktop\RAG-Project\data"
EMBEDDINGS_FOLDER = Path(DATA_FOLDER) / "embeddings"
OUTPUT_FOLDER = Path(DATA_FOLDER) / "visualizations"

# Create output folder
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def load_embeddings(embeddings_folder):
    """Load embeddings quickly"""
    logger.info("📂 Loading embeddings...")
    
    embeddings_folder = Path(embeddings_folder)
    all_data = {}
    
    for npy_file in embeddings_folder.glob("*_embeddings.npy"):
        base_name = npy_file.stem.replace('_embeddings', '')
        
        # Load embeddings
        logger.info(f"   Loading {npy_file.name}...")
        embeddings = np.load(npy_file)
        
        # Load metadata
        metadata_file = embeddings_folder / f"{base_name}_metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        all_data[base_name] = {
            'embeddings': embeddings,
            'chunks': metadata['chunks'],
            'metadata': metadata['metadata']
        }
        
        logger.info(f"   ✅ Loaded: {base_name} - Shape: {embeddings.shape}")
    
    return all_data


def create_pca_2d(embeddings, chunks, doc_name, output_folder):
    """Create 2D PCA visualization (FAST!)"""
    logger.info(f"📊 Creating PCA 2D plot...")
    start = time.time()
    
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go
    
    # PCA reduction (very fast)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Colors by page
    colors = [c['source']['page_number'] for c in chunks]
    
    # Hover text (simplified for speed)
    hover_texts = []
    for chunk in chunks:
        text = chunk['text'][:80] + '...'
        hover = f"Chunk {chunk['chunk_index']} | Page {chunk['source']['page_number']}<br>{text}"
        hover_texts.append(hover)
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Page'),
            line=dict(width=0.5, color='white')
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    variance = pca.explained_variance_ratio_
    
    fig.update_layout(
        title=f"{doc_name} - PCA 2D (Variance: {variance.sum():.1%})",
        xaxis_title=f"PC1 ({variance[0]:.1%})",
        yaxis_title=f"PC2 ({variance[1]:.1%})",
        width=1000,
        height=800,
        template='plotly_white'
    )
    
    # Save
    output_path = output_folder / f"{doc_name}_pca_2d.html"
    fig.write_html(output_path)
    
    elapsed = time.time() - start
    logger.info(f"   ✅ Created in {elapsed:.1f}s - Saved: {output_path.name}")
    
    return output_path


def create_stats_plot(embeddings, chunks, doc_name, output_folder):
    """Create statistics dashboard (FAST!)"""
    logger.info(f"📊 Creating statistics plot...")
    start = time.time()
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from collections import Counter
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Chunk Length Distribution',
            'Embedding Magnitude',
            'Chunks per Page',
            'Text Length vs Embedding Norm'
        )
    )
    
    # 1. Chunk lengths
    lengths = [c['length'] for c in chunks]
    fig.add_trace(
        go.Histogram(x=lengths, nbinsx=30, name='Length'),
        row=1, col=1
    )
    
    # 2. Embedding norms
    norms = np.linalg.norm(embeddings, axis=1)
    fig.add_trace(
        go.Histogram(x=norms, nbinsx=30, name='Norm'),
        row=1, col=2
    )
    
    # 3. Chunks per page
    pages = [c['source']['page_number'] for c in chunks]
    page_counts = Counter(pages)
    fig.add_trace(
        go.Bar(
            x=list(page_counts.keys()),
            y=list(page_counts.values()),
            name='Chunks'
        ),
        row=2, col=1
    )
    
    # 4. Scatter: length vs norm
    fig.add_trace(
        go.Scatter(
            x=lengths,
            y=norms,
            mode='markers',
            marker=dict(size=5, color=pages, colorscale='Viridis'),
            name='Correlation'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"{doc_name} - Statistics",
        showlegend=False,
        template='plotly_white'
    )
    
    # Save
    output_path = output_folder / f"{doc_name}_stats.html"
    fig.write_html(output_path)
    
    elapsed = time.time() - start
    logger.info(f"   ✅ Created in {elapsed:.1f}s - Saved: {output_path.name}")
    
    return output_path


def create_similarity_heatmap(embeddings, chunks, doc_name, output_folder, max_chunks=50):
    """Create similarity heatmap (LIMITED for speed)"""
    logger.info(f"📊 Creating similarity heatmap...")
    start = time.time()
    
    from sklearn.metrics.pairwise import cosine_similarity
    import plotly.graph_objects as go
    
    # Limit chunks for speed
    if len(embeddings) > max_chunks:
        logger.info(f"   ⚠️  Limiting to {max_chunks} chunks for speed")
        indices = np.linspace(0, len(embeddings)-1, max_chunks, dtype=int)
        embeddings_subset = embeddings[indices]
        chunks_subset = [chunks[i] for i in indices]
    else:
        embeddings_subset = embeddings
        chunks_subset = chunks
    
    # Calculate similarity
    similarity = cosine_similarity(embeddings_subset)
    
    # Labels
    labels = [f"P{c['source']['page_number']}-C{c['chunk_index']}" for c in chunks_subset]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0.5,
        colorbar=dict(title='Similarity')
    ))
    
    fig.update_layout(
        title=f'{doc_name} - Chunk Similarity',
        width=900,
        height=900,
        template='plotly_white'
    )
    
    # Save
    output_path = output_folder / f"{doc_name}_similarity.html"
    fig.write_html(output_path)
    
    elapsed = time.time() - start
    logger.info(f"   ✅ Created in {elapsed:.1f}s - Saved: {output_path.name}")
    
    return output_path


def create_3d_pca(embeddings, chunks, doc_name, output_folder):
    """Create 3D PCA plot (FAST!)"""
    logger.info(f"📊 Creating 3D PCA plot...")
    start = time.time()
    
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go
    
    # PCA reduction
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(embeddings)
    
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    colors = [c['source']['page_number'] for c in chunks]
    
    # Hover text
    hover_texts = [f"Chunk {c['chunk_index']} | Page {c['source']['page_number']}" 
                   for c in chunks]
    
    # Create 3D plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Page')
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>'
    )])
    
    variance = pca.explained_variance_ratio_
    
    fig.update_layout(
        title=f"{doc_name} - PCA 3D (Variance: {variance.sum():.1%})",
        scene=dict(
            xaxis_title=f"PC1 ({variance[0]:.1%})",
            yaxis_title=f"PC2 ({variance[1]:.1%})",
            zaxis_title=f"PC3 ({variance[2]:.1%})"
        ),
        width=1000,
        height=800,
        template='plotly_white'
    )
    
    # Save
    output_path = output_folder / f"{doc_name}_pca_3d.html"
    fig.write_html(output_path)
    
    elapsed = time.time() - start
    logger.info(f"   ✅ Created in {elapsed:.1f}s - Saved: {output_path.name}")
    
    return output_path


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("📊 FAST EMBEDDING VISUALIZER")
    print("="*70)
    print(f"📁 Embeddings: {EMBEDDINGS_FOLDER}")
    print(f"📁 Output: {OUTPUT_FOLDER}")
    print("="*70 + "\n")
    
    # Check embeddings exist
    if not EMBEDDINGS_FOLDER.exists():
        print("❌ Embeddings folder not found!")
        print(f"   Run pipeline first to generate embeddings.\n")
        return
    
    embedding_files = list(EMBEDDINGS_FOLDER.glob("*_embeddings.npy"))
    
    if not embedding_files:
        print("❌ No embeddings found!")
        return
    
    print(f"📄 Found {len(embedding_files)} file(s)")
    print()
    
    total_start = time.time()
    
    # Load all embeddings
    all_data = load_embeddings(EMBEDDINGS_FOLDER)
    
    if not all_data:
        print("❌ Failed to load embeddings\n")
        return
    
    print()
    
    # Process each document
    all_files = []
    
    for doc_name, data in all_data.items():
        logger.info(f"📄 Processing: {doc_name}")
        logger.info(f"   Chunks: {len(data['chunks'])}, Embedding dim: {data['embeddings'].shape[1]}")
        print()
        
        embeddings = data['embeddings']
        chunks = data['chunks']
        
        # Create visualizations (only fast ones!)
        files = []
        
        # 1. PCA 2D (very fast)
        files.append(create_pca_2d(embeddings, chunks, doc_name, OUTPUT_FOLDER))
        
        # 2. PCA 3D (fast)
        files.append(create_3d_pca(embeddings, chunks, doc_name, OUTPUT_FOLDER))
        
        # 3. Statistics (fast)
        files.append(create_stats_plot(embeddings, chunks, doc_name, OUTPUT_FOLDER))
        
        # 4. Similarity heatmap (limited)
        files.append(create_similarity_heatmap(embeddings, chunks, doc_name, OUTPUT_FOLDER))
        
        all_files.extend(files)
        print()
    
    total_time = time.time() - total_start
    
    # Summary
    print("="*70)
    print("✅ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📁 Output: {OUTPUT_FOLDER}")
    print(f"\nGenerated {len(all_files)} files:")
    
    for f in sorted(all_files):
        size_kb = f.stat().st_size / 1024
        print(f"   • {f.name} ({size_kb:.0f} KB)")
    
    print("\n💡 Open HTML files in your browser!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user\n")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}\n")