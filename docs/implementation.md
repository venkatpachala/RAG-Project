# Implementation Guide — Data Flow & Architecture

This document explains how data flows through the RAG pipeline, what each module does, and how they connect together.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Ingestion Pipeline](#ingestion-pipeline)
3. [Query Pipeline](#query-pipeline)
4. [Module Reference](#module-reference)
5. [Data Storage](#data-storage)
6. [Configuration System](#configuration-system)
7. [Logging System](#logging-system)
8. [Frontend Architecture](#frontend-architecture)

---

## Architecture Overview

The system uses a **modular pipeline architecture** where each stage of the RAG workflow is handled by a dedicated module in `src/rag/`. The `query_engine.py` acts as a thin coordinator that wires all modules together, and `app.py` provides the Streamlit UI.

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                    │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │ 💬 Chat  │     │ 📁 Documents │     │ 📊 Visualize │       │
│   └────┬─────┘     └──────┬───────┘     └──────┬───────┘       │
└────────┼──────────────────┼────────────────────┼───────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  QUERY ENGINE (query_engine.py)                  │
│                     Thin Coordinator Layer                       │
└───────┬───────────────────┬────────────────────┬───────────────┘
        │                   │                    │
   QUERY PIPELINE      INGEST PIPELINE     VISUALIZATIONS
        │                   │                    │
        ▼                   ▼                    ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
│  Retriever   │  │ Loader→Chunker→  │  │  Visualizer  │
│  + Generator │  │ Embedder→Store   │  │  (Plotly)    │
└──────────────┘  └──────────────────┘  └──────────────┘
```

---

## Ingestion Pipeline

**Purpose:** Transform a raw PDF file into searchable vector embeddings stored in ChromaDB.

### Step-by-Step Data Flow

```
PDF File
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: LOADING (src/rag/loader.py → SimplePDFLoader)        │
│                                                              │
│ Input:  file_path (string)                                   │
│ Action: Opens PDF, extracts text page-by-page using pypdf    │
│ Output: {                                                    │
│           'filename': 'document.pdf',                        │
│           'pages': [                                         │
│             {'page_number': 1, 'text': '...', 'length': 847},│
│             {'page_number': 2, 'text': '...', 'length': 1203}│
│           ],                                                 │
│           'metadata': {                                      │
│             'total_pages': 35,                               │
│             'total_characters': 28450,                       │
│             'filename': 'document.pdf'                       │
│           }                                                  │
│         }                                                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: CHUNKING (src/rag/chunker.py → TextChunker)          │
│                                                              │
│ Input:  PDF data dict from Step 1                            │
│ Action: Splits text into overlapping chunks                  │
│         - chunk_size: 800 chars (configurable)               │
│         - chunk_overlap: 150 chars (configurable)            │
│         - Preserves page and source metadata per chunk       │
│ Output: List of chunk dicts:                                 │
│         [                                                    │
│           {                                                  │
│             'text': 'chunk content here...',                 │
│             'chunk_index': 0,                                │
│             'length': 798,                                   │
│             'source': {                                      │
│               'filename': 'document.pdf',                    │
│               'page_number': 1                               │
│             }                                                │
│           },                                                 │
│           ...                                                │
│         ]                                                    │
│ Side Effect: Saved to data/chunks/<filename>_chunks.json     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: EMBEDDING (src/rag/embedder.py → EmbeddingGenerator) │
│                                                              │
│ Input:  List of chunk dicts from Step 2                      │
│ Action: Converts each chunk's text into a 384-dim vector     │
│         using sentence-transformers (all-MiniLM-L6-v2)       │
│ Output: {                                                    │
│           'embeddings': numpy.ndarray (shape: [N, 384]),     │
│           'chunks': [...original chunk dicts...],            │
│           'metadata': {                                      │
│             'model_name': 'all-MiniLM-L6-v2',               │
│             'dimension': 384,                                │
│             'total_chunks': N                                │
│           }                                                  │
│         }                                                    │
│ Side Effect: Saved to data/embeddings/<filename>_embeddings  │
│              (.npy for vectors, _metadata.json for chunks)   │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: STORAGE (src/rag/vector_store.py → VectorStore)      │
│                                                              │
│ Input:  Embedded data dict from Step 3                       │
│ Action: Stores vectors + metadata in ChromaDB collection     │
│         - Collection name: 'knowledge_base'                  │
│         - Persists to: ./vector_db/                          │
│         - Batches inserts for performance                    │
│ Output: ChromaDB collection with searchable vectors          │
│                                                              │
│ Stored per document:                                         │
│   - id: unique chunk identifier                              │
│   - embedding: 384-dim float vector                          │
│   - document: original chunk text                            │
│   - metadata: {source_file, page_number, chunk_index}        │
└──────────────────────────────────────────────────────────────┘
```

---

## Query Pipeline

**Purpose:** Take a user's natural language question, find the most relevant chunks, and generate a structured answer with citations.

### Step-by-Step Data Flow

```
User Question: "What are the key investment strategies?"
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: RETRIEVAL (src/rag/retriever.py → Retriever)         │
│                                                              │
│ Substep A — Embed Query:                                     │
│   Input:  "What are the key investment strategies?"          │
│   Action: EmbeddingGenerator converts query to 384-dim vector│
│   Output: [0.023, -0.145, 0.089, ...]  (384 floats)         │
│                                                              │
│ Substep B — Search ChromaDB:                                 │
│   Input:  Query vector + top_k=5                             │
│   Action: Cosine similarity search in ChromaDB               │
│   Output: Top 5 most similar chunks with metadata:           │
│     [                                                        │
│       {                                                      │
│         'text': 'Portfolio diversification involves...',     │
│         'source_file': 'investment-guide.pdf',               │
│         'page_number': 12,                                   │
│         'score': 0.87,                                       │
│         'chunk_index': 45                                    │
│       },                                                     │
│       ...4 more results                                      │
│     ]                                                        │
│                                                              │
│ Optional: filter_source parameter limits search to one file  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: GENERATION (src/services/generator.py → Gemini)      │
│                                                              │
│ Input:  User question + 5 context chunks with metadata       │
│                                                              │
│ Action: Constructs a prompt that includes:                   │
│   1. All 5 chunks labeled as [Document 1-5 | Source, Page]   │
│   2. The user's question                                     │
│   3. Formatting instructions for topic/subtopic structure    │
│   4. Rules requiring inline citations [1], [2], etc.         │
│                                                              │
│ Sends to Gemini API → receives structured response           │
│                                                              │
│ Output (example):                                            │
│   ## Key Investment Strategies                               │
│                                                              │
│   An overview paragraph... [1]                               │
│                                                              │
│   ### Diversification                                        │
│   - Spread investments across asset classes [1]              │
│   - Reduce risk through allocation [2]                       │
│                                                              │
│   ### Value Investing                                        │
│   - Focus on undervalued securities [3]                      │
│                                                              │
│   ---                                                        │
│   **References:**                                            │
│   1. *investment-guide.pdf* — Page 12                        │
│   2. *investment-guide.pdf* — Page 45                        │
│   3. *investment-guide.pdf* — Page 8                         │
└──────────────────────────────────────────────────────────────┘
```

---

## Module Reference

| Module | Class | Key Methods | Purpose |
|--------|-------|-------------|---------|
| `src/rag/loader.py` | `SimplePDFLoader` | `load_pdf(filename)` | Extract text from PDFs page by page |
| `src/rag/chunker.py` | `TextChunker` | `chunk_pdf_data(data)`, `save_chunks(...)` | Split text into overlapping chunks |
| `src/rag/embedder.py` | `EmbeddingGenerator` | `embed_chunks(chunks)`, `embed_texts(texts)`, `save_embeddings(...)` | Generate vector embeddings |
| `src/rag/vector_store.py` | `VectorStore` | `connect()`, `store_embedded_data(data)`, `search(vector, n)`, `get_stats()` | ChromaDB CRUD operations |
| `src/rag/retriever.py` | `Retriever` | `retrieve(query, top_k, filter)` | End-to-end: embed query → search → return results with metadata |
| `src/services/generator.py` | `GeminiGenerator` | `generate(query, context_chunks)` | Send context + question to Gemini, get structured answer |
| `src/core/config.py` | `Settings` | Properties for all paths | Centralized pydantic-settings config from `.env` |
| `src/core/logger.py` | — | `get_logger(name)` | Rotating file handler to `data/logs/` |

---

## Data Storage

All generated data lives under the `data/` directory (gitignored):

```
data/
├── chunks/
│   └── document_chunks.json       # Text chunks with metadata
├── embeddings/
│   ├── document_embeddings.npy    # Numpy arrays of vectors
│   └── document_metadata.json     # Chunk text + source info
├── logs/
│   └── rag_app.log                # Rotating application log (5MB max)
└── visualizations/
    └── *.html                     # Plotly interactive charts
```

The vector database is stored separately:

```
vector_db/
└── chroma.sqlite3                 # ChromaDB persistent storage
```

---

## Configuration System

All settings flow from a single source of truth: `src/core/config.py`

```python
from src.core.config import settings

# Access any setting
settings.GOOGLE_API_KEY     # from .env
settings.CHUNK_SIZE          # 800 (default, or from .env)
settings.chunks_dir          # Path('./data/chunks')
settings.embeddings_dir      # Path('./data/embeddings')
```

Settings are loaded in this priority order:
1. Environment variables (highest priority)
2. `.env` file values
3. Default values in `config.py`

---

## Logging System

The logging system (`src/core/logger.py`) provides:

- **Console output** — colored, formatted logs during development
- **File logging** — rotating log files in `data/logs/rag_app.log`
  - Max file size: 5MB
  - Keeps 3 backup files
- **Per-module loggers** — each module gets its own named logger

```python
from src.core.logger import get_logger
logger = get_logger(__name__)

logger.info("Processing document...")
logger.error("Failed to generate embeddings")
```

---

## Frontend Architecture

The Streamlit app (`app.py`) uses a **tabbed layout** with three sections:

### 💬 Chat Tab
- Displays conversation history
- Sidebar controls: retrieval count slider, document filter dropdown
- Shows live metrics (chunks in DB, message count)
- Renders Gemini's structured answers with markdown
- Displays citation cards below each answer

### 📁 Documents Tab
- Multi-file PDF uploader with drag-and-drop
- Progress bar during ingestion
- Knowledge base stats card
- List of ingested source files
- Clear all data button

### 📊 Visualize Tab
- Four visualization types:
  - **2D PCA Plot** — scatter plot of chunk embeddings
  - **3D PCA Plot** — interactive 3D rotation
  - **Similarity Heatmap** — chunk-to-chunk similarity matrix
  - **Statistics Dashboard** — chunk length distribution + metrics
- All charts are interactive Plotly (zoom, pan, hover)

---

## Adding New Data Formats

The system is designed for easy extension. To add a new file format (e.g., `.docx`):

1. Add a new loader method in `src/rag/loader.py`
2. Update `query_engine.py`'s `add_document()` to handle the new extension
3. Update the `file_uploader` in `app.py` to accept the new type

The chunking, embedding, and storage stages remain unchanged since they work on plain text.
