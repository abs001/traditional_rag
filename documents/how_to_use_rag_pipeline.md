# Traditional RAG Pipeline: Usage Guide

This document provides a comprehensive guide on how to configure, run, and extend this **Traditional Retrieval-Augmented Generation (RAG)** pipeline.

---

## 1. Architecture Overview

A Traditional RAG pipeline augments Large Language Models (LLMs) with private or domain-specific knowledge stored in a vector database.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion"]
        A[PDF / Text Documents] --> B[LangChain Document Loaders]
        B --> C[Document Chunks + Metadata]
    end

    subgraph Embedding ["2. Embedding & Indexing"]
        C --> D[SentenceTransformer: all-MiniLM-L6-v2]
        D --> E[(ChromaDB: data/vecor_store)]
    end

    subgraph Retrieval ["3. Retrieval Phase"]
        Q[User Natural Language Query] --> QE[Embed Query Vector]
        QE --> RS[ChromaDB Vector Search]
        E -. Query .-> RS
        RS --> F1[Similarity Search + L2 Scored]
        RS --> F2[Maximal Marginal Relevance - MMR]
        F1 --> G[Top-K Filtered Document Chunks]
        F2 --> G
    end

    subgraph Augmentation ["4. Generation Prep"]
        G --> H[Context Formatter]
        H --> P[Augmented Prompt with Citations]
        P --> LLM[LLM Generation]
    end
```

### Key Components
| Component | Module | Purpose |
|---|---|---|
| **Document Loaders** | `langchain_community.document_loaders` (`PyMuPDFLoader`, `DirectoryLoader`, `TextLoader`) | Reads raw PDFs and text files with metadata (e.g. source, page). |
| **Embedding Manager** | `retriever.EmbeddingManager` | Encodes text into 384-dimensional dense vectors using `all-MiniLM-L6-v2`. |
| **Vector Store** | `retriever.VectorStore` | Persists embeddings, chunk texts, and metadata in ChromaDB (`data/vecor_store/`). |
| **Retriever** | `retriever.RAGRetriever` | Handles similarity search, similarity score normalization, MMR diversity retrieval, and prompt context formatting. |
| **CLI Demo** | `main.py` | Command-line interface to execute queries against the vector store. |
| **Interactive Notebook**| `notebook/document.ipynb` | Step-by-step notebook walking through every phase of the pipeline. |

---

## 2. Directory Structure

```text
traditional_rag/
│
├── data/
│   ├── how_car_engine_works.txt            # Raw text sample
│   ├── text_files/                         # Text documents directory
│   ├── pdf_files/                          # PDF source documents
│   └── vecor_store/                        # ChromaDB persistent database
│
├── notebook/
│   └── document.ipynb                      # Step-by-step interactive workflow
│
├── documents/
│   └── how_to_use_rag_pipeline.md          # This user guide
│
├── retriever.py                            # Core retrieval and vector store module
├── main.py                                 # Command-line interface for querying
├── requirements.txt                        # Project dependencies
└── pyproject.toml                          # Project configuration
```

---

## 3. Quick Start & Prerequisites

### 3.1 Python Environment
Ensure virtual environment is activated and dependencies are installed:
```powershell
# From project root: h:\AI\traditional_rag
.\.venv\Scripts\Activate.ps1

# Or install dependencies if setting up from scratch:
pip install -r requirements.txt
```

---

## 4. How to Use the Pipeline

### Method A: Command-Line Interface (`main.py`)

You can run queries directly from the command line using `uv run` or your `.venv` python:

#### 1. Basic Query
```powershell
uv run python main.py --query "How does a four-stroke engine cycle work?"
# or:
.\.venv\Scripts\python.exe main.py --query "How does a four-stroke engine cycle work?"
```

#### 2. Specifying Top-K Results
Control the number of retrieved chunks (default is 3):
```powershell
uv run python main.py --query "What is the function of the crankshaft?" --top-k 2
```

#### 3. Maximal Marginal Relevance (MMR) for Diversity
MMR balances query relevance with diversity across candidate chunks to prevent repetitive context:
```powershell
.\.venv\Scripts\python.exe main.py --query "What are the key engine parts?" --top-k 3 --mmr
```

#### 4. Applying a Minimum Similarity Score Threshold
Filter out chunks that fall below a confidence cutoff (0.0 to 1.0):
```powershell
.\.venv\Scripts\python.exe main.py --query "How does the starter motor work?" --threshold 0.5
```

---

### Method B: Python API (`retriever.py`)

You can import and use `RAGRetriever` in any Python script or web application:

```python
from retriever import RAGRetriever, VectorStore, EmbeddingManager

# 1. Connect to the existing vector store
vector_store = VectorStore(
    collection_name="pdf_documents",
    persist_directory="data/vecor_store"
)

# 2. Initialize embedding manager
embedding_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")

# 3. Create the retriever
retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_mgr)

# 4. Standard Retrieval with Normalized Similarity Scores
query = "What is the difference between intake and exhaust valves?"
scored_results = retriever.retrieve_with_scores(query=query, top_k=3)

for doc, score in scored_results:
    print(f"Similarity Score: {score:.4f}")
    print(f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
    print(f"Content: {doc.page_content.strip()}\n")

# 5. Diverse Retrieval with MMR
diverse_docs = retriever.retrieve_mmr(
    query=query,
    top_k=3,
    fetch_k=10,
    lambda_mult=0.6  # 1.0 = pure relevance, 0.0 = pure diversity
)

# 6. Format Retrieved Documents for an LLM Prompt
llm_context = retriever.format_context(diverse_docs)
print("=== Formatted LLM Context ===")
print(llm_context)
```

---

### Method C: Interactive Jupyter Notebook (`notebook/document.ipynb`)

1. Open `notebook/document.ipynb` in your IDE or JupyterLab.
2. Select the `.venv` Python kernel.
3. Run through sections 1 to 4 to review document loading, chunking, and embedding creation.
4. Scroll to **Section 5: Retrieval Pipeline** to run queries, test MMR diversity, and inspect augmented prompt formatting interactively.

---

## 5. Ingesting and Updating Documents

A unified ingestion script [ingest.py](file:///h:/AI/traditional_rag/ingest.py) is provided to automatically discover, load, chunk, embed, and store all documents (PDFs and text files) from the `data/` directory into ChromaDB.

### Automated Ingestion via CLI
Run the ingestion pipeline anytime you add, modify, or remove files:
```powershell
uv run python ingest.py
```

#### Available Flags:
- `--reset`: Clears the collection before ingesting to prevent duplicate chunks (default: enabled).
- `--no-reset`: Appends new documents without clearing existing stored chunks.
- `--chunk-size <int>`: Maximum chunk character length (default: `800`).
- `--chunk-overlap <int>`: Overlap between consecutive chunks (default: `100`).

```powershell
# Example: Ingest with custom chunk size and overlap
uv run python ingest.py --chunk-size 1000 --chunk-overlap 150
```

### Supported Data Locations:
1. `data/pdf_files/*.pdf` (Loaded via `PyMuPDFLoader` with page numbers)
2. `data/text_files/*.txt` (Loaded via `TextLoader`)
3. `data/*.txt` (Root text files)

---

## 6. Downstream LLM Integration (Augmentation & Generation)

Once `retriever.format_context(...)` produces the retrieved context, pass it to your LLM prompt:

```python
prompt_template = f"""You are a helpful assistant. Use only the following context to answer the user's question accurately.

Context:
{retriever.format_context(retrieved_docs)}

Question: {user_query}
Answer:"""

# Send prompt_template to your LLM (OpenAI, Gemini, Ollama, Anthropic, etc.)
```
