"""
ingest.py
Unified Document Ingestion Pipeline for Traditional RAG.

Scans and ingests all documents (PDFs and Text files) into ChromaDB:
- data/pdf_files/*.pdf
- data/text_files/*.txt
- data/*.txt
"""

import os
import sys

# Ensure Windows terminal handles special unicode characters safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retriever import VectorStore, EmbeddingManager


def load_all_documents(data_dir: str = "data") -> List[Document]:
    """Load all PDF and text documents from the data directory."""
    raw_documents: List[Document] = []

    # 1. Load PDFs from data/pdf_files
    pdf_dir = os.path.join(data_dir, "pdf_files")
    if os.path.exists(pdf_dir):
        for file in os.listdir(pdf_dir):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(pdf_dir, file)
                print(f"[+] Loading PDF: {file_path}")
                try:
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source_type"] = "pdf"
                        d.metadata["file_name"] = file
                    raw_documents.extend(docs)
                except Exception as e:
                    print(f"[-] Error loading PDF {file}: {e}")

    # 2. Load TXT files from data/text_files
    text_dir = os.path.join(data_dir, "text_files")
    if os.path.exists(text_dir):
        for file in os.listdir(text_dir):
            if file.lower().endswith(".txt"):
                file_path = os.path.join(text_dir, file)
                print(f"[+] Loading Text file: {file_path}")
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source_type"] = "txt"
                        d.metadata["file_name"] = file
                        d.metadata["page"] = 1
                    raw_documents.extend(docs)
                except Exception as e:
                    print(f"[-] Error loading text file {file}: {e}")

    # 3. Load any standalone .txt files directly in data/
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.lower().endswith(".txt"):
                file_path = os.path.join(data_dir, file)
                print(f"[+] Loading Root Text file: {file_path}")
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source_type"] = "txt"
                        d.metadata["file_name"] = file
                        d.metadata["page"] = 1
                    raw_documents.extend(docs)
                except Exception as e:
                    print(f"[-] Error loading {file}: {e}")

    return raw_documents


def ingest(
    data_dir: str = "data",
    collection_name: str = "pdf_documents",
    reset_collection: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
):
    print("=" * 70)
    print(" TRADITIONAL RAG - DOCUMENT INGESTION PIPELINE ")
    print("=" * 70)

    # 1. Load raw documents
    documents = load_all_documents(data_dir=data_dir)
    if not documents:
        print("[-] No documents found to ingest!")
        return

    print(f"\n[i] Loaded {len(documents)} raw document page(s)/file(s).")

    # 2. Chunk documents for optimal retrieval granularity
    print(f"[i] Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[i] Produced {len(chunks)} text chunk(s).")

    # 3. Initialize Embedding Manager
    print("\n[i] Initializing embedding model (all-MiniLM-L6-v2)...")
    embedding_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")

    # 4. Initialize Vector Store
    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=os.path.join(data_dir, "vecor_store")
    )

    if reset_collection:
        print(f"[i] Resetting existing collection '{collection_name}' before ingestion...")
        vector_store.reset()

    # 5. Insert documents (if not already existing, generating embeddings only for new chunks)
    print(f"[i] Checking ChromaDB and inserting chunks (if not exists)...")
    stats = vector_store.add_documents_if_not_exists(
        documents=chunks,
        embedding_manager=embedding_mgr,
    )

    total_count = vector_store.count()
    print("=" * 70)
    print(f"[OK] Ingestion complete!")
    print(f"     Total checked:  {stats['total_checked']}")
    print(f"     New inserted:   {stats['inserted']}")
    print(f"     Skipped exists: {stats['skipped']}")
    print(f"     Current total chunks in ChromaDB: {total_count}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Ingest all PDFs and TXT files into ChromaDB")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing source files")
    parser.add_argument("--collection", type=str, default="pdf_documents", help="ChromaDB collection name")
    parser.add_argument("--reset", action="store_true", default=True, help="Clear existing collection before ingesting (default: True)")
    parser.add_argument("--no-reset", dest="reset", action="store_false", help="Append without clearing existing collection")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for splitting (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap between chunks (default: 100)")

    args = parser.parse_args()
    ingest(
        data_dir=args.data_dir,
        collection_name=args.collection,
        reset_collection=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
