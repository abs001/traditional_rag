"""
main.py
CLI Demonstration of the Retrieval Functionality in Traditional RAG.
"""

import argparse
import sys
from retriever import RAGRetriever, VectorStore, EmbeddingManager

# Ensure Windows terminal handles special unicode characters safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_retrieval(query: str, top_k: int = 3, use_mmr: bool = False, score_threshold: float = None):
    print("=" * 70)
    print(" TRADITIONAL RAG - RETRIEVAL SYSTEM ")
    print("=" * 70)

    # 1. Connect to the existing Chroma vector store
    vector_store = VectorStore(
        collection_name="pdf_documents",
        persist_directory="data/vecor_store"
    )
    doc_count = vector_store.count()
    print(f"[i] Connected to Vector Store '{vector_store.collection_name}' with {doc_count} document chunks.")

    # 2. Initialize the Embedding Manager
    embedding_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")

    # 3. Initialize the Retriever
    retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_mgr)

    print(f"\n[?] Query: '{query}'")
    print(f"[i] Retrieval Mode: {'Maximal Marginal Relevance (MMR)' if use_mmr else 'Similarity Search'}")
    print(f"[i] Top-K: {top_k}")
    if score_threshold:
        print(f"[i] Score Threshold: {score_threshold}")
    print("-" * 70)

    if use_mmr:
        retrieved_docs = retriever.retrieve_mmr(query=query, top_k=top_k, fetch_k=10, lambda_mult=0.6)
        scored_docs = [(doc, None) for doc in retrieved_docs]
    else:
        scored_docs = retriever.retrieve_with_scores(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )
        retrieved_docs = [doc for doc, _ in scored_docs]

    if not scored_docs:
        print("No documents found matching the criteria.")
        return

    print(f"\n Retrieved {len(scored_docs)} relevant chunk(s):\n")
    for i, (doc, score) in enumerate(scored_docs, 1):
        score_info = f" | Score: {score:.4f}" if score is not None else ""
        source = doc.metadata.get("source", "N/A")
        page = doc.metadata.get("page", "N/A")
        print(f"--- Chunk #{i}{score_info} ---")
        print(f"Source: {source} (Page {page})")
        print("Content:")
        print(doc.page_content.strip())
        print()

    # 4. Show LLM Context formatting
    print("=" * 70)
    print(" FORMATTED AUGMENTED CONTEXT (Ready for LLM Prompt) ")
    print("=" * 70)
    context_str = retriever.format_context(retrieved_docs)
    print(context_str)


def main():
    parser = argparse.ArgumentParser(description="Query the Traditional RAG Vector Store")
    parser.add_argument(
        "--query",
        type=str,
        default="How does a four-stroke engine cycle work?",
        help="The question to search for in the document store"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)"
    )
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Use Maximal Marginal Relevance (MMR) for diverse retrieval"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum similarity score threshold (0.0 to 1.0)"
    )

    args = parser.parse_args()
    run_retrieval(
        query=args.query,
        top_k=args.top_k,
        use_mmr=args.mmr,
        score_threshold=args.threshold
    )


if __name__ == "__main__":
    main()
