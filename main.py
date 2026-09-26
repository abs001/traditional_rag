"""
main.py
CLI Demonstration of the Traditional RAG Pipeline:
1. Retrieval (Similarity search / MMR over ChromaDB)
2. Prompt Augmentation (Context assembly with strict grounding instructions)
3. LLM Generation (Google Gemini via langchain-google-genai)
"""

import os
import sys
import argparse
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from retriever import RAGRetriever, VectorStore, EmbeddingManager

# Load environment variables (.env file)
load_dotenv()

# Ensure Windows terminal handles special unicode characters safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SYSTEM_PROMPT = """You are an expert technical automotive assistant answering questions based on retrieved documentation.

Strict Rules:
1. Grounding: Answer the question using ONLY the facts explicitly stated in the provided context below. Do NOT assume, extrapolate, or use outside knowledge.
2. Missing Information: If the context does not contain the answer, explicitly state: "The provided documents do not contain enough information to answer this question."
3. Citations: Cite your sources for every major claim using the format [Document X, Source: filename, Page: Y] matching the context header.
4. Tone: Be clear, structured, and technically precise.
"""


def build_rag_prompt(query: str, context_str: str) -> str:
    """Combines system instructions, retrieved context, and the query into a unified prompt string."""
    return f"""{SYSTEM_PROMPT}

----------------------------------------
RETRIEVED CONTEXT:
----------------------------------------
{context_str}
----------------------------------------

USER QUESTION:
{query}

ANSWER (grounded in context, citing sources):"""


def generate_llm_answer(
    query: str,
    retrieved_docs: List[Document],
    retriever: RAGRetriever,
    model_name: str = "gemini-3.8-flash",
    temperature: float = 0.2,
    stream: bool = True,
) -> Optional[str]:
    """Generates an answer from Google Gemini using the augmented context."""
    # 1. Format retrieved documents into context
    context_str = retriever.format_context(retrieved_docs)

    # 2. Check for Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n" + "=" * 70)
        print(" [!] NOTICE: GEMINI_API_KEY is not set.")
        print("=" * 70)
        print(" To enable LLM generation with Google Gemini:")
        print("   1. Create a '.env' file in this folder (or copy from .env.example):")
        print("      GEMINI_API_KEY=your_gemini_api_key_here")
        print("   2. Get a free API key at: https://aistudio.google.com/app/apikey")
        print("=" * 70)
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("[X] Error: langchain-google-genai is not installed. Run: uv add langchain-google-genai")
        return None

    print("\n" + "=" * 70)
    print(f" GENERATING LLM RESPONSE (Model: {model_name}) ")
    print("=" * 70)

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )

        user_content = f"Context:\n{context_str}\n\nQuestion: {query}"
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        if stream:
            full_response = []
            for chunk in llm.stream(messages):
                token = chunk.content
                if isinstance(token, str):
                    print(token, end="", flush=True)
                    full_response.append(token)
                elif isinstance(token, list):
                    for item in token:
                        if isinstance(item, dict) and "text" in item:
                            print(item["text"], end="", flush=True)
                            full_response.append(item["text"])
                        elif isinstance(item, str):
                            print(item, end="", flush=True)
                            full_response.append(item)
            print("\n")
            return "".join(full_response)
        else:
            response = llm.invoke(messages)
            answer = response.content
            print(answer)
            print()
            return str(answer)

    except Exception as e:
        print(f"\n[X] Error during Gemini LLM generation: {e}")
        return None


def run_pipeline(
    query: str,
    top_k: int = 3,
    use_mmr: bool = False,
    score_threshold: Optional[float] = None,
    run_llm: bool = True,
    model_name: str = "gemini-3.8-flash",
    temperature: float = 0.2,
    stream: bool = True,
):
    print("=" * 70)
    print(" TRADITIONAL RAG - RETRIEVAL & GENERATION PIPELINE ")
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
    if score_threshold is not None:
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
        print("[!] No documents found matching the search criteria.")
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

    # 5. LLM Answer Generation
    if run_llm:
        generate_llm_answer(
            query=query,
            retrieved_docs=retrieved_docs,
            retriever=retriever,
            model_name=model_name,
            temperature=temperature,
            stream=stream,
        )
    else:
        print("\n[i] LLM generation skipped (--no-llm flag specified).")


def main():
    parser = argparse.ArgumentParser(description="Traditional RAG Pipeline: Query Vector Store and Generate Answers with LLM")
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
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.8-flash",
        help="Google Gemini model name (default: gemini-3.8-flash)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2 for factual RAG)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation and only run retrieval and context formatting"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output for LLM generation"
    )

    args = parser.parse_args()
    run_pipeline(
        query=args.query,
        top_k=args.top_k,
        use_mmr=args.mmr,
        score_threshold=args.threshold,
        run_llm=not args.no_llm,
        model_name=args.model,
        temperature=args.temperature,
        stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()
