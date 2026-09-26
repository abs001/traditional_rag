"""
retriever.py
Core Retrieval functionality for the Traditional RAG Pipeline.

This module provides:
1. EmbeddingManager: Handles generating query and document vector embeddings using SentenceTransformers.
2. VectorStore: ChromaDB wrapper with support for persistence, document insertion, and similarity queries.
3. RAGRetriever: High-level retriever supporting:
   - Top-K similarity search with score calculation
   - Score threshold filtering
   - Metadata filtering
   - Maximal Marginal Relevance (MMR) for diverse, non-redundant retrieval
   - Prompt-ready context formatting for downstream LLM generation
"""

import os
import uuid
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


def _compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between 2D arrays using pure NumPy (eliminates sklearn dependency)."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.where(a_norm == 0, 1e-10, a_norm)
    b_norm = np.where(b_norm == 0, 1e-10, b_norm)
    return np.dot(a / a_norm, (b / b_norm).T)


class EmbeddingManager:
    """Manages text embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate vector embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Model is not initialized.")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a vector embedding for a single user query."""
        if not self.model:
            raise ValueError("Model is not initialized.")
        return self.model.encode(query, show_progress_bar=False)


class VectorStore:
    """Persistent ChromaDB vector store wrapper for storing and querying documents."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vecor_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        # Resolve path relative to project root or current working dir
        target_path = self.persist_directory
        if not os.path.exists(target_path):
            alt_path = os.path.join(os.path.dirname(__file__), "..", target_path)
            if os.path.exists(alt_path):
                target_path = alt_path
            elif os.path.exists(os.path.join(os.path.dirname(__file__), target_path)):
                target_path = os.path.join(os.path.dirname(__file__), target_path)

        try:
            self.client = chromadb.PersistentClient(path=target_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF documents embeddings for RAG"},
            )
        except Exception as e:
            print(f"Error initializing vector store at '{target_path}': {e}")
            raise

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[np.ndarray] = None,
    ):
        """Add LangChain Document objects and their vector embeddings to ChromaDB."""
        if not documents:
            return

        ids = []
        metadatas = []
        document_texts = []
        embeddings_list = []

        for i, doc in enumerate(documents):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Clean metadata so Chroma doesn't reject None or nested dicts
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is not None:
                    clean_meta[k] = str(v)
            clean_meta["content_length"] = len(doc.page_content)
            metadatas.append(clean_meta)

            document_texts.append(doc.page_content)
            if embeddings is not None:
                emb = embeddings[i]
                embeddings_list.append(emb.tolist() if isinstance(emb, np.ndarray) else emb)

        add_kwargs = {
            "ids": ids,
            "metadatas": metadatas,
            "documents": document_texts,
        }
        if embeddings_list:
            add_kwargs["embeddings"] = embeddings_list

        self.collection.add(**add_kwargs)
        print(f"Successfully added {len(documents)} document chunk(s) to collection '{self.collection_name}'.")

    @staticmethod
    def generate_deterministic_id(doc: Document) -> str:
        """
        Generate a unique, deterministic ID for a Document based on source, page, and content.
        Ensures the same document chunk always produces the exact same ID across runs.
        """
        source = str(doc.metadata.get("source", doc.metadata.get("file_name", "unknown")))
        page = str(doc.metadata.get("page", 0))
        normalized_content = doc.page_content.strip()

        # Build a unique signature from metadata and content
        signature = f"{source}::p{page}::{normalized_content}"
        content_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]

        clean_src = os.path.basename(source).replace(" ", "_").replace(".", "_")
        return f"doc_{clean_src}_p{page}_{content_hash}"

    def get_existing_ids(self, ids: List[str]) -> Set[str]:
        """Check which of the provided candidate IDs already exist in ChromaDB."""
        if not ids:
            return set()

        existing: Set[str] = set()
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            try:
                records = self.collection.get(ids=batch)
                if records and records.get("ids"):
                    existing.update(records["ids"])
            except Exception as e:
                print(f"Notice while checking existing IDs: {e}")
        return existing

    def add_documents_if_not_exists(
        self,
        documents: List[Document],
        embedding_manager: Optional[EmbeddingManager] = None,
        precomputed_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        """
        Insert documents into ChromaDB ONLY IF they do not already exist.

        Steps:
        1. Generates deterministic SHA-256 content IDs for all candidate documents.
        2. Queries ChromaDB to check which IDs are already stored.
        3. Filters out already-existing documents (saves time and avoids duplication).
        4. Computes vector embeddings ONLY for new documents.
        5. Inserts the new documents into ChromaDB.

        Returns:
            Dict[str, int]: Summary stats {'total_checked': int, 'inserted': int, 'skipped': int}
        """
        if not documents:
            return {"total_checked": 0, "inserted": 0, "skipped": 0}

        # 1. Generate deterministic IDs for all candidate chunks
        candidate_ids = [self.generate_deterministic_id(doc) for doc in documents]

        # 2. Check existing IDs in ChromaDB
        existing_ids = self.get_existing_ids(candidate_ids)

        new_docs: List[Document] = []
        new_ids: List[str] = []
        new_texts: List[str] = []
        new_metadatas: List[Dict[str, Any]] = []
        new_embeddings: List[List[float]] = []

        # 3. Filter candidates
        for i, (doc, doc_id) in enumerate(zip(documents, candidate_ids)):
            if doc_id in existing_ids:
                continue

            new_docs.append(doc)
            new_ids.append(doc_id)
            new_texts.append(doc.page_content)

            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is not None:
                    clean_meta[k] = str(v)
            clean_meta["content_length"] = len(doc.page_content)
            clean_meta["doc_id"] = doc_id
            new_metadatas.append(clean_meta)

            if precomputed_embeddings is not None:
                emb = precomputed_embeddings[i]
                new_embeddings.append(emb.tolist() if isinstance(emb, np.ndarray) else emb)

        skipped_count = len(documents) - len(new_docs)

        # 4. If all documents already exist, skip insertion
        if not new_docs:
            print(f"[i] All {len(documents)} document chunk(s) already exist in '{self.collection_name}'. Skipping insertion.")
            return {"total_checked": len(documents), "inserted": 0, "skipped": skipped_count}

        # 5. Compute embeddings ONLY for new documents if not precomputed
        if not new_embeddings:
            if embedding_manager is not None:
                print(f"[i] Generating embeddings for {len(new_docs)} new chunk(s) (skipped {skipped_count} existing duplicate(s))...")
                emb_array = embedding_manager.generate_embeddings(new_texts)
                new_embeddings = [e.tolist() for e in emb_array]
            else:
                raise ValueError("Must provide either 'embedding_manager' or 'precomputed_embeddings' to embed new documents.")

        # 6. Insert new records into ChromaDB
        self.collection.add(
            ids=new_ids,
            documents=new_texts,
            metadatas=new_metadatas,
            embeddings=new_embeddings,
        )
        print(f"[OK] Successfully inserted {len(new_docs)} new chunk(s) into '{self.collection_name}' (skipped {skipped_count} duplicate(s)).")
        return {"total_checked": len(documents), "inserted": len(new_docs), "skipped": skipped_count}

    def reset(self):
        """Clear all documents in the collection to re-index from scratch."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF and Text documents embeddings for RAG"},
            )
            print(f"Collection '{self.collection_name}' reset successfully.")
        except Exception as e:
            print(f"Notice during collection reset: {e}")

    def query_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Query ChromaDB with a query embedding vector."""
        includes = ["documents", "metadatas", "distances"]
        if include_embeddings:
            includes.append("embeddings")

        query_list = (
            query_embedding.tolist()
            if isinstance(query_embedding, np.ndarray)
            else query_embedding
        )

        query_kwargs = {
            "query_embeddings": [query_list],
            "n_results": top_k,
            "include": includes,
        }
        if filter_dict:
            query_kwargs["where"] = filter_dict

        return self.collection.query(**query_kwargs)


class RAGRetriever:
    """High-level retriever for Traditional RAG pipelines."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedding_manager = embedding_manager or EmbeddingManager()

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top_k documents with their computed similarity scores.

        Chroma returns squared L2 distances (d) by default.
        Normalized similarity score: 1.0 / (1.0 + distance)
        where 1.0 represents an exact match.
        """
        query_emb = self.embedding_manager.embed_query(query)
        raw_results = self.vector_store.query_similarity(
            query_embedding=query_emb,
            top_k=top_k,
            filter_dict=filter_dict,
            include_embeddings=False,
        )

        docs = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        scored_results: List[Tuple[Document, float]] = []
        for text, meta, dist in zip(docs, metadatas, distances):
            # Compute normalized similarity score from L2 distance
            similarity = 1.0 / (1.0 + float(dist))

            if score_threshold is not None and similarity < score_threshold:
                continue

            doc = Document(page_content=text, metadata=meta or {})
            scored_results.append((doc, similarity))

        return scored_results

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve top_k Document objects matching the query."""
        scored = self.retrieve_with_scores(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_dict=filter_dict,
        )
        return [doc for doc, _ in scored]

    def retrieve_mmr(
        self,
        query: str,
        top_k: int = 3,
        fetch_k: int = 10,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Maximal Marginal Relevance (MMR) retrieval.

        Balances query relevance and diversity to prevent retrieving near-duplicate chunks.
        MMR Score = lambda_mult * Sim(query, doc) - (1 - lambda_mult) * max(Sim(doc, selected_doc))
        """
        query_emb = self.embedding_manager.embed_query(query)
        total_docs = self.vector_store.count()
        fetch_k = min(fetch_k, total_docs)
        if fetch_k <= 0:
            return []

        raw_results = self.vector_store.query_similarity(
            query_embedding=query_emb,
            top_k=fetch_k,
            filter_dict=filter_dict,
            include_embeddings=True,
        )

        docs = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        candidate_embeddings = raw_results.get("embeddings", [[]])[0]

        if not docs or not candidate_embeddings:
            return []

        cand_embs = np.array(candidate_embeddings)
        q_emb_2d = np.array(query_emb).reshape(1, -1)

        # Relevance to query (cosine similarity)
        query_sims = _compute_cosine_similarity(cand_embs, q_emb_2d).reshape(-1)

        selected_indices: List[int] = []
        remaining_indices = list(range(len(docs)))

        # Greedily pick the highest MMR candidate
        for _ in range(min(top_k, len(docs))):
            best_mmr_score = -float("inf")
            best_idx = None

            for idx in remaining_indices:
                relevance = query_sims[idx]

                if not selected_indices:
                    diversity_penalty = 0.0
                else:
                    # Maximum similarity with already selected documents
                    selected_embs = cand_embs[selected_indices]
                    doc_emb_2d = cand_embs[idx].reshape(1, -1)
                    sims_to_selected = _compute_cosine_similarity(doc_emb_2d, selected_embs).reshape(-1)
                    diversity_penalty = float(np.max(sims_to_selected))

                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity_penalty
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        selected_documents = [
            Document(page_content=docs[i], metadata=metadatas[i] or {})
            for i in selected_indices
        ]
        return selected_documents

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a structured context string suitable for an LLM prompt.
        """
        if not documents:
            return "No relevant documents found."

        context_blocks = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", None)
            page_info = f" | Page: {page}" if page is not None else ""

            block = (
                f"[Document {i}]\n"
                f"Source: {source}{page_info}\n"
                f"Content:\n{doc.page_content.strip()}"
            )
            context_blocks.append(block)

        return "\n\n" + ("=" * 50) + "\n\n".join(context_blocks) + "\n\n" + ("=" * 50)
