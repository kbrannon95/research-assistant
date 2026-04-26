"""Hybrid retrieval combining semantic search with BM25 keyword matching."""

import numpy as np
from rank_bm25 import BM25Okapi

from ..vectorstore.faiss_store import FAISSStore
from ..vectorstore.embeddings import EmbeddingGenerator


class HybridRetriever:
    """Combines FAISS semantic search with BM25 keyword matching.
    
    Semantic search captures meaning; BM25 catches exact terms and acronyms
    that embedding models sometimes miss.
    """

    def __init__(
        self,
        store: FAISSStore,
        embedder: EmbeddingGenerator,
        alpha: float = 0.7,
    ):
        self.store = store
        self.embedder = embedder
        self.alpha = alpha  # Weight for semantic (1-alpha for BM25)

        # Build BM25 index from chunk texts
        self.chunk_texts = [c["text"] for c in store.chunks]
        tokenized = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the most relevant chunks using hybrid search.
        
        Combines cosine similarity scores from FAISS with BM25 keyword scores,
        weighted by alpha.
        
        Args:
            query: User query
            top_k: Number of chunks to return
        
        Returns:
            List of chunk dicts with combined scores, sorted by relevance
        """
        # Semantic search — get more than top_k to allow re-ranking
        fetch_k = min(top_k * 3, self.store.size)
        query_embedding = self.embedder.embed_query(query)
        semantic_results = self.store.search(query_embedding, top_k=fetch_k)

        # BM25 keyword search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Normalize BM25 scores to [0, 1]
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_normalized = bm25_scores / bm25_max

        # Combine scores
        scored_chunks = {}

        for result in semantic_results:
            chunk_idx = result["chunk_index"]
            semantic_score = result["similarity_score"]

            bm25_score = bm25_normalized[chunk_idx] if chunk_idx < len(bm25_normalized) else 0

            combined_score = self.alpha * semantic_score + (1 - self.alpha) * bm25_score

            result["semantic_score"] = semantic_score
            result["bm25_score"] = float(bm25_score)
            result["combined_score"] = float(combined_score)
            scored_chunks[chunk_idx] = result

        # Also check top BM25 results that might not be in semantic results
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
        for idx in bm25_top_indices:
            if idx not in scored_chunks and idx < len(self.store.chunks):
                chunk_data = self.store.chunks[idx].copy()

                # Get semantic score for this chunk
                chunk_embedding_score = 0.0  # Would need the embedding; approximate as 0

                bm25_score = bm25_normalized[idx]
                combined_score = (1 - self.alpha) * bm25_score

                chunk_data["semantic_score"] = chunk_embedding_score
                chunk_data["bm25_score"] = float(bm25_score)
                chunk_data["combined_score"] = float(combined_score)
                chunk_data["similarity_score"] = chunk_embedding_score
                scored_chunks[idx] = chunk_data

        # Sort by combined score and return top_k
        ranked = sorted(
            scored_chunks.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )

        return ranked[:top_k]

    def retrieve_semantic_only(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve using only semantic search (for comparison)."""
        query_embedding = self.embedder.embed_query(query)
        return self.store.search(query_embedding, top_k=top_k)
