"""FAISS-based local vector store with persistence."""

import os
import json
import numpy as np
import faiss

from ..ingestion.chunker import Chunk


class FAISSStore:
    """Local FAISS vector store with metadata persistence.
    
    Stores embeddings in a FAISS index and chunk metadata in a JSON sidecar.
    """

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine after normalization)
        self.chunks: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]):
        """Add embeddings and their corresponding chunks to the store.
        
        Normalizes embeddings for cosine similarity via inner product.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        self.index.add(normalized)
        self.chunks.extend([c.to_dict() for c in chunks])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search for the most similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of dicts with chunk data and similarity score
        """
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_data = self.chunks[idx].copy()
            chunk_data["similarity_score"] = float(score)
            results.append(chunk_data)

        return results

    def save(self, directory: str):
        """Persist the index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(directory, "index.faiss")
        faiss.write_index(self.index, index_path)

        # Save chunk metadata
        meta_path = os.path.join(directory, "chunks.json")
        with open(meta_path, "w") as f:
            json.dump({
                "embedding_dim": self.embedding_dim,
                "chunks": self.chunks,
            }, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "FAISSStore":
        """Load a persisted index from disk."""
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "chunks.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No index found at {directory}")

        with open(meta_path) as f:
            meta = json.load(f)

        store = cls(embedding_dim=meta["embedding_dim"])
        store.index = faiss.read_index(index_path)
        store.chunks = meta["chunks"]

        return store

    @property
    def size(self) -> int:
        return self.index.ntotal

    def get_stats(self) -> dict:
        """Return index statistics."""
        source_files = set(c["source_file"] for c in self.chunks)
        return {
            "total_chunks": self.size,
            "total_documents": len(source_files),
            "documents": sorted(source_files),
            "embedding_dim": self.embedding_dim,
        }
