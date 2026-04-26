"""Command-line interface for rag-papers."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag_papers.config import RAGConfig
from rag_papers.ingestion.pdf_parser import parse_directory
from rag_papers.ingestion.chunker import chunk_document
from rag_papers.vectorstore.embeddings import EmbeddingGenerator
from rag_papers.vectorstore.faiss_store import FAISSStore
from rag_papers.retrieval.hybrid_search import HybridRetriever
from rag_papers.generation.generator import AnswerGenerator
from rag_papers.evaluation.metrics import evaluate_retrieval


def cmd_index(args):
    """Build the vector index from PDFs."""
    config = RAGConfig(
        papers_dir=args.papers_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_strategy=args.strategy,
    )

    print(f"\n📄 Indexing papers from {config.papers_dir}")
    print("=" * 50)

    # Parse PDFs
    documents = parse_directory(config.papers_dir)
    if not documents:
        print("No PDF files found!")
        return

    # Chunk documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, strategy=config.chunk_strategy, chunk_size=config.chunk_size)
        all_chunks.extend(chunks)
        print(f"  Chunked: {doc.filename} → {len(chunks)} chunks")

    print(f"\n  Total chunks: {len(all_chunks)}")

    # Generate embeddings
    print("\n  Generating embeddings...")
    embedder = EmbeddingGenerator(model=config.embedding_model, api_key=args.api_key)
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_texts(texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Build and save FAISS index
    store = FAISSStore(embedding_dim=config.embedding_dim)
    store.add(embeddings, all_chunks)
    store.save(config.index_dir)

    stats = store.get_stats()
    print(f"\n  Index saved to {config.index_dir}")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print("=" * 50)


def cmd_query(args):
    """Query the index."""
    config = RAGConfig(index_dir=args.index_dir)

    # Load index
    store = FAISSStore.load(config.index_dir)
    embedder = EmbeddingGenerator(model=config.embedding_model, api_key=args.api_key)
    retriever = HybridRetriever(store, embedder, alpha=config.hybrid_alpha)
    generator = AnswerGenerator(model=config.llm_model, api_key=args.api_key)

    query = args.query
    print(f"\n🔍 Query: {query}")
    print("=" * 50)

    # Retrieve
    chunks = retriever.retrieve(query, top_k=config.top_k)

    # Evaluate retrieval
    metrics = evaluate_retrieval(query, chunks)
    print(f"\n  Retrieved {metrics.n_retrieved} chunks (avg similarity: {metrics.avg_similarity:.3f})")

    # Generate answer
    answer = generator.generate(query, chunks)

    print(f"\n{'=' * 50}")
    print(answer.formatted)
    print(f"{'=' * 50}\n")


def cmd_interactive(args):
    """Interactive query mode."""
    config = RAGConfig(index_dir=args.index_dir)

    store = FAISSStore.load(config.index_dir)
    embedder = EmbeddingGenerator(model=config.embedding_model, api_key=args.api_key)
    retriever = HybridRetriever(store, embedder, alpha=config.hybrid_alpha)
    generator = AnswerGenerator(model=config.llm_model, api_key=args.api_key)

    stats = store.get_stats()
    print(f"\n📄 rag-papers — Interactive Mode")
    print(f"  Index: {stats['total_chunks']} chunks from {stats['total_documents']} documents")
    print(f"  Type 'quit' to exit, 'clear' to reset conversation\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "clear":
            generator.clear_history()
            print("  Conversation cleared.\n")
            continue

        chunks = retriever.retrieve(query, top_k=config.top_k)
        answer = generator.generate(query, chunks)

        print(f"\nAssistant: {answer.answer}\n")
        for i, c in enumerate(answer.citations, 1):
            print(f"  [{i}] {c.label}")
        print()


def cmd_stats(args):
    """Show index statistics."""
    config = RAGConfig(index_dir=args.index_dir)
    store = FAISSStore.load(config.index_dir)
    stats = store.get_stats()

    print(f"\n📊 Index Statistics")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Embedding dim: {stats['embedding_dim']}")
    print(f"  Documents:")
    for doc in stats['documents']:
        print(f"    - {doc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="RAG application for querying AI/ML research papers",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    idx = subparsers.add_parser("index", help="Build vector index from PDFs")
    idx.add_argument("--papers-dir", default="papers", help="Directory with PDF files")
    idx.add_argument("--index-dir", default="index", help="Where to save the index")
    idx.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens")
    idx.add_argument("--strategy", choices=["fixed", "recursive"], default="recursive")
    idx.add_argument("--api-key", help="OpenAI API key")

    # Query command
    qry = subparsers.add_parser("query", help="Query the index")
    qry.add_argument("query", help="Question to ask")
    qry.add_argument("--index-dir", default="index")
    qry.add_argument("--api-key", help="OpenAI API key")

    # Interactive command
    inter = subparsers.add_parser("interactive", help="Interactive query mode")
    inter.add_argument("--index-dir", default="index")
    inter.add_argument("--api-key", help="OpenAI API key")

    # Stats command
    st = subparsers.add_parser("stats", help="Show index statistics")
    st.add_argument("--index-dir", default="index")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
