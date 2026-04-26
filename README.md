# 📄 rag-papers

A retrieval-augmented generation (RAG) application for querying AI/ML research papers. Drop in PDFs, build a vector index, and ask questions — get answers grounded in the source material with page-level citations.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## What It Does

`rag-papers` builds a searchable knowledge base from research papers and answers questions using retrieval-augmented generation:

1. **Ingest** — Parse PDFs, extract text, and split into semantically meaningful chunks
2. **Index** — Embed chunks using OpenAI embeddings and store in a local FAISS vector index
3. **Retrieve** — Find the most relevant chunks using hybrid search (semantic + keyword)
4. **Generate** — Produce answers grounded in retrieved context with source citations

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/rag-papers.git
cd rag-papers
pip install -r requirements.txt
```

### Set your API key

```bash
export OPENAI_API_KEY="your-key-here"
```

### Add papers

Drop PDF files into the `papers/` directory, or specify a custom path.

### Build the index

```bash
python -m rag_papers.cli index --papers-dir papers/
```

### Ask questions

```bash
python -m rag_papers.cli query "What are the main approaches to RLHF?"
```

### Launch the Streamlit app

```bash
streamlit run app.py
```

## Architecture

```
PDFs → Text Extraction → Chunking → Embedding → FAISS Index
                                                      │
                                          User Query → Hybrid Search
                                                      │
                                          Top-K Chunks → LLM + Prompt
                                                      │
                                          Answer with Citations
```

## Features

- **PDF ingestion** with page-level tracking for accurate citations
- **Configurable chunking** — fixed-size, sentence-based, or recursive splitting
- **Hybrid search** — combines semantic similarity (cosine) with BM25 keyword matching
- **Source citations** — every answer references specific papers and page numbers
- **Conversation memory** — ask follow-up questions with full context
- **Retrieval evaluation** — built-in metrics to measure retrieval quality
- **Local vector store** — FAISS index stored on disk, no external database needed

## Project Structure

```
rag-papers/
├── app.py                          # Streamlit interface
├── requirements.txt
├── README.md
├── papers/                         # Drop PDFs here
├── src/
│   └── rag_papers/
│       ├── __init__.py
│       ├── cli.py                  # CLI interface
│       ├── config.py               # Configuration
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── pdf_parser.py       # PDF text extraction
│       │   └── chunker.py          # Text chunking strategies
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   ├── embeddings.py       # Embedding generation
│       │   └── faiss_store.py      # FAISS index management
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── hybrid_search.py    # Semantic + keyword retrieval
│       ├── generation/
│       │   ├── __init__.py
│       │   └── generator.py        # LLM answer generation
│       └── evaluation/
│           ├── __init__.py
│           └── metrics.py          # Retrieval quality metrics
└── tests/
    ├── __init__.py
    └── test_rag.py
```

## Configuration

Key settings in `src/rag_papers/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 512 | Target tokens per chunk |
| `chunk_overlap` | 50 | Overlap between adjacent chunks |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `llm_model` | `gpt-4o-mini` | Model for answer generation |
| `top_k` | 5 | Number of chunks to retrieve |
| `hybrid_alpha` | 0.7 | Weight for semantic vs keyword search |

## Extending

Add custom document loaders by implementing the `BaseParser` interface, or swap in a different vector store by implementing the `BaseVectorStore` interface.

## Disclaimer

This tool uses OpenAI's API for embeddings and generation. Ensure you have appropriate rights to process any documents you ingest.

## License

MIT
