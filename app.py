"""Streamlit interface for rag-papers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import plotly.graph_objects as go
import json

from rag_papers.config import RAGConfig
from rag_papers.ingestion.pdf_parser import parse_directory, parse_pdf
from rag_papers.ingestion.chunker import chunk_document
from rag_papers.vectorstore.embeddings import EmbeddingGenerator
from rag_papers.vectorstore.faiss_store import FAISSStore
from rag_papers.retrieval.hybrid_search import HybridRetriever
from rag_papers.generation.generator import AnswerGenerator
from rag_papers.evaluation.metrics import evaluate_retrieval, evaluate_answer_groundedness


# -- Page config --
st.set_page_config(
    page_title="rag-papers",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .citation-box {
        background: rgba(74, 158, 255, 0.1);
        border-left: 3px solid #4a9eff;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -- Configuration --
config = RAGConfig()
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")

# -- Sidebar --
st.sidebar.markdown("# 📄 rag-papers")
st.sidebar.markdown("Research Paper Q&A")
st.sidebar.divider()

api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Or set OPENAI_API_KEY env var")
effective_key = api_key or os.getenv("OPENAI_API_KEY")

st.sidebar.divider()

page = st.sidebar.radio("Navigation", ["Chat", "Index Management", "Retrieval Analysis"])

st.sidebar.divider()

# Settings
with st.sidebar.expander("Settings"):
    top_k = st.slider("Chunks to retrieve", 1, 15, 5)
    hybrid_alpha = st.slider("Semantic weight", 0.0, 1.0, 0.7, 0.05,
                             help="1.0 = pure semantic, 0.0 = pure keyword")
    llm_model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
    chunk_strategy = st.selectbox("Chunking Strategy", ["recursive", "fixed"])
    chunk_size = st.slider("Chunk size (tokens)", 128, 1024, 512, 64)


# -- Session state --
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "generator" not in st.session_state:
    st.session_state.generator = None


def get_index_status():
    """Check if index exists and return stats."""
    try:
        store = FAISSStore.load(INDEX_DIR)
        return store.get_stats()
    except FileNotFoundError:
        return None


# ============================================================
# PAGE: Chat
# ============================================================
if page == "Chat":
    st.markdown("# 📄 Research Paper Q&A")

    index_stats = get_index_status()
    if not index_stats:
        st.warning("No index found. Go to **Index Management** to ingest papers first.")
        st.stop()

    st.caption(f"Index: {index_stats['total_chunks']} chunks from {index_stats['total_documents']} papers")
    st.divider()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "citations" in msg:
                with st.expander("📚 Sources"):
                    for i, citation in enumerate(msg["citations"], 1):
                        st.markdown(
                            f'<div class="citation-box">'
                            f'<strong>[{i}]</strong> {citation["label"]} '
                            f'(score: {citation["score"]:.3f})<br>'
                            f'<em>{citation["excerpt"]}</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    if query := st.chat_input("Ask a question about your papers..."):
        if not effective_key:
            st.error("Please provide an OpenAI API key.")
            st.stop()

        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching papers..."):
                store = FAISSStore.load(INDEX_DIR)
                embedder = EmbeddingGenerator(api_key=effective_key)
                retriever = HybridRetriever(store, embedder, alpha=hybrid_alpha)

                if st.session_state.generator is None:
                    st.session_state.generator = AnswerGenerator(
                        model=llm_model, api_key=effective_key
                    )

                chunks = retriever.retrieve(query, top_k=top_k)
                answer = st.session_state.generator.generate(query, chunks)

            st.markdown(answer.answer)

            citations_data = [
                {
                    "label": c.label,
                    "score": c.relevance_score,
                    "excerpt": c.excerpt,
                }
                for c in answer.citations
            ]

            with st.expander("📚 Sources"):
                for i, citation in enumerate(citations_data, 1):
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<strong>[{i}]</strong> {citation["label"]} '
                        f'(score: {citation["score"]:.3f})<br>'
                        f'<em>{citation["excerpt"]}</em>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer.answer,
                "citations": citations_data,
            })

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            if st.session_state.generator:
                st.session_state.generator.clear_history()
            st.rerun()

# ============================================================
# PAGE: Index Management
# ============================================================
elif page == "Index Management":
    st.markdown("# Index Management")
    st.markdown("Ingest papers and build the search index")
    st.divider()

    # Current index status
    index_stats = get_index_status()
    if index_stats:
        st.success(f"Index loaded: {index_stats['total_chunks']} chunks from {index_stats['total_documents']} documents")
        with st.expander("Indexed Documents"):
            for doc in index_stats["documents"]:
                st.markdown(f"- {doc}")
    else:
        st.info("No index found. Upload papers and build the index below.")

    st.divider()

    # Upload PDFs
    st.markdown("### Upload Papers")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        os.makedirs(PAPERS_DIR, exist_ok=True)
        for f in uploaded_files:
            path = os.path.join(PAPERS_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} files to {PAPERS_DIR}")

    # List papers directory
    if os.path.exists(PAPERS_DIR):
        pdfs = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]
        if pdfs:
            st.markdown(f"### Papers Directory ({len(pdfs)} files)")
            for pdf in sorted(pdfs):
                st.markdown(f"- {pdf}")

    st.divider()

    # Build index
    st.markdown("### Build Index")
    if st.button("🔨 Build Index", type="primary", use_container_width=True):
        if not effective_key:
            st.error("Please provide an OpenAI API key.")
            st.stop()

        if not os.path.exists(PAPERS_DIR):
            st.error(f"No papers directory found at {PAPERS_DIR}")
            st.stop()

        with st.spinner("Parsing PDFs..."):
            documents = parse_directory(PAPERS_DIR)
            if not documents:
                st.error("No PDF files found!")
                st.stop()

        progress = st.progress(0)
        status = st.empty()

        # Chunk documents
        all_chunks = []
        for i, doc in enumerate(documents):
            status.text(f"Chunking {doc.filename}...")
            chunks = chunk_document(doc, strategy=chunk_strategy, chunk_size=chunk_size)
            all_chunks.extend(chunks)
            progress.progress((i + 1) / (len(documents) + 2))

        status.text(f"Generating embeddings for {len(all_chunks)} chunks...")
        embedder = EmbeddingGenerator(api_key=effective_key)
        texts = [c.text for c in all_chunks]
        embeddings = embedder.embed_texts(texts)
        progress.progress((len(documents) + 1) / (len(documents) + 2))

        status.text("Building FAISS index...")
        store = FAISSStore(embedding_dim=config.embedding_dim)
        store.add(embeddings, all_chunks)
        store.save(INDEX_DIR)
        progress.progress(1.0)

        status.text("Done!")
        st.success(f"Index built: {store.size} chunks from {len(documents)} documents")
        st.rerun()

# ============================================================
# PAGE: Retrieval Analysis
# ============================================================
elif page == "Retrieval Analysis":
    st.markdown("# Retrieval Analysis")
    st.markdown("Analyze retrieval quality and search behavior")
    st.divider()

    index_stats = get_index_status()
    if not index_stats:
        st.warning("No index found. Build an index first.")
        st.stop()

    test_query = st.text_input("Test query", placeholder="Enter a query to analyze retrieval...")

    if test_query and effective_key:
        store = FAISSStore.load(INDEX_DIR)
        embedder = EmbeddingGenerator(api_key=effective_key)
        retriever = HybridRetriever(store, embedder, alpha=hybrid_alpha)

        # Get results from both methods
        hybrid_results = retriever.retrieve(test_query, top_k=top_k)
        semantic_results = retriever.retrieve_semantic_only(test_query, top_k=top_k)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Hybrid Search Results")
            metrics = evaluate_retrieval(test_query, hybrid_results)
            st.metric("Avg Score", f"{metrics.avg_similarity:.4f}")
            st.metric("Source Diversity", metrics.source_diversity)

            for i, chunk in enumerate(hybrid_results, 1):
                with st.expander(f"[{i}] {chunk.get('source_file', '?')} — {chunk.get('combined_score', 0):.4f}"):
                    st.markdown(f"**Semantic:** {chunk.get('semantic_score', 0):.4f} | **BM25:** {chunk.get('bm25_score', 0):.4f}")
                    st.markdown(f"**Pages:** {chunk.get('page_numbers', [])}")
                    st.text(chunk.get("text", "")[:500])

        with col2:
            st.markdown("### Semantic-Only Results")
            sem_metrics = evaluate_retrieval(test_query, semantic_results)
            st.metric("Avg Score", f"{sem_metrics.avg_similarity:.4f}")
            st.metric("Source Diversity", sem_metrics.source_diversity)

            for i, chunk in enumerate(semantic_results, 1):
                with st.expander(f"[{i}] {chunk.get('source_file', '?')} — {chunk.get('similarity_score', 0):.4f}"):
                    st.markdown(f"**Pages:** {chunk.get('page_numbers', [])}")
                    st.text(chunk.get("text", "")[:500])

        # Score comparison chart
        st.divider()
        st.markdown("### Score Distribution")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Chunk {i+1}" for i in range(len(hybrid_results))],
            y=[c.get("combined_score", 0) for c in hybrid_results],
            name="Hybrid",
            marker_color="#4a9eff",
        ))
        fig.add_trace(go.Bar(
            x=[f"Chunk {i+1}" for i in range(len(semantic_results))],
            y=[c.get("similarity_score", 0) for c in semantic_results],
            name="Semantic Only",
            marker_color="#50c878",
        ))
        fig.update_layout(
            barmode="group",
            yaxis_title="Score",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400,
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
