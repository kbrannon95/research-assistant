"""LLM answer generation with source citations."""

import os
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class Citation:
    """A source citation for a generated answer."""
    source_file: str
    page_numbers: list[int]
    relevance_score: float
    excerpt: str = ""

    @property
    def label(self) -> str:
        pages = self.page_numbers
        if len(pages) == 1:
            return f"{self.source_file}, p.{pages[0]}"
        return f"{self.source_file}, pp.{pages[0]}-{pages[-1]}"


@dataclass
class GeneratedAnswer:
    """An answer generated from retrieved context."""
    answer: str
    citations: list[Citation]
    query: str
    model: str
    context_chunks: list[dict]

    @property
    def formatted(self) -> str:
        """Format answer with inline citations."""
        lines = [self.answer, "", "**Sources:**"]
        for i, citation in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {citation.label} (relevance: {citation.relevance_score:.2f})")
        return "\n".join(lines)


SYSTEM_PROMPT = """You are a research assistant that answers questions based on provided research paper excerpts. Follow these rules:

1. Answer the question using ONLY the information in the provided context. If the context doesn't contain enough information, say so.
2. Be specific and technical — the user is reading research papers.
3. When referencing information, mention which source it comes from using [Source N] notation.
4. If different sources present conflicting information, note the disagreement.
5. Keep answers concise but thorough."""


def build_context_prompt(query: str, chunks: list[dict]) -> str:
    """Build the prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file", "Unknown")
        pages = chunk.get("page_numbers", [])
        page_str = f"pp.{pages[0]}-{pages[-1]}" if len(pages) > 1 else f"p.{pages[0]}" if pages else ""

        context_parts.append(
            f"[Source {i}] {source} ({page_str}):\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""Context from research papers:

{context}

---

Question: {query}

Answer based on the context above, citing sources with [Source N] notation:"""


class AnswerGenerator:
    """Generate answers from retrieved context using an LLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.conversation_history: list[dict] = []

    def generate(
        self,
        query: str,
        chunks: list[dict],
        use_history: bool = True,
    ) -> GeneratedAnswer:
        """Generate an answer from retrieved chunks.
        
        Args:
            query: User question
            chunks: Retrieved context chunks with metadata
            use_history: Whether to include conversation history
        
        Returns:
            GeneratedAnswer with citations
        """
        user_prompt = build_context_prompt(query, chunks)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if use_history and self.conversation_history:
            # Include recent history for follow-up questions
            messages.extend(self.conversation_history[-6:])  # Last 3 exchanges

        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1024,
        )

        answer_text = response.choices[0].message.content or ""

        # Build citations from chunks
        citations = [
            Citation(
                source_file=chunk.get("source_file", "Unknown"),
                page_numbers=chunk.get("page_numbers", []),
                relevance_score=chunk.get("combined_score", chunk.get("similarity_score", 0)),
                excerpt=chunk["text"][:150] + "...",
            )
            for chunk in chunks
        ]

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer_text})

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            query=query,
            model=self.model,
            context_chunks=chunks,
        )

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
