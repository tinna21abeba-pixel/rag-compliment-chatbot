"""
rag_pipeline.py
---------------
Orchestrates the full Retrieval-Augmented Generation pipeline:

    User Question
         │
         ▼
   [QueryEmbedder]  ──► 384-dim query vector
         │
         ▼
   [Retriever]      ──► Top-k RetrievedChunk objects
         │
         ▼
   [build_prompt]   ──► Formatted prompt with injected context
         │
         ▼
   [Generator]      ──► Natural-language answer
         │
         ▼
   RAGResponse (answer + sources)

This module is the single entry-point used by both the Streamlit UI (app.py)
and the evaluation notebook.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.embedder import get_embedder, DEFAULT_MODEL_NAME
from src.generator import build_generator, build_prompt
from src.retriever import build_retriever, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response data class
# ---------------------------------------------------------------------------
@dataclass
class RAGResponse:
    """Structured output from the RAG pipeline."""
    question: str
    answer: str
    sources: List[RetrievedChunk]
    latency_seconds: float
    product_filter: Optional[str] = None
    error: Optional[str] = None

    def formatted_sources(self) -> str:
        """Return a markdown-formatted source summary for display."""
        lines = []
        for i, chunk in enumerate(self.sources, start=1):
            lines.append(
                f"**Source {i}** — `{chunk.complaint_id}` | "
                f"{chunk.product_category} | {chunk.issue} | "
                f"{chunk.company} | Score: {chunk.score:.3f}\n\n"
                f"> {chunk.text[:300]}{'...' if len(chunk.text) > 300 else ''}"
            )
        return "\n\n---\n\n".join(lines)


# ---------------------------------------------------------------------------
# RAG Pipeline class
# ---------------------------------------------------------------------------
class RAGPipeline:
    """
    End-to-end RAG pipeline for CrediTrust complaint analysis.

    Usage
    -----
    >>> pipeline = RAGPipeline.from_config(config)
    >>> response = pipeline.query("Why are customers unhappy with credit cards?")
    >>> print(response.answer)
    >>> print(response.formatted_sources())

    Parameters
    ----------
    retriever : ChromaRetriever | FAISSRetriever
    generator : HFPipelineGenerator | HFHubGenerator
    top_k : int
        Default number of chunks to retrieve per query.
    """

    def __init__(self, retriever, generator, top_k: int = 5) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        logger.info(
            "RAGPipeline initialised. top_k=%d, retriever=%s, generator=%s",
            top_k,
            type(retriever).__name__,
            type(generator).__name__,
        )

    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        product_filter: Optional[str] = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a given question.

        Parameters
        ----------
        question : str
            User's natural-language question.
        top_k : int | None
            Override the default top_k for this query.
        product_filter : str | None
            Restrict retrieval to a specific product category.

        Returns
        -------
        RAGResponse
        """
        if not question or not question.strip():
            return RAGResponse(
                question=question,
                answer="Please provide a non-empty question.",
                sources=[],
                latency_seconds=0.0,
                error="Empty question",
            )

        k = top_k or self.top_k
        t0 = time.perf_counter()

        try:
            # Step 1: Retrieve relevant chunks
            logger.info("Retrieving top-%d chunks for: '%s'", k, question[:80])
            chunks = self.retriever.retrieve(
                query=question,
                top_k=k,
                product_filter=product_filter,
            )

            if not chunks:
                return RAGResponse(
                    question=question,
                    answer=(
                        "No relevant complaint data was found for your query. "
                        "Try rephrasing or selecting a different product filter."
                    ),
                    sources=[],
                    latency_seconds=time.perf_counter() - t0,
                    product_filter=product_filter,
                )

            # Step 2: Build prompt
            prompt = build_prompt(context_chunks=chunks, question=question)
            logger.debug("Prompt length: %d characters", len(prompt))

            # Step 3: Generate answer
            logger.info("Generating answer via LLM...")
            answer = self.generator.generate(prompt)

            latency = time.perf_counter() - t0
            logger.info("Query completed in %.2f seconds.", latency)

            return RAGResponse(
                question=question,
                answer=answer,
                sources=chunks,
                latency_seconds=latency,
                product_filter=product_filter,
            )

        except Exception as exc:
            logger.exception("RAG pipeline error: %s", exc)
            return RAGResponse(
                question=question,
                answer=f"An error occurred while processing your query: {exc}",
                sources=[],
                latency_seconds=time.perf_counter() - t0,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "RAGPipeline":
        """
        Construct a RAGPipeline from a configuration dictionary.

        Expected keys
        -------------
        retriever_backend : str          ('chromadb' | 'faiss')
        chroma_persist_dir : str
        chroma_collection : str
        faiss_index_path : str
        faiss_metadata_path : str
        generator_backend : str          ('hf_pipeline' | 'hf_hub')
        model_id : str
        max_new_tokens : int
        temperature : float
        top_k : int
        embedding_model : str
        device : int
        load_in_4bit : bool

        Example
        -------
        >>> config = {
        ...     "retriever_backend": "chromadb",
        ...     "chroma_persist_dir": "vector_store/chroma",
        ...     "chroma_collection": "cfpb_complaints",
        ...     "generator_backend": "hf_hub",
        ...     "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        ...     "max_new_tokens": 512,
        ...     "temperature": 0.1,
        ...     "top_k": 5,
        ... }
        >>> pipeline = RAGPipeline.from_config(config)
        """
        embedding_model = config.get("embedding_model", DEFAULT_MODEL_NAME)
        embedder = get_embedder(embedding_model)

        retriever = build_retriever(
            backend=config.get("retriever_backend", "chromadb"),
            embedder=embedder,
            chroma_persist_dir=config.get("chroma_persist_dir", "vector_store/chroma"),
            chroma_collection=config.get("chroma_collection", "cfpb_complaints"),
            faiss_index_path=config.get("faiss_index_path", "vector_store/faiss/complaints.faiss"),
            faiss_metadata_path=config.get("faiss_metadata_path", "vector_store/faiss/metadata.parquet"),
        )

        generator = build_generator(
            backend=config.get("generator_backend", "hf_hub"),
            model_id=config.get("model_id", "mistralai/Mistral-7B-Instruct-v0.2"),
            max_new_tokens=config.get("max_new_tokens", 512),
            temperature=config.get("temperature", 0.1),
            device=config.get("device", -1),
            load_in_4bit=config.get("load_in_4bit", False),
        )

        return cls(
            retriever=retriever,
            generator=generator,
            top_k=config.get("top_k", 5),
        )


# ---------------------------------------------------------------------------
# Default configuration (override via environment variables or config file)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "retriever_backend": "chromadb",
    "chroma_persist_dir": "vector_store/chroma",
    "chroma_collection": "cfpb_complaints",
    "generator_backend": "hf_hub",
    "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_k": 5,
    "embedding_model": DEFAULT_MODEL_NAME,
}