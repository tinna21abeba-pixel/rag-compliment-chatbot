"""
retriever.py
------------
Loads the persisted ChromaDB (or FAISS) vector store and retrieves the
top-k most semantically similar complaint chunks for a given query.

Design decisions:
  - Supports both ChromaDB and FAISS backends via a common interface
  - Returns rich metadata alongside chunk text so the UI can surface sources
  - Product-category filtering is optionally applied at retrieval time
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class for a single retrieved chunk
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    """Represents one complaint chunk returned by the retriever."""
    text: str
    complaint_id: str
    product_category: str
    product: str
    issue: str
    company: str
    state: str
    date_received: str
    chunk_index: int
    score: float
    metadata: dict = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Human-readable context string injected into the prompt."""
        return (
            f"[Complaint {self.complaint_id} | {self.product_category} | "
            f"{self.issue} | {self.company}]\n{self.text}"
        )


# ---------------------------------------------------------------------------
# ChromaDB Retriever
# ---------------------------------------------------------------------------
class ChromaRetriever:
    """
    Retriever backed by a persisted ChromaDB collection.

    Parameters
    ----------
    persist_dir : str | Path
        Directory where the ChromaDB collection is persisted.
    collection_name : str
        Name of the ChromaDB collection.
    embedder : QueryEmbedder
        Shared embedder instance (avoids duplicate model loads).
    """

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedder,
    ) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            ) from exc

        self.embedder = embedder
        persist_dir = str(Path(persist_dir).resolve())
        logger.info("Loading ChromaDB from: %s / %s", persist_dir, collection_name)

        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_collection(collection_name)
        logger.info(
            "ChromaDB collection loaded. Total documents: %d",
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        product_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-k complaint chunks most relevant to `query`.

        Parameters
        ----------
        query : str
            User's plain-English question.
        top_k : int
            Number of chunks to return.
        product_filter : str | None
            If provided, restrict results to this product category
            (e.g. "Credit Card").

        Returns
        -------
        list[RetrievedChunk]
        """
        query_vector = self.embedder.embed_query(query).tolist()

        where_clause = (
            {"product_category": {"$eq": product_filter}}
            if product_filter
            else None
        )

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[RetrievedChunk] = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB distances are L2; convert to a 0–1 similarity score
            score = float(1 / (1 + dist))
            chunks.append(
                RetrievedChunk(
                    text=text,
                    complaint_id=str(meta.get("complaint_id", "N/A")),
                    product_category=meta.get("product_category", "Unknown"),
                    product=meta.get("product", "Unknown"),
                    issue=meta.get("issue", "Unknown"),
                    company=meta.get("company", "Unknown"),
                    state=meta.get("state", "N/A"),
                    date_received=meta.get("date_received", "N/A"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=score,
                    metadata=meta,
                )
            )
        logger.debug("Retrieved %d chunks for query: '%s'", len(chunks), query[:60])
        return chunks


# ---------------------------------------------------------------------------
# FAISS Retriever
# ---------------------------------------------------------------------------
class FAISSRetriever:
    """
    Retriever backed by a persisted FAISS index + a parallel metadata store
    saved as a JSON or Parquet file.

    Parameters
    ----------
    index_path : str | Path
        Path to the .faiss index file.
    metadata_path : str | Path
        Path to a Parquet or JSON file containing per-chunk metadata,
        indexed by position (row i ↔ FAISS vector i).
    embedder : QueryEmbedder
    """

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        embedder,
    ) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        import pandas as pd

        self.embedder = embedder
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        logger.info("Loading FAISS index from: %s", index_path)
        self.index = faiss.read_index(str(index_path))

        logger.info("Loading FAISS metadata from: %s", metadata_path)
        if metadata_path.suffix == ".parquet":
            self.metadata_df = pd.read_parquet(metadata_path)
        else:
            self.metadata_df = pd.read_json(metadata_path)

        logger.info("FAISS index loaded. Vectors: %d", self.index.ntotal)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        product_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks using FAISS inner-product similarity search.
        """
        query_vector = self.embedder.embed_query(query).reshape(1, -1)

        # Increase k if filtering, so we have enough post-filter results
        search_k = top_k * 5 if product_filter else top_k
        distances, indices = self.index.search(query_vector, search_k)

        chunks: List[RetrievedChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS sentinel for "not enough results"
                continue
            row = self.metadata_df.iloc[idx]
            if product_filter and row.get("product_category") != product_filter:
                continue
            chunks.append(
                RetrievedChunk(
                    text=row.get("text", ""),
                    complaint_id=str(row.get("complaint_id", "N/A")),
                    product_category=row.get("product_category", "Unknown"),
                    product=row.get("product", "Unknown"),
                    issue=row.get("issue", "Unknown"),
                    company=row.get("company", "Unknown"),
                    state=row.get("state", "N/A"),
                    date_received=row.get("date_received", "N/A"),
                    chunk_index=int(row.get("chunk_index", 0)),
                    score=float(dist),
                    metadata=row.to_dict(),
                )
            )
            if len(chunks) >= top_k:
                break

        logger.debug("Retrieved %d chunks for query: '%s'", len(chunks), query[:60])
        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_retriever(
    backend: str,
    embedder,
    chroma_persist_dir: str = "vector_store/chroma",
    chroma_collection: str = "cfpb_complaints",
    faiss_index_path: str = "vector_store/faiss/complaints.faiss",
    faiss_metadata_path: str = "vector_store/faiss/metadata.parquet",
):
    """
    Factory function that returns the appropriate retriever based on `backend`.

    Parameters
    ----------
    backend : str
        'chromadb' or 'faiss'
    embedder : QueryEmbedder
    """
    backend = backend.lower().strip()
    if backend == "chromadb":
        return ChromaRetriever(
            persist_dir=chroma_persist_dir,
            collection_name=chroma_collection,
            embedder=embedder,
        )
    elif backend == "faiss":
        return FAISSRetriever(
            index_path=faiss_index_path,
            metadata_path=faiss_metadata_path,
            embedder=embedder,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'chromadb' or 'faiss'.")
