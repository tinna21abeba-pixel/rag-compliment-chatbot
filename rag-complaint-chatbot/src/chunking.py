"""
chunking.py

Text chunking utilities for splitting cleaned complaint narratives into
overlapping chunks suitable for embedding, using LangChain's
RecursiveCharacterTextSplitter.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter(chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter configured for complaint narratives.

    Defaults (chunk_size=500, chunk_overlap=50) match the spec used for the
    pre-built vector store provided for Tasks 3-4, so chunks produced here
    are directly comparable.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks,
        which helps preserve context across chunk boundaries.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_complaint(
    complaint_id: Any,
    text: str,
    metadata: Dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> List[Dict[str, Any]]:
    """
    Split a single complaint narrative into chunks with attached metadata.

    Parameters
    ----------
    complaint_id : Any
        Original complaint identifier (used to trace chunks back to source).
    text : str
        Cleaned narrative text to chunk.
    metadata : dict
        Additional metadata to attach to every chunk (product_category,
        product, issue, sub_issue, company, state, date_received, etc.)
    splitter : RecursiveCharacterTextSplitter
        Configured splitter instance (reuse one across the whole dataset
        for efficiency rather than creating a new one per complaint).

    Returns
    -------
    list of dict
        One dict per chunk with keys: 'text', 'metadata'. The metadata dict
        includes complaint_id, chunk_index, and total_chunks in addition to
        whatever was passed in.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    raw_chunks = splitter.split_text(text)
    total_chunks = len(raw_chunks)

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_metadata = dict(metadata)
        chunk_metadata.update(
            {
                "complaint_id": complaint_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
            }
        )
        chunks.append({"text": chunk_text, "metadata": chunk_metadata})

    return chunks
