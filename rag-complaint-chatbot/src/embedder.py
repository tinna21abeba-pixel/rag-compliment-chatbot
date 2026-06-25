"""
embedder.py
------------
Wrapper around Sentence-Transformers for query and document embedding.

This module provides a small adapter with a stable `embed_query`
method used by the retriever and supports a configurable default
model used throughout the RAG pipeline.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str):
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]

    def embed_documents(self, texts: List[str]):
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")
        return self.model.encode(texts, convert_to_numpy=True)


def get_embedder(model_name: str = DEFAULT_MODEL_NAME):
    return SentenceTransformerEmbedder(model_name)
