

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import get_text_splitter, chunk_complaint


def test_short_text_produces_one_chunk():
    splitter = get_text_splitter(chunk_size=500, chunk_overlap=50)
    text = "This is a short complaint narrative."
    result = chunk_complaint("c1", text, {"product_category": "Credit Card"}, splitter)
    assert len(result) == 1
    assert result[0]["text"] == text
    assert result[0]["metadata"]["complaint_id"] == "c1"
    assert result[0]["metadata"]["chunk_index"] == 0
    assert result[0]["metadata"]["total_chunks"] == 1


def test_long_text_produces_multiple_chunks():
    splitter = get_text_splitter(chunk_size=100, chunk_overlap=20)
    text = "word " * 200
    result = chunk_complaint("c2", text, {"product_category": "Personal Loan"}, splitter)
    assert len(result) > 1
    indices = [c["metadata"]["chunk_index"] for c in result]
    assert indices == list(range(len(result)))
    assert all(c["metadata"]["total_chunks"] == len(result) for c in result)


def test_metadata_is_preserved_and_attached_to_every_chunk():
    splitter = get_text_splitter(chunk_size=50, chunk_overlap=10)
    text = "a" * 200
    metadata = {"product_category": "Savings Account", "company": "Test Bank"}
    result = chunk_complaint("c3", text, metadata, splitter)
    for chunk in result:
        assert chunk["metadata"]["product_category"] == "Savings Account"
        assert chunk["metadata"]["company"] == "Test Bank"
        assert chunk["metadata"]["complaint_id"] == "c3"


def test_empty_text_produces_no_chunks():
    splitter = get_text_splitter()
    assert chunk_complaint("c4", "", {}, splitter) == []
    assert chunk_complaint("c5", None, {}, splitter) == []
    assert chunk_complaint("c6", "   ", {}, splitter) == []
