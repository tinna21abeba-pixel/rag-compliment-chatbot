"""Unit tests for src/text_cleaning.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from text_cleaning import clean_narrative, remove_boilerplate, word_count


def test_clean_narrative_lowercases():
    assert clean_narrative("THIS IS A TEST") == "this is a test"


def test_clean_narrative_removes_boilerplate():
    text = "I am writing to file a complaint about my credit card."
    result = clean_narrative(text)
    assert "i am writing to file a complaint" not in result
    assert "credit card" in result


def test_clean_narrative_removes_redaction_placeholders():
    text = "I called on XX/XX/2023 and spoke with XXXX about my account."
    result = clean_narrative(text)
    assert "xxxx" not in result
    assert "xx" not in result.split()


def test_clean_narrative_handles_empty_input():
    assert clean_narrative("") == ""
    assert clean_narrative(None) == ""
    assert clean_narrative("   ") == ""


def test_clean_narrative_removes_special_characters():
    text = "My balance was $500.00 & it's wrong!! #fraud @company"
    result = clean_narrative(text)
    assert "#" not in result
    assert "@" not in result
    assert "&" not in result


def test_remove_boilerplate_case_insensitive():
    text = "I am writing to file a complaint regarding fees."
    result = remove_boilerplate(text.lower())
    assert "i am writing to file a complaint" not in result


def test_word_count_basic():
    assert word_count("this is four words") == 4


def test_word_count_empty():
    assert word_count("") == 0
    assert word_count(None) == 0
