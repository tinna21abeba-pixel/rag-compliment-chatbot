"""
text_cleaning.py

Utility functions for cleaning and normalizing CFPB consumer complaint
narratives. Kept separate from the EDA notebook so the logic is testable
and reusable in Task 2 (chunking) without copy-pasting.
"""

import re

# Common boilerplate openers consumers use when filing complaints.
# These add no signal for retrieval/embedding and are stripped out.
BOILERPLATE_PATTERNS = [
    r"i am writing to file a complaint\s*",
    r"i am writing to (file|submit) a complaint (about|regarding|against)\s*",
    r"this letter is to inform you\s*",
    r"to whom it may concern[,:]?\s*",
    r"i would like to (file|lodge) a complaint\s*",
    r"i am writing this complaint\s*",
    r"this is regarding\s*",
]

# Pattern to redact common PII placeholders CFPB already masks
# (e.g. "XX/XX/2023", "XXXX") so they don't pollute embeddings as tokens.
REDACTION_PATTERN = re.compile(r"\bx{2,}\b", re.IGNORECASE)


def remove_boilerplate(text: str) -> str:
    """Strip common complaint-opening boilerplate phrases."""
    cleaned = text
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def clean_narrative(text: str) -> str:
    """
    Clean a single consumer complaint narrative.

    Steps:
      1. Lowercase the text.
      2. Remove CFPB redaction placeholders (e.g. 'XXXX', 'XX/XX/XXXX').
      3. Remove boilerplate complaint-opening phrases.
      4. Remove special characters (keep letters, numbers, basic punctuation).
      5. Collapse repeated whitespace.
      6. Strip leading/trailing whitespace.

    Parameters
    ----------
    text : str
        Raw consumer complaint narrative.

    Returns
    -------
    str
        Cleaned narrative text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text.lower()
    cleaned = REDACTION_PATTERN.sub("", cleaned)
    cleaned = remove_boilerplate(cleaned)

    # Keep letters, digits, whitespace, and basic sentence punctuation.
    cleaned = re.sub(r"[^a-z0-9\s.,!?'%$-]", " ", cleaned)

    # Collapse multiple spaces/newlines into a single space.
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())
