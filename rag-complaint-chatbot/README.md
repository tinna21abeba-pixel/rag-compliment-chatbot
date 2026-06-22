# RAG Complaint Chatbot — CrediTrust Financial

An internal Retrieval-Augmented Generation (RAG) tool that lets Product,
Support, and Compliance teams at CrediTrust Financial ask plain-English
questions about customer complaints and get synthesized, evidence-backed
answers — instead of manually reading through thousands of complaint
narratives.

## Status: Interim Submission (Task 1 + Task 2)

This submission covers:
- **Task 1** — EDA and preprocessing of the full CFPB complaint dataset
- **Task 2** — Stratified sampling, text chunking, embedding, and ChromaDB
  vector store indexing

Tasks 3 (RAG pipeline + evaluation) and 4 (Gradio/Streamlit UI) will be
completed for the final submission.

## Project Structure

```
rag-complaint-chatbot/
├── .github/workflows/unittests.yml   # CI: runs pytest on push/PR
├── data/
│   ├── raw/                          # Place the downloaded CFPB CSV here
│   └── processed/                    # EDA chart outputs land here
├── notebooks/
│   ├── 01_eda_preprocessing.py       # Task 1 (jupytext "light" format)
│   └── 02_chunking_embedding_indexing.py  # Task 2
├── src/
│   ├── text_cleaning.py              # Narrative cleaning utilities
│   └── chunking.py                   # Text chunking utilities
├── tests/
│   ├── test_text_cleaning.py
│   └── test_chunking.py
├── vector_store/                     # Persisted ChromaDB collection (generated)
├── requirements.txt
└── reports/interim_report.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

### 1. Get the data
Download the CFPB complaint dataset and place it at:
```
data/raw/complaints.csv
```

### 2. Run Task 1 — EDA & Preprocessing

The notebook files (`notebooks/*.py`) are written in
[Jupytext "light" format](https://jupytext.readthedocs.io/) — plain Python
scripts with `# %%` cell markers. You have two options:

**Option A — Open directly as a notebook in VS Code/Jupyter (recommended):**
```bash
pip install jupytext
jupytext --to notebook notebooks/01_eda_preprocessing.py
jupyter notebook notebooks/01_eda_preprocessing.ipynb
```

**Option B — Run as a plain script:**
```bash
cd notebooks
python 01_eda_preprocessing.py
```

This produces `data/filtered_complaints.csv` and chart images in
`data/processed/`.

### 3. Run Task 2 — Chunking, Embedding, Indexing

```bash
cd notebooks
jupytext --to notebook 02_chunking_embedding_indexing.py
jupyter notebook 02_chunking_embedding_indexing.ipynb
```

or as a script:
```bash
cd notebooks
python 02_chunking_embedding_indexing.py
```

This produces a persisted ChromaDB collection in `vector_store/`. The
first run downloads the `all-MiniLM-L6-v2` model (~80MB) from Hugging
Face, so an internet connection is required.

### 4. Run tests

```bash
pytest tests/ -v
```

## Key Design Decisions

See `reports/interim_report.md` for the full write-up. Summary:

- **Sampling**: stratified by `product_category`, proportional to each
  category's real share of the cleaned dataset (not equal-sized buckets),
  to keep the sample's complaint mix representative.
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`,
  `chunk_size=500`, `chunk_overlap=50` — matching the spec of the
  pre-built vector store used in later tasks, for consistency.
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` — small
  (384-dim, ~80MB), fast on CPU, strong semantic-similarity performance
  for its size, and matches the pre-built vector store's model so our
  pipeline is directly comparable.
- **Vector store**: ChromaDB — simpler local persistence and built-in
  metadata filtering compared to managing a raw FAISS index by hand.
