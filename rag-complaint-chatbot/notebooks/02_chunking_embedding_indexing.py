# %% [markdown]
# # Task 2: Text Chunking, Embedding, and Vector Store Indexing
#
# **CrediTrust Financial — Intelligent Complaint Analysis**
#
# Objective: Convert the cleaned complaint narratives from Task 1 into a
# stratified sample, chunk them, embed each chunk, and index everything
# in a persisted ChromaDB vector store.
#
# **Input:**  `data/filtered_complaints.csv` (output of Task 1)
# **Output:** `vector_store/` (persisted ChromaDB collection)

# %%
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import pandas as pd
import numpy as np
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import get_text_splitter, chunk_complaint

INPUT_PATH = "../data/filtered_complaints.csv"
VECTOR_STORE_DIR = "../vector_store"
COLLECTION_NAME = "complaint_chunks"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SAMPLE_SIZE = 12000  # within the 10,000-15,000 range specified in the assignment
RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load the cleaned dataset from Task 1

# %%
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df):,} cleaned complaints")
print(df["product_category"].value_counts())

# %% [markdown]
# ## 2. Stratified sampling
#
# **Sampling strategy:** We draw a sample proportional to each product
# category's share of the cleaned dataset, capped at `SAMPLE_SIZE` total.
# This preserves the real-world product mix (e.g. if Credit Card
# complaints are 40% of the cleaned data, they'll be ~40% of the sample)
# rather than forcing equal counts per category, which would distort the
# retrieval corpus relative to actual complaint volume.

# %%
def stratified_sample(data: pd.DataFrame, total_size: int, strat_col: str, random_state: int) -> pd.DataFrame:
    """Sample `total_size` rows from `data`, proportional to the
    distribution of `strat_col`."""
    proportions = data[strat_col].value_counts(normalize=True)
    sampled_frames = []
    for category, proportion in proportions.items():
        n_for_category = max(1, round(total_size * proportion))
        category_df = data[data[strat_col] == category]
        n_for_category = min(n_for_category, len(category_df))
        sampled_frames.append(
            category_df.sample(n=n_for_category, random_state=random_state)
        )
    return pd.concat(sampled_frames).sample(frac=1, random_state=random_state).reset_index(drop=True)


sample_df = stratified_sample(df, SAMPLE_SIZE, "product_category", RANDOM_STATE)
print(f"Sampled {len(sample_df):,} complaints")
print(sample_df["product_category"].value_counts())
print("\nProportions match original distribution:")
print(
    pd.DataFrame(
        {
            "original": df["product_category"].value_counts(normalize=True),
            "sample": sample_df["product_category"].value_counts(normalize=True),
        }
    )
)

# %% [markdown]
# ## 3. Text chunking
#
# **Chunking approach:** We use LangChain's `RecursiveCharacterTextSplitter`
# with `chunk_size=500` and `chunk_overlap=50` characters. This matches the
# specification of the pre-built vector store used later in Tasks 3-4,
# keeping our own experiments directly comparable.
#
# Why these values:
# - **500 characters** (~80-100 words) is long enough to preserve a
#   complete thought/complaint detail, but short enough that each chunk
#   stays topically focused for embedding — large chunks dilute the
#   semantic signal with multiple unrelated points.
# - **50 character overlap** (10% of chunk size) prevents losing context
#   at chunk boundaries (e.g. a sentence split mid-way) without
#   significantly inflating the index size.

# %%
splitter = get_text_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

all_chunks = []
for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Chunking complaints"):
    metadata = {
        "product_category": row.get("product_category", ""),
        "product": str(row.get("product", "")),
        "issue": str(row.get("issue", "")),
        "sub_issue": str(row.get("sub_issue", "")),
        "company": str(row.get("company", "")),
        "state": str(row.get("state", "")),
        "date_received": str(row.get("date_received", "")),
    }
    chunks = chunk_complaint(
        complaint_id=row.get("complaint_id", row.name),
        text=row["cleaned_narrative"],
        metadata=metadata,
        splitter=splitter,
    )
    all_chunks.extend(chunks)

print(f"Produced {len(all_chunks):,} chunks from {len(sample_df):,} complaints")
print(f"Average chunks per complaint: {len(all_chunks) / len(sample_df):.2f}")

# %% [markdown]
# ## 4. Embedding model
#
# **Model choice: `sentence-transformers/all-MiniLM-L6-v2`**
#
# Reasons:
# - **Speed/size tradeoff**: only 384 dimensions and ~80MB, so encoding
#   12K+ chunks finishes in minutes on CPU — important since this sample
#   pipeline needs to run on standard hardware.
# - **Quality**: it's a well-established sentence-embedding model with
#   strong performance on semantic similarity benchmarks relative to its
#   size, making it a sensible default for short complaint-narrative text.
# - **Consistency**: this is the same model used to build the pre-built
#   vector store for Tasks 3-4, so our own embeddings here are directly
#   comparable and our retrieval code transfers without changes.

# %%
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chunk_texts = [c["text"] for c in all_chunks]

print(f"Encoding {len(chunk_texts):,} chunks...")
embeddings = embedding_model.encode(
    chunk_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Embeddings shape: {embeddings.shape}")

# %% [markdown]
# ## 5. Build and persist the ChromaDB vector store
#
# Each chunk is stored with its embedding plus metadata (complaint_id,
# product_category, product, issue, sub_issue, company, state,
# date_received, chunk_index, total_chunks) so retrieved chunks can always
# be traced back to their source complaint.

# %%
client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

# Drop any existing collection with this name so re-running this script
# doesn't duplicate entries.
existing_collections = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing_collections:
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"embedding_model": EMBEDDING_MODEL_NAME, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
)

# Chroma requires string IDs and JSON-serializable metadata (no NaN/None
# for numeric-looking fields), so we sanitize before inserting.
ids = [f"{c['metadata']['complaint_id']}_{c['metadata']['chunk_index']}" for c in all_chunks]
metadatas = []
for c in all_chunks:
    clean_meta = {k: ("" if v is None else v) for k, v in c["metadata"].items()}
    metadatas.append(clean_meta)

BATCH_SIZE = 500
for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Indexing into ChromaDB"):
    batch_ids = ids[i : i + BATCH_SIZE]
    batch_texts = chunk_texts[i : i + BATCH_SIZE]
    batch_embeddings = embeddings[i : i + BATCH_SIZE].tolist()
    batch_metadatas = metadatas[i : i + BATCH_SIZE]

    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_texts,
        metadatas=batch_metadatas,
    )

print(f"Indexed {collection.count():,} chunks into ChromaDB collection '{COLLECTION_NAME}'")
print(f"Vector store persisted at: {VECTOR_STORE_DIR}")

# %% [markdown]
# ## 6. Sanity check: run a test query

# %%
test_query = "Why are customers unhappy with credit card billing?"
test_embedding = embedding_model.encode([test_query]).tolist()

results = collection.query(
    query_embeddings=test_embedding,
    n_results=3,
)

for i, (doc, meta, dist) in enumerate(
    zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
):
    print(f"\n--- Result {i+1} (distance={dist:.4f}) ---")
    print(f"Product: {meta.get('product_category')} | Complaint ID: {meta.get('complaint_id')}")
    print(f"Text: {doc[:200]}...")
