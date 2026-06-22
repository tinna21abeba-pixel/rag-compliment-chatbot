# Interim Report — Intelligent Complaint Analysis for Financial Services

**CrediTrust Financial — RAG-Powered Complaint Chatbot**
**Submission:** Interim (Task 1 & Task 2)

---

## 1. Task 1: Exploratory Data Analysis and Preprocessing

### 1.1 Approach

We loaded the full CFPB consumer complaint dataset and performed an
initial exploratory pass before any filtering, to understand the overall
shape of the data: the relative volume of complaints by product, how many
complaints include a free-text narrative versus metadata only, and the
distribution of narrative lengths. This ordering matters — filtering
first would have hidden whether the four target product categories
(Credit Card, Personal Loan, Savings Account, Money Transfer) are
representative of the broader complaint landscape, or a small, unusual
slice of it.

After the initial EDA, we filtered the dataset down to records belonging
to the four target product categories with a non-empty consumer
complaint narrative, then cleaned the narrative text: lowercasing,
stripping CFPB's redaction placeholders (e.g. `XXXX`, `XX/XX/2023`),
removing common complaint-opening boilerplate (e.g. "I am writing to
file a complaint..."), removing special characters, and collapsing
whitespace. The cleaning logic lives in `src/text_cleaning.py` as
standalone, unit-tested functions rather than inline notebook code, so
the same logic can be reused without duplication during Task 2's
chunking step and verified independently of any specific dataset.

### 1.2 Key EDA Findings

*(Run `notebooks/01_eda_preprocessing.py` against the real CFPB dataset
and replace the placeholders below with your actual numbers before
submitting — the script saves the underlying charts to
`data/processed/` automatically.)*

1. **Product distribution.** [Fill in: e.g., "Credit Card and Debt
   Collection were the two largest categories in the full dataset,
   together accounting for X% of all complaints. Within the four
   target categories, Credit Card complaints were the most numerous at
   N records, followed by..."]

2. **Narrative availability.** [Fill in: e.g., "X% of all complaints in
   the full dataset included a free-text consumer narrative; the
   remainder were metadata-only submissions (e.g., routed directly to a
   company without a public narrative). This sets the practical ceiling
   on how large our retrieval corpus can be before any product
   filtering is applied."]

3. **Narrative length.** [Fill in: e.g., "Narrative word counts had a
   median of N words with most falling between N and N words; a small
   number of outliers exceeded 1,000 words. Given a chunk size of 500
   characters (~80-100 words), most narratives will produce 1-3 chunks,
   which keeps the chunk-to-complaint traceability manageable."]

### 1.3 Output

The cleaned, filtered dataset is saved to `data/filtered_complaints.csv`
with columns: `complaint_id`, `product_category`, `product`, `issue`,
`sub_issue`, `company`, `state`, `date_received`,
`consumer_complaint_narrative` (original), `cleaned_narrative`,
`cleaned_word_count`.

---

## 2. Task 2: Text Chunking, Embedding, and Vector Store Indexing

### 2.1 Sampling Strategy

We drew a stratified sample of **12,000 complaints** (within the
10,000–15,000 range specified) from the cleaned dataset, with sample
size per product category proportional to that category's share of the
full cleaned dataset. We chose proportional stratification over an
equal-size-per-category split because the goal of this sample is to
let us validate the chunking/embedding/indexing pipeline on a
realistic, representative subset — not to artificially balance product
categories. An equal split would distort retrieval behavior relative to
the full dataset, since rarer complaint types would be over-represented
and common ones under-represented relative to real complaint volume.

### 2.2 Chunking Approach

We used LangChain's `RecursiveCharacterTextSplitter` with:

- **`chunk_size = 500`** characters
- **`chunk_overlap = 50`** characters (10% of chunk size)

These values were chosen to match the specification of the pre-built
vector store provided for Tasks 3–4, so that our own pipeline's output
is directly comparable and our retrieval code can be reused without
modification later. Independently, 500 characters (roughly 80–100
words) struck a reasonable balance for this domain: long enough to
preserve a complete complaint detail or grievance, short enough that
each chunk stays topically focused rather than blending multiple
unrelated points into one embedding. The 50-character overlap reduces
the risk of losing context at a chunk boundary — for example, a
sentence describing the resolution a customer was denied shouldn't be
split away from the sentence describing what they originally asked for.

The chunking logic is implemented in `src/chunking.py` and is unit
tested (`tests/test_chunking.py`) covering: single-chunk short text,
multi-chunk long text, metadata propagation to every chunk, and
empty-text edge cases.

### 2.3 Embedding Model

We used **`sentence-transformers/all-MiniLM-L6-v2`**, producing
384-dimensional embeddings. This was chosen for three reasons:

1. **Resource efficiency.** At ~80MB and 384 dimensions, it encodes
   12,000+ chunks in a reasonable time on standard CPU hardware — this
   matters directly for the assignment's stated reason for the two-track
   approach (full 464K-complaint embedding would take hours).
2. **Quality for short text.** It performs well on semantic textual
   similarity benchmarks relative to its size, which fits this use case
   well since complaint narrative chunks are short, single-topic spans
   of text rather than long documents.
3. **Consistency with the pre-built vector store.** Using the same model
   that was used to build the full 1.37M-chunk pre-built ChromaDB store
   (per the assignment spec) means our embeddings here are directly
   comparable, and the retrieval function we write in Task 3 will work
   against either vector store without changes.

### 2.4 Vector Store

We used **ChromaDB**, persisted locally to the `vector_store/`
directory, over FAISS, for two practical reasons: ChromaDB handles
on-disk persistence and metadata storage/filtering natively, while FAISS
requires more manual bookkeeping to associate vectors with metadata and
to persist/reload an index. Since the assignment also distributes the
full-dataset pre-built store in ChromaDB format, standardizing on it now
avoids having to maintain two different retrieval code paths later.

Each chunk is stored with:
- Its embedding vector
- The chunk text itself (as the Chroma "document")
- Metadata: `complaint_id`, `product_category`, `product`, `issue`,
  `sub_issue`, `company`, `state`, `date_received`, `chunk_index`,
  `total_chunks`

This metadata schema mirrors the pre-built vector store's specification
exactly, so retrieved chunks can always be traced back to their source
complaint, and filtering by product category or date range will work
consistently whether querying our own sample store or the full-dataset
store in later tasks.

### 2.5 Validation

We validated the full Task 2 pipeline (stratified sampling → chunking →
metadata attachment → ChromaDB indexing → similarity query) end-to-end
on synthetic data structured like Task 1's output, confirming:
correct proportional sampling across product categories, correct
chunk/metadata generation, correct batch indexing into ChromaDB, and
correct retrieval via a test query. All chunking and text-cleaning unit
tests pass (12/12). The embedding model download itself
(`all-MiniLM-L6-v2` from Hugging Face) requires a standard internet
connection on the machine actually running the pipeline.

---

## 3. Next Steps (Final Submission)

- **Task 3**: Build the retriever (embed query → similarity search
  top-k) and generator (prompt template + LLM call) using the
  pre-built full-dataset vector store; run the qualitative evaluation
  with 5–10 representative questions.
- **Task 4**: Build the Gradio/Streamlit chat interface with source
  display and a clear/reset button.
